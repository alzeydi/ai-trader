"""Order execution: paper or live.

`OrderExecutor` is the only place that talks to the Binance REST API for
order placement. In paper mode it simulates a fill at `entry_price` and
records the trade in the DB; live mode also records the trade after sending
the bracket (entry market + reduce-only stop + reduce-only take-profit).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from sqlalchemy import select

import ccxt

from src.config import settings
from src.execution.types import ExecutionResult, TradeOrder
from src.llm.veto import invalidate_cooldown
from src.persistence.db import SessionFactory, Trade, make_session_factory

if TYPE_CHECKING:
    from src.data.binance_client import BinanceClient

log = logging.getLogger(__name__)


def _is_reduce_only(o: dict[str, Any]) -> bool:
    info = o.get("info") or {}
    if o.get("reduceOnly") is True or info.get("reduceOnly") in (True, "true"):
        return True
    # Binance fapi sometimes only fills `closePosition` or carries the flag
    # under `ro`; treat anything that can only reduce a position as such.
    if info.get("closePosition") in (True, "true"):
        return True
    return False


def _is_conditional(o: dict[str, Any]) -> bool:
    info = o.get("info") or {}
    otype = (o.get("type") or info.get("type") or "").upper()
    return any(
        tag in otype
        for tag in ("STOP", "TAKE_PROFIT", "TRAILING_STOP")
    )


def _fetch_open_orders_with_retry(ex, symbol: str, attempts: int = 3) -> list[dict[str, Any]] | None:
    """Returns None on terminal failure (caller should NOT proceed assuming clean state)."""
    last_exc: Exception | None = None
    delay = 0.5
    for i in range(attempts):
        try:
            return ex.fetch_open_orders(symbol)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            time.sleep(delay)
            delay *= 2
    log.warning("fetch_open_orders %s failed after %d attempts: %s",
                symbol, attempts, last_exc)
    return None


# ── Algo (conditional) order endpoints ─────────────────────────────────────
#
# 2025-12-09: Binance USDⓈ-M Futures migrated all conditional orders
# (STOP_MARKET, TAKE_PROFIT_MARKET, TRAILING_STOP_MARKET, STOP, TAKE_PROFIT)
# from the legacy /fapi/v1/order endpoint to a separate "Algo Service" at
# /fapi/v1/algoOrder. The legacy bulk endpoint /fapi/v1/allOpenOrders
# (used by ccxt's `cancel_all_orders`) does NOT touch this new bucket, and
# `fetch_open_orders` doesn't return entries from it either. That is why
# orphan SL/TP keep piling up on a symbol after a position closes — the
# orders are physically there but in a bucket our cleanup never reads.
#
# These helpers hit the new endpoints directly. They prefer ccxt's implicit
# methods when available (newer versions register them as
# `fapiPrivateDeleteAlgoOpenOrders` etc.) and fall back to a raw signed
# `request()` call if not.

def _algo_request(
    ex,
    path: str,
    method: str,
    params: dict[str, Any],
) -> Any:
    """Invoke an `fapiPrivate` algo endpoint, preferring ccxt's implicit
    method if registered, else falling back to the generic `request`.
    Re-raises on failure so the caller can decide how to log/recover.
    """
    method_name = f"fapiPrivate{method.title()}{path[0].upper()}{path[1:]}"
    impl = getattr(ex, method_name, None)
    if impl is not None:
        return impl(params)
    return ex.request(path, api="fapiPrivate", method=method, params=params)


def _cancel_algo_open_orders(ex, symbol: str) -> bool:
    """Bulk-cancel algo (conditional) orders on `symbol`.

    Endpoint: DELETE /fapi/v1/algoOpenOrders. Returns True on success.
    """
    try:
        market = ex.market(symbol)
    except Exception as exc:  # noqa: BLE001
        log.warning("algo cancel: market lookup %s failed: %s", symbol, exc)
        return False
    try:
        _algo_request(ex, "algoOpenOrders", "DELETE", {"symbol": market["id"]})
        log.info("algoOpenOrders DELETE OK on %s", symbol)
        return True
    except Exception as exc:  # noqa: BLE001
        log.warning("algoOpenOrders DELETE %s failed: %s", symbol, exc)
        return False


def _fetch_algo_open_orders(ex, symbol: str | None = None) -> list[dict[str, Any]]:
    """List open algo orders. With `symbol=None`, returns across the whole
    account. Always returns a list — empty on any failure or missing
    endpoint, since the caller can degrade gracefully (regular bucket
    is still being cleaned).
    """
    params: dict[str, Any] = {}
    if symbol is not None:
        try:
            market = ex.market(symbol)
        except Exception as exc:  # noqa: BLE001
            log.warning("algo fetch: market lookup %s failed: %s", symbol, exc)
            return []
        params["symbol"] = market["id"]
    try:
        resp = _algo_request(ex, "openAlgoOrders", "GET", params)
    except Exception as exc:  # noqa: BLE001
        log.debug("openAlgoOrders GET (%s) failed: %s", symbol or "ALL", exc)
        return []
    # Binance wraps the array as {"orders": [...]}; some ccxt versions
    # may already unwrap it. Handle both.
    if isinstance(resp, dict):
        orders = resp.get("orders") or resp.get("data") or []
    elif isinstance(resp, list):
        orders = resp
    else:
        orders = []
    return list(orders)


def _algo_id(o: dict[str, Any]) -> str:
    """Pull algoId out of either the unified ccxt shape or raw Binance
    response shape. Returns "" if missing.
    """
    info = o.get("info") if isinstance(o.get("info"), dict) else None
    for src in (o, info):
        if not src:
            continue
        for key in ("algoId", "algoID", "id"):
            v = src.get(key)
            if v not in (None, ""):
                return str(v)
    return ""


def _cancel_algo_order(ex, symbol: str, algo_id: str) -> bool:
    try:
        market = ex.market(symbol)
    except Exception:  # noqa: BLE001
        return False
    try:
        _algo_request(
            ex, "algoOrder", "DELETE",
            {"symbol": market["id"], "algoId": algo_id},
        )
        return True
    except Exception as exc:  # noqa: BLE001
        log.warning("algoOrder DELETE %s/%s failed: %s", symbol, algo_id, exc)
        return False


def cancel_symbol_conditionals(
    ex,
    symbol: str,
    *,
    keep_ids: tuple[str, ...] = (),
    only_stop: bool = False,
) -> int:
    """Cancel every reduce-only conditional order on `symbol`.

    Returns the count cancelled. `keep_ids` are skipped (e.g. the freshly
    placed SL during a trail tick). With `only_stop=True`, leaves
    TAKE_PROFIT orders alone — used by trailing-stop replacement.

    Hits BOTH Binance buckets:
      * legacy `cancel_all_orders` (DELETE /fapi/v1/allOpenOrders) for any
        residual non-algo orders, plus
      * the new algo endpoint (DELETE /fapi/v1/algoOpenOrders) which is
        where STOP_MARKET / TAKE_PROFIT_MARKET / TRAILING_STOP_MARKET have
        lived since the 2025-12-09 migration.

    Without the algo path, the legacy bulk silently succeeds while leaving
    every SL/TP intact — the symptom we kept seeing as orphan SL/TP stacks.

    Falls back to per-order cancel when keep_ids or only_stop forbid bulk,
    routing each id to the correct endpoint by bucket.
    """
    use_bulk = not keep_ids and not only_stop
    if use_bulk:
        try:
            ex.cancel_all_orders(symbol)
            log.info("cancel_all_orders OK on %s", symbol)
        except Exception as exc:  # noqa: BLE001
            log.warning("cancel_all_orders %s failed, falling back: %s",
                        symbol, exc)
        # The new algo bucket is independent of the call above; always hit
        # it too. A failure here is not fatal — the per-id sweep below acts
        # as the second line of defence.
        _cancel_algo_open_orders(ex, symbol)

    # Verification fetch from BOTH buckets so per-id cleanup catches
    # whatever survived the bulk passes (or whatever the bulk skipped
    # because keep_ids / only_stop disallowed it).
    legacy_orders = _fetch_open_orders_with_retry(ex, symbol)
    algo_orders = _fetch_algo_open_orders(ex, symbol)
    if legacy_orders is None:
        # Couldn't enumerate the legacy bucket. Bulk was already attempted
        # at the top when use_bulk was True. With keep_ids/only_stop we
        # MUST NOT bulk-cancel here — that would nuke the orders we wanted
        # to keep. Continue with whatever algo info we have; report -1 if
        # algo also empty so the caller knows status is unknown.
        legacy_orders = []
        if not algo_orders:
            return -1

    cancelled = 0

    # Legacy (regular) order book — typically empty for conditionals after
    # 2025-12-09 but may still hold compatibility-shimmed entries.
    for o in legacy_orders:
        oid = str(o.get("id") or "")
        if not oid or oid in keep_ids:
            continue
        if not _is_reduce_only(o) or not _is_conditional(o):
            continue
        if only_stop:
            info = o.get("info") or {}
            otype = (o.get("type") or info.get("type") or "").upper()
            if "TAKE_PROFIT" in otype:
                continue
        try:
            ex.cancel_order(oid, symbol)
            cancelled += 1
        except Exception as exc:  # noqa: BLE001
            log.warning("cancel_order %s on %s failed: %s", oid, symbol, exc)

    # Algo bucket — where the bulk of orphan SL/TP actually lives now.
    for o in algo_orders:
        aid = _algo_id(o)
        if not aid or aid in keep_ids:
            continue
        info = o.get("info") if isinstance(o.get("info"), dict) else o
        otype = str(info.get("type") or o.get("type") or "").upper()
        if not any(tag in otype for tag in ("STOP", "TAKE_PROFIT", "TRAILING_STOP")):
            continue
        if only_stop and "TAKE_PROFIT" in otype:
            continue
        # Algo orders are reduce-only by construction in our code path; we
        # don't gate on the flag here because Binance's response shape
        # does not consistently expose it.
        if _cancel_algo_order(ex, symbol, aid):
            cancelled += 1

    if cancelled:
        log.info("cancelled %d conditional order(s) on %s", cancelled, symbol)
    return cancelled


@dataclass
class OrderExecutor:
    client: "BinanceClient | None" = None
    paper: bool = field(default_factory=lambda: settings.paper_trading)
    session_factory: SessionFactory = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.session_factory is None:
            self.session_factory = make_session_factory()

    # ------------------------------------------------------------------
    def open_position(self, order: TradeOrder) -> ExecutionResult:
        # Idempotency: never stack a second position on a symbol that
        # already has an open trade in the DB. Without this guard, a
        # cached LLM TAKE verdict (cooldown) re-fires open_position every
        # cycle, drains margin, and the second call's SL placement may
        # leave the first position naked.
        with self.session_factory() as s:
            existing = s.execute(
                select(Trade)
                .where(Trade.symbol == order.symbol)
                .where(Trade.closed_at.is_(None))
                .order_by(Trade.opened_at.desc())
            ).scalars().first()
            if existing is not None:
                log.info(
                    "skip open: %s already has open trade id=%d (idempotent)",
                    order.symbol, existing.id,
                )
                return ExecutionResult(
                    success=False,
                    paper=self.paper,
                    trade_id=existing.id,
                    reason="already_open",
                )
        if self.paper:
            return self._open_paper(order)
        return self._open_live(order)

    def close_position(self, symbol: str, reason: str) -> ExecutionResult:
        if self.paper:
            return self._close_paper(symbol, reason)
        return self._close_live(symbol, reason)

    # --- paper -------------------------------------------------------
    def _open_paper(self, order: TradeOrder) -> ExecutionResult:
        trade_id = self._record_open(order, paper=True)
        log.info(
            "PAPER OPEN %s %s qty=%.6f entry=%.4f sl=%.4f tp=%.4f trade_id=%s",
            order.symbol, order.side, order.quantity,
            order.entry_price, order.stop_loss, order.take_profit, trade_id,
        )
        return ExecutionResult(
            success=True,
            paper=True,
            trade_id=trade_id,
            fill_price=order.entry_price,
            reason="paper-fill",
        )

    def _close_paper(self, symbol: str, reason: str) -> ExecutionResult:
        with self.session_factory() as s:
            row = s.execute(
                select(Trade)
                .where(Trade.symbol == symbol)
                .where(Trade.closed_at.is_(None))
                .order_by(Trade.opened_at.desc())
            ).scalars().first()
            if row is None:
                return ExecutionResult(
                    success=False, paper=True, reason=f"no open paper trade for {symbol}"
                )

            close_price = self._mark_price(symbol) or row.entry
            gross_pnl = self._compute_pnl(
                side=row.side, qty=row.quantity, entry=row.entry, exit_=close_price
            )
            # Mirror live + backtest accounting: round-trip taker fees so
            # paper, live and backtest produce comparable PnL on the same
            # trades. Without this, paper systematically over-estimates
            # net returns vs the other two paths.
            taker_pct = float(settings.taker_fee_pct)
            fees = (
                float(row.quantity) * float(row.entry) * taker_pct
                + float(row.quantity) * float(close_price) * taker_pct
            )
            pnl = gross_pnl - fees
            entry_notional = float(row.entry) * float(row.quantity)
            pnl_pct = (gross_pnl / entry_notional * 100.0) if entry_notional > 0 else 0.0
            row.closed_at = datetime.now(tz=timezone.utc)
            row.close_price = close_price
            row.close_reason = reason
            row.pnl_usd = pnl
            row.pnl_pct = pnl_pct
            row.fees = fees
            s.commit()
            trade_id = row.id

        log.info(
            "PAPER CLOSE %s reason=%s exit=%.4f gross=%.2f fees=%.2f net_pnl=%.2f "
            "trade_id=%s",
            symbol, reason, close_price, gross_pnl, fees, pnl, trade_id,
        )
        return ExecutionResult(
            success=True,
            paper=True,
            trade_id=trade_id,
            fill_price=close_price,
            pnl_usd=pnl,
            reason=reason,
        )

    # --- live --------------------------------------------------------
    def _open_live(self, order: TradeOrder) -> ExecutionResult:
        if self.client is None:
            raise RuntimeError("OrderExecutor in live mode requires a BinanceClient")
        ex = self.client.exchange

        try:
            ex.set_leverage(order.leverage, order.symbol)
        except Exception as exc:  # noqa: BLE001
            log.warning("set_leverage failed for %s: %s", order.symbol, exc)

        # Pre-open sweep: idempotency already ensured no DB row is open for
        # this symbol, so any reduce-only order still on the book is an
        # orphan from a prior position. Wipe everything via bulk cancel so
        # a brand-new position can't inherit a stale SL/TP.
        cancel_symbol_conditionals(ex, order.symbol)

        # Pre-flight margin check: sizing already caps notional by free
        # margin, but the wallet can have moved (other open positions, fees)
        # between sizing and execution. Skip cleanly instead of blowing up
        # with a -2019 traceback.
        required_margin = (order.quantity * order.entry_price) / max(order.leverage, 1)
        try:
            bal = ex.fetch_balance()
            usdt = (bal.get("USDT") or {}) if isinstance(bal, dict) else {}
            free = float(usdt.get("free") or 0.0)
        except Exception as exc:  # noqa: BLE001
            log.warning("pre-flight fetch_balance failed for %s: %s", order.symbol, exc)
            free = float("inf")  # don't block on a transient balance error
        if free < required_margin:
            log.info(
                "skip open %s: insufficient margin (need=%.2f free=%.2f)",
                order.symbol, required_margin, free,
            )
            return ExecutionResult(
                success=False,
                paper=False,
                reason=f"insufficient_margin: need={required_margin:.2f} free={free:.2f}",
            )

        entry_side = "buy" if order.side == "long" else "sell"
        exit_side = "sell" if order.side == "long" else "buy"

        try:
            entry_resp = ex.create_order(
                order.symbol, "market", entry_side, order.quantity,
                None, {"reduceOnly": False},
            )
        except ccxt.InsufficientFunds as exc:
            # Race with another fill that consumed margin between the
            # pre-flight check and this call. Treat as soft skip rather than
            # an exception so the trader cycle can continue with the rest of
            # the universe.
            log.info("skip open %s: exchange reported insufficient funds: %s",
                     order.symbol, exc)
            return ExecutionResult(
                success=False, paper=False, reason=f"insufficient_funds: {exc}"
            )
        except ccxt.InvalidOrder as exc:
            # -4164 (notional below 5 USDT) and other "won't accept this
            # order" cases. Sizing already gates on min_notional_usd, but
            # exchange-side step-size rounding or a stale price reference
            # can still slip a sub-floor order through. Soft-skip.
            log.info("skip open %s: invalid order rejected by exchange: %s",
                     order.symbol, exc)
            return ExecutionResult(
                success=False, paper=False, reason=f"invalid_order: {exc}"
            )
        entry_id = str(entry_resp.get("id") or "")

        # Anchor SL/TP to the actual fill, not the stale candle reference.
        # Without this, a small adverse move between signal generation and
        # market fill makes the planned SL sit on the wrong side of the
        # fill price, the exchange rejects with -2021 "Order would
        # immediately trigger", and the rollback bleeds the round-trip
        # spread + slippage. Sizing's distance is preserved; only the
        # anchor point moves.
        fill_price = float(entry_resp.get("average") or 0.0) or order.entry_price
        sl_distance = abs(order.entry_price - order.stop_loss)
        tp_distance = abs(order.take_profit - order.entry_price)
        if order.side == "long":
            sl_price = fill_price - sl_distance
            tp_price = fill_price + tp_distance
        else:
            sl_price = fill_price + sl_distance
            tp_price = fill_price - tp_distance
        try:
            sl_price = float(ex.price_to_precision(order.symbol, sl_price))
            tp_price = float(ex.price_to_precision(order.symbol, tp_price))
        except Exception:  # noqa: BLE001
            pass

        # Critical: protect the freshly-opened position with a stop-loss
        # before doing anything else. If the SL placement fails (e.g.
        # -2021 "Order would immediately trigger" because the market
        # moved past our stop between signal and order) we MUST roll the
        # entry back. A naked perp position is not acceptable.
        sl_params: dict[str, Any] = {"stopPrice": sl_price, "reduceOnly": True}
        try:
            sl_resp = ex.create_order(
                order.symbol, "STOP_MARKET", exit_side, order.quantity, None, sl_params
            )
        except Exception as sl_exc:  # noqa: BLE001
            log.error(
                "CRITICAL: SL placement failed for %s after entry %s — "
                "rolling back position: %s",
                order.symbol, entry_id, sl_exc,
            )
            try:
                ex.create_order(
                    order.symbol, "market", exit_side, order.quantity,
                    None, {"reduceOnly": True},
                )
                log.info("rollback OK: %s position closed", order.symbol)
            except Exception as rb_exc:  # noqa: BLE001
                log.error(
                    "ROLLBACK FAILED for %s — manual intervention required: %s",
                    order.symbol, rb_exc,
                )
            # Drop the cached LLM verdict so the next cycle issues a fresh
            # call instead of looping on the same stale TAKE that just
            # produced a self-rolling-back trade.
            invalidate_cooldown(order.symbol)
            return ExecutionResult(
                success=False,
                paper=False,
                entry_order_id=entry_id,
                reason=f"sl_failed_rolled_back: {sl_exc}",
            )
        sl_id = str(sl_resp.get("id") or "")

        # TP failure is non-fatal: SL still protects downside, monitor's
        # max-hold and invalidation paths will still close the trade.
        tp_id = ""
        tp_params: dict[str, Any] = {"stopPrice": tp_price, "reduceOnly": True}
        try:
            tp_resp = ex.create_order(
                order.symbol, "TAKE_PROFIT_MARKET", exit_side, order.quantity,
                None, tp_params,
            )
            tp_id = str(tp_resp.get("id") or "")
        except Exception as tp_exc:  # noqa: BLE001
            log.warning(
                "TP placement failed for %s (SL still active): %s",
                order.symbol, tp_exc,
            )

        # Post-place sweep: the pre-open `cancel_symbol_conditionals` can
        # silently no-op if both bulk cancel and the verifying fetch fail
        # (returns -1, caller can't tell). Now that we know the IDs of the
        # SL/TP we just placed, cancel anything else still on the book.
        # Without this, an old bracket from a previous position can survive
        # alongside the new one (observed: 4 reduce-only orders on ZECUSDT).
        keep_ids = tuple(oid for oid in (sl_id, tp_id) if oid)
        try:
            cancel_symbol_conditionals(ex, order.symbol, keep_ids=keep_ids)
        except Exception as exc:  # noqa: BLE001
            log.warning("post-open sweep failed for %s: %s", order.symbol, exc)

        trade_id = self._record_open(
            order,
            paper=False,
            entry_order_id=entry_id,
            stop_order_id=sl_id,
            take_order_id=tp_id,
            entry_override=fill_price,
            stop_override=sl_price,
            take_override=tp_price,
        )
        return ExecutionResult(
            success=True,
            paper=False,
            trade_id=trade_id,
            entry_order_id=entry_id,
            stop_order_id=sl_id,
            take_order_id=tp_id,
            fill_price=float(entry_resp.get("average") or order.entry_price),
            reason="live-fill",
        )

    def _close_live(self, symbol: str, reason: str) -> ExecutionResult:
        if self.client is None:
            raise RuntimeError("OrderExecutor in live mode requires a BinanceClient")
        ex = self.client.exchange

        with self.session_factory() as s:
            row = s.execute(
                select(Trade)
                .where(Trade.symbol == symbol)
                .where(Trade.closed_at.is_(None))
                .order_by(Trade.opened_at.desc())
            ).scalars().first()
            if row is None:
                return ExecutionResult(
                    success=False, paper=False, reason=f"no open trade for {symbol}"
                )
            row_id = row.id
            qty = float(row.quantity)
            side = row.side
            stop_oid = row.stop_order_id
            take_oid = row.take_order_id
            entry_oid = row.entry_order_id
            entry_px = float(row.entry) if row.entry is not None else 0.0
            opened_at = row.opened_at

        opened_at_ms: int | None = None
        if opened_at is not None:
            if opened_at.tzinfo is None:
                opened_at = opened_at.replace(tzinfo=timezone.utc)
            opened_at_ms = int(opened_at.timestamp() * 1000)

        fill: float = 0.0

        if reason == "sl_or_tp_hit":
            # Exchange already closed the position via SL or TP. Recover the
            # fill price from whichever bracket order is in a filled state;
            # cancel the sibling so it doesn't linger.
            for oid in (stop_oid, take_oid):
                if not oid:
                    continue
                try:
                    o = ex.fetch_order(oid, symbol)
                except Exception as exc:  # noqa: BLE001
                    log.warning("fetch_order %s failed: %s", oid, exc)
                    continue
                status = (o.get("status") or "").lower()
                avg = o.get("average") or o.get("price")
                if status in ("closed", "filled") and avg:
                    try:
                        fill = float(avg)
                    except (TypeError, ValueError):
                        fill = 0.0
                    break
            for oid in (stop_oid, take_oid):
                if not oid:
                    continue
                try:
                    ex.cancel_order(oid, symbol)
                except Exception as exc:  # noqa: BLE001
                    log.debug("cancel_order %s skipped: %s", oid, exc)
        else:
            # Manual / max-hold / invalidation: cancel brackets, market-exit.
            for oid in (stop_oid, take_oid):
                if not oid:
                    continue
                try:
                    ex.cancel_order(oid, symbol)
                except Exception as exc:  # noqa: BLE001
                    log.warning("cancel_order %s failed: %s", oid, exc)

        # Wipe every conditional order on the symbol. Bulk cancel is atomic
        # and survives transient fetch failures, so ghosts can't accumulate
        # across closes (the previous per-id sweep silently no-op'd whenever
        # fetch_open_orders failed, leaving stacks of stale SLs behind).
        cancel_symbol_conditionals(ex, symbol)

        if reason == "sl_or_tp_hit":
            # SL/TP path is finished — nothing else to do; PnL is recorded below.
            pass
        else:
            exit_side = "sell" if side == "long" else "buy"
            try:
                close_resp = ex.create_order(
                    symbol, "market", exit_side, qty, None, {"reduceOnly": True}
                )
            except Exception as exc:  # noqa: BLE001
                log.error("close_position market exit failed for %s: %s", symbol, exc)
                return ExecutionResult(
                    success=False, paper=False, trade_id=row_id, reason=str(exc)
                )
            fill = float(close_resp.get("average") or close_resp.get("price") or 0.0)

        # Recover ground-truth exit VWAP and fees from userTrades. The
        # `fetch_order` path above is unreliable for SL/TP because conditional
        # orders moved to the algo endpoint (Binance migration 2025-12-09)
        # and the create_order response only carries an average for the
        # market-close branch. Relying on `_mark_price` after the fact
        # silently mis-prices closes that slipped through the trigger
        # (observed STX/USDT case: SL=0.2490 → fill VWAP=0.2482, ~$5
        # PnL error). userTrades is the only source that always reflects
        # the actual fills, regardless of trigger type.
        # Binance derivatives trade endpoint:
        # https://developers.binance.com/docs/derivatives/usds-margined-futures/trade
        my_trades: list[dict[str, Any]] = []
        if opened_at_ms is not None:
            try:
                my_trades = ex.fetch_my_trades(symbol, since=opened_at_ms) or []
            except Exception as exc:  # noqa: BLE001
                log.warning("fetch_my_trades failed for %s: %s", symbol, exc)

        exit_fills, entry_fills = self._split_fills(my_trades, entry_oid)
        vwap_exit = self._vwap(exit_fills)
        fee_exit = self._sum_fee_usdt(exit_fills)
        fee_entry = self._sum_fee_usdt(entry_fills)

        if vwap_exit > 0:
            fill = vwap_exit
        if fill <= 0:
            fill = self._mark_price(symbol) or 0.0
        if fill <= 0:
            # Last-resort fallback so we still record a closure.
            fill = entry_px

        # If userTrades didn't surface fees (BNB-burn currency, or fetch
        # failed entirely), approximate from the configured taker rate.
        # Note: ignores BNB-burn discounts → conservative (overstates fees).
        taker_pct = float(settings.taker_fee_pct)
        if fee_entry <= 0:
            fee_entry = qty * entry_px * taker_pct
        if fee_exit <= 0:
            fee_exit = qty * fill * taker_pct

        gross_pnl = self._compute_pnl(side=side, qty=qty, entry=entry_px, exit_=fill)
        pnl = gross_pnl - fee_entry - fee_exit
        entry_notional = float(entry_px) * float(qty)
        pnl_pct = (gross_pnl / entry_notional * 100.0) if entry_notional > 0 else 0.0
        total_fees = float(fee_entry) + float(fee_exit)

        with self.session_factory() as s:
            row = s.execute(select(Trade).where(Trade.id == row_id)).scalar_one()
            row.closed_at = datetime.now(tz=timezone.utc)
            row.close_price = fill
            row.close_reason = reason
            row.pnl_usd = pnl
            row.pnl_pct = pnl_pct
            row.fees = total_fees
            s.commit()

        log.info(
            "LIVE CLOSE %s reason=%s exit=%.4f gross=%.2f fees=%.2f net_pnl=%.2f "
            "trade_id=%s",
            symbol, reason, fill, gross_pnl, fee_entry + fee_exit, pnl, row_id,
        )
        return ExecutionResult(
            success=True,
            paper=False,
            trade_id=row_id,
            fill_price=fill,
            pnl_usd=pnl,
            reason=reason,
        )

    # ------------------------------------------------------------------
    def replace_stop(self, trade_id: int, new_stop: float) -> bool:
        """Cancel the live SL order and place a tighter one. Update DB.

        Returns True on success. Used by the trailing-stop logic in the
        position monitor. Paper trades just update the DB row.
        """
        with self.session_factory() as s:
            row = s.execute(select(Trade).where(Trade.id == trade_id)).scalars().first()
            if row is None or row.closed_at is not None:
                return False
            symbol = row.symbol
            side = row.side
            qty = float(row.quantity)
            old_oid = row.stop_order_id

        if self.paper or self.client is None:
            new_oid = ""
        else:
            ex = self.client.exchange
            try:
                price_str = ex.price_to_precision(symbol, new_stop)
                new_stop_priced = float(price_str)
            except Exception:  # noqa: BLE001
                new_stop_priced = float(new_stop)
            exit_side = "sell" if side == "long" else "buy"
            try:
                resp = ex.create_order(
                    symbol, "STOP_MARKET", exit_side, qty, None,
                    {"stopPrice": new_stop_priced, "reduceOnly": True},
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("replace_stop: new SL placement failed for %s: %s", symbol, exc)
                return False
            new_oid = str(resp.get("id") or "")
            # Sweep every other STOP-class reduce-only order, keeping the
            # freshly placed one and leaving TP_MARKET alone. Targeted
            # cancel of `old_oid` is unreliable on demo-fapi.
            cancel_symbol_conditionals(
                ex, symbol, keep_ids=(new_oid,), only_stop=True
            )
            new_stop = new_stop_priced

        with self.session_factory() as s:
            row = s.execute(select(Trade).where(Trade.id == trade_id)).scalars().first()
            if row is None:
                return False
            row.stop = float(new_stop)
            row.stop_order_id = new_oid
            s.commit()
        log.info(
            "trail: SL tightened %s side=%s -> %.6f trade_id=%d",
            symbol, side, new_stop, trade_id,
        )
        return True

    # --- shared helpers ----------------------------------------------
    def _record_open(
        self,
        order: TradeOrder,
        *,
        paper: bool,
        entry_order_id: str | None = None,
        stop_order_id: str | None = None,
        take_order_id: str | None = None,
        entry_override: float | None = None,
        stop_override: float | None = None,
        take_override: float | None = None,
    ) -> int:
        with self.session_factory() as s:
            entry_val = entry_override if entry_override is not None else order.entry_price
            stop_val = stop_override if stop_override is not None else order.stop_loss
            tp_val = take_override if take_override is not None else order.take_profit
            row = Trade(
                symbol=order.symbol,
                side=order.side,
                type=order.entry_type,
                entry=entry_val,
                stop=stop_val,
                original_stop=stop_val,
                take_profit=tp_val,
                quantity=order.quantity,
                leverage=order.leverage,
                risk_usd=order.risk_usd,
                pnl_usd=0.0,
                paper=paper,
                entry_order_id=entry_order_id,
                stop_order_id=stop_order_id,
                take_order_id=take_order_id,
                invalidation_signal=order.invalidation_signal,
                llm_confidence=order.llm_confidence,
            )
            s.add(row)
            s.commit()
            return int(row.id)

    @staticmethod
    def _compute_pnl(*, side: str, qty: float, entry: float, exit_: float) -> float:
        if side == "long":
            return float(qty) * (float(exit_) - float(entry))
        return float(qty) * (float(entry) - float(exit_))

    @staticmethod
    def _split_fills(
        trades: list[dict[str, Any]],
        entry_order_id: str | None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Partition userTrades into (exit_fills, entry_fills).

        Exit fills = reduceOnly trades (closing/decreasing the position).
        Entry fills = trades belonging to the recorded entry order id, OR
        any non-reduceOnly trades if the order id can't be matched (the
        entry response in `_open_live` is the only place we capture the
        entry id, so a missing id falls back to the side heuristic).
        """
        exits: list[dict[str, Any]] = []
        entries: list[dict[str, Any]] = []
        for t in trades:
            info = t.get("info") or {}
            ro = info.get("reduceOnly")
            is_reduce = ro in (True, "true") or t.get("reduceOnly") is True
            if is_reduce:
                exits.append(t)
                continue
            oid = t.get("order") or info.get("orderId")
            if entry_order_id and str(oid) == str(entry_order_id):
                entries.append(t)
            elif not entry_order_id:
                entries.append(t)
        return exits, entries

    @staticmethod
    def _vwap(fills: list[dict[str, Any]]) -> float:
        notional = 0.0
        amount = 0.0
        for f in fills:
            try:
                p = float(f.get("price") or 0.0)
                a = float(f.get("amount") or 0.0)
            except (TypeError, ValueError):
                continue
            if p > 0 and a > 0:
                notional += p * a
                amount += a
        return notional / amount if amount > 0 else 0.0

    @staticmethod
    def _sum_fee_usdt(fills: list[dict[str, Any]]) -> float:
        """Sum trade fees in USDT.

        Only USDT-denominated fees are counted. BNB-burn fees are ignored
        here — caller falls back to `taker_fee_pct` estimate. We don't
        convert BNB→USDT mid-close to avoid an extra ticker fetch on every
        position closure (and BNB-burn is opt-in; default is USDT).
        """
        total = 0.0
        for f in fills:
            fee = f.get("fee") or {}
            ccy = (fee.get("currency") or "").upper()
            cost = fee.get("cost")
            if cost is None or ccy not in ("USDT", ""):
                continue
            try:
                total += float(cost)
            except (TypeError, ValueError):
                continue
        return total

    def _mark_price(self, symbol: str) -> float | None:
        if self.client is None:
            return None
        try:
            ticker = self.client.fetch_ticker(symbol)
        except Exception as exc:  # noqa: BLE001
            log.debug("mark_price fetch failed for %s: %s", symbol, exc)
            return None
        last = ticker.get("last")
        try:
            return float(last) if last is not None else None
        except (TypeError, ValueError):
            return None


# --- legacy bracket helper (kept so backtest/tests don't break) ----------

@dataclass
class BracketOrders:
    entry_id: str | None
    stop_id: str | None
    take_id: str | None


def place_bracket(client, order, dry_run: bool | None = None) -> BracketOrders:
    """Thin wrapper around `OrderExecutor.open_position` for older callers."""
    paper = settings.paper_trading if dry_run is None else dry_run
    executor = OrderExecutor(client=client, paper=paper)
    res = executor.open_position(order)
    return BracketOrders(
        entry_id=res.entry_order_id,
        stop_id=res.stop_order_id,
        take_id=res.take_order_id,
    )
