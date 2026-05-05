"""Order execution: paper or live.

`OrderExecutor` is the only place that talks to the Binance REST API for
order placement. In paper mode it simulates a fill at `entry_price` and
records the trade in the DB; live mode also records the trade after sending
the bracket (entry market + reduce-only stop + reduce-only take-profit).
"""

from __future__ import annotations

import logging
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
            pnl = self._compute_pnl(
                side=row.side, qty=row.quantity, entry=row.entry, exit_=close_price
            )
            row.closed_at = datetime.now(tz=timezone.utc)
            row.close_price = close_price
            row.close_reason = reason
            row.pnl_usd = pnl
            s.commit()
            trade_id = row.id

        log.info(
            "PAPER CLOSE %s reason=%s exit=%.4f pnl=%.2f trade_id=%s",
            symbol, reason, close_price, pnl, trade_id,
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

        # Critical: protect the freshly-opened position with a stop-loss
        # before doing anything else. If the SL placement fails (e.g.
        # -2021 "Order would immediately trigger" because the market
        # moved past our stop between signal and order) we MUST roll the
        # entry back. A naked perp position is not acceptable.
        sl_params: dict[str, Any] = {"stopPrice": order.stop_loss, "reduceOnly": True}
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
        tp_params: dict[str, Any] = {"stopPrice": order.take_profit, "reduceOnly": True}
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

        trade_id = self._record_open(
            order,
            paper=False,
            entry_order_id=entry_id,
            stop_order_id=sl_id,
            take_order_id=tp_id,
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

        # Sweep any leftover reduce-only orders. Catches:
        #   * the previous SL when `replace_stop` placed a new one but failed
        #     to cancel the old one,
        #   * unrelated brackets placed manually before adoption,
        #   * any order whose ID was lost because the row was wiped.
        # Without this sweep these orders linger forever (and re-trigger if
        # a future position re-opens on the same symbol).
        try:
            leftovers = ex.fetch_open_orders(symbol)
        except Exception as exc:  # noqa: BLE001
            log.warning("close sweep: fetch_open_orders %s failed: %s", symbol, exc)
            leftovers = []
        for o in leftovers:
            info = o.get("info") or {}
            if not (o.get("reduceOnly") or info.get("reduceOnly") in (True, "true")):
                continue
            oid = str(o.get("id") or "")
            if not oid:
                continue
            try:
                ex.cancel_order(oid, symbol)
                log.info("close sweep: cancelled leftover order %s on %s", oid, symbol)
            except Exception as exc:  # noqa: BLE001
                log.warning("close sweep: cancel %s failed: %s", oid, exc)

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

        if fill <= 0:
            fill = self._mark_price(symbol) or 0.0
        if fill <= 0:
            # Last-resort fallback so we still record a closure.
            with self.session_factory() as s:
                fallback_row = s.execute(
                    select(Trade).where(Trade.id == row_id)
                ).scalar_one()
                fill = float(fallback_row.entry)

        pnl = self._compute_pnl(side=side, qty=qty, entry=row.entry, exit_=fill)

        with self.session_factory() as s:
            row = s.execute(select(Trade).where(Trade.id == row_id)).scalar_one()
            row.closed_at = datetime.now(tz=timezone.utc)
            row.close_price = fill
            row.close_reason = reason
            row.pnl_usd = pnl
            s.commit()

        log.info(
            "LIVE CLOSE %s reason=%s exit=%.4f pnl=%.2f trade_id=%s",
            symbol, reason, fill, pnl, row_id,
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
            # Sweep ALL other reduce-only STOP_MARKET orders for this symbol
            # instead of cancelling only `old_oid`. On demo-fapi (and
            # occasionally on prod) targeted cancel_order silently fails or
            # returns "Unknown order"; once the DB pointer is overwritten
            # those orphans become unreachable and stack on the exchange,
            # so each trail tick previously left a fresh ghost SL behind.
            try:
                open_orders = ex.fetch_open_orders(symbol)
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "replace_stop: fetch_open_orders %s failed, falling back to "
                    "targeted cancel of %s: %s", symbol, old_oid, exc,
                )
                open_orders = []
                if old_oid:
                    try:
                        ex.cancel_order(old_oid, symbol)
                    except Exception as inner:  # noqa: BLE001
                        log.warning(
                            "replace_stop: targeted cancel of %s on %s also "
                            "failed: %s", old_oid, symbol, inner,
                        )
            for o in open_orders:
                info = o.get("info") or {}
                oid = str(o.get("id") or "")
                if not oid or oid == new_oid:
                    continue
                if not (o.get("reduceOnly") or info.get("reduceOnly") in (True, "true")):
                    continue
                otype = (o.get("type") or info.get("type") or "").upper()
                # Only sweep STOP-class orders. Leave TAKE_PROFIT_MARKET alone:
                # it's the same trade's TP and cancelling it would leave the
                # position with no upside target until close.
                if "STOP" not in otype or "TAKE_PROFIT" in otype:
                    continue
                try:
                    ex.cancel_order(oid, symbol)
                    log.info(
                        "replace_stop: swept stale SL %s on %s", oid, symbol,
                    )
                except Exception as exc:  # noqa: BLE001
                    log.warning(
                        "replace_stop: sweep cancel %s on %s failed: %s",
                        oid, symbol, exc,
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
    ) -> int:
        with self.session_factory() as s:
            row = Trade(
                symbol=order.symbol,
                side=order.side,
                type=order.entry_type,
                entry=order.entry_price,
                stop=order.stop_loss,
                original_stop=order.stop_loss,
                take_profit=order.take_profit,
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
