"""Open-position monitoring: SL/TP sync, max-hold, LLM invalidation checks.

Designed to run on its own schedule (e.g. every 60 s) in parallel with the
main trading loop. `PositionMonitor.tick()` is the single unit of work and
is safe to call from a thread.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

from src.config import settings
from src.execution.orders import (
    OrderExecutor,
    _fetch_algo_open_orders,
    cancel_symbol_conditionals,
)
from src.persistence.db import (
    SessionFactory,
    Trade,
    list_open_trades,
    make_session_factory,
)

if TYPE_CHECKING:
    from src.data.binance_client import BinanceClient
    from src.llm.client import ClaudeClient
    from src.notify.telegram import TelegramNotifier
    from src.risk.safety_mode import SafetyMode

log = logging.getLogger(__name__)


@dataclass
class PositionSnapshot:
    symbol: str
    side: str
    quantity: float
    entry_price: float
    mark_price: float
    unrealized_pnl: float


def fetch_open_positions(client: "BinanceClient") -> list[PositionSnapshot]:
    """Snapshot of open exchange positions; empty list on any failure."""
    out: list[PositionSnapshot] = []
    try:
        positions: list[dict[str, Any]] = client.exchange.fetch_positions()  # type: ignore[assignment]
    except Exception as exc:  # noqa: BLE001
        log.warning("monitor.fetch_positions failed: %s", exc)
        return out

    for p in positions:
        contracts = float(p.get("contracts") or 0.0)
        if contracts == 0:
            continue
        out.append(
            PositionSnapshot(
                symbol=str(p.get("symbol")),
                side=str(p.get("side") or "").upper(),
                quantity=contracts,
                entry_price=float(p.get("entryPrice") or 0.0),
                mark_price=float(p.get("markPrice") or 0.0),
                unrealized_pnl=float(p.get("unrealizedPnl") or 0.0),
            )
        )
    return out


# --- LLM invalidation prompt -----------------------------------------------

_INVALIDATION_PROMPT = (
    "You evaluate whether a single, observable invalidation condition is "
    "TRUE right now for a live crypto futures position.\n"
    "Reply with exactly one token: yes, no, or unclear.\n"
    "Do not explain, do not output anything else."
)


def _parse_yes_no_unclear(text: str) -> str:
    """Coerce arbitrary LLM output to one of yes/no/unclear."""
    if not text:
        return "unclear"
    token = re.split(r"\s+", text.strip().lower())[0]
    token = token.strip(".,!?:'\"")
    if token in {"yes", "y", "true"}:
        return "yes"
    if token in {"no", "n", "false"}:
        return "no"
    return "unclear"


@dataclass
class PositionMonitor:
    client: "BinanceClient | None" = None
    executor: OrderExecutor | None = None
    safety: "SafetyMode | None" = None
    llm_client: "ClaudeClient | None" = None
    notifier: "TelegramNotifier | None" = None
    session_factory: SessionFactory = field(default=None)  # type: ignore[assignment]
    max_hold_hours: int = field(default_factory=lambda: settings.max_hold_hours)
    invalidation_interval_min: int = field(
        default_factory=lambda: settings.invalidation_check_interval_min
    )

    def __post_init__(self) -> None:
        if self.session_factory is None:
            self.session_factory = make_session_factory()
        if self.executor is None:
            self.executor = OrderExecutor(
                client=self.client, session_factory=self.session_factory
            )

    # ------------------------------------------------------------------
    def tick(self) -> list[str]:
        """One pass over all open trades; returns symbols that were closed."""
        closed_symbols: list[str] = []
        for trade in list_open_trades(self.session_factory):
            if settings.trail_enabled and not trade.paper:
                try:
                    self._update_trail(trade)
                except Exception as exc:  # noqa: BLE001
                    log.warning("trail update failed for %s: %s", trade.symbol, exc)
            try:
                reason = self._evaluate(trade)
            except Exception as exc:  # noqa: BLE001
                log.exception("monitor.evaluate failed for %s: %s", trade.symbol, exc)
                continue
            if reason is None:
                continue
            self._close_and_record(trade, reason)
            closed_symbols.append(trade.symbol)
        # Belt-and-suspenders: anything still hanging on the book for a
        # symbol with no live position is, by definition, an orphan from
        # an earlier close whose sweep failed (or a trail tick that
        # placed a new SL but couldn't cancel the old one). Wipe them.
        try:
            self._reconcile_orphan_orders()
        except Exception as exc:  # noqa: BLE001
            log.warning("monitor: reconcile failed: %s", exc)
        return closed_symbols

    def _reconcile_orphan_orders(self) -> None:
        """Sweep stale conditional orders.

        Two cases handled in one pass:
        1. Symbol has reduce-only conditional orders but no live position
           and no DB trade → wipe everything (classic orphans from a close
           whose cleanup failed).
        2. Symbol has a live DB trade but extra conditional orders beyond
           the trade's tracked SL/TP → wipe the extras. This is the case
           that bit us on ZECUSDT: a previous bracket survived the close
           and the open's pre-sweep, so the symbol ended up with two SLs
           and two TPs after a new entry.
        """
        if self.client is None:
            return
        ex = self.client.exchange
        symbols_with_orders: set[str] = set()

        # Legacy bucket (compat-shimmed conditionals, plain reduce-only
        # market orders left behind, etc.)
        try:
            open_orders = ex.fetch_open_orders()
        except Exception as exc:  # noqa: BLE001
            log.debug("reconcile: fetch_open_orders (all) failed: %s", exc)
            open_orders = []
        for o in open_orders:
            sym = o.get("symbol")
            if not sym:
                continue
            otype = (o.get("type") or (o.get("info") or {}).get("type") or "").upper()
            if any(tag in otype for tag in ("STOP", "TAKE_PROFIT", "TRAILING_STOP")):
                symbols_with_orders.add(str(sym))

        # New algo bucket — the actual home of conditional orders since the
        # 2025-12-09 Binance migration. Without this pass the reconciler
        # was effectively a no-op for conditional cleanup.
        markets_by_id = getattr(ex, "markets_by_id", {}) or {}
        for o in _fetch_algo_open_orders(ex):
            info = o.get("info") if isinstance(o.get("info"), dict) else o
            raw_sym = o.get("symbol") or info.get("symbol") or ""
            if not raw_sym:
                continue
            # The algo endpoint returns Binance-native ids ("ZECUSDT");
            # convert to the unified ccxt symbol ("ZEC/USDT:USDT") so it
            # matches DB rows and `cancel_symbol_conditionals` calls.
            unified = raw_sym
            mkt = markets_by_id.get(raw_sym)
            if isinstance(mkt, list) and mkt:
                mkt = mkt[0]
            if isinstance(mkt, dict) and mkt.get("symbol"):
                unified = mkt["symbol"]
            symbols_with_orders.add(str(unified))

        if not symbols_with_orders:
            return

        open_trades_by_symbol: dict[str, Trade] = {}
        for t in list_open_trades(self.session_factory):
            if not t.paper:
                open_trades_by_symbol[t.symbol] = t

        live_symbols = {
            s.symbol for s in fetch_open_positions(self.client) if s.quantity > 0
        }

        for sym in symbols_with_orders:
            trade = open_trades_by_symbol.get(sym)
            if trade is None:
                if sym in live_symbols:
                    # Live position with no DB row — adoption hasn't run
                    # yet for this cycle. Leave the orders alone; the next
                    # adopt/reconcile pass on restart picks them up.
                    continue
                n = cancel_symbol_conditionals(ex, sym)
                if n:
                    log.info(
                        "reconcile: %s had no position, wiped %s order(s)", sym, n
                    )
                continue

            keep = tuple(
                oid for oid in (
                    trade.stop_order_id, trade.trail_stop_order_id, trade.take_order_id
                ) if oid
            )
            if not keep:
                # DB row exists but we don't know which orders belong to it
                # (e.g. a TP failed at placement and the SL id wasn't
                # recorded). Don't touch — risk cancelling the live SL.
                continue
            n = cancel_symbol_conditionals(ex, sym, keep_ids=keep)
            if n and n > 0:
                log.info(
                    "reconcile: %s had %d extra conditional(s) beyond tracked SL/TP",
                    sym, n,
                )

    # ------------------------------------------------------------------
    def _update_trail(self, trade: Trade) -> None:
        """Notional-percent breakeven trailing.

        Phase 1 (activation): when unrealised net profit (after round-trip
        taker fees) reaches `breakeven_activate_pct` of entry notional
        (qty × entry), push the stop to a price that locks in at least
        `breakeven_lock_pct` of notional in net profit.
        Phase 2 (trail): on each tick, lock in
        `max(breakeven_lock_pct, HWM_net_profit − trail_distance_pct)`.
        Stops only tighten; never loosen.
        """
        if self.client is None or self.executor is None:
            return
        if trade.entry is None or trade.quantity is None or trade.side is None:
            return
        qty = float(trade.quantity)
        entry = float(trade.entry)
        if qty <= 0 or entry <= 0:
            return

        try:
            ticker = self.client.fetch_ticker(trade.symbol)
        except Exception as exc:  # noqa: BLE001
            log.debug("trail: mark fetch failed for %s: %s", trade.symbol, exc)
            return
        last = ticker.get("last")
        if last is None:
            return
        mark = float(last)

        side = trade.side.lower()
        # Round-trip fees: taker on entry + taker on exit at current mark.
        fee_pct = float(settings.taker_fee_pct)
        fees = qty * entry * fee_pct + qty * mark * fee_pct

        if side == "long":
            gross = qty * (mark - entry)
        else:
            gross = qty * (entry - mark)
        net_profit = gross - fees

        # Convert pct knobs to USD using entry notional as the base. Doing
        # the conversion here (instead of carrying pct everywhere) keeps
        # the rest of the math identical to the previous USD-denominated
        # version — only the thresholds are now self-scaling.
        notional = qty * entry
        activate_usd = notional * float(settings.breakeven_activate_pct)
        lock_usd = notional * float(settings.breakeven_lock_pct)
        distance_usd = notional * float(settings.trail_distance_pct)
        be_arm_usd = notional * float(settings.be_protect_arm_pct)
        be_lock_usd = notional * float(settings.be_protect_lock_pct)

        prev_hwm = float(trade.trail_high_water) if trade.trail_high_water is not None else 0.0
        hwm_profit = max(prev_hwm, net_profit)
        was_armed = bool(trade.trail_armed)

        # ── Pre-arm HWM tracking ──────────────────────────────────────────
        # Persist peak net profit before the main trail activates so
        # be_protect can see how high the position reached. trail_armed stays
        # False until the full breakeven_activate_pct threshold is hit.
        if not was_armed and net_profit > 0 and hwm_profit > prev_hwm:
            with self.session_factory() as s:
                row = s.get(Trade, trade.id)
                if row is not None:
                    row.trail_high_water = float(hwm_profit)
                    s.commit()

        # ── BE-protect (phase 0.5) ────────────────────────────────────────
        # Once HWM ≥ be_protect_arm_pct of notional AND the position starts
        # retracing (current < HWM), move SL to lock in be_protect_lock_pct
        # net profit. Fires only once: subsequent ticks find the new stop
        # already tighter and skip. The regular trail takes over if price
        # recovers to breakeven_activate_pct.
        if not was_armed and hwm_profit >= be_arm_usd and net_profit < hwm_profit:
            if side == "long":
                be_stop = (be_lock_usd / qty + entry * (1.0 + fee_pct)) / (1.0 - fee_pct)
            else:
                be_stop = (entry * (1.0 - fee_pct) - be_lock_usd / qty) / (1.0 + fee_pct)
            try:
                price_str = self.client.exchange.price_to_precision(trade.symbol, be_stop)
                be_stop = float(price_str)
            except Exception:  # noqa: BLE001
                pass
            be_tighter = (
                (side == "long" and be_stop > float(trade.stop))
                or (side == "short" and be_stop < float(trade.stop))
            )
            if be_tighter and self.executor.replace_stop(trade.id, be_stop):
                log.info(
                    "be_protect: SL → %.6f for %s (hwm_net=%.2f cur_net=%.2f)",
                    be_stop, trade.symbol, hwm_profit, net_profit,
                )
                if self.notifier is not None:
                    try:
                        self.notifier.send(
                            f"🛡 *BE PROTECT* `{trade.symbol}` {side.upper()}\n"
                            f"SL → `{be_stop:.6f}`\n"
                            f"Locked: `${be_lock_usd:.2f}` net"
                        )
                    except Exception as exc:  # noqa: BLE001
                        log.debug("be_protect notify failed: %s", exc)

        if net_profit < activate_usd:
            return

        target_locked = max(lock_usd, hwm_profit - distance_usd)

        # Convert locked NET profit -> SL price.
        # net = qty*(stop−entry) − qty*entry*fee − qty*stop*fee  (long)
        #     = qty*stop*(1−fee) − qty*entry*(1+fee)
        # → stop = (net/qty + entry*(1+fee)) / (1−fee)
        # Symmetric for short.
        if side == "long":
            target_stop = (target_locked / qty + entry * (1.0 + fee_pct)) / (1.0 - fee_pct)
        else:
            target_stop = (entry * (1.0 - fee_pct) - target_locked / qty) / (1.0 + fee_pct)

        # Round to exchange precision BEFORE the tighter-than-current check.
        # Without this, the raw float drifts microscopically each tick while
        # the stored value is the rounded one — so `target_stop > trade.stop`
        # is true on every tick by 1e-6, replace_stop fires, sweep cancels
        # the identical SL, and the user gets a TRAIL notification per
        # minute showing the same "0.113070 → 0.113072" no-op move.
        if self.client is not None:
            try:
                price_str = self.client.exchange.price_to_precision(
                    trade.symbol, target_stop
                )
                target_stop = float(price_str)
            except Exception:  # noqa: BLE001
                pass

        if side == "long":
            tighter = target_stop > float(trade.stop)
        else:
            tighter = target_stop < float(trade.stop)

        # Persist HWM/armed even if the stop is not moving yet, so a
        # restart preserves arming state.
        if (
            trade.trail_high_water is None
            or hwm_profit != prev_hwm
            or not trade.trail_armed
        ):
            with self.session_factory() as s:
                row = s.get(Trade, trade.id)
                if row is not None:
                    row.trail_high_water = float(hwm_profit)
                    row.trail_armed = True
                    s.commit()

        if not tighter:
            return

        if not self.executor.replace_stop(trade.id, target_stop):
            return
        # Notify only on the first time trailing arms; subsequent SL tightening
        # ticks stay silent and the user gets the next push only on close.
        # Per-tick notifications spammed Telegram (one push every ~14 s on a
        # fast-moving symbol).
        if self.notifier is not None and not was_armed:
            try:
                self.notifier.send(
                    f"🔒 *TRAIL ARMED* `{trade.symbol}` {side.upper()}\n"
                    f"SL → `{target_stop:.6f}`\n"
                    f"Locked: `${target_locked:.2f}` net"
                )
            except Exception as exc:  # noqa: BLE001
                log.debug("trail notify failed: %s", exc)

    def run_forever(self, interval_sec: int | None = None) -> None:
        """Block forever, ticking every `interval_sec` (default from settings)."""
        delay = interval_sec or settings.monitor_interval_sec
        while True:
            try:
                self.tick()
            except Exception:  # noqa: BLE001
                log.exception("monitor: tick failed")
            time.sleep(delay)

    # ------------------------------------------------------------------
    def _evaluate(self, trade: Trade) -> str | None:
        """Return a close-reason string, or `None` to keep the trade open."""
        # 1. Was the position closed on the exchange (SL/TP hit)?
        if not trade.paper and self.client is not None:
            if not self._exchange_position_open(trade.symbol):
                return "sl_or_tp_hit"

        # 2. Max-hold timeout.
        opened_at = trade.opened_at
        if opened_at is not None:
            if opened_at.tzinfo is None:
                opened_at = opened_at.replace(tzinfo=timezone.utc)
            age = datetime.now(tz=timezone.utc) - opened_at
            if age >= timedelta(hours=self.max_hold_hours):
                return "max_hold_timeout"

        # 3. LLM invalidation poll (rate-limited per trade).
        if self._should_run_invalidation_check(trade):
            verdict = self._check_invalidation(trade)
            self._record_invalidation_check(trade.id)
            if verdict == "yes":
                return "invalidation_signal"

        return None

    def _exchange_position_open(self, symbol: str) -> bool:
        """True iff the exchange still reports a non-zero position for symbol."""
        if self.client is None:
            return True
        snaps = fetch_open_positions(self.client)
        return any(s.symbol == symbol and s.quantity > 0 for s in snaps)

    def _should_run_invalidation_check(self, trade: Trade) -> bool:
        if self.llm_client is None or not trade.invalidation_signal:
            return False
        last = trade.last_invalidation_check_at
        if last is None:
            return True
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        return (datetime.now(tz=timezone.utc) - last) >= timedelta(
            minutes=self.invalidation_interval_min
        )

    def _check_invalidation(self, trade: Trade) -> str:
        """Ask the LLM whether the invalidation condition has triggered."""
        assert self.llm_client is not None
        condition = trade.invalidation_signal or ""
        user = (
            f"Symbol: {trade.symbol}\n"
            f"Side: {trade.side}\n"
            f"Invalidation condition: {condition}\n\n"
            "Is this condition TRUE right now? Reply yes/no/unclear."
        )
        try:
            text, _, _ = self.llm_client._call(_INVALIDATION_PROMPT, user)
        except Exception as exc:  # noqa: BLE001
            log.warning("invalidation check failed for %s: %s", trade.symbol, exc)
            return "unclear"
        verdict = _parse_yes_no_unclear(text)
        log.info("invalidation %s: condition=%r verdict=%s", trade.symbol, condition, verdict)
        return verdict

    def _record_invalidation_check(self, trade_id: int) -> None:
        try:
            with self.session_factory() as s:
                row = s.get(Trade, trade_id)
                if row is None:
                    return
                row.last_invalidation_check_at = datetime.now(tz=timezone.utc)
                s.commit()
        except Exception as exc:  # noqa: BLE001
            log.warning("monitor: failed to stamp invalidation check: %s", exc)

    def _close_and_record(self, trade: Trade, reason: str) -> None:
        assert self.executor is not None
        result = self.executor.close_position(trade.symbol, reason)
        log.info("monitor: closed %s reason=%s pnl=%s", trade.symbol, reason, result.pnl_usd)
        if self.notifier is not None and result.pnl_usd is not None:
            try:
                snapshot = {
                    "symbol": trade.symbol,
                    "side": trade.side,
                    "entry": float(trade.entry or 0.0),
                    "exit": float(result.fill_price or trade.entry or 0.0),
                    "quantity": float(trade.quantity or 0.0),
                    "pnl_usd": float(result.pnl_usd),
                    "leverage": int(trade.leverage or 1),
                    # result.fees_usd carries the actual computed fees from _close_live /
                    # _close_paper; trade.fees at this point is still the pre-close DB
                    # value (0 or entry-only) because the session was committed after
                    # ExecutionResult was returned.
                    "fees": float(result.fees_usd if result.fees_usd is not None else trade.fees or 0.0),
                    "close_reason": reason,
                }
                self.notifier.send_close_signal(snapshot)
            except Exception as exc:  # noqa: BLE001
                log.warning("notifier close failed: %s", exc)
        if self.safety is not None and result.pnl_usd is not None:
            self.safety.check_after_close(result.pnl_usd)
