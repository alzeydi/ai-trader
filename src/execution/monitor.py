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
from src.execution.orders import OrderExecutor
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
            try:
                reason = self._evaluate(trade)
            except Exception as exc:  # noqa: BLE001
                log.exception("monitor.evaluate failed for %s: %s", trade.symbol, exc)
                continue
            if reason is None:
                continue
            self._close_and_record(trade, reason)
            closed_symbols.append(trade.symbol)
        return closed_symbols

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
                self.notifier.close(trade.symbol, result.pnl_usd)
            except Exception as exc:  # noqa: BLE001
                log.warning("notifier.close failed: %s", exc)
        if self.safety is not None and result.pnl_usd is not None:
            self.safety.check_after_close(result.pnl_usd)
