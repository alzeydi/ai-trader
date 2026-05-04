"""Layer 3 — orchestrate sizing + risk gates + order placement."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.data.binance_client import BinanceClient
from src.execution.orders import BracketOrders, place_bracket
from src.risk.limits import RiskState, check
from src.risk.safety_mode import SafetyMode
from src.risk.sizing import size_position
from src.signal.types import CandidateSignal

log = logging.getLogger(__name__)


@dataclass
class TradeOutcome:
    accepted: bool
    reason: str
    quantity: float = 0.0
    notional: float = 0.0
    risk_usd: float = 0.0
    orders: BracketOrders | None = None


@dataclass
class Trader:
    client: BinanceClient
    safety: SafetyMode

    def execute(self, signal: CandidateSignal, risk_state: RiskState) -> TradeOutcome:
        if self.safety.is_paused():
            return TradeOutcome(False, "safety mode paused")

        ok, reason = check(risk_state)
        if not ok:
            return TradeOutcome(False, f"risk gate: {reason}")

        sizing = size_position(signal, risk_state.equity_usd)
        if sizing.quantity <= 0:
            return TradeOutcome(False, "sizing returned 0")

        orders = place_bracket(self.client, signal, sizing.quantity)
        return TradeOutcome(
            accepted=True,
            reason="ok",
            quantity=sizing.quantity,
            notional=sizing.notional,
            risk_usd=sizing.risk_usd,
            orders=orders,
        )
