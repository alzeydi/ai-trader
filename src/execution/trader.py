"""Layer 3 — sizing, risk gates, and order placement.

Converts a vetted CandidateSignal into a fully-specified TradeOrder,
then delegates placement to orders.py.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

from src.config import settings
from src.data.binance_client import BinanceClient
from src.execution.orders import BracketOrders, place_bracket
from src.llm.veto import VetoResponse
from src.risk.limits import RiskState, check
from src.risk.safety_mode import SafetyMode
from src.signal.types import CandidateSignal

log = logging.getLogger(__name__)


class TradeOrder(BaseModel):
    symbol: str
    side: Literal["long", "short"]
    quantity: float
    leverage: int
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_usd: float
    invalidation_signal: str
    llm_confidence: float


@dataclass
class TradeOutcome:
    accepted: bool
    reason: str
    order: TradeOrder | None = None
    brackets: BracketOrders | None = None


@dataclass
class Trader:
    client: BinanceClient
    safety: SafetyMode

    def execute(
        self,
        signal: CandidateSignal,
        veto: VetoResponse,
        risk_state: RiskState,
        equity_usd: float,
    ) -> TradeOutcome:
        if self.safety.is_paused():
            return TradeOutcome(False, "safety mode paused")

        ok, reason = check(risk_state)
        if not ok:
            return TradeOutcome(False, f"risk gate: {reason}")

        # Compute SL/TP from ATR, then size from fixed-risk %.
        atr = signal.atr_14
        entry = signal.entry_price_ref
        mult_sl = settings.atr_stop_multiplier
        mult_tp = settings.atr_take_multiplier

        if signal.side == "long":
            stop_loss   = entry - mult_sl * atr
            take_profit = entry + mult_tp * atr
        else:
            stop_loss   = entry + mult_sl * atr
            take_profit = entry - mult_tp * atr

        stop_dist = abs(entry - stop_loss)
        if stop_dist <= 0 or entry <= 0:
            return TradeOutcome(False, "degenerate stop distance")

        risk_usd = equity_usd * (settings.risk_per_trade_pct / 100.0)
        quantity = risk_usd / stop_dist

        order = TradeOrder(
            symbol=signal.symbol,
            side=signal.side,
            quantity=round(quantity, 6),
            leverage=settings.leverage,
            entry_price=entry,
            stop_loss=round(stop_loss, 6),
            take_profit=round(take_profit, 6),
            risk_usd=round(risk_usd, 4),
            invalidation_signal=veto.invalidation_signal,
            llm_confidence=veto.confidence,
        )

        brackets = place_bracket(self.client, order)
        return TradeOutcome(accepted=True, reason="ok", order=order, brackets=brackets)
