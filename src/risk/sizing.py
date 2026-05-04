"""Position sizing — risk-per-trade scaled by ATR-based stop distance."""

from __future__ import annotations

from dataclasses import dataclass

from src.config import settings
from src.signal.types import CandidateSignal


@dataclass
class SizingResult:
    quantity: float
    notional: float
    risk_usd: float


def size_position(signal: CandidateSignal, equity_usd: float) -> SizingResult:
    risk_usd = equity_usd * (settings.risk_per_trade_pct / 100.0)
    stop_distance = abs(signal.entry - signal.stop)
    if stop_distance <= 0 or signal.entry <= 0:
        return SizingResult(0.0, 0.0, 0.0)
    quantity = risk_usd / stop_distance
    notional = quantity * signal.entry
    return SizingResult(quantity=quantity, notional=notional, risk_usd=risk_usd)
