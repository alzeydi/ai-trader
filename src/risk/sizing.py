"""Position sizing (pure arithmetic).

`size_position` is the primary entry point used by `Trader.run_cycle`.
`size_from_risk` is preserved for backtest/legacy callers.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.config import settings
from src.execution.types import TradeOrder
from src.llm.types import VetoResponse
from src.risk.types import AccountState
from src.signal.types import CandidateSignal


@dataclass
class SizingResult:
    quantity: float
    notional: float
    risk_usd: float


def size_from_risk(
    equity_usd: float,
    entry_price: float,
    stop_price: float,
    risk_pct: float | None = None,
) -> SizingResult:
    """Legacy fixed-risk helper used by the backtest harness."""
    rp = risk_pct if risk_pct is not None else settings.risk_per_trade_pct
    risk_usd = equity_usd * (rp / 100.0)
    stop_dist = abs(entry_price - stop_price)
    if stop_dist <= 0 or entry_price <= 0:
        return SizingResult(0.0, 0.0, 0.0)
    quantity = risk_usd / stop_dist
    return SizingResult(
        quantity=quantity,
        notional=quantity * entry_price,
        risk_usd=risk_usd,
    )


_RISK_BY_TYPE: dict[str, str] = {
    "A": "risk_per_trade_type_a",
    "B": "risk_per_trade_type_b",
    "C": "risk_per_trade_type_c",
}


def size_position(
    candidate: CandidateSignal,
    account: AccountState,
    veto: VetoResponse,
) -> TradeOrder | None:
    """Convert a vetted candidate into a fully-specified order.

    Returns `None` when the trade is too small to be profitable after fees
    (target distance < 3 × round-trip taker fees).
    """
    risk_pct = getattr(settings, _RISK_BY_TYPE[candidate.entry_type])
    risk_usd = account.equity * risk_pct

    sl_distance = candidate.atr_14 * settings.atr_stop_multiplier
    tp_distance = sl_distance * (settings.atr_take_multiplier / settings.atr_stop_multiplier)

    entry = candidate.entry_price_ref
    if entry <= 0 or sl_distance <= 0:
        return None

    # Round-trip fee = 2 × taker fee. Reject if the target is smaller than 3×
    # round-trip — anything tighter is dominated by transaction cost.
    round_trip_fee_pct = settings.taker_fee_pct * 2.0
    if (tp_distance / entry) < (round_trip_fee_pct * 3.0):
        return None

    quantity = risk_usd / sl_distance
    if quantity <= 0:
        return None

    if candidate.side == "long":
        sl = entry - sl_distance
        tp = entry + tp_distance
    else:
        sl = entry + sl_distance
        tp = entry - tp_distance

    return TradeOrder(
        symbol=candidate.symbol,
        side=candidate.side,
        entry_type=candidate.entry_type,
        quantity=round(quantity, 6),
        leverage=settings.leverage,
        entry_price=round(entry, 6),
        stop_loss=round(sl, 6),
        take_profit=round(tp, 6),
        risk_usd=round(risk_usd, 4),
        invalidation_signal=veto.invalidation_signal,
        llm_confidence=veto.confidence,
    )
