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

    # Type C trades the 1h range, so its stop must clear 1h noise — using
    # the 15m ATR here produced sub-1% stops that got nuked on the next
    # candle. Fall back to atr_14 if the 1h figure isn't populated (legacy
    # fixtures / backtest harness).
    atr = (
        candidate.atr_14_1h
        if candidate.entry_type == "C" and candidate.atr_14_1h is not None
        else candidate.atr_14
    )
    sl_distance = atr * settings.atr_stop_multiplier

    entry = candidate.entry_price_ref
    if entry <= 0 or sl_distance <= 0:
        return None

    # Volatility floor: on calm symbols even a 1h ATR can resolve to ~0.3 %
    # of price, which is well inside typical bid/ask noise. Force the stop
    # to clear at least `min_stop_pct` of the entry price.
    min_stop_distance = entry * settings.min_stop_pct
    if sl_distance < min_stop_distance:
        sl_distance = min_stop_distance

    tp_distance = sl_distance * (settings.atr_take_multiplier / settings.atr_stop_multiplier)

    # Round-trip fee = 2 × taker fee. Reject if the target is smaller than 3×
    # round-trip — anything tighter is dominated by transaction cost.
    round_trip_fee_pct = settings.taker_fee_pct * 2.0
    if (tp_distance / entry) < (round_trip_fee_pct * 3.0):
        return None

    quantity = risk_usd / sl_distance
    if quantity <= 0:
        return None

    # Per-trade margin policy cap: don't let a single trade consume more
    # than `max_margin_per_trade_pct` of equity in initial margin. This is
    # what equalises position sizes across symbols — without it, low-price
    # coins blow out to absurd notionals because qty = risk / sl_distance
    # explodes when sl_distance is small in absolute terms.
    if settings.leverage > 0 and settings.max_margin_per_trade_pct > 0:
        max_margin = account.equity * settings.max_margin_per_trade_pct
        max_qty_by_policy = (max_margin * settings.leverage) / entry
        if max_qty_by_policy <= 0:
            return None
        if quantity > max_qty_by_policy:
            quantity = max_qty_by_policy

    # Cap quantity by free margin so we never request a notional the wallet
    # cannot back. Required margin per contract = entry / leverage; reserve
    # ~10 % headroom for taker fees, slippage and price drift between sizing
    # and execution. Skipped when free margin is unknown (backtest/tests).
    if account.free_margin_usd is not None and settings.leverage > 0:
        max_notional = account.free_margin_usd * settings.leverage * 0.9
        max_qty = max_notional / entry
        if max_qty <= 0:
            return None
        if quantity > max_qty:
            quantity = max_qty

    # Exchange minimum notional (Binance -4164). If sizing landed below the
    # floor — typically because free margin is tiny — reject cleanly so the
    # trader cycle continues instead of firing a doomed order.
    if quantity * entry < settings.min_notional_usd:
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
