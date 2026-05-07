"""Position sizing (pure arithmetic).

`size_position` is the primary entry point used by `Trader.run_cycle`.
`size_from_risk` is preserved for backtest/legacy callers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.config import settings
from src.execution.types import TradeOrder
from src.llm.types import VetoResponse
from src.risk.types import AccountState
from src.signal.types import CandidateSignal

log = logging.getLogger(__name__)


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
    "D": "risk_per_trade_type_d",
}


def size_position(
    candidate: CandidateSignal,
    account: AccountState,
    veto: VetoResponse,
    *,
    market_max_qty: float | None = None,
) -> TradeOrder | None:
    """Convert a vetted candidate into a fully-specified order.

    `market_max_qty` is the per-order MAX_QTY filter from
    `exchange.markets[symbol]["limits"]["amount"]["max"]`. When provided and
    our computed qty exceeds it, we silently clamp down — losing some of the
    intended size on cheap pairs (e.g. STRK at ~$0.20) is much better than
    eating a Binance -4005 "Quantity greater than max quantity." rejection
    every cycle. None disables the clamp (used by backtest / unit tests
    where there is no live markets cache).

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
    last_cap = "risk_dollar"
    qty_uncapped = quantity

    # Per-trade margin policy cap: don't let a single trade consume more
    # than `max_margin_per_trade_pct` of equity in initial margin. This is
    # what equalises position sizes across symbols — without it, low-price
    # coins blow out to absurd notionals because qty = risk / sl_distance
    # explodes when sl_distance is small in absolute terms.
    policy_max_notional = 0.0
    if settings.leverage > 0 and settings.max_margin_per_trade_pct > 0:
        max_margin = account.equity * settings.max_margin_per_trade_pct
        policy_max_notional = max_margin * settings.leverage
        max_qty_by_policy = policy_max_notional / entry
        if max_qty_by_policy <= 0:
            return None
        if quantity > max_qty_by_policy:
            quantity = max_qty_by_policy
            last_cap = "policy_margin"

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
            last_cap = "free_margin"

    # Exchange per-order MAX_QTY (Binance -4005). Cheap perps (STRK, ZIL,
    # 1000PEPE…) have caps in the 50k–1M contracts range; risk-USD-based
    # sizing easily blows past them when sl_distance is small in absolute
    # terms. Clamp here so the order never gets rejected — the only knock-on
    # effect is a slightly under-sized position on those pairs.
    if market_max_qty is not None and market_max_qty > 0 and quantity > market_max_qty:
        quantity = market_max_qty
        last_cap = "market_max_qty"

    # "Meaningful notional" floor. A trade whose final notional is a tiny
    # fraction of the policy cap is almost always the result of the
    # free-margin cap binding when many positions are already open
    # (observed 2026-05-06: 10 parallel opens drained the demo wallet, the
    # next ~12 candidates all sized to $5–$15 notional and produced PnL
    # smaller than fees). Skip those instead of opening micro-positions —
    # they distort stats and have no meaningful R:R after fees. The floor
    # scales with equity via `policy_max_notional`, so the rule still does
    # the right thing on a $50 testnet wallet vs a $50k live one.
    notional = quantity * entry
    floor_pct = float(settings.min_notional_pct_of_policy)
    floor_from_policy = policy_max_notional * floor_pct if floor_pct > 0 else 0.0
    effective_min_notional = max(settings.min_notional_usd, floor_from_policy)
    if notional < effective_min_notional:
        log.info(
            "%s %s: skip — %s cap binds notional=%.2f < min=%.2f "
            "(policy_max=%.2f, free_margin=%s, equity=%.2f)",
            candidate.symbol, candidate.side, last_cap,
            notional, effective_min_notional, policy_max_notional,
            f"{account.free_margin_usd:.2f}" if account.free_margin_usd is not None else "n/a",
            account.equity,
        )
        return None

    log.info(
        "%s %s sized: qty=%.6f notional=%.2f cap=%s "
        "(uncapped_qty=%.6f, policy_max=%.2f, equity=%.2f)",
        candidate.symbol, candidate.side, quantity, notional, last_cap,
        qty_uncapped, policy_max_notional, account.equity,
    )

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
