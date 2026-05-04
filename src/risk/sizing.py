"""Position sizing helpers (pure arithmetic, no side-effects).

Primary sizing is done inline in `Trader.execute`, but these helpers are
exported for scripts and tests that need standalone calculations.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.config import settings


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
