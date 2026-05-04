"""Hard risk gates evaluated before any order is sent."""

from __future__ import annotations

from dataclasses import dataclass

from src.config import settings


@dataclass
class RiskState:
    equity_usd: float
    open_positions: int
    daily_pnl_pct: float
    drawdown_pct: float


def check(state: RiskState) -> tuple[bool, str]:
    if state.open_positions >= settings.max_open_positions:
        return False, f"max_open_positions={settings.max_open_positions} reached"
    if state.daily_pnl_pct <= -settings.max_daily_loss_pct:
        return False, f"daily loss limit hit: {state.daily_pnl_pct:.2f}%"
    if state.drawdown_pct >= settings.max_drawdown_pct:
        return False, f"max drawdown breached: {state.drawdown_pct:.2f}%"
    if state.equity_usd <= 0:
        return False, "equity is zero or negative"
    return True, "ok"
