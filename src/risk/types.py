"""Shared Pydantic types for the risk layer."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class AccountState(BaseModel):
    """Snapshot of the account used by `risk.limits.can_take_signal`."""

    model_config = ConfigDict(extra="allow")

    equity: float = Field(gt=0)
    open_positions: int = 0
    daily_pnl_pct: float = 0.0  # percent, e.g. -1.5 == -1.5 %
    consecutive_losses: int = 0
    # Free USDT margin available on the futures wallet. None means "unknown"
    # and disables the margin-cap branch in `size_position` (used by backtest
    # and unit tests where there is no live wallet).
    free_margin_usd: float | None = None
