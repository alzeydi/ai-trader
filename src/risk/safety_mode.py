"""Safety mode (NoFx pattern).

After N consecutive losing trades, trading is paused for `pause_hours`. The
state is intentionally in-memory; persistence is handled elsewhere if needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from src.config import settings


@dataclass
class SafetyMode:
    consecutive_losses: int = 0
    paused_until: datetime | None = field(default=None)

    def record_trade(self, pnl_usd: float) -> None:
        if pnl_usd < 0:
            self.consecutive_losses += 1
            if self.consecutive_losses >= settings.safety_max_consecutive_losses:
                self.paused_until = datetime.now(tz=timezone.utc) + timedelta(
                    hours=settings.safety_pause_hours
                )
        else:
            self.consecutive_losses = 0
            self.paused_until = None

    def is_paused(self) -> bool:
        if self.paused_until is None:
            return False
        if datetime.now(tz=timezone.utc) >= self.paused_until:
            self.paused_until = None
            self.consecutive_losses = 0
            return False
        return True
