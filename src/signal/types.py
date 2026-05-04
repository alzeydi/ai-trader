"""Signal types shared across the pipeline."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


class CandidateSignal(BaseModel):
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc)
    )
    symbol: str
    side: Literal["long", "short"]
    entry_type: Literal["A", "B", "C"]
    signal_strength: float = Field(ge=0.0, le=1.0)
    entry_price_ref: float
    atr_14: float
    swing_high_1h: float
    swing_low_1h: float
