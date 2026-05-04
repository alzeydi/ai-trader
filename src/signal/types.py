"""Signal taxonomy.

Type A — high-conviction trend continuation (all 3 timeframes aligned).
Type B — counter-trend reversal at structural level (1h swing flip).
Type C — range / mean-reversion at well-defined band edges.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class SignalType(str, Enum):
    A = "A"  # trend continuation
    B = "B"  # reversal
    C = "C"  # range


class Side(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class CandidateSignal:
    symbol: str
    side: Side
    type: SignalType
    entry: float
    stop: float
    take_profit: float
    confidence: float                       # 0..1, produced by the rule layer
    rationale: str                          # short human-readable summary
    features: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(
        default_factory=lambda: datetime.now(tz=timezone.utc)
    )

    def risk_reward(self) -> float:
        risk = abs(self.entry - self.stop)
        reward = abs(self.take_profit - self.entry)
        return reward / risk if risk > 0 else 0.0
