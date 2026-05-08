"""Signal types shared across the pipeline."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field


class CandidateSignal(BaseModel):
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(tz=UTC)
    )
    symbol: str
    side: Literal["long", "short"]
    entry_type: Literal["A", "B", "C", "D", "E", "F", "G"]
    signal_strength: float = Field(ge=0.0, le=1.0)
    entry_price_ref: float
    atr_14: float
    # Higher-timeframe ATR (1h) used by Type C sizing so the stop clears
    # typical 1h noise instead of getting nuked by 15m wiggles. Optional for
    # backwards compatibility with backtest harness and old fixtures.
    atr_14_1h: float | None = None
    swing_high_1h: float
    swing_low_1h: float
    # Optional structural stop hint (e.g. bounce_low for Type E). When set,
    # size_position uses this price directly instead of atr * multiplier.
    # This lets bounce/structural strategies define the SL from the chart
    # rather than from volatility noise. The ATR floor (min_stop_pct) still
    # applies as a last-resort guard.
    sl_price_hint: float | None = None
    # When True, the symbol is in a self-driven trend matching `side` over
    # the last N 30m bars (configured via `a_self_trend_bars`). Set by the
    # signal engine for Type A only; consumed by the BTC-regime gate to
    # let the trade through even when BTC's H1 regime is against side.
    # Engine never sets this to True for other entry types.
    self_trend_override: bool = False
