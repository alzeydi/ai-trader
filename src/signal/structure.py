"""1h structure: rolling swing highs and lows.

A swing high at index i requires `lookback` lower highs on each side; same idea
for swing lows. The most recent confirmed swing in each direction is returned.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class Structure:
    last_swing_high: float | None
    last_swing_low: float | None


def find_structure(df_1h: pd.DataFrame, lookback: int = 5) -> Structure:
    if len(df_1h) < lookback * 2 + 1:
        return Structure(None, None)

    highs = df_1h["high"]
    lows = df_1h["low"]

    swing_high: float | None = None
    swing_low: float | None = None

    for i in range(len(df_1h) - lookback - 1, lookback, -1):
        window_high = highs.iloc[i - lookback : i + lookback + 1]
        window_low = lows.iloc[i - lookback : i + lookback + 1]
        if swing_high is None and highs.iloc[i] == window_high.max():
            swing_high = float(highs.iloc[i])
        if swing_low is None and lows.iloc[i] == window_low.min():
            swing_low = float(lows.iloc[i])
        if swing_high is not None and swing_low is not None:
            break

    return Structure(swing_high, swing_low)
