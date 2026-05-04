"""15m entry triggers.

Given a trend bias and structure context, look for a confirmation candle that
breaks/retests the relevant level with momentum (RSI > 50 long, < 50 short).
"""

from __future__ import annotations

import pandas as pd

from src.data.indicators import rsi14
from src.signal.structure import Structure
from src.signal.trend import TrendBias


def find_trigger(
    df_15m: pd.DataFrame,
    bias: TrendBias,
    structure: Structure,
) -> tuple[bool, str]:
    """Return (fired, reason)."""
    if bias == "neutral":
        return False, "no trend bias"
    rsi_last = rsi14(df_15m).iloc[-1]
    close_last = df_15m["close"].iloc[-1]

    if bias == "long" and structure.last_swing_high is not None:
        if close_last > structure.last_swing_high and not pd.isna(rsi_last) and rsi_last > 50:
            return True, "break of 1h swing high with RSI>50"
    if bias == "short" and structure.last_swing_low is not None:
        if close_last < structure.last_swing_low and not pd.isna(rsi_last) and rsi_last < 50:
            return True, "break of 1h swing low with RSI<50"
    return False, "no trigger"
