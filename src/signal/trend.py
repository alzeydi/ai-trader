"""4h trend bias detection.

Returns "long" / "short" / "neutral" based on EMA stack (close vs ema50 vs ema200).
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

from src.data.indicators import ema

TrendBias = Literal["long", "short", "neutral"]


def detect_bias(df_4h: pd.DataFrame) -> TrendBias:
    if len(df_4h) < 220:
        return "neutral"
    ema50 = ema(df_4h["close"], 50).iloc[-1]
    ema200 = ema(df_4h["close"], 200).iloc[-1]
    close = df_4h["close"].iloc[-1]
    if pd.isna(ema50) or pd.isna(ema200):
        return "neutral"
    if close > ema50 > ema200:
        return "long"
    if close < ema50 < ema200:
        return "short"
    return "neutral"
