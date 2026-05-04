"""4h trend bias detection.

Returns "long" / "short" / "neutral" based on EMA stack and recent slope.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

from src.data.indicators import add_emas

TrendBias = Literal["long", "short", "neutral"]


def detect_bias(df_4h: pd.DataFrame) -> TrendBias:
    if len(df_4h) < 220:
        return "neutral"
    feat = add_emas(df_4h, lengths=(50, 200))
    last = feat.iloc[-1]
    if pd.isna(last["ema_50"]) or pd.isna(last["ema_200"]):
        return "neutral"
    if last["close"] > last["ema_50"] > last["ema_200"]:
        return "long"
    if last["close"] < last["ema_50"] < last["ema_200"]:
        return "short"
    return "neutral"
