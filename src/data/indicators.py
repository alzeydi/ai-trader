"""Technical indicators (pandas_ta backed).

All df-based helpers expect OHLCV columns (open/high/low/close/volume) and a
DatetimeIndex. They return Series/DataFrame aligned to the input index.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandas_ta as ta


def ema(series: pd.Series, length: int) -> pd.Series:
    out = ta.ema(series, length=length)
    if out is None:
        return pd.Series(np.nan, index=series.index, name=f"ema_{length}")
    return out.rename(f"ema_{length}")


def rsi14(df: pd.DataFrame) -> pd.Series:
    out = ta.rsi(df["close"], length=14)
    if out is None:
        return pd.Series(np.nan, index=df.index, name="rsi_14")
    return out.rename("rsi_14")


def macd(df: pd.DataFrame) -> pd.DataFrame:
    """Return MACD line, signal line, and histogram (cols: macd/signal/hist)."""
    raw = ta.macd(df["close"])
    cols = ["macd", "signal", "hist"]
    if raw is None:
        return pd.DataFrame(np.nan, index=df.index, columns=cols)
    # pandas_ta column order: MACD, MACDh (hist), MACDs (signal)
    raw.columns = ["macd", "hist", "signal"]
    return raw[cols]


def atr14(df: pd.DataFrame) -> pd.Series:
    out = ta.atr(df["high"], df["low"], df["close"], length=14)
    if out is None:
        return pd.Series(np.nan, index=df.index, name="atr_14")
    return out.rename("atr_14")


def swing_high_low(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """Mark confirmed swing pivots.

    Bar `i` is a swing high iff its high equals the maximum across the window
    [i-lookback, i+lookback] (inclusive); analogous rule for lows. The result
    has two columns:
        swing_high — pivot price where confirmed, NaN otherwise
        swing_low  — pivot price where confirmed, NaN otherwise

    Bars within `lookback` of either edge cannot be confirmed and are NaN.
    """
    n = len(df)
    sh = pd.Series(np.nan, index=df.index, name="swing_high", dtype=float)
    sl = pd.Series(np.nan, index=df.index, name="swing_low", dtype=float)
    if n < 2 * lookback + 1:
        return pd.concat([sh, sl], axis=1)

    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()

    for i in range(lookback, n - lookback):
        window_high = highs[i - lookback : i + lookback + 1]
        window_low = lows[i - lookback : i + lookback + 1]
        if highs[i] == window_high.max():
            sh.iloc[i] = float(highs[i])
        if lows[i] == window_low.min():
            sl.iloc[i] = float(lows[i])
    return pd.concat([sh, sl], axis=1)
