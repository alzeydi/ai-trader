"""Self-trend detection: aggregates 15m bars into 30m and checks the last
N closed 30m candles' direction.

Used by the BTC-regime gate override for Type A: when a symbol has been
moving on its own (against BTC's H1 regime) it is more likely to be a
genuine independent trend than a BTC-correlated bounce/dump. Pure data
function — no I/O, no settings.

A 30m candle covers two consecutive 15m bars on a wall-clock boundary:
the first 15m has timestamp minute ∈ {0, 30}, the second has minute ∈
{15, 45}. We use index timestamps (not just position) to enforce that
alignment, so a misaligned head/tail (a partial 30m window at either
end of the fetch) is silently skipped.
"""

from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger(__name__)


def compute_self_trend(
    df_15m: pd.DataFrame,
    side: str,
    bars: int,
) -> bool:
    """Return True iff the last `bars` CLOSED 30m candles match `side`.

    "In trend" for long  → close > open
    "In trend" for short → close < open

    The current (in-progress) 30m candle is NOT counted. If the most
    recent 15m bar has minute ∈ {0, 30}, it opens a new 30m candle that
    has not closed yet; we look strictly at the pairs before it.
    """
    if bars <= 0 or side not in ("long", "short"):
        return False
    if df_15m is None or df_15m.empty:
        return False
    if not isinstance(df_15m.index, pd.DatetimeIndex):
        log.debug("self_trend: non-datetime index, skipping")
        return False
    # Need at least 2 × bars rows to build the pairs, plus 1 in case the
    # tail bar is the open of an in-progress 30m window.
    if len(df_15m) < bars * 2 + 1:
        return False

    minutes = df_15m.index.minute

    # Find the index of the last bar that closes a 30m candle.
    # Closed-30m end bar: minute ∈ {15, 45}.
    last_close = None
    for j in range(len(df_15m) - 1, 0, -1):
        if minutes[j] in (15, 45) and minutes[j - 1] in (0, 30):
            # Verify exact 15-minute spacing — guards against gapped data.
            delta = df_15m.index[j] - df_15m.index[j - 1]
            if delta == pd.Timedelta(minutes=15):
                last_close = j
                break

    if last_close is None:
        return False

    # We need `bars` complete pairs ending at `last_close`. Each pair
    # occupies two rows and is preceded by another pair: pair k ends at
    # last_close - 2*k.
    needed_open_idx = last_close - 1 - 2 * (bars - 1)
    if needed_open_idx < 0:
        return False

    for k in range(bars):
        close_i = last_close - 2 * k
        open_i = close_i - 1
        # Re-verify alignment for each pair so a hole anywhere in the
        # window (data outage) makes us return False conservatively.
        if minutes[close_i] not in (15, 45):
            return False
        if minutes[open_i] not in (0, 30):
            return False
        if df_15m.index[close_i] - df_15m.index[open_i] != pd.Timedelta(minutes=15):
            return False
        if k > 0:
            prev_close = last_close - 2 * (k - 1)
            # Pairs must abut: this pair's close is exactly 30 min before
            # the previous pair's close.
            if df_15m.index[prev_close] - df_15m.index[close_i] != pd.Timedelta(minutes=30):
                return False

        op = float(df_15m["open"].iloc[open_i])
        cl = float(df_15m["close"].iloc[close_i])
        if side == "long" and not (cl > op):
            return False
        if side == "short" and not (cl < op):
            return False

    return True
