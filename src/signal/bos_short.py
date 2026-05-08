"""Break-of-Structure short signal detector (Strategy F).

Pure function — no I/O, no side-effects, no mutable state.

The setup:
  1. Look back N closed 1h bars and locate the lowest low (the "swing low" SL).
  2. Between SL and the latest closed bar, find the highest high (the
     "lower-high retest" LH). LH must be the price that the market reached
     while pulling back from SL but failed to break above prior structure.
  3. The current 1h bar closes BELOW SL — break of structure confirmed.
  4. 15m volume on the breakdown is at least f_min_vol_ratio × the 20-bar
     trailing mean (rejection of low-volume fakes).
  5. Close has not already run more than f_max_excess_pct past SL — fresh
     breaks only, no chasing.
  6. The LH formed within f_max_lower_high_age bars of the current one —
     stale retests do not count.

LH is used as the structural stop (sl_price_hint above entry). Entry side
is short by construction. The 4h-bias HTF gate is applied by the engine,
not here, so this detector stays pure-data.

All tunable thresholds live in src/config/settings.py.
"""

from __future__ import annotations

import logging
from typing import TypedDict

import pandas as pd

from src.config import settings

log = logging.getLogger(__name__)


class BoSShortSignal(TypedDict):
    entry_price: float       # current 1h close (the breakdown bar close)
    swing_low: float         # the broken structural low
    lower_high: float        # retest high — used as structural stop
    excess_pct: float        # (swing_low - close) / swing_low — freshness
    bars_since_lh: int       # how many 1h bars ago LH printed
    vol_ratio: float         # 15m volume / 20-bar mean (informational)


def detect_bos_short(
    symbol: str,
    df_1h: pd.DataFrame,
    vol_ratio_15m: float,
) -> BoSShortSignal | None:
    """Return a BoS-short signal when all conditions hold, else None.

    Parameters
    ----------
    symbol:
        Ticker string used only for log messages.
    df_1h:
        ≥ (f_swing_lookback_bars + 2) rows of 1h OHLC, ascending. The LAST
        row is the breakdown bar (running close used for the BoS check).
    vol_ratio_15m:
        Volume ratio computed by the caller on the 15m frame
        (last bar / 20-bar mean). Passed in instead of recomputed so this
        function stays pure-shape and does not duplicate engine work.
    """
    lookback = max(2, int(settings.f_swing_lookback_bars))
    if len(df_1h) < lookback + 2:
        log.debug(
            "%s: bos_short needs ≥%d bars, got %d",
            symbol, lookback + 2, len(df_1h),
        )
        return None

    # Use only CLOSED bars for the swing-low search — the live bar carries
    # its own running low which can be artificially deep mid-formation.
    closed = df_1h.iloc[-(lookback + 1):-1]
    if len(closed) < lookback:
        return None

    swing_low = float(closed["low"].min())
    if swing_low <= 0:
        return None
    swing_low_pos = int(closed["low"].values.argmin())  # 0..len(closed)-1
    swing_low_iloc = -(lookback + 1) + swing_low_pos    # negative index in df_1h

    # Lower-high search: bars strictly AFTER swing_low and strictly BEFORE
    # the live bar. If swing_low is the most recent closed bar there is no
    # room for a retest yet — bail out.
    after_lo_start = swing_low_iloc + 1
    if after_lo_start >= -1:
        log.debug(
            "%s: bos_short no room for LH after swing_low (start=%d)",
            symbol, after_lo_start,
        )
        return None

    after_lo = df_1h.iloc[after_lo_start:-1]
    if after_lo.empty:
        return None

    lh_pos = int(after_lo["high"].values.argmax())
    lower_high = float(after_lo["high"].iloc[lh_pos])
    # Negative index of LH inside df_1h. The live bar is at -1; the most
    # recent CLOSED bar is at -2, so bars_since_lh ≥ 1 by construction.
    lh_iloc = after_lo_start + lh_pos
    bars_since_lh = (-1) - lh_iloc

    if lower_high <= swing_low:
        # Pullback didn't actually go up — no retest, just continuation.
        log.debug(
            "%s: bos_short FAIL LH=%.5f ≤ swing_low=%.5f",
            symbol, lower_high, swing_low,
        )
        return None

    max_age = max(1, int(settings.f_max_lower_high_age))
    if bars_since_lh > max_age:
        log.debug(
            "%s: bos_short FAIL stale LH (age=%d > %d)",
            symbol, bars_since_lh, max_age,
        )
        return None

    close_now = float(df_1h["close"].iloc[-1])
    if close_now >= swing_low:
        log.debug(
            "%s: bos_short FAIL no break (close=%.5f ≥ swing_low=%.5f)",
            symbol, close_now, swing_low,
        )
        return None

    excess_pct = (swing_low - close_now) / swing_low
    max_excess = float(settings.f_max_excess_pct)
    if excess_pct > max_excess:
        log.debug(
            "%s: bos_short FAIL stale break (excess=%.3f%% > %.3f%%)",
            symbol, excess_pct * 100, max_excess * 100,
        )
        return None

    min_vol = float(settings.f_min_vol_ratio)
    if vol_ratio_15m < min_vol:
        log.debug(
            "%s: bos_short FAIL vol (%.2fx < %.2fx)",
            symbol, vol_ratio_15m, min_vol,
        )
        return None

    log.info(
        "%s: bos_short SIGNAL entry=%.5f swing_low=%.5f LH=%.5f "
        "excess=%.3f%% age=%d vol=%.2fx",
        symbol, close_now, swing_low, lower_high,
        excess_pct * 100, bars_since_lh, vol_ratio_15m,
    )

    return BoSShortSignal(
        entry_price=close_now,
        swing_low=swing_low,
        lower_high=lower_high,
        excess_pct=excess_pct,
        bars_since_lh=bars_since_lh,
        vol_ratio=vol_ratio_15m,
    )
