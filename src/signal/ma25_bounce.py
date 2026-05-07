"""MA25 Confirmed Bounce entry signal detector (Strategy X).

Pure function — no I/O, no side-effects, no mutable state.

Run right after each H4 close (UTC :00 multiples of 4). Only pass CLOSED
bars; drop the in-progress candle before calling.

Three-filter pipeline:
  1. Trend filter  — MA25 rising + MA25 > MA99
  2. Prior context — price was already above MA25 before the bounce
  3. Bounce shape  — green close above MA25, touch zone, not too late

All tunable thresholds live in src/config/settings.py under the
"MA25 Bounce" section. Do NOT hardcode replacements here.
"""

from __future__ import annotations

import logging
from typing import TypedDict

import pandas as pd

from src.config import settings

log = logging.getLogger(__name__)

# Minimum closed candles required: 99 for MA99 + 2 extra so the slope
# look-back window (candles[-37:-12]) never runs off the start of the array.
_MIN_CANDLES = 101


class MA25BounceSignal(TypedDict):
    entry_price: float
    ma25: float
    slope_pct: float
    prior_above: float   # 0..1, fraction of 10 prior bars closed above MA25
    bounce_low: float


def detect_ma25_bounce(
    symbol: str,
    candles_4h: pd.DataFrame,
) -> MA25BounceSignal | None:
    """Return a signal dict when all MA25-bounce conditions hold, else None.

    Parameters
    ----------
    symbol:
        Ticker string used only for log messages.
    candles_4h:
        ≥101 most-recent 4-hour CLOSED candles, ascending by time.
        Required columns: open, high, low, close (quote_volume optional).
    """
    if len(candles_4h) < _MIN_CANDLES:
        log.debug(
            "%s: ma25_bounce needs ≥%d candles, got %d",
            symbol, _MIN_CANDLES, len(candles_4h),
        )
        return None

    closes = candles_4h["close"]

    # ── 1. Trend filter ───────────────────────────────────────────────────────
    ma25_now  = float(closes.iloc[-25:].mean())
    ma99_now  = float(closes.iloc[-99:].mean())

    # Slope: compare MA25 now vs MA25 as of 12 bars ago.
    # past_window = candles_4h[-(25+12) : -12]  →  closes[-37:-12]
    ma25_past = float(closes.iloc[-37:-12].mean())

    if ma25_past <= 0:
        return None

    slope_pct = (ma25_now - ma25_past) / ma25_past * 100

    if slope_pct < settings.ma25_slope_min_pct:
        log.debug(
            "%s: ma25_bounce FAIL trend/slope: slope_pct=%.3f < %.3f",
            symbol, slope_pct, settings.ma25_slope_min_pct,
        )
        return None

    if ma25_now <= ma99_now:
        log.debug(
            "%s: ma25_bounce FAIL trend/stack: MA25=%.5f ≤ MA99=%.5f",
            symbol, ma25_now, ma99_now,
        )
        return None

    # ── 2. Bounce confirmation ────────────────────────────────────────────────
    last = candles_4h.iloc[-1]
    last_close = float(last["close"])
    last_open  = float(last["open"])

    # A. Prior-above context: 10 bars BEFORE the current bar.
    # Excludes the current bar so its own close does not bias the ratio.
    prior = candles_4h.iloc[-11:-1]
    above_ratio = float((prior["close"] > ma25_now).sum()) / 10.0

    if above_ratio < settings.ma25_prior_above_min:
        log.debug(
            "%s: ma25_bounce FAIL prior_above=%.2f < %.2f",
            symbol, above_ratio, settings.ma25_prior_above_min,
        )
        return None

    # B. Current bar must be green (close > open — wicks alone do not count).
    if last_close <= last_open:
        log.debug(
            "%s: ma25_bounce FAIL not green: close=%.5f open=%.5f",
            symbol, last_close, last_open,
        )
        return None

    # C. Current bar closed above MA25.
    if last_close <= ma25_now:
        log.debug(
            "%s: ma25_bounce FAIL close below MA25: close=%.5f MA25=%.5f",
            symbol, last_close, ma25_now,
        )
        return None

    # D. Not too late — reject if price has already run away from MA25.
    pct_above = (last_close - ma25_now) / ma25_now * 100
    if pct_above > settings.ma25_pct_above_max:
        log.debug(
            "%s: ma25_bounce FAIL too late: pct_above=%.2f%% > %.2f%%",
            symbol, pct_above, settings.ma25_pct_above_max,
        )
        return None

    # E. Touch with bounded depth (lows of last 3 bars including current).
    # Uses LOW — a wick that dipped to MA25 and recovered counts as a touch;
    # entry check uses CLOSE (above), so wicks alone never trigger.
    touch_max  = ma25_now * (1.0 + settings.ma25_touch_max_pct)
    touch_min  = ma25_now * (1.0 - settings.ma25_touch_min_pct)
    lookback   = candles_4h.iloc[-3:]
    bounce_low = float(lookback["low"].min())

    if bounce_low > touch_max:
        log.debug(
            "%s: ma25_bounce FAIL no touch: bounce_low=%.5f > touch_max=%.5f",
            symbol, bounce_low, touch_max,
        )
        return None

    if bounce_low < touch_min:
        log.debug(
            "%s: ma25_bounce FAIL deep wick: bounce_low=%.5f < touch_min=%.5f",
            symbol, bounce_low, touch_min,
        )
        return None

    log.info(
        "%s: ma25_bounce SIGNAL entry=%.5f ma25=%.5f "
        "slope_pct=%.3f prior_above=%.2f bounce_low=%.5f",
        symbol, last_close, ma25_now, slope_pct, above_ratio, bounce_low,
    )

    return MA25BounceSignal(
        entry_price=last_close,
        ma25=ma25_now,
        slope_pct=slope_pct,
        prior_above=above_ratio,
        bounce_low=bounce_low,
    )
