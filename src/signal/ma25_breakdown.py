"""MA25 Confirmed Breakdown signal detector (Strategy G — bearish mirror of E).

Pure function — no I/O, no side-effects, no mutable state.

IMPORTANT: this detector USES the in-progress H4 bar. The LAST element of
candles_4h MUST be the currently-forming bar — do not strip it. Conditions
are deliberately re-evaluated against the live bar so a setup can be entered
as soon as it looks confirmed, instead of waiting up to 4 hours for bar
close. The trade-off (bar may reverse before close) is absorbed by the host
bot's stop-loss; per-symbol 24h cooldown handles intra-bar dedup.

Mirror of detect_ma25_bounce(): MA25 falling, MA25 < MA99, ≥70 % of last 10
prior bars closed BELOW MA25, current bar red and closed below MA25 with
the upper-wick of the last 3 bars touching the MA25 zone from below.
Re-uses the same touch-zone width but applied above MA25 instead of below.

All tunable thresholds live in src/config/settings.py under "MA25 Breakdown".
"""

from __future__ import annotations

import logging
from typing import TypedDict

import pandas as pd

from src.config import settings

log = logging.getLogger(__name__)

# Same warm-up rule as ma25_bounce: 99 bars for MA99 + 2 extra so the
# slope window (closes[-37:-12]) never runs off the start of the array.
_MIN_CANDLES = 101


class MA25BreakdownSignal(TypedDict):
    entry_price: float
    ma25: float
    slope_pct: float
    prior_below: float       # 0..1, fraction of 10 prior bars closed BELOW MA25
    bounce_high: float       # local high used as structural stop


def detect_ma25_breakdown(
    symbol: str,
    candles_4h: pd.DataFrame,
) -> MA25BreakdownSignal | None:
    """Return a signal dict when all MA25-breakdown conditions hold, else None.

    Parameters mirror detect_ma25_bounce(): see that module's docstring.
    """
    if len(candles_4h) < _MIN_CANDLES:
        log.debug(
            "%s: ma25_breakdown needs ≥%d candles, got %d",
            symbol, _MIN_CANDLES, len(candles_4h),
        )
        return None

    closes = candles_4h["close"]

    # ── 1. Trend filter ───────────────────────────────────────────────────────
    ma25_now  = float(closes.iloc[-25:].mean())
    ma99_now  = float(closes.iloc[-99:].mean())
    ma25_past = float(closes.iloc[-37:-12].mean())

    if ma25_past <= 0:
        return None

    slope_pct = (ma25_now - ma25_past) / ma25_past * 100

    if slope_pct > settings.ma25_breakdown_slope_max_pct:
        log.debug(
            "%s: ma25_breakdown FAIL trend/slope: slope_pct=%.3f > %.3f",
            symbol, slope_pct, settings.ma25_breakdown_slope_max_pct,
        )
        return None

    if ma25_now >= ma99_now:
        log.debug(
            "%s: ma25_breakdown FAIL trend/stack: MA25=%.5f ≥ MA99=%.5f",
            symbol, ma25_now, ma99_now,
        )
        return None

    # ── 2. Breakdown confirmation ─────────────────────────────────────────────
    last = candles_4h.iloc[-1]
    last_close = float(last["close"])
    last_open  = float(last["open"])

    # A. Prior-below context: 10 bars BEFORE the live bar.
    prior = candles_4h.iloc[-11:-1]
    below_ratio = float((prior["close"] < ma25_now).sum()) / 10.0

    if below_ratio < settings.ma25_breakdown_prior_below_min:
        log.debug(
            "%s: ma25_breakdown FAIL prior_below=%.2f < %.2f",
            symbol, below_ratio, settings.ma25_breakdown_prior_below_min,
        )
        return None

    # B. Current bar must be red so far (running close < open).
    if last_close >= last_open:
        log.debug(
            "%s: ma25_breakdown FAIL not red: close=%.5f open=%.5f",
            symbol, last_close, last_open,
        )
        return None

    # C. Current bar's running close is below MA25.
    if last_close >= ma25_now:
        log.debug(
            "%s: ma25_breakdown FAIL close above MA25: close=%.5f MA25=%.5f",
            symbol, last_close, ma25_now,
        )
        return None

    # D. Not too late — reject if price has already collapsed from MA25.
    pct_below = (ma25_now - last_close) / ma25_now * 100
    if pct_below > settings.ma25_breakdown_pct_below_max:
        log.debug(
            "%s: ma25_breakdown FAIL too late: pct_below=%.2f%% > %.2f%%",
            symbol, pct_below, settings.ma25_breakdown_pct_below_max,
        )
        return None

    # E. Touch with bounded depth. Mirror of bounce: bounded zone around
    # MA25 from below. Highs of last 3 bars (incl. live) define the
    # rejection wick. Allow at most touch_max_pct UNDER MA25 (the high
    # made it close to MA25) and at most touch_min_pct OVER MA25
    # (overshoots above are allowed up to 3 % — beyond that the high is
    # not a clean MA25 retest but a separate impulse).
    touch_min = ma25_now * (1.0 - settings.ma25_breakdown_touch_max_pct)
    touch_max = ma25_now * (1.0 + settings.ma25_breakdown_touch_min_pct)
    lookback   = candles_4h.iloc[-3:]
    bounce_high = float(lookback["high"].max())

    if bounce_high < touch_min:
        log.debug(
            "%s: ma25_breakdown FAIL no touch: bounce_high=%.5f < touch_min=%.5f",
            symbol, bounce_high, touch_min,
        )
        return None

    if bounce_high > touch_max:
        log.debug(
            "%s: ma25_breakdown FAIL deep wick above: bounce_high=%.5f > touch_max=%.5f",
            symbol, bounce_high, touch_max,
        )
        return None

    log.info(
        "%s: ma25_breakdown SIGNAL entry=%.5f ma25=%.5f "
        "slope_pct=%.3f prior_below=%.2f bounce_high=%.5f",
        symbol, last_close, ma25_now, slope_pct, below_ratio, bounce_high,
    )

    return MA25BreakdownSignal(
        entry_price=last_close,
        ma25=ma25_now,
        slope_pct=slope_pct,
        prior_below=below_ratio,
        bounce_high=bounce_high,
    )
