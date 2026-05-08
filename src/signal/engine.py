"""Layer 1 — multi-timeframe signal engine.

Three signal types:
  A — with-trend pullback (4h bullish/bearish + 1h near EMA50 + 15m reversal)
  B — counter-trend extreme (4h RSI >75 or <25 + 15m reversal confirmation)
  C — range fade (4h neutral + close near 1h swing boundary + 15m reversal)

Public API:
  SignalEngine.generate_signals(symbol) -> CandidateSignal | None
  SignalEngine._evaluate(symbol, df_4h, df_1h, df_15m) -> CandidateSignal | None
    (separated for testability with synthetic data)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime

import pandas as pd

from src.config import settings
from src.data.binance_client import BinanceClient
from src.data.indicators import atr14, ema, macd, rsi14, swing_high_low
from src.signal.bos_short import detect_bos_short
from src.signal.ma25_bounce import detect_ma25_bounce
from src.signal.ma25_breakdown import detect_ma25_breakdown
from src.signal.self_trend import compute_self_trend
from src.signal.types import CandidateSignal

log = logging.getLogger(__name__)

_MIN_4H = 210   # need EMA200 warmup
_MIN_1H = 55    # need EMA50 + swing lookback
_MIN_15M = 35   # need ATR14 + MACD warmup + volume window


# ── composite scorers ──────────────────────────────────────────────────────

def _score_a(ema_gap_pct: float, rsi_delta: float, vol_ratio: float) -> float:
    """Score Type A trend-pullback candidates.

    Constants saturate each component at its "strong enough" threshold:
        ema_gap_pct / 0.08 — 8 % gap between EMA50/EMA200 caps the trend
            component (typical strong-trend ranges on 4h crypto majors).
        rsi_delta / 8.0    — an 8-point RSI swing on 1h is a confident
            pullback-then-resume signature in our backtests.
        (vol_ratio - 1) / 1.0 — 2× the 20-bar mean volume saturates the
            confirmation component.
    Weights 0.40 / 0.35 / 0.25 were fitted on the 2024 universe; if you
    re-fit, keep the sum at 1.0 so the cap below stays meaningful.
    """
    trend   = min(abs(ema_gap_pct) / 0.08, 1.0) * 0.40
    rev     = min(abs(rsi_delta)   / 8.0,  1.0) * 0.35
    volume  = min(max(vol_ratio - 1.0, 0.0) / 1.0, 1.0) * 0.25
    # Round to 2 decimals so the value compared against signal_min_confidence
    # matches what is printed in logs (%.2f). Sub-cent precision was false:
    # 0.4994 used to log as "0.50" while still failing a 0.50 floor.
    return round(min(trend + rev + volume, 0.99), 2)


def _score_b(rsi_4h: float, side: str, hist_delta: float) -> float:
    """Score Type B mean-reversion candidates.

    75 / 25 are the canonical RSI overbought / oversold thresholds; the
    excess past the threshold is normalised by the remaining 25 points so
    a fully-pegged RSI (100 / 0) saturates the component. hist_delta is
    the MACD-histogram slope on 15m: 0.01 saturates at "clear reversal"
    based on observed turn magnitudes in the 2024 sample.
    """
    excess   = (rsi_4h - 75) / 25.0 if side == "short" else (25 - rsi_4h) / 25.0
    reversal = min(abs(hist_delta) / 0.01, 1.0) * 0.50
    return round(min(0.50 * min(excess, 1.0) + reversal, 0.99), 2)


def _score_c(proximity: float, rsi_delta: float) -> float:
    """Score Type C range-bounce candidates.

    50 / 50 weighting between range-edge proximity and RSI swing magnitude;
    rsi_delta is normalised by 8 points (same scale as Type A — an 8-point
    1h RSI move is the empirical "confident reversal" boundary).
    """
    return round(min(0.50 * proximity + 0.50 * min(abs(rsi_delta) / 8.0, 1.0), 0.99), 2)


def _score_e(slope_pct: float, prior_above: float, pct_above: float) -> float:
    """Score Type E MA25-bounce candidates.

    slope_pct:   MA25 rise over 12-bar window (%). Floor 0.5 % already held.
    prior_above: fraction of 10 prior bars closed above MA25 (floor 0.7).
    pct_above:   how far current close sits above MA25 (0–4 %). Lower = fresher.
    """
    # Stronger slope → better-established uptrend behind the bounce.
    slope = min(max(slope_pct - 0.5, 0.0) / 2.5, 1.0) * 0.40
    # More prior bars above MA25 → cleaner pullback, not a reclaim-from-below.
    ctx   = min((prior_above - 0.7) / 0.3, 1.0) * 0.35
    # Entry closer to MA25 → fresher signal, less chasing.
    fresh = max(0.0, 1.0 - pct_above / 4.0) * 0.25
    return round(min(slope + ctx + fresh, 0.99), 2)


def _score_g(slope_pct: float, prior_below: float, pct_below: float) -> float:
    """Score Type G MA25-breakdown candidates (mirror of _score_e).

    slope_pct:   MA25 fall over 12-bar window (%, negative). We score on |slope|.
    prior_below: fraction of 10 prior bars closed below MA25 (floor 0.7).
    pct_below:   how far current close sits below MA25 (0–4 %). Lower = fresher.
    """
    slope = min(max(abs(slope_pct) - 0.5, 0.0) / 2.5, 1.0) * 0.40
    ctx   = min((prior_below - 0.7) / 0.3, 1.0) * 0.35
    fresh = max(0.0, 1.0 - pct_below / 4.0) * 0.25
    return round(min(slope + ctx + fresh, 0.99), 2)


def _score_f(vol_ratio: float, excess_pct: float, lh_age: int, max_age: int) -> float:
    """Score Type F (BoS short) candidates.

    vol_ratio: 15m volume / 20-bar mean. Higher = more conviction on the break.
    excess_pct: how far the breakdown bar closed past swing_low (relative).
        Smaller = fresher entry.
    lh_age: bars elapsed since the lower-high printed (1 = last closed bar).
        Smaller = fresher retest.
    """
    volume = min(max(vol_ratio - 1.0, 0.0) / 1.5, 1.0) * 0.40
    fresh_break = max(0.0, 1.0 - excess_pct / 0.02) * 0.30
    fresh_lh = max(0.0, 1.0 - (lh_age - 1) / max(1, max_age - 1)) * 0.30
    return round(min(volume + fresh_break + fresh_lh, 0.99), 2)


def _score_d(vol_ratio: float, excess_pct: float, close_position: float) -> float:
    """Score Type D breakout candidates.

    `excess_pct` = how far the current 1h close is past the breakout
    level (relative to that level). Smaller is fresher.
    `close_position` = (close_1h − low_1h) / (high_1h − low_1h) for
    longs, mirrored for shorts. 1.0 = closed at the bar high (decisive
    break), 0.5 = mid-bar (indecisive).
    """
    volume = min(max(vol_ratio - 1.0, 0.0) / 1.5, 1.0) * 0.40
    # Penalise stale breakouts: we want fresh ones, not 5%-late chases.
    fresh = max(0.0, 1.0 - excess_pct / 0.02) * 0.30
    decisive = max(0.0, min(close_position, 1.0)) * 0.30
    return round(min(volume + fresh + decisive, 0.99), 2)


# ── indicator snapshot ─────────────────────────────────────────────────────

@dataclass
class SignalEngine:
    client: BinanceClient

    def generate_signals(self, symbol: str) -> CandidateSignal | None:
        df_4h = self.client.fetch_ohlcv(symbol, "4h",  limit=300)
        df_1h = self.client.fetch_ohlcv(symbol, "1h",  limit=100)
        df_15m = self.client.fetch_ohlcv(symbol, "15m", limit=200)
        return self._evaluate(symbol, df_4h, df_1h, df_15m)

    def _evaluate(
        self,
        symbol: str,
        df_4h: pd.DataFrame,
        df_1h: pd.DataFrame,
        df_15m: pd.DataFrame,
    ) -> CandidateSignal | None:

        if len(df_4h) < _MIN_4H or len(df_1h) < _MIN_1H or len(df_15m) < _MIN_15M:
            log.debug("%s: insufficient data (4h=%d 1h=%d 15m=%d)",
                      symbol, len(df_4h), len(df_1h), len(df_15m))
            return None

        # ── 4h indicators ──────────────────────────────────────────────────
        c4 = df_4h["close"]
        ema50_4h  = ema(c4, 50).iloc[-1]
        ema200_4h = ema(c4, 200).iloc[-1]
        rsi_4h    = rsi14(df_4h).iloc[-1]
        macd_4h   = macd(df_4h)
        hist_4h   = macd_4h["hist"]
        close_4h  = float(c4.iloc[-1])
        atr_4h_val = float(atr14(df_4h).iloc[-1])

        # ── 1h indicators ──────────────────────────────────────────────────
        c1 = df_1h["close"]
        ema50_1h_series = ema(c1, 50)
        ema50_1h = ema50_1h_series.iloc[-1]
        close_1h = float(c1.iloc[-1])
        atr_1h_val = float(atr14(df_1h).iloc[-1])
        # Slope of EMA50(1h) over the last `b_trend_lookback_bars`. Used by
        # the Type-B gate below: fading a coin whose 1h trend has been
        # running hard against us is a tax on the strategy.
        b_lookback = max(2, int(settings.b_trend_lookback_bars))
        if len(ema50_1h_series) > b_lookback:
            ema50_1h_then = float(ema50_1h_series.iloc[-1 - b_lookback])
            if ema50_1h_then > 0:
                ema50_1h_slope_pct = (
                    float(ema50_1h) - ema50_1h_then
                ) / ema50_1h_then
            else:
                ema50_1h_slope_pct = 0.0
        else:
            ema50_1h_slope_pct = 0.0
        swings   = swing_high_low(df_1h, lookback=20)
        sh_valid = swings["swing_high"].dropna()
        sl_valid = swings["swing_low"].dropna()
        swing_h: float | None = float(sh_valid.iloc[-1]) if not sh_valid.empty else None
        swing_l: float | None = float(sl_valid.iloc[-1]) if not sl_valid.empty else None

        # ── 15m indicators ─────────────────────────────────────────────────
        rsi_15 = rsi14(df_15m)
        rsi_now  = rsi_15.iloc[-1]
        rsi_prev = rsi_15.iloc[-2]
        atr_val  = float(atr14(df_15m).iloc[-1])
        macd_15  = macd(df_15m)
        hist_15  = macd_15["hist"]
        vol      = df_15m["volume"]
        vol_ratio = (
            float(vol.iloc[-1]) / float(vol.iloc[-20:].mean())
            if float(vol.iloc[-20:].mean()) > 0 else 1.0
        )

        # NaN guard — any key indicator being NaN aborts
        guards = [ema50_4h, ema200_4h, rsi_4h, hist_4h.iloc[-1], hist_4h.iloc[-2],
                  ema50_1h, atr_1h_val, rsi_now, rsi_prev, atr_val, atr_4h_val,
                  hist_15.iloc[-1], hist_15.iloc[-2]]
        if any(pd.isna(g) for g in guards):
            log.debug("%s: one or more indicators are NaN, skipping", symbol)
            return None

        # ── 4h trend bias ─────────────────────────────────────────────────
        bullish = (close_4h > float(ema50_4h) > float(ema200_4h))
        bearish = (close_4h < float(ema50_4h) < float(ema200_4h))
        # HTF extension vs 4h-EMA200. Used by Type B (counter-trend gate) and
        # Type F (BoS short, distribution-zone gate). Computed here so both
        # blocks see the same value.
        htf_ext = (
            (close_4h - float(ema200_4h)) / float(ema200_4h)
            if float(ema200_4h) > 0 else 0.0
        )

        # ── shared reversal conditions (15m) ──────────────────────────────
        rsi_up   = bool(rsi_now > rsi_prev) and bool(rsi_prev < 45)
        rsi_down = bool(rsi_now < rsi_prev) and bool(rsi_prev > 55)

        # ── near 1h EMA50 ─────────────────────────────────────────────────
        near_ema50 = abs(close_1h - float(ema50_1h)) / float(ema50_1h) < 0.03

        # fallback swing levels (used if no confirmed pivot yet)
        sh_ref = swing_h if swing_h is not None else close_1h * 1.02
        sl_ref = swing_l if swing_l is not None else close_1h * 0.98

        now = datetime.now(tz=UTC)

        # ════════════════════════════════════════════════════════════════════
        # TYPE A — with-trend pullback
        # ════════════════════════════════════════════════════════════════════
        if bullish and near_ema50 and rsi_up:
            strength = _score_a(
                ema_gap_pct=(close_4h - float(ema200_4h)) / float(ema200_4h),
                rsi_delta=float(rsi_now - rsi_prev),
                vol_ratio=vol_ratio,
            )
            if strength >= 0.25:
                self_trend = (
                    settings.a_self_trend_override_enabled
                    and compute_self_trend(
                        df_15m, "long", int(settings.a_self_trend_bars)
                    )
                )
                log.info(
                    "%s: Type A LONG strength=%.2f self_trend=%s",
                    symbol, strength, self_trend,
                )
                return CandidateSignal(
                    timestamp=now, symbol=symbol, side="long",
                    entry_type="A", signal_strength=strength,
                    entry_price_ref=close_4h, atr_14=atr_val,
                    swing_high_1h=sh_ref, swing_low_1h=sl_ref, atr_14_1h=atr_1h_val,
                    self_trend_override=self_trend,
                )

        if bearish and near_ema50 and rsi_down:
            strength = _score_a(
                ema_gap_pct=(float(ema200_4h) - close_4h) / float(ema200_4h),
                rsi_delta=float(rsi_prev - rsi_now),
                vol_ratio=vol_ratio,
            )
            if strength >= 0.25:
                self_trend = (
                    settings.a_self_trend_override_enabled
                    and compute_self_trend(
                        df_15m, "short", int(settings.a_self_trend_bars)
                    )
                )
                log.info(
                    "%s: Type A SHORT strength=%.2f self_trend=%s",
                    symbol, strength, self_trend,
                )
                return CandidateSignal(
                    timestamp=now, symbol=symbol, side="short",
                    entry_type="A", signal_strength=strength,
                    entry_price_ref=close_4h, atr_14=atr_val,
                    swing_high_1h=sh_ref, swing_low_1h=sl_ref, atr_14_1h=atr_1h_val,
                    self_trend_override=self_trend,
                )

        # ════════════════════════════════════════════════════════════════════
        # TYPE D — with-trend breakout (fires when A's pullback never came)
        # ════════════════════════════════════════════════════════════════════
        # Type A only enters on a pullback into 1h-EMA50; on a strong run
        # the pullback never materialises and the bot just watches. D
        # complements A by entering on the 1h close that breaks the prior
        # N-bar high (long) or low (short), with 4h trend agreement and
        # 15m volume confirmation. Stale breakouts (already > d_max_excess
        # past the level) are skipped — chasing a 5 %-late break is the
        # canonical losing version of this setup.
        d_lookback = max(2, int(settings.d_breakout_lookback_bars))
        if len(df_1h) > d_lookback + 1:
            prior_1h = df_1h.iloc[-(d_lookback + 1):-1]
            prior_high = float(prior_1h["high"].max())
            prior_low = float(prior_1h["low"].min())
            high_1h = float(df_1h["high"].iloc[-1])
            low_1h = float(df_1h["low"].iloc[-1])
            bar_range = max(high_1h - low_1h, 1e-12)
            d_min_vol = float(settings.d_min_vol_ratio)
            d_max_excess = float(settings.d_max_excess_pct)
            d_max_ema_dist = float(settings.d_max_ema_dist_pct)
            ema_dist_1h = abs(close_1h - float(ema50_1h)) / float(ema50_1h)

            if (
                bullish
                and close_1h > prior_high
                and prior_high > 0
                and vol_ratio >= d_min_vol
            ):
                if ema_dist_1h > d_max_ema_dist:
                    log.info(
                        "%s: Type D LONG skipped, entry too far from EMA50(1h) "
                        "(%.2f%% > %.2f%%)",
                        symbol, ema_dist_1h * 100, d_max_ema_dist * 100,
                    )
                else:
                    excess = (close_1h - prior_high) / prior_high
                    if excess > d_max_excess:
                        log.info(
                            "%s: Type D LONG skipped, breakout stale "
                            "(close %.2f%% above %d-bar high > %.2f%%)",
                            symbol, excess * 100, d_lookback, d_max_excess * 100,
                        )
                    else:
                        close_pos = (close_1h - low_1h) / bar_range
                        strength = _score_d(vol_ratio, excess, close_pos)
                        if strength >= 0.30:
                            log.info(
                                "%s: Type D LONG (excess=%.2f%% ema_dist=%.2f%% vol=%.2fx) strength=%.2f",
                                symbol, excess * 100, ema_dist_1h * 100, vol_ratio, strength,
                            )
                            return CandidateSignal(
                                timestamp=now, symbol=symbol, side="long",
                                entry_type="D", signal_strength=strength,
                                entry_price_ref=close_1h, atr_14=atr_val,
                                swing_high_1h=sh_ref, swing_low_1h=sl_ref,
                                atr_14_1h=atr_1h_val,
                            )

            if (
                bearish
                and close_1h < prior_low
                and prior_low > 0
                and vol_ratio >= d_min_vol
            ):
                if ema_dist_1h > d_max_ema_dist:
                    log.info(
                        "%s: Type D SHORT skipped, entry too far from EMA50(1h) "
                        "(%.2f%% > %.2f%%)",
                        symbol, ema_dist_1h * 100, d_max_ema_dist * 100,
                    )
                else:
                    excess = (prior_low - close_1h) / prior_low
                    if excess > d_max_excess:
                        log.info(
                            "%s: Type D SHORT skipped, breakdown stale "
                            "(close %.2f%% below %d-bar low > %.2f%%)",
                            symbol, excess * 100, d_lookback, d_max_excess * 100,
                        )
                    else:
                        # Mirror of long: 1.0 = closed at bar low (decisive break).
                        close_pos = (high_1h - close_1h) / bar_range
                        strength = _score_d(vol_ratio, excess, close_pos)
                        if strength >= 0.30:
                            log.info(
                                "%s: Type D SHORT (excess=%.2f%% ema_dist=%.2f%% vol=%.2fx) strength=%.2f",
                                symbol, excess * 100, ema_dist_1h * 100, vol_ratio, strength,
                            )
                            return CandidateSignal(
                                timestamp=now, symbol=symbol, side="short",
                                entry_type="D", signal_strength=strength,
                                entry_price_ref=close_1h, atr_14=atr_val,
                                swing_high_1h=sh_ref, swing_low_1h=sl_ref,
                                atr_14_1h=atr_1h_val,
                            )

        # ════════════════════════════════════════════════════════════════════
        # TYPE E — MA25 Confirmed Bounce (Strategy X)
        #
        # With-trend long on the 4h MA25. Fires only when A and D have
        # already declined (otherwise one of them would have returned above).
        # Unlike A (near 1h-EMA50 with RSI turn) and D (1h breakout),
        # E is a pure 4h structural trade: the bounce low is the natural
        # stop and 4h-ATR is stored for informational context.
        # ════════════════════════════════════════════════════════════════════
        # Pass the FULL H4 frame including the in-progress bar. Strategy E is
        # designed to evaluate the live bar so a setup can be entered as soon
        # as conditions look satisfied, instead of waiting up to 4h for bar
        # close (spec §1 — "USES the in-progress H4 bar"). The host bot's
        # stop-loss absorbs the risk of the live bar reversing before close;
        # the per-symbol 24h cooldown in trader.py prevents the same symbol
        # from re-firing every 60 s while the live bar continues to qualify.
        e_sig = detect_ma25_bounce(symbol, df_4h)
        if e_sig is not None:
            entry_e = e_sig["entry_price"]
            ma25_e  = e_sig["ma25"]
            pct_above_e = (entry_e - ma25_e) / ma25_e * 100.0 if ma25_e > 0 else 0.0
            strength_e = _score_e(
                slope_pct=e_sig["slope_pct"],
                prior_above=e_sig["prior_above"],
                pct_above=pct_above_e,
            )
            if strength_e >= 0.25:
                log.info(
                    "%s: Type E MA25-bounce LONG "
                    "slope=%.2f%% prior_above=%.2f pct_above=%.2f%% "
                    "bounce_low=%.5f strength=%.2f",
                    symbol, e_sig["slope_pct"], e_sig["prior_above"],
                    pct_above_e, e_sig["bounce_low"], strength_e,
                )
                return CandidateSignal(
                    timestamp=now, symbol=symbol, side="long",
                    entry_type="E", signal_strength=strength_e,
                    entry_price_ref=entry_e,
                    atr_14=atr_4h_val,        # 4h ATR: context, not used for SL
                    atr_14_1h=None,
                    swing_high_1h=sh_ref,
                    swing_low_1h=float(e_sig["bounce_low"]),
                    sl_price_hint=float(e_sig["bounce_low"]),  # structural stop
                )

        # ════════════════════════════════════════════════════════════════════
        # TYPE G — MA25 Confirmed Breakdown (bearish mirror of E)
        #
        # Pure 4h structural short. Fires when MA25 is falling, MA25 < MA99,
        # most recent prior bars closed BELOW MA25, current bar is red and
        # closed below MA25 with the upper-wick of the last 3 bars touching
        # the MA25 zone from below. The local high is the natural stop.
        # ════════════════════════════════════════════════════════════════════
        g_sig = detect_ma25_breakdown(symbol, df_4h)
        if g_sig is not None:
            entry_g = g_sig["entry_price"]
            ma25_g  = g_sig["ma25"]
            pct_below_g = (ma25_g - entry_g) / ma25_g * 100.0 if ma25_g > 0 else 0.0
            strength_g = _score_g(
                slope_pct=g_sig["slope_pct"],
                prior_below=g_sig["prior_below"],
                pct_below=pct_below_g,
            )
            if strength_g >= 0.25:
                log.info(
                    "%s: Type G MA25-breakdown SHORT "
                    "slope=%.2f%% prior_below=%.2f pct_below=%.2f%% "
                    "bounce_high=%.5f strength=%.2f",
                    symbol, g_sig["slope_pct"], g_sig["prior_below"],
                    pct_below_g, g_sig["bounce_high"], strength_g,
                )
                return CandidateSignal(
                    timestamp=now, symbol=symbol, side="short",
                    entry_type="G", signal_strength=strength_g,
                    entry_price_ref=entry_g,
                    atr_14=atr_4h_val,        # 4h ATR: context, not used for SL
                    atr_14_1h=None,
                    swing_high_1h=float(g_sig["bounce_high"]),
                    swing_low_1h=sl_ref,
                    sl_price_hint=float(g_sig["bounce_high"]),  # structural stop
                )

        # ════════════════════════════════════════════════════════════════════
        # TYPE F — Break-of-Structure short (1h structural, with-short-trend)
        #
        # Fills the early-downtrend gap left by A and D. A needs the 4h to
        # already be bearish (close < EMA50 < EMA200) and D needs an
        # additional N-bar low break — both wait for the trend to confirm.
        # F triggers as soon as the 1h structure breaks: a fresh swing low
        # was printed, price made a lower-high retest, then the next 1h
        # close closes below the swing low. Allowed in neutral and bearish
        # 4h unconditionally; allowed in bullish 4h only when close_4h is
        # not more than `f_htf_extension_pct` ABOVE 4h-EMA200 — blocks
        # mid-trend pump fades while preserving the canonical "distribution
        # zone" early-reversal short.
        # ════════════════════════════════════════════════════════════════════
        if not bullish or htf_ext <= float(settings.f_htf_extension_pct):
            f_sig = detect_bos_short(symbol, df_1h, vol_ratio_15m=vol_ratio)
            if f_sig is not None:
                strength_f = _score_f(
                    vol_ratio=f_sig["vol_ratio"],
                    excess_pct=f_sig["excess_pct"],
                    lh_age=f_sig["bars_since_lh"],
                    max_age=int(settings.f_max_lower_high_age),
                )
                if strength_f >= 0.25:
                    log.info(
                        "%s: Type F BoS SHORT entry=%.5f swing_low=%.5f LH=%.5f "
                        "excess=%.3f%% age=%d vol=%.2fx strength=%.2f",
                        symbol, f_sig["entry_price"], f_sig["swing_low"],
                        f_sig["lower_high"], f_sig["excess_pct"] * 100,
                        f_sig["bars_since_lh"], f_sig["vol_ratio"], strength_f,
                    )
                    return CandidateSignal(
                        timestamp=now, symbol=symbol, side="short",
                        entry_type="F", signal_strength=strength_f,
                        entry_price_ref=float(f_sig["entry_price"]),
                        atr_14=atr_val,
                        atr_14_1h=atr_1h_val,
                        swing_high_1h=float(f_sig["lower_high"]),
                        swing_low_1h=float(f_sig["swing_low"]),
                        sl_price_hint=float(f_sig["lower_high"]),  # structural stop
                    )
        elif bullish:
            log.debug(
                "%s: Type F SHORT skipped, 4h bullish + close %.2f%% above "
                "EMA200 > %.2f%%",
                symbol, htf_ext * 100, float(settings.f_htf_extension_pct) * 100,
            )

        # ════════════════════════════════════════════════════════════════════
        # TYPE B — counter-trend extreme
        # ════════════════════════════════════════════════════════════════════
        hist_delta_15 = float(hist_15.iloc[-1]) - float(hist_15.iloc[-2])
        b_block = float(settings.b_trend_block_pct)
        b_htf = float(settings.b_htf_extension_pct)
        # `htf_ext` is computed in the shared section above (used by Type F too).

        if float(rsi_4h) > 75 and rsi_down:
            # Refuse to short into a 1h that has been climbing — the most
            # common failure mode of Type B.
            if ema50_1h_slope_pct > b_block:
                log.info(
                    "%s: Type B SHORT blocked by 1h trend gate "
                    "(slope=%.2f%% > %.2f%%)",
                    symbol, ema50_1h_slope_pct * 100, b_block * 100,
                )
            elif htf_ext > b_htf:
                # 4h close is well above 4h-EMA200 — this is a mid-trend
                # pump, not an exhaustion top. Fading it is the canonical
                # losing trade (BNB/USDT 2026-05-06 case study).
                log.info(
                    "%s: Type B SHORT blocked by 4h HTF gate "
                    "(close %.2f%% above 4h-EMA200 > %.2f%%)",
                    symbol, htf_ext * 100, b_htf * 100,
                )
            else:
                strength = _score_b(float(rsi_4h), "short", hist_delta_15)
                if strength >= 0.20:
                    log.info("%s: Type B SHORT (RSI4h=%.1f) strength=%.2f",
                             symbol, rsi_4h, strength)
                    return CandidateSignal(
                        timestamp=now, symbol=symbol, side="short",
                        entry_type="B", signal_strength=strength,
                        entry_price_ref=close_4h, atr_14=atr_val,
                        swing_high_1h=sh_ref, swing_low_1h=sl_ref, atr_14_1h=atr_1h_val,
                    )

        if float(rsi_4h) < 25 and rsi_up:
            if ema50_1h_slope_pct < -b_block:
                log.info(
                    "%s: Type B LONG blocked by 1h trend gate "
                    "(slope=%.2f%% < -%.2f%%)",
                    symbol, ema50_1h_slope_pct * 100, b_block * 100,
                )
            elif htf_ext < -b_htf:
                # Symmetric: 4h close well below 4h-EMA200 → mid-downtrend,
                # not an exhaustion bottom. Don't fade.
                log.info(
                    "%s: Type B LONG blocked by 4h HTF gate "
                    "(close %.2f%% below 4h-EMA200 < -%.2f%%)",
                    symbol, htf_ext * 100, b_htf * 100,
                )
            else:
                strength = _score_b(float(rsi_4h), "long", hist_delta_15)
                if strength >= 0.20:
                    log.info("%s: Type B LONG (RSI4h=%.1f) strength=%.2f",
                             symbol, rsi_4h, strength)
                    return CandidateSignal(
                        timestamp=now, symbol=symbol, side="long",
                        entry_type="B", signal_strength=strength,
                        entry_price_ref=close_4h, atr_14=atr_val,
                        swing_high_1h=sh_ref, swing_low_1h=sl_ref, atr_14_1h=atr_1h_val,
                    )

        # ════════════════════════════════════════════════════════════════════
        # TYPE C — range fade (only when 4h is neutral)
        # ════════════════════════════════════════════════════════════════════
        if (not bullish) and (not bearish) and swing_h is not None and swing_l is not None:
            rng = swing_h - swing_l
            if rng > 0:
                pos = (close_1h - swing_l) / rng   # 0 = at low, 1 = at high

                # Hard bound: pos outside [0, 1] means the 1h range has
                # already broken (price closed above swing_high or below
                # swing_low). Fading a confirmed breakout is exactly the
                # opposite of a range-fade trade — bail out instead.
                if pos < 0.0 or pos > 1.0:
                    log.debug(
                        "%s: Type C rejected, range broken (pos=%.2f)",
                        symbol, pos,
                    )
                    return None

                # Extra safety: require 1h price to actually be hugging its
                # EMA50 (no strong 1h directional trend). Without this a
                # neutral 4h with a fast 1h leg keeps producing fade signals
                # against the active move.
                if not near_ema50:
                    log.debug(
                        "%s: Type C rejected, 1h not flat (close=%.4f ema50=%.4f)",
                        symbol, close_1h, float(ema50_1h),
                    )
                    return None

                if pos < 0.20 and rsi_up:           # near low → long
                    proximity = 1.0 - pos / 0.20
                    strength = _score_c(proximity, float(rsi_now - rsi_prev))
                    if strength >= 0.20:
                        log.info("%s: Type C LONG (pos=%.2f) strength=%.2f",
                                 symbol, pos, strength)
                        return CandidateSignal(
                            timestamp=now, symbol=symbol, side="long",
                            entry_type="C", signal_strength=strength,
                            entry_price_ref=close_1h, atr_14=atr_val,
                            swing_high_1h=swing_h, swing_low_1h=swing_l, atr_14_1h=atr_1h_val,
                        )

                if pos > 0.80 and rsi_down:         # near high → short
                    proximity = (pos - 0.80) / 0.20
                    strength = _score_c(proximity, float(rsi_prev - rsi_now))
                    if strength >= 0.20:
                        log.info("%s: Type C SHORT (pos=%.2f) strength=%.2f",
                                 symbol, pos, strength)
                        return CandidateSignal(
                            timestamp=now, symbol=symbol, side="short",
                            entry_type="C", signal_strength=strength,
                            entry_price_ref=close_1h, atr_14=atr_val,
                            swing_high_1h=swing_h, swing_low_1h=swing_l, atr_14_1h=atr_1h_val,
                        )

        return None
