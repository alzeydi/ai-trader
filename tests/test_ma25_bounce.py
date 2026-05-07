"""Tests for detect_ma25_bounce() — synthetic OHLCV data, no network."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.signal.ma25_bounce import detect_ma25_bounce

# ── fixture builder ───────────────────────────────────────────────────────────

def _make_df(n: int = 110) -> pd.DataFrame:
    """Rising trend 90→120 with a clean bounce pattern at the end.

    After construction all conditions pass:
      trend  — MA25 ≈ 116.5 > MA99 ≈ 106.8, slope ≈ 2.7 %
      prior  — all 10 prior bars close above MA25
      green  — last bar: open < close
      above  — last close ≈ 2 % above MA25
      touch  — lows[-2] dips into the touch zone (within 0.5 % above MA25)
    """
    base = np.linspace(90.0, 120.0, n)
    closes = base.copy()

    # Estimate MA25 from the raw trend; patch is ≤2 % so the real MA25
    # computed inside detect_ma25_bounce stays within 1 % of this estimate.
    ma25_est = float(np.mean(closes[-25:]))

    # Shape last 3 bars as a bounce: two bars near MA25, current bar green
    closes[-3] = ma25_est * 1.010
    closes[-2] = ma25_est * 1.008
    closes[-1] = ma25_est * 1.020   # entry bar

    opens = closes.copy()
    opens[-1] = closes[-1] * 0.997  # green last bar

    highs = closes * 1.002
    lows  = closes * 0.998

    # Make lows[-2] dip into the touch zone (within 0.5 % above MA25).
    # Recompute MA25 from the patched closes so the threshold is exact.
    actual_ma25 = float(np.mean(closes[-25:]))
    lows[-2] = actual_ma25 * 1.002

    idx = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": 1000.0},
        index=idx,
    )


# ── happy path ────────────────────────────────────────────────────────────────

def test_happy_path_returns_signal() -> None:
    df = _make_df()
    sig = detect_ma25_bounce("BTCUSDT", df)

    assert sig is not None
    assert sig["entry_price"] > sig["ma25"]
    assert sig["slope_pct"] >= 0.5
    assert sig["prior_above"] >= 0.7
    assert sig["bounce_low"] <= sig["ma25"] * 1.005
    assert sig["bounce_low"] >= sig["ma25"] * 0.97


def test_signal_is_typed_dict() -> None:
    """Return value must satisfy MA25BounceSignal TypedDict keys."""
    df = _make_df()
    sig = detect_ma25_bounce("BTCUSDT", df)
    assert sig is not None
    assert set(sig.keys()) == {"entry_price", "ma25", "slope_pct", "prior_above", "bounce_low"}


# ── trend filter failures ─────────────────────────────────────────────────────

def test_fail_insufficient_candles() -> None:
    df = _make_df(n=100)          # one bar short of the 101 minimum
    assert detect_ma25_bounce("X", df) is None


def test_fail_slope_too_low() -> None:
    # Last 37 bars flat at 100 → MA25_now = MA25_past = 100, slope = 0 %.
    # Earlier 73 bars at 90 keep MA99 ≈ 93.7, so MA25 > MA99 (stack passes)
    # and slope is the only failing gate.
    n = 110
    closes = np.concatenate([np.full(73, 90.0), np.full(37, 100.0)])
    opens  = closes * 0.997
    highs  = closes * 1.002
    lows   = closes * 0.998
    idx = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    df = pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": 1000.0},
        index=idx,
    )
    assert detect_ma25_bounce("X", df) is None


def test_fail_ma25_not_above_ma99() -> None:
    # Reverse the trend so early candles are high and recent ones are low.
    n = 110
    base = np.linspace(120.0, 90.0, n)   # downtrend → MA25 < MA99
    closes = base.copy()
    opens  = closes * 0.997
    highs  = closes * 1.002
    lows   = closes * 0.998
    idx = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    df = pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": 1000.0},
        index=idx,
    )
    assert detect_ma25_bounce("X", df) is None


# ── bounce confirmation failures ──────────────────────────────────────────────

def test_fail_prior_above_too_low() -> None:
    df = _make_df()
    closes = df["close"].values.copy()
    lows   = df["low"].values.copy()

    # 4 of the 10 prior bars (candles[-11:-1] = indices 99..108) close
    # clearly below MA25 → above_ratio = 6/10 = 0.60 < 0.70.
    ma25 = float(np.mean(closes[-25:]))
    closes[99:103] = ma25 * 0.95   # clearly below MA25
    lows[99:103]   = ma25 * 0.95 * 0.998

    df["close"] = closes
    df["low"]   = lows

    assert detect_ma25_bounce("X", df) is None


def test_fail_bar_not_green() -> None:
    df = _make_df()
    opens = df["open"].values.copy()
    # Make the last bar red: open above close.
    opens[-1] = df["close"].values[-1] * 1.005
    df["open"] = opens
    assert detect_ma25_bounce("X", df) is None


def test_fail_close_below_ma25() -> None:
    df = _make_df()
    closes = df["close"].values.copy()
    opens  = df["open"].values.copy()

    ma25 = float(np.mean(closes[-25:]))
    # Last bar closes just below MA25 but is still green.
    closes[-1] = ma25 * 0.995
    opens[-1]  = ma25 * 0.990

    df["close"] = closes
    df["open"]  = opens

    assert detect_ma25_bounce("X", df) is None


def test_fail_too_far_above_ma25() -> None:
    df = _make_df()
    closes = df["close"].values.copy()
    opens  = df["open"].values.copy()

    ma25 = float(np.mean(closes[-25:]))
    # 5 % above MA25 → past the 4 % ceiling.
    closes[-1] = ma25 * 1.05
    opens[-1]  = ma25 * 1.045   # still green

    df["close"] = closes
    df["open"]  = opens

    assert detect_ma25_bounce("X", df) is None


def test_fail_never_touched_ma25() -> None:
    df = _make_df()
    lows = df["low"].values.copy()
    closes = df["close"].values

    ma25 = float(np.mean(closes[-25:]))
    # All 3 lows in the lookback sit 1 % above MA25 → above touch_max (0.5 %).
    lows[-3] = ma25 * 1.01
    lows[-2] = ma25 * 1.01
    lows[-1] = ma25 * 1.01

    df["low"] = lows

    assert detect_ma25_bounce("X", df) is None


def test_fail_deep_wick() -> None:
    df = _make_df()
    lows = df["low"].values.copy()
    closes = df["close"].values

    ma25 = float(np.mean(closes[-25:]))
    # One wick pierces 4 % below MA25 → below touch_min (3 %).
    lows[-2] = ma25 * 0.96

    df["low"] = lows

    assert detect_ma25_bounce("X", df) is None


# ── edge: exactly-at-threshold values ────────────────────────────────────────

def test_pct_above_at_boundary_passes() -> None:
    """Close exactly at pct_above_max (4.0 %) should still pass."""
    df = _make_df()
    closes = df["close"].values.copy()
    opens  = df["open"].values.copy()

    ma25 = float(np.mean(closes[-25:]))
    closes[-1] = ma25 * 1.040
    opens[-1]  = ma25 * 1.035   # green

    df["close"] = closes
    df["open"]  = opens

    # Recompute MA25 from patched closes and adjust lows so touch still holds.
    actual_ma25 = float(np.mean(df["close"].values[-25:]))
    lows = df["low"].values.copy()
    lows[-2] = actual_ma25 * 1.002
    df["low"] = lows

    result = detect_ma25_bounce("X", df)
    assert result is not None


def test_prior_above_exactly_07_passes() -> None:
    """Exactly 7/10 prior bars above MA25 must pass."""
    df = _make_df()
    closes = df["close"].values.copy()
    lows   = df["low"].values.copy()

    ma25 = float(np.mean(closes[-25:]))
    # Set 3 prior bars (of 10) just below MA25.
    closes[99:102] = ma25 * 0.995
    lows[99:102]   = ma25 * 0.995 * 0.998
    df["close"] = closes
    df["low"]   = lows

    result = detect_ma25_bounce("X", df)
    assert result is not None
