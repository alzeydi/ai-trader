"""Tests for detect_bos_short() and engine Type F integration."""

from __future__ import annotations

import unittest.mock as mock

import numpy as np
import pandas as pd

from src.signal.bos_short import detect_bos_short

# ── fixture builder ───────────────────────────────────────────────────────────

def _make_df_1h(
    *,
    n: int = 30,
    swing_low_at: int = -10,   # negative iloc of the swing-low bar
    swing_low: float = 100.0,
    lower_high_at: int = -3,
    lower_high: float = 102.0,
    breakdown_close: float = 99.0,
) -> pd.DataFrame:
    """Build a 1h frame that satisfies BoS-short conditions by construction.

    All bars default to a flat 101.0; the swing-low bar dips to `swing_low`,
    the lower-high bar pokes up to `lower_high`, and the LIVE bar (iloc[-1])
    closes at `breakdown_close`. Volume is constant.
    """
    closes = np.full(n, 101.0)
    highs  = closes * 1.001
    lows   = closes * 0.999

    # Swing low.
    sl_idx = swing_low_at
    closes[sl_idx] = swing_low * 1.001
    lows[sl_idx]   = swing_low

    # Lower high.
    lh_idx = lower_high_at
    closes[lh_idx] = lower_high * 0.999
    highs[lh_idx]  = lower_high

    # Live bar = breakdown.
    closes[-1] = breakdown_close
    lows[-1]   = breakdown_close * 0.999
    highs[-1]  = breakdown_close * 1.001

    opens = closes.copy()
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": 1000.0},
        index=idx,
    )


# ── happy path ────────────────────────────────────────────────────────────────

def test_happy_path() -> None:
    df = _make_df_1h()
    sig = detect_bos_short("BTCUSDT", df, vol_ratio_15m=1.5)

    assert sig is not None
    assert sig["entry_price"] == 99.0
    assert sig["swing_low"] == 100.0
    assert sig["lower_high"] == 102.0
    assert sig["bars_since_lh"] == 2   # lower_high_at=-3 → 2 bars before live
    assert sig["excess_pct"] > 0
    assert sig["vol_ratio"] == 1.5


# ── failure modes ─────────────────────────────────────────────────────────────

def test_fail_no_break() -> None:
    """Live close above swing_low → no break."""
    df = _make_df_1h(breakdown_close=100.5)
    assert detect_bos_short("X", df, vol_ratio_15m=1.5) is None


def test_fail_volume_too_low() -> None:
    df = _make_df_1h()
    assert detect_bos_short("X", df, vol_ratio_15m=1.0) is None


def test_fail_stale_break() -> None:
    """Close 5 % below swing_low — breakdown bar already gone."""
    df = _make_df_1h(breakdown_close=95.0)
    assert detect_bos_short("X", df, vol_ratio_15m=1.5) is None


def test_fail_no_lower_high() -> None:
    """LH equals swing_low → no real retest."""
    df = _make_df_1h(lower_high=99.5)
    assert detect_bos_short("X", df, vol_ratio_15m=1.5) is None


def test_fail_lh_too_old() -> None:
    """Lower-high formed >6 bars before the breakdown bar."""
    df = _make_df_1h(swing_low_at=-12, lower_high_at=-9)
    # bars_since_lh = 8, > f_max_lower_high_age default 6 → fail.
    assert detect_bos_short("X", df, vol_ratio_15m=1.5) is None


def test_fail_too_few_bars() -> None:
    df = _make_df_1h(n=10)
    assert detect_bos_short("X", df, vol_ratio_15m=1.5) is None


def test_fail_swing_low_is_last_closed_bar() -> None:
    """Swing low is at iloc[-2] — no room for an LH between SL and live bar."""
    df = _make_df_1h(swing_low_at=-2, lower_high_at=-3)
    # The "after_lo" slice would be empty → return None.
    assert detect_bos_short("X", df, vol_ratio_15m=1.5) is None


# ── engine integration (Type F) ───────────────────────────────────────────────

def _flat_4h() -> pd.DataFrame:
    """Neutral 4h: mild noise around 100, ~300 bars. Avoids A/B/D/E/G triggers."""
    rng = np.random.default_rng(42)
    closes = 100.0 + rng.normal(0, 0.5, 300)
    opens = closes
    highs = closes * 1.001
    lows  = closes * 0.999
    idx = pd.date_range("2024-01-01", periods=300, freq="4h", tz="UTC")
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": 1000.0},
        index=idx,
    )


def _bos_1h() -> pd.DataFrame:
    """1h that satisfies BoS-short structurally.

    100 bars (engine min). Bars 0..78 walk down 110→101 to push EMA50 well
    above the live close (so near_ema50 is False — Type C is blocked).
    Bar 80 prints the swing low at 100. Bar 97 prints the lower-high at 102.
    The live bar (99) closes at 99.0 — break of structure. Volume on the
    last bar is 4× the trailing mean to satisfy f_min_vol_ratio.
    """
    n = 100
    closes = np.full(n, 110.0)
    # Decline 0..78
    closes[:79] = np.linspace(110.0, 101.0, 79)
    # Light noise 79..96
    rng = np.random.default_rng(7)
    closes[79:97] = 101.0 + rng.normal(0, 0.2, 18)
    # Swing low at idx 80
    closes[80] = 100.5
    # Lower high at idx 97
    closes[97] = 101.9
    # Pre-live bar
    closes[98] = 100.6
    # Live = breakdown
    closes[99] = 99.0

    highs = closes * 1.001
    lows  = closes * 0.999
    lows[80]  = 100.0
    highs[97] = 102.0

    opens = closes.copy()
    volumes = np.full(n, 1000.0)
    volumes[-1] = 4000.0   # 15m volume? No — engine uses 15m for vol_ratio.
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=idx,
    )


def _bos_15m_with_vol_spike() -> pd.DataFrame:
    """15m frame with the last bar carrying 2× the 20-bar mean volume."""
    n = 200
    closes = np.tile([99.1, 98.9], n // 2).astype(float)
    opens = closes
    highs = closes * 1.001
    lows  = closes * 0.999
    volumes = np.full(n, 1000.0)
    volumes[-1] = 2500.0
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=idx,
    )


def test_engine_emits_type_f_candidate() -> None:
    """A 4h-neutral / 1h-BoS / 15m-vol-spike combo must yield Type F SHORT."""
    from src.signal.engine import SignalEngine

    df_4h = _flat_4h()
    df_1h = _bos_1h()
    df_15m = _bos_15m_with_vol_spike()

    engine = SignalEngine(client=mock.MagicMock())
    sig = engine._evaluate("BTCUSDT", df_4h, df_1h, df_15m)

    assert sig is not None, "expected a CandidateSignal, got None"
    assert sig.entry_type == "F"
    assert sig.side == "short"
    assert sig.sl_price_hint is not None
    assert sig.sl_price_hint > sig.entry_price_ref


def test_engine_blocks_f_in_strong_bullish_4h() -> None:
    """When 4h is bullish AND close_4h >> EMA200(4h) by > 5 %, F must NOT fire."""
    from src.signal.engine import SignalEngine

    # Build a strongly bullish 4h: close 10 % above EMA200.
    n4 = 300
    base = np.linspace(100.0, 200.0, n4)   # huge sustained rally
    df_4h = pd.DataFrame(
        {"open": base, "high": base * 1.002, "low": base * 0.998,
         "close": base, "volume": 1000.0},
        index=pd.date_range("2024-01-01", periods=n4, freq="4h", tz="UTC"),
    )
    df_1h = _bos_1h()
    df_15m = _bos_15m_with_vol_spike()

    engine = SignalEngine(client=mock.MagicMock())
    sig = engine._evaluate("BTCUSDT", df_4h, df_1h, df_15m)

    if sig is not None:
        assert sig.entry_type != "F", (
            "F must be blocked in strong bullish 4h (mid-trend pump territory)"
        )
