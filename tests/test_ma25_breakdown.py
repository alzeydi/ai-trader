"""Tests for detect_ma25_breakdown() and engine Type G integration.

Mirrors tests/test_ma25_bounce.py to keep symmetry between the two strategies.
"""

from __future__ import annotations

import unittest.mock as mock

import numpy as np
import pandas as pd

from src.risk.sizing import size_position
from src.risk.types import AccountState
from src.signal.ma25_breakdown import detect_ma25_breakdown
from src.signal.types import CandidateSignal

# ── fixture builder ───────────────────────────────────────────────────────────

def _make_df(n: int = 110) -> pd.DataFrame:
    """Falling trend 120→90 with a clean breakdown pattern at the end.

    After construction all conditions pass:
      trend  — MA25 ≈ 93.5 < MA99 ≈ 103.2, slope ≈ −2.7 %
      prior  — all 10 prior bars close below MA25
      red    — last bar: open > close
      below  — last close ≈ 2 % below MA25
      touch  — highs[-2] tags the touch zone (within 0.5 % below MA25)
    """
    base = np.linspace(120.0, 90.0, n)
    closes = base.copy()

    ma25_est = float(np.mean(closes[-25:]))

    # Shape last 3 bars as a breakdown: two bars near MA25, current bar red
    closes[-3] = ma25_est * 0.990
    closes[-2] = ma25_est * 0.992
    closes[-1] = ma25_est * 0.980   # entry bar

    opens = closes.copy()
    opens[-1] = closes[-1] * 1.003   # red last bar

    highs = closes * 1.002
    lows  = closes * 0.998

    actual_ma25 = float(np.mean(closes[-25:]))
    highs[-2] = actual_ma25 * 0.998   # tag zone within 0.5 % below MA25

    idx = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": 1000.0},
        index=idx,
    )


# ── happy path ────────────────────────────────────────────────────────────────

def test_happy_path_returns_signal() -> None:
    df = _make_df()
    sig = detect_ma25_breakdown("BTCUSDT", df)

    assert sig is not None
    assert sig["entry_price"] < sig["ma25"]
    assert sig["slope_pct"] <= -0.5
    assert sig["prior_below"] >= 0.7
    # bounce_high lives in the touch zone: close to MA25 from below or
    # within touch_min_pct above (we put it just below MA25).
    assert sig["bounce_high"] >= sig["ma25"] * (1.0 - 0.005)
    assert sig["bounce_high"] <= sig["ma25"] * 1.03


def test_signal_keys() -> None:
    df = _make_df()
    sig = detect_ma25_breakdown("BTCUSDT", df)
    assert sig is not None
    assert set(sig.keys()) == {"entry_price", "ma25", "slope_pct", "prior_below", "bounce_high"}


# ── trend filter failures ─────────────────────────────────────────────────────

def test_fail_insufficient_candles() -> None:
    df = _make_df(n=100)
    assert detect_ma25_breakdown("X", df) is None


def test_fail_slope_not_falling_enough() -> None:
    # Last 37 bars flat at 100; earlier 73 bars at 110 → MA99 ≈ 106.3 > MA25 = 100,
    # so MA25 < MA99 (stack passes). Slope = 0 % which is > −0.5 % → fails.
    n = 110
    closes = np.concatenate([np.full(73, 110.0), np.full(37, 100.0)])
    opens  = closes * 1.003
    highs  = closes * 1.002
    lows   = closes * 0.998
    idx = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    df = pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": 1000.0},
        index=idx,
    )
    assert detect_ma25_breakdown("X", df) is None


def test_fail_ma25_not_below_ma99() -> None:
    # Reverse: uptrend → MA25 > MA99.
    n = 110
    base = np.linspace(90.0, 120.0, n)
    closes = base.copy()
    opens  = closes * 0.997
    highs  = closes * 1.002
    lows   = closes * 0.998
    idx = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    df = pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": 1000.0},
        index=idx,
    )
    assert detect_ma25_breakdown("X", df) is None


# ── breakdown confirmation failures ───────────────────────────────────────────

def test_fail_prior_below_too_low() -> None:
    df = _make_df()
    closes = df["close"].values.copy()
    highs  = df["high"].values.copy()

    ma25 = float(np.mean(closes[-25:]))
    # 4 of the 10 prior bars (indices 99..108) close clearly above MA25
    # → below_ratio = 6/10 = 0.60 < 0.70.
    closes[99:103] = ma25 * 1.05
    highs[99:103]  = ma25 * 1.05 * 1.002

    df["close"] = closes
    df["high"]  = highs

    assert detect_ma25_breakdown("X", df) is None


def test_fail_bar_not_red() -> None:
    df = _make_df()
    opens = df["open"].values.copy()
    # Make the last bar green: open below close.
    opens[-1] = df["close"].values[-1] * 0.995
    df["open"] = opens
    assert detect_ma25_breakdown("X", df) is None


def test_fail_close_above_ma25() -> None:
    df = _make_df()
    closes = df["close"].values.copy()
    opens  = df["open"].values.copy()

    ma25 = float(np.mean(closes[-25:]))
    closes[-1] = ma25 * 1.005   # closes above MA25
    opens[-1]  = ma25 * 1.010   # still red

    df["close"] = closes
    df["open"]  = opens

    assert detect_ma25_breakdown("X", df) is None


def test_fail_too_far_below_ma25() -> None:
    df = _make_df()
    closes = df["close"].values.copy()
    opens  = df["open"].values.copy()

    ma25 = float(np.mean(closes[-25:]))
    # 5 % below MA25 → past the 4 % ceiling.
    closes[-1] = ma25 * 0.95
    opens[-1]  = ma25 * 0.955   # still red

    df["close"] = closes
    df["open"]  = opens

    assert detect_ma25_breakdown("X", df) is None


def test_fail_never_touched_ma25() -> None:
    df = _make_df()
    highs = df["high"].values.copy()
    closes = df["close"].values

    ma25 = float(np.mean(closes[-25:]))
    # All 3 highs in the lookback sit 1 % below MA25 → below touch_max
    # (which is 0.5 % below). The pullback never reached MA25.
    highs[-3] = ma25 * 0.99
    highs[-2] = ma25 * 0.99
    highs[-1] = ma25 * 0.99

    df["high"] = highs

    assert detect_ma25_breakdown("X", df) is None


def test_fail_overshoot_too_far_above() -> None:
    df = _make_df()
    highs = df["high"].values.copy()
    closes = df["close"].values

    ma25 = float(np.mean(closes[-25:]))
    # One wick pierces 4 % above MA25 → above touch_min_pct (3 %).
    highs[-2] = ma25 * 1.04

    df["high"] = highs

    assert detect_ma25_breakdown("X", df) is None


# ── engine integration (Type G) ───────────────────────────────────────────────

def _make_engine_dfs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(df_4h, df_1h, df_15m) where ONLY Type G fires."""
    # df_4h: falling 120→90, breakdown at end (301 bars).
    n4 = 301
    base4 = np.linspace(120.0, 90.0, n4)
    closes4 = base4.copy()
    ma25_est = float(np.mean(closes4[-25:]))
    closes4[-3] = ma25_est * 0.992
    closes4[-2] = ma25_est * 0.990
    closes4[-1] = ma25_est * 0.980
    opens4 = closes4.copy()
    opens4[-1] = closes4[-1] * 1.003
    highs4 = closes4 * 1.002
    lows4  = closes4 * 0.998
    actual_ma25 = float(np.mean(closes4[-25:]))
    highs4[-3] = actual_ma25 * 0.998
    idx4 = pd.date_range("2024-01-01", periods=n4, freq="4h", tz="UTC")
    df_4h = pd.DataFrame(
        {"open": opens4, "high": highs4, "low": lows4,
         "close": closes4, "volume": 1000.0},
        index=idx4,
    )

    # df_1h: 99 bars at 80, last bar at 91 → close > EMA50, near_ema50=False.
    # No 1h structural break (close went UP, not below the 20-bar low) →
    # Type D and Type F SHORT cannot fire.
    closes1 = np.concatenate([np.full(99, 80.0), [91.0]])
    idx1 = pd.date_range("2024-01-01", periods=100, freq="1h", tz="UTC")
    df_1h = pd.DataFrame(
        {"open": closes1, "high": closes1 * 1.002,
         "low": closes1 * 0.998, "close": closes1, "volume": 1000.0},
        index=idx1,
    )

    # df_15m: alternating around 91 → RSI ≈ 50, no reversal triggers.
    closes15 = np.tile([91.1, 90.9], 100).astype(float)
    idx15 = pd.date_range("2024-01-01", periods=200, freq="15min", tz="UTC")
    df_15m = pd.DataFrame(
        {"open": closes15, "high": closes15 * 1.002,
         "low": closes15 * 0.998, "close": closes15, "volume": 1000.0},
        index=idx15,
    )

    return df_4h, df_1h, df_15m


def test_engine_emits_type_g_candidate() -> None:
    """_evaluate with a valid MA25-breakdown 4h series must return entry_type='G'."""
    from src.signal.engine import SignalEngine

    df_4h, df_1h, df_15m = _make_engine_dfs()
    engine = SignalEngine(client=mock.MagicMock())
    sig = engine._evaluate("BTCUSDT", df_4h, df_1h, df_15m)

    assert sig is not None, "expected a CandidateSignal, got None"
    assert sig.entry_type == "G"
    assert sig.side == "short"
    assert sig.sl_price_hint is not None
    assert sig.sl_price_hint > sig.entry_price_ref  # stop is above entry


def test_sizing_uses_bounce_high_as_stop() -> None:
    """size_position with sl_price_hint must place SL at bounce_high for G."""
    entry   = 100.0
    b_high  = 102.5   # 2.5 % above entry — the rejection high
    atr_4h  = 3.0

    candidate = CandidateSignal(
        symbol="BTCUSDT", side="short", entry_type="G",
        signal_strength=0.60,
        entry_price_ref=entry, atr_14=atr_4h, atr_14_1h=None,
        swing_high_1h=b_high, swing_low_1h=95.0,
        sl_price_hint=b_high,
    )
    account = AccountState(
        equity=10_000.0, open_positions=0,
        daily_pnl_pct=0.0, consecutive_losses=0,
        free_margin_usd=10_000.0,
    )
    veto = mock.MagicMock()
    veto.confidence = 0.85
    veto.invalidation_signal = "test"

    order = size_position(candidate, account, veto)

    assert order is not None
    # SL must be at bounce_high (within float rounding).
    assert abs(order.stop_loss - b_high) < 1e-4

    # TP for Type G is anchored to 4h ATR × atr_take_multiplier (mirror of E).
    from src.config import settings
    expected_tp = entry - atr_4h * settings.atr_take_multiplier
    assert abs(order.take_profit - expected_tp) < 1e-4
