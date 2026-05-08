"""Tests for compute_self_trend (15m → 30m aggregation)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.signal.self_trend import compute_self_trend


def _df_15m(
    *,
    n: int = 30,
    pattern: list[str] | None = None,
    start: str = "2024-01-01 00:00",
) -> pd.DataFrame:
    """Build a 15m frame aligned to wall-clock 30m boundaries.

    `pattern` is a list of `"green"` / `"red"` / `"flat"` per 30m candle,
    applied to the LAST `len(pattern)` 30m candles. Earlier bars are flat.
    Bars before/after the pattern have open == close so they don't
    accidentally satisfy direction checks.
    """
    idx = pd.date_range(start, periods=n, freq="15min", tz="UTC")
    base = 100.0
    closes = np.full(n, base)
    opens  = np.full(n, base)
    highs  = np.full(n, base * 1.001)
    lows   = np.full(n, base * 0.999)

    if pattern:
        minutes = idx.minute
        last_close_pos = None
        for j in range(n - 1, 0, -1):
            if minutes[j] in (15, 45) and minutes[j - 1] in (0, 30):
                last_close_pos = j
                break
        assert last_close_pos is not None, "fixture must contain a closed 30m"

        for k, kind in enumerate(reversed(pattern)):
            close_i = last_close_pos - 2 * k
            open_i = close_i - 1
            if kind == "green":
                opens[open_i]   = base
                closes[open_i]  = base + 0.5
                opens[close_i]  = base + 0.5
                closes[close_i] = base + 1.0
            elif kind == "red":
                opens[open_i]   = base
                closes[open_i]  = base - 0.5
                opens[close_i]  = base - 0.5
                closes[close_i] = base - 1.0
            else:  # flat
                opens[open_i]   = base
                closes[open_i]  = base
                opens[close_i]  = base
                closes[close_i] = base

    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows,
         "close": closes, "volume": 1000.0},
        index=idx,
    )


# ── happy paths ───────────────────────────────────────────────────────────────

def test_5_green_long_returns_true() -> None:
    df = _df_15m(pattern=["green"] * 5)
    assert compute_self_trend(df, "long", 5) is True


def test_5_red_short_returns_true() -> None:
    df = _df_15m(pattern=["red"] * 5)
    assert compute_self_trend(df, "short", 5) is True


def test_5_red_long_returns_false() -> None:
    df = _df_15m(pattern=["red"] * 5)
    assert compute_self_trend(df, "long", 5) is False


def test_4_green_1_red_long_returns_false() -> None:
    df = _df_15m(pattern=["green", "green", "green", "green", "red"])
    # The most recent 30m is red — 5-bar streak broken.
    assert compute_self_trend(df, "long", 5) is False


def test_3_green_streak_with_bars_3() -> None:
    df = _df_15m(pattern=["red", "red", "green", "green", "green"])
    # Last 3 are green, but bars=5 must look at the older red ones too.
    assert compute_self_trend(df, "long", 3) is True
    assert compute_self_trend(df, "long", 4) is False
    assert compute_self_trend(df, "long", 5) is False


def test_flat_bar_breaks_streak() -> None:
    df = _df_15m(pattern=["green", "green", "flat", "green", "green"])
    # close == open is not "in trend" for either side.
    assert compute_self_trend(df, "long", 5) is False


# ── edge cases ────────────────────────────────────────────────────────────────

def test_too_few_bars() -> None:
    df = _df_15m(n=5)
    assert compute_self_trend(df, "long", 5) is False


def test_invalid_side() -> None:
    df = _df_15m(pattern=["green"] * 5)
    assert compute_self_trend(df, "neutral", 5) is False  # type: ignore[arg-type]


def test_zero_bars() -> None:
    df = _df_15m(pattern=["green"] * 5)
    assert compute_self_trend(df, "long", 0) is False


def test_non_datetime_index() -> None:
    df = _df_15m(pattern=["green"] * 5).reset_index(drop=True)
    assert compute_self_trend(df, "long", 5) is False


def test_disabled_at_engine_level() -> None:
    """Engine consults the setting; helper itself ignores it.

    Verifies that the helper still works regardless of the global toggle —
    the gate is the trader's responsibility, not the helper's.
    """
    df = _df_15m(pattern=["green"] * 5)
    assert compute_self_trend(df, "long", 5) is True


# ── alignment robustness ──────────────────────────────────────────────────────

def test_misaligned_tail_in_progress_30m() -> None:
    """When the LAST 15m bar opens a new (incomplete) 30m, that bar is
    skipped — we only count the 5 closed pairs that precede it."""
    # 30m candles aligned at :00 and :30. Add ONE extra bar at :30 with
    # the same OHLC as the previous (in-progress 30m has no close yet).
    base_pattern = ["green"] * 5
    df_full = _df_15m(pattern=base_pattern, n=12)   # 12 = 6 pairs of 15m
    # Append a "live" bar at the next :00 — opens a 7th 30m candle.
    last_ts = df_full.index[-1]
    next_ts = last_ts + pd.Timedelta(minutes=15)
    extra = pd.DataFrame(
        {"open": [100.0], "high": [100.1], "low": [99.9],
         "close": [100.0], "volume": [1000.0]},
        index=pd.DatetimeIndex([next_ts]),
    )
    df = pd.concat([df_full, extra])
    # The 5 most recent CLOSED pairs are still all green, so override fires.
    assert compute_self_trend(df, "long", 5) is True
