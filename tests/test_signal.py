"""Signal-engine tests using fully synthetic OHLCV data (no network)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.signal.engine import SignalEngine
from src.signal.types import CandidateSignal


# ── OHLCV builder ────────────────────────────────────────────────────────────

def _df(closes: list[float], volumes: list[float] | None = None, freq_min: int = 240) -> pd.DataFrame:
    n = len(closes)
    vols = volumes or [1000.0] * n
    idx = pd.date_range("2024-01-01", periods=n, freq=f"{freq_min}min", tz="UTC")
    return pd.DataFrame(
        {
            "open":   closes,
            "high":   [c * 1.002 for c in closes],
            "low":    [c * 0.998 for c in closes],
            "close":  closes,
            "volume": vols,
        },
        index=idx,
    )


# ── shared 4h uptrend (bullish, RSI high but ≤99) ───────────────────────────

def _4h_bullish() -> list[float]:
    """300-bar noisy uptrend → close > EMA50 > EMA200, MACD hist > 0."""
    rng = np.random.default_rng(0)
    c = [100.0]
    for _ in range(299):
        c.append(c[-1] + 0.10 + rng.normal(0, 0.08))
    return c


# ── 15m setups ───────────────────────────────────────────────────────────────

def _15m_rsi_turning_up() -> tuple[list[float], list[float]]:
    """178 uptrend + 20 decline + 1 uptick → rsi_prev<45, rsi_now>rsi_prev."""
    c = [100.0]
    for _ in range(178): c.append(c[-1] + 0.05)
    for _ in range(20):  c.append(c[-1] - 0.35)
    c.append(c[-1] + 0.60)              # uptick
    vols = [1000.0] * 200
    vols[-1] = 2500.0
    return c, vols


def _15m_rsi_turning_down() -> list[float]:
    """197 uptrend + 2 declines of -1.5 → rsi_prev≈61>55, rsi_now≈43<rsi_prev."""
    c = [100.0]
    for _ in range(197): c.append(c[-1] + 0.18)
    c.append(c[-1] - 1.5)
    c.append(c[-1] - 1.5)
    return c


def _15m_flat() -> list[float]:
    """Steady uptrend → RSI≈100, not reversing in either direction."""
    c = [100.0]
    for _ in range(199): c.append(c[-1] + 0.05)
    return c


# ── engine with no real client ────────────────────────────────────────────────

def _engine() -> SignalEngine:
    return SignalEngine(client=None)  # type: ignore[arg-type]


# ════════════════════════════════════════════════════════════════════════════════
# Test 1: Type A LONG — 4h bullish + 1h near EMA50 + 15m RSI turning up
# ════════════════════════════════════════════════════════════════════════════════

def test_type_a_long():
    # 4h: noisy uptrend → bullish
    df_4h = _df(_4h_bullish(), freq_min=240)

    # 1h: 70-bar uptrend then 30-bar pullback → close near EMA50
    c1 = [100.0]
    for _ in range(69): c1.append(c1[-1] + 0.10)
    for _ in range(30): c1.append(c1[-1] - 0.22)
    df_1h = _df(c1, freq_min=60)

    # 15m: uptrend → decline → uptick + volume spike
    c15, v15 = _15m_rsi_turning_up()
    df_15m = _df(c15, volumes=v15, freq_min=15)

    sig = _engine()._evaluate("BTC/USDT:USDT", df_4h, df_1h, df_15m)

    assert sig is not None, "Expected a signal, got None"
    assert isinstance(sig, CandidateSignal)
    assert sig.entry_type == "A"
    assert sig.side == "long"
    assert 0.0 < sig.signal_strength <= 1.0
    assert sig.atr_14 > 0


# ════════════════════════════════════════════════════════════════════════════════
# Test 2: Type B SHORT — 4h RSI>75 overbought + 15m RSI turning down
# ════════════════════════════════════════════════════════════════════════════════

def test_type_b_short_overbought():
    # 4h: mixed uptrend then blowoff → RSI >99 (well above 75)
    c4b = [100.0]
    for i in range(269):
        c4b.append(c4b[-1] + (0.12 if i % 3 != 2 else -0.04))
    for _ in range(30): c4b.append(c4b[-1] + 1.2)
    df_4h = _df(c4b, freq_min=240)

    # 1h: simple uptrend (just needs ≥55 bars for EMA50/swing warmup)
    c1b = [100.0]
    for _ in range(99): c1b.append(c1b[-1] + 0.10)
    df_1h = _df(c1b, freq_min=60)

    # 15m: strong uptrend then 2-bar decline from overbought → rsi_prev>55
    df_15m = _df(_15m_rsi_turning_down(), freq_min=15)

    sig = _engine()._evaluate("BTC/USDT:USDT", df_4h, df_1h, df_15m)

    assert sig is not None, "Expected a signal, got None"
    assert sig.entry_type == "B"
    assert sig.side == "short"
    assert 0.0 < sig.signal_strength <= 1.0


# ════════════════════════════════════════════════════════════════════════════════
# Test 3: Type C LONG — 4h neutral + close near 1h swing low + 15m RSI up
# ════════════════════════════════════════════════════════════════════════════════

def test_type_c_long_range():
    # 4h: 200-bar uptrend, 20-bar decline, 80-bar alternating flat
    # → close < EMA50 > EMA200 (neutral), RSI≈48 (not extreme → no TypeB)
    c4c = [100.0]
    for _ in range(199): c4c.append(c4c[-1] + 0.15)
    for _ in range(20):  c4c.append(c4c[-1] - 0.30)
    for i in range(80):  c4c.append(c4c[-1] + (0.15 if i % 2 == 0 else -0.15))
    df_4h = _df(c4c, freq_min=240)

    # 1h: sine wave ending at trough (pos ≈ 0.008 < 0.20)
    t = np.linspace(0, 3.5 * np.pi, 100)
    c1c = (100 + 10 * np.sin(t)).tolist()
    df_1h = _df(c1c, freq_min=60)

    # 15m: decline then uptick → rsi_up=True
    c15c, v15c = _15m_rsi_turning_up()
    df_15m = _df(c15c, volumes=v15c, freq_min=15)

    sig = _engine()._evaluate("BTC/USDT:USDT", df_4h, df_1h, df_15m)

    assert sig is not None, "Expected a signal, got None"
    assert sig.entry_type == "C"
    assert sig.side == "long"
    assert sig.swing_low_1h < sig.swing_high_1h


# ════════════════════════════════════════════════════════════════════════════════
# Test 4: No signal — bullish 4h but 1h far above EMA50, 15m RSI not reversing
# ════════════════════════════════════════════════════════════════════════════════

def test_no_signal_when_no_setup():
    df_4h = _df(_4h_bullish(), freq_min=240)     # bullish (no TypeC/TypeB check)

    # 1h: sharp uptrend, close >> EMA50 → near_ema50=False
    c1n = [100.0]
    for _ in range(99): c1n.append(c1n[-1] + 0.20)
    df_1h = _df(c1n, freq_min=60)

    # 15m: monotonic uptrend → RSI≈100, rsi_up/rsi_down both False
    df_15m = _df(_15m_flat(), freq_min=15)

    sig = _engine()._evaluate("BTC/USDT:USDT", df_4h, df_1h, df_15m)

    assert sig is None, f"Expected None, got {sig}"


# ════════════════════════════════════════════════════════════════════════════════
# Structural / type tests
# ════════════════════════════════════════════════════════════════════════════════

def test_candidate_signal_is_pydantic():
    sig = CandidateSignal(
        symbol="ETH/USDT:USDT",
        side="long",
        entry_type="A",
        signal_strength=0.75,
        entry_price_ref=2000.0,
        atr_14=50.0,
        swing_high_1h=2100.0,
        swing_low_1h=1900.0,
    )
    assert sig.side == "long"
    assert sig.signal_strength == 0.75
    assert sig.swing_high_1h > sig.swing_low_1h


def test_candidate_signal_strength_bounds():
    with pytest.raises(Exception):  # pydantic validation error
        CandidateSignal(
            symbol="X",
            side="long",
            entry_type="A",
            signal_strength=1.5,   # out of range
            entry_price_ref=100.0,
            atr_14=1.0,
            swing_high_1h=102.0,
            swing_low_1h=98.0,
        )


def test_insufficient_data_returns_none():
    df_tiny = _df([100.0] * 10)
    sig = _engine()._evaluate("BTC/USDT:USDT", df_tiny, df_tiny, df_tiny)
    assert sig is None
