"""Tests for the data layer (no real network calls)."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.data.binance_client import BinanceClient
from src.data.context import MarketContext, get_market_context
from src.data.indicators import atr14, ema, macd, rsi14, swing_high_low
from src.persistence.db import make_engine, make_session_factory


# ---------- fixtures ----------

@pytest.fixture()
def session_factory(tmp_path):
    """Per-test SQLite DB so cache state never leaks across tests."""
    return make_session_factory(make_engine(tmp_path / "test.db"))


@pytest.fixture()
def mock_exchange():
    """A bare ccxt-like exchange surface used by BinanceClient."""
    ex = MagicMock()
    ex.has = {"fetchLiquidations": True}
    return ex


@pytest.fixture()
def client(session_factory, mock_exchange, monkeypatch):
    """BinanceClient with the underlying ccxt exchange replaced by a mock."""
    monkeypatch.setattr(
        "src.data.binance_client.ccxt.binanceusdm",
        lambda *a, **kw: mock_exchange,
    )
    bc = BinanceClient(testnet=False, session_factory=session_factory)
    return bc


def _ohlcv_rows(n: int, start_ms: int, step_ms: int = 60_000) -> list[list[float]]:
    """Build a deterministic OHLCV list as ccxt would return it."""
    rng = np.random.default_rng(42)
    closes = 100 + np.cumsum(rng.normal(0, 0.5, n))
    rows: list[list[float]] = []
    for i in range(n):
        c = float(closes[i])
        rows.append([start_ms + i * step_ms, c - 0.1, c + 0.5, c - 0.5, c, 1.0])
    return rows


def _recent_ohlcv(n: int, step_ms: int = 60_000) -> list[list[float]]:
    """OHLCV ending at 'now' so cache freshness checks treat it as current."""
    end_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
    start_ms = end_ms - (n - 1) * step_ms
    return _ohlcv_rows(n, start_ms, step_ms)


# ---------- indicators ----------

class TestIndicators:
    def test_ema_constant_series_equals_value(self):
        s = pd.Series(np.full(30, 7.5))
        out = ema(s, length=10)
        assert out.name == "ema_10"
        # EMA of a constant series is the constant itself.
        assert out.iloc[-1] == pytest.approx(7.5, rel=1e-9)

    def test_ema_strictly_increasing_input_is_monotonic(self):
        s = pd.Series(np.arange(1, 51, dtype=float))
        out = ema(s, length=10).dropna()
        # EMA must be monotonically non-decreasing on a strictly increasing series.
        diffs = out.diff().dropna()
        assert (diffs >= 0).all()
        # And bounded above by the latest input.
        assert out.iloc[-1] <= s.iloc[-1]

    def test_rsi14_saturates_on_monotonic_up(self):
        df = pd.DataFrame({"close": np.arange(1, 60, dtype=float)})
        out = rsi14(df)
        assert out.name == "rsi_14"
        # Every gain, no losses → RSI must hit 100 once warmed up.
        assert out.iloc[-1] == pytest.approx(100.0, abs=1e-6)

    def test_atr14_constant_range(self):
        n = 30
        df = pd.DataFrame(
            {
                "high": np.full(n, 11.0),
                "low": np.full(n, 9.0),
                "close": np.full(n, 10.0),
            }
        )
        out = atr14(df)
        assert out.name == "atr_14"
        # True range is always 2.0 → ATR converges to 2.0.
        assert out.iloc[-1] == pytest.approx(2.0, rel=1e-6)

    def test_macd_columns_present(self):
        df = pd.DataFrame({"close": np.linspace(100, 200, 80)})
        out = macd(df)
        assert list(out.columns) == ["macd", "signal", "hist"]
        assert not out["macd"].iloc[-1] != out["macd"].iloc[-1]  # not NaN

    def test_swing_high_low_known_peak(self):
        # A clear peak at index 5; a clear trough at index 12.
        prices = [10, 11, 12, 13, 14, 20, 14, 13, 12, 11, 10, 9, 5, 9, 10, 11]
        df = pd.DataFrame({"high": prices, "low": [p - 1 for p in prices]})
        out = swing_high_low(df, lookback=3)
        # The 5th bar must be marked as a swing high at price 20.
        assert out["swing_high"].iloc[5] == 20
        # The 12th bar must be marked as a swing low at price 4 (low = 5 - 1).
        assert out["swing_low"].iloc[12] == 4
        # Edges must remain NaN.
        assert pd.isna(out["swing_high"].iloc[0])
        assert pd.isna(out["swing_low"].iloc[-1])


# ---------- BinanceClient ----------

class TestBinanceClient:
    def test_fetch_ohlcv_returns_dataframe_and_caches(
        self, client: BinanceClient, mock_exchange, session_factory
    ):
        mock_exchange.fetch_ohlcv.return_value = _recent_ohlcv(50)
        df = client.fetch_ohlcv("BTC/USDT:USDT", "1m", limit=50)
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert len(df) == 50
        assert df.index.tz is not None
        mock_exchange.fetch_ohlcv.assert_called_once()

        # Second call within the candle period should serve from cache.
        df2 = client.fetch_ohlcv("BTC/USDT:USDT", "1m", limit=50)
        assert len(df2) == 50
        # Still only one remote hit total.
        assert mock_exchange.fetch_ohlcv.call_count == 1

    def test_fetch_funding_rate_passes_through(self, client: BinanceClient, mock_exchange):
        mock_exchange.fetch_funding_rate.return_value = {
            "symbol": "BTC/USDT:USDT",
            "fundingRate": 0.0001,
        }
        out = client.fetch_funding_rate("BTC/USDT:USDT")
        assert out["fundingRate"] == 0.0001

    def test_fetch_open_interest_passes_through(self, client: BinanceClient, mock_exchange):
        mock_exchange.fetch_open_interest.return_value = {
            "symbol": "BTC/USDT:USDT",
            "openInterestAmount": 1234.5,
        }
        out = client.fetch_open_interest("BTC/USDT:USDT")
        assert out["openInterestAmount"] == 1234.5

    def test_fetch_liquidations_returns_empty_when_unsupported(
        self, client: BinanceClient, mock_exchange
    ):
        mock_exchange.has = {"fetchLiquidations": False}
        assert client.fetch_liquidations("BTC/USDT:USDT") == []
        mock_exchange.fetch_liquidations.assert_not_called()

    def test_fetch_liquidations_uses_since_datetime(
        self, client: BinanceClient, mock_exchange
    ):
        mock_exchange.fetch_liquidations.return_value = [
            {"price": 100.0, "amount": 2.0, "quoteVolume": 200.0}
        ]
        since_dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
        out = client.fetch_liquidations("BTC/USDT:USDT", since=since_dt)
        assert out and out[0]["quoteVolume"] == 200.0
        called_since = mock_exchange.fetch_liquidations.call_args.kwargs["since"]
        assert called_since == int(since_dt.timestamp() * 1000)


# ---------- MarketContext ----------

class TestMarketContext:
    def test_get_market_context_assembles_fields(
        self, client: BinanceClient, mock_exchange
    ):
        start = int(datetime.now(tz=timezone.utc).timestamp() * 1000) - 3_600_000
        mock_exchange.fetch_ticker.return_value = {"last": 30000.0}
        mock_exchange.fetch_funding_rate.return_value = {"fundingRate": 0.00012}
        mock_exchange.fetch_open_interest.return_value = {"openInterestAmount": 5000.0}
        mock_exchange.fetch_open_interest_history.return_value = [
            {"openInterestAmount": 4900.0, "timestamp": start},
            {"openInterestAmount": 5000.0, "timestamp": start + 3_600_000},
        ]
        mock_exchange.fetch_liquidations.return_value = [
            {"price": 30000.0, "amount": 0.1, "quoteVolume": 3000.0},
            {"price": 30000.0, "amount": 0.2, "quoteVolume": 6000.0},
        ]
        mock_exchange.fetch_ohlcv.return_value = _recent_ohlcv(5, step_ms=3_600_000)

        ctx = get_market_context("BTC/USDT:USDT", client)
        assert isinstance(ctx, MarketContext)
        assert ctx.last_price == 30000.0
        assert ctx.funding_rate == pytest.approx(0.00012)
        assert ctx.open_interest == 5000.0
        assert ctx.open_interest_delta_1h == pytest.approx(100.0)
        assert ctx.liquidations_4h_usd == pytest.approx(9000.0)
        assert ctx.btc_1h_direction in {"up", "down", "flat"}
        assert ctx.btc_dominance is None  # not yet wired to an external source
