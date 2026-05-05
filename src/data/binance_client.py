"""Binance USDT-M Futures client (ccxt wrapper).

- Honors `BINANCE_TESTNET` flag.
- Built-in ccxt rate limiting (`enableRateLimit=True`).
- Tenacity-based retry (3 attempts, exponential backoff) on network errors.
- OHLCV candles are cached in SQLite (`src.persistence.db.OhlcvCandle`) so
  repeated calls within a candle window don't hit the REST endpoint.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import ccxt
import pandas as pd
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config import settings
from src.persistence.db import (
    SessionFactory,
    make_session_factory,
    read_cached_candles,
    upsert_candles,
)

log = logging.getLogger(__name__)

# Network-class errors that are worth retrying. Exchange-class errors
# (e.g. InsufficientFunds, BadSymbol) are not — let them propagate.
NETWORK_ERRORS: tuple[type[Exception], ...] = (
    ccxt.NetworkError,
    ccxt.RequestTimeout,
    ccxt.ExchangeNotAvailable,
    ccxt.DDoSProtection,
)

_retry = retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type(NETWORK_ERRORS),
)


def _timeframe_seconds(timeframe: str) -> int:
    """Convert ccxt timeframe ('1m','5m','1h','4h','1d') to seconds."""
    unit = timeframe[-1]
    n = int(timeframe[:-1])
    return n * {"m": 60, "h": 3600, "d": 86400, "w": 604800}[unit]


@dataclass
class BinanceClient:
    testnet: bool = field(default_factory=lambda: settings.binance_testnet)
    session_factory: SessionFactory | None = None
    _exchange: ccxt.Exchange | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        params: dict[str, Any] = {
            "apiKey": settings.binance_api_key.get_secret_value() or None,
            "secret": settings.binance_api_secret.get_secret_value() or None,
            "enableRateLimit": True,
            "options": {
                "defaultType": "future",
                # ccxt's load_markets() otherwise calls SAPI
                # (api.binance.com/sapi/v1/capital/config/getall) to enrich
                # currency metadata. Demo Trading exposes only fapi, and our
                # futures-only keys can't sign SAPI requests, which surfaces
                # as a misleading "Invalid Api-Key ID". We don't need
                # currency metadata for futures trading anyway.
                "fetchCurrencies": False,
            },
        }
        self._exchange = ccxt.binanceusdm(params)
        if self.testnet:
            # Binance retired the classic testnet (testnet.binancefuture.com)
            # and replaced it with "Demo Trading" at demo-fapi.binance.com.
            # ccxt's set_sandbox_mode still points at the dead testnet host,
            # so we patch URLs manually.
            api_urls = self._exchange.urls.get("api", {})
            if isinstance(api_urls, dict):
                patched = {
                    k: (v.replace("fapi.binance.com", "demo-fapi.binance.com")
                        if isinstance(v, str) and "fapi.binance.com" in v else v)
                    for k, v in api_urls.items()
                }
                self._exchange.urls["api"] = patched
            log.info("binance: using demo trading endpoint demo-fapi.binance.com")
        if self.session_factory is None:
            self.session_factory = make_session_factory()

    @property
    def exchange(self) -> ccxt.Exchange:
        assert self._exchange is not None
        return self._exchange

    # ----- OHLCV with SQLite cache -----

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 500,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Return at most `limit` recent candles, indexed by UTC timestamp.

        Cache strategy: if the cache holds enough rows and the latest one is
        younger than one timeframe period, the cache is returned directly.
        Otherwise we fetch from Binance, upsert into the cache, and return the
        merged result.
        """
        assert self.session_factory is not None
        if not force_refresh:
            cached = read_cached_candles(self.session_factory, symbol, timeframe, limit)
            if not cached.empty and len(cached) >= limit:
                age = (datetime.now(tz=timezone.utc) - cached.index[-1]).total_seconds()
                if age < _timeframe_seconds(timeframe):
                    return cached

        raw = self._fetch_ohlcv_remote(symbol, timeframe, limit)
        df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
        if df.empty:
            return df
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df.set_index("ts", inplace=True)

        try:
            upsert_candles(self.session_factory, symbol, timeframe, df)
        except Exception as exc:  # noqa: BLE001 — cache failures must not break trading
            log.warning("ohlcv cache upsert failed for %s %s: %s", symbol, timeframe, exc)

        return df

    @_retry
    def _fetch_ohlcv_remote(self, symbol: str, timeframe: str, limit: int) -> list[list[Any]]:
        return self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

    # ----- Funding / OI / Liquidations -----

    @_retry
    def fetch_funding_rate(self, symbol: str) -> dict[str, Any]:
        return self.exchange.fetch_funding_rate(symbol)

    @_retry
    def fetch_open_interest(self, symbol: str) -> dict[str, Any]:
        return self.exchange.fetch_open_interest(symbol)

    @_retry
    def fetch_open_interest_history(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 2,
    ) -> list[dict[str, Any]]:
        fn = getattr(self.exchange, "fetch_open_interest_history", None)
        if fn is None:
            return []
        return fn(symbol, timeframe=timeframe, limit=limit)

    def fetch_liquidations(
        self,
        symbol: str,
        since: int | datetime | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return recent liquidation events (best-effort; empty list on no support)."""
        if not self.exchange.has.get("fetchLiquidations"):
            log.debug("fetch_liquidations not supported by this exchange")
            return []
        since_ms: int | None
        if isinstance(since, datetime):
            since_ms = int(since.timestamp() * 1000)
        else:
            since_ms = since
        try:
            return self._fetch_liquidations_remote(symbol, since_ms, limit)
        except (ccxt.NotSupported, ccxt.PermissionDenied) as exc:
            log.debug("liquidations endpoint unavailable: %s", exc)
            return []

    @_retry
    def _fetch_liquidations_remote(
        self,
        symbol: str,
        since_ms: int | None,
        limit: int | None,
    ) -> list[dict[str, Any]]:
        return self.exchange.fetch_liquidations(symbol, since=since_ms, limit=limit)

    # ----- Misc -----

    @_retry
    def fetch_ticker(self, symbol: str) -> dict[str, Any]:
        return self.exchange.fetch_ticker(symbol)

    @_retry
    def fetch_balance(self) -> dict[str, Any]:
        return self.exchange.fetch_balance()
