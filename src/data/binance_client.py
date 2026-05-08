"""Binance USDT-M Futures client (ccxt wrapper).

- Honors `BINANCE_TESTNET` flag.
- Built-in ccxt rate limiting (`enableRateLimit=True`).
- Tenacity-based retry (3 attempts, exponential backoff) on network errors.
- OHLCV candles use a two-layer cache: closed historical bars come from the
  SQLite cache (`src.persistence.db.OhlcvCandle`); the last 2 bars (the most
  recent closed bar + the in-progress running bar) are ALWAYS fetched fresh
  on every call. Without this, the in-progress bar can stay stale for up to
  one timeframe period (4h on H4 candles), causing detectors to fire on
  outdated data while the actual market has moved.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
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

# Number of trailing bars we always refresh from the exchange. Two bars
# covers (last closed bar, currently-forming bar) — the running bar is the
# critical one for signal correctness; the previous closed bar is included
# so a bar boundary crossed between calls is also captured cleanly.
_FRESH_TAIL_BARS = 2

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

        Two-layer strategy:

        - HOT PATH (cache fully warm). The historical bars (everything except
          the last `_FRESH_TAIL_BARS`) come from SQLite. The last 2 bars are
          ALWAYS fetched fresh, so the in-progress running bar reflects the
          true running close on every call. Cost: 1 weight per request
          (`limit=2` is in the lowest fapi-klines tier).
        - COLD PATH. If the cache has fewer than `limit` bars or
          `force_refresh=True`, do a full `limit`-bar fetch. Used on cold
          start, after a long downtime, and by the backtest harness.

        The previous strategy returned the cache verbatim while the latest
        cached row was younger than one timeframe period. For 4h that meant a
        stale running bar could be served for up to 4 hours — detectors saw
        running close from hours ago while market orders filled at live spot,
        producing 2-3% gaps between recorded entry and actual fill.
        """
        assert self.session_factory is not None

        if force_refresh:
            return self._fetch_full(symbol, timeframe, limit)

        cached = read_cached_candles(self.session_factory, symbol, timeframe, limit)
        if cached.empty or len(cached) < limit:
            return self._fetch_full(symbol, timeframe, limit)

        fresh = self._fetch_tail(symbol, timeframe, _FRESH_TAIL_BARS)
        if fresh is None:
            # Network/exchange failure on the tail fetch. The cache is at
            # least minutes old in the worst case (we just enforced
            # len >= limit); serving stale is preferable to returning
            # nothing and stalling the cycle.
            log.warning(
                "fetch_ohlcv: %s %s tail-refresh failed, serving cache",
                symbol, timeframe,
            )
            return cached
        if fresh.empty:
            # Symbol soft-skipped (-1122 etc.) — treat as no data so engine's
            # length guard short-circuits the symbol for this cycle.
            return fresh

        # Replace any cached rows that overlap with the fresh window, then
        # append fresh and trim to `limit`. fresh.index[0] is the oldest of
        # the freshly-fetched bars; we drop everything ≥ that timestamp from
        # the cached frame so the fresh values win on the overlap.
        keep = cached[cached.index < fresh.index[0]]
        merged = pd.concat([keep, fresh]).tail(limit)

        try:
            upsert_candles(self.session_factory, symbol, timeframe, fresh)
        except Exception as exc:  # noqa: BLE001 — cache failures must not break trading
            log.warning("ohlcv cache upsert failed for %s %s: %s", symbol, timeframe, exc)

        return merged

    # --- internal fetch helpers ---------------------------------------------

    def _fetch_full(
        self, symbol: str, timeframe: str, limit: int
    ) -> pd.DataFrame:
        """Full-window fetch with cache seed. Used on cold start and
        force_refresh."""
        try:
            raw = self._fetch_ohlcv_remote(symbol, timeframe, limit)
        except ccxt.BadRequest as exc:
            # -1122 "Invalid symbol status" — symbol is listed but in
            # maintenance / pre-trading / delisted state. Soft-skip with a
            # warning instead of raising so a single broken pair doesn't
            # surface as a stack trace every cycle. Engine will short-circuit
            # on the resulting empty frame via its _MIN_* length guard.
            msg = str(exc)
            if "-1122" in msg or "Invalid symbol status" in msg:
                log.warning("fetch_ohlcv: %s %s soft-skipped (%s)",
                            symbol, timeframe, msg)
                return pd.DataFrame()
            raise
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

    def _fetch_tail(
        self, symbol: str, timeframe: str, count: int
    ) -> pd.DataFrame | None:
        """Fetch the last `count` bars only (closed-tail + running).

        Returns:
          - DataFrame with ≤ count rows on success (may be empty for
            symbol soft-skip).
          - None on transient network/exchange errors, signalling the caller
            to fall back on cached data.
        """
        try:
            raw = self._fetch_ohlcv_remote(symbol, timeframe, count)
        except ccxt.BadRequest as exc:
            msg = str(exc)
            if "-1122" in msg or "Invalid symbol status" in msg:
                log.warning("fetch_ohlcv: %s %s soft-skipped (%s)",
                            symbol, timeframe, msg)
                return pd.DataFrame()
            raise
        except Exception as exc:  # noqa: BLE001
            log.warning("fetch_ohlcv tail: %s %s remote error: %s",
                        symbol, timeframe, exc)
            return None

        df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
        if df.empty:
            return df
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df.set_index("ts", inplace=True)
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
