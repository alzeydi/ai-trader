"""Market context features for the LLM veto layer.

Each `MarketContext` snapshot bundles the off-OHLCV signals the model uses to
sanity-check a candidate trade: funding pressure, OI delta, recent
liquidation pulse, and the wider regime (BTC dominance + BTC short-term
direction).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Literal

from pydantic import BaseModel, Field

from src.data.binance_client import BinanceClient

log = logging.getLogger(__name__)

BTC_REFERENCE = "BTC/USDT:USDT"

Direction = Literal["up", "down", "flat"]


class MarketContext(BaseModel):
    """Snapshot of non-OHLCV market context for a single symbol."""

    symbol: str
    last_price: float | None = None
    funding_rate: float | None = None
    open_interest: float | None = None
    open_interest_delta_1h: float | None = None
    liquidations_4h_usd: float | None = None
    btc_dominance: float | None = None
    btc_1h_direction: Direction | None = None
    fetched_at: datetime = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc)
    )


def _safe(call_label: str, fn, default=None):
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001
        log.debug("context.%s failed: %s", call_label, exc)
        return default


def _compute_oi_delta(client: BinanceClient, symbol: str) -> float | None:
    history = _safe(
        "fetch_open_interest_history",
        lambda: client.fetch_open_interest_history(symbol, timeframe="1h", limit=2),
        default=[],
    )
    if not history or len(history) < 2:
        return None
    prev = history[-2].get("openInterestAmount") or history[-2].get("openInterestValue")
    curr = history[-1].get("openInterestAmount") or history[-1].get("openInterestValue")
    if prev is None or curr is None:
        return None
    try:
        return float(curr) - float(prev)
    except (TypeError, ValueError):
        return None


def _compute_liquidations_4h(client: BinanceClient, symbol: str) -> float | None:
    since_dt = datetime.now(tz=timezone.utc) - timedelta(hours=4)
    events = client.fetch_liquidations(symbol, since=since_dt)
    if not events:
        return None
    total = 0.0
    for ev in events:
        # ccxt unified shape: {"price": float, "amount": float, ...}.
        # Some exchanges return notional in `quoteVolume` directly.
        notional = ev.get("quoteVolume")
        if notional is None:
            price = ev.get("price")
            amount = ev.get("amount")
            if price is not None and amount is not None:
                try:
                    notional = float(price) * float(amount)
                except (TypeError, ValueError):
                    notional = None
        if notional is not None:
            total += float(notional)
    return total


def _compute_btc_direction(client: BinanceClient) -> Direction | None:
    try:
        df = client.fetch_ohlcv(BTC_REFERENCE, timeframe="1h", limit=3)
    except Exception as exc:  # noqa: BLE001
        log.debug("context.btc_direction fetch failed: %s", exc)
        return None
    if df.empty or len(df) < 2:
        return None
    last = float(df["close"].iloc[-1])
    prev = float(df["close"].iloc[-2])
    if prev == 0:
        return None
    delta_pct = (last - prev) / prev * 100.0
    if delta_pct > 0.1:
        return "up"
    if delta_pct < -0.1:
        return "down"
    return "flat"


def get_market_context(symbol: str, client: BinanceClient) -> MarketContext:
    """Build a `MarketContext` snapshot for `symbol`.

    All fields are best-effort; any individual fetch failure leaves that field
    as `None` rather than aborting the whole snapshot.
    """
    last_price: float | None = None
    funding_rate: float | None = None
    open_interest: float | None = None

    ticker = _safe("fetch_ticker", lambda: client.fetch_ticker(symbol), default={})
    if ticker:
        last_price = ticker.get("last")
        if last_price is not None:
            try:
                last_price = float(last_price)
            except (TypeError, ValueError):
                last_price = None

    funding = _safe("fetch_funding_rate", lambda: client.fetch_funding_rate(symbol), default={})
    if funding:
        rate = funding.get("fundingRate")
        if rate is not None:
            try:
                funding_rate = float(rate)
            except (TypeError, ValueError):
                funding_rate = None

    oi_payload = _safe(
        "fetch_open_interest", lambda: client.fetch_open_interest(symbol), default={}
    )
    if oi_payload:
        oi_raw = (
            oi_payload.get("openInterestAmount")
            or oi_payload.get("openInterestValue")
            or oi_payload.get("openInterest")
        )
        if oi_raw is not None:
            try:
                open_interest = float(oi_raw)
            except (TypeError, ValueError):
                open_interest = None

    return MarketContext(
        symbol=symbol,
        last_price=last_price,
        funding_rate=funding_rate,
        open_interest=open_interest,
        open_interest_delta_1h=_compute_oi_delta(client, symbol),
        liquidations_4h_usd=_compute_liquidations_4h(client, symbol),
        btc_dominance=None,  # requires off-exchange data source (e.g. coingecko)
        btc_1h_direction=_compute_btc_direction(client),
    )


# Back-compat shim for code that called the old helper.
def fetch_context(client: BinanceClient, symbol: str) -> MarketContext:
    return get_market_context(symbol, client)
