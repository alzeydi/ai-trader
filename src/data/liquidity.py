"""Per-cycle liquidity screen for the trading universe.

Why: bracket SL on Binance USDT-M futures is a STOP_MARKET — when triggered
it sweeps the book at any price. On thin symbols this slipped a STX/USDT
close 8 ticks (~0.32%) past the SL trigger, swinging recorded PnL by ~$5
on a single trade. This module drops symbols whose 24h quoteVolume or
bid/ask spread are outside acceptable bounds before the signal engine
ever looks at them, so we don't structurally bleed to slippage on
illiquid pairs.

Fetched in one `fetch_tickers()` call (single REST request on Binance),
cached for `settings.liquidity_filter_cache_sec` seconds to survive
transient ticker failures.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.config import settings

if TYPE_CHECKING:
    from src.data.binance_client import BinanceClient

log = logging.getLogger(__name__)


@dataclass
class _CacheEntry:
    monotonic_ts: float
    kept: list[str]
    dropped_low_volume: list[tuple[str, float]]
    dropped_wide_spread: list[tuple[str, float]]
    dropped_no_data: list[str]


_CACHE: _CacheEntry | None = None


def _spread_bps(ticker: dict[str, Any]) -> float | None:
    """Bid/ask spread in basis points (1 bps = 0.01%). None if unavailable."""
    bid = ticker.get("bid")
    ask = ticker.get("ask")
    try:
        bid_f = float(bid) if bid is not None else 0.0
        ask_f = float(ask) if ask is not None else 0.0
    except (TypeError, ValueError):
        return None
    if bid_f <= 0 or ask_f <= 0 or ask_f < bid_f:
        return None
    mid = (ask_f + bid_f) / 2.0
    if mid <= 0:
        return None
    return (ask_f - bid_f) / mid * 10_000.0


def _quote_volume(ticker: dict[str, Any]) -> float | None:
    qv = ticker.get("quoteVolume")
    try:
        return float(qv) if qv is not None else None
    except (TypeError, ValueError):
        return None


def filter_by_liquidity(
    client: "BinanceClient",
    symbols: list[str],
    *,
    force_refresh: bool = False,
) -> list[str]:
    """Return the subset of `symbols` that passes the liquidity screen.

    Pure side-effect-free except for module-level cache and structured
    logging on each refresh. On any failure (network, cache miss with
    no fallback) returns the input list unchanged — fail-open is the
    right default here, since fail-closed would silently disable the
    bot whenever Binance hiccups.
    """
    global _CACHE
    now = time.monotonic()

    if (
        not force_refresh
        and _CACHE is not None
        and now - _CACHE.monotonic_ts < settings.liquidity_filter_cache_sec
    ):
        return [s for s in symbols if s in set(_CACHE.kept)]

    try:
        tickers = client.exchange.fetch_tickers(symbols)
    except Exception as exc:  # noqa: BLE001
        log.warning("liquidity: fetch_tickers failed: %s — keeping cache", exc)
        if _CACHE is not None:
            return [s for s in symbols if s in set(_CACHE.kept)]
        return symbols  # fail-open on first call

    min_vol = float(settings.min_quote_volume_24h_usdt)
    max_spread = float(settings.max_spread_bps)

    kept: list[str] = []
    dropped_low_volume: list[tuple[str, float]] = []
    dropped_wide_spread: list[tuple[str, float]] = []
    dropped_no_data: list[str] = []

    for sym in symbols:
        t = tickers.get(sym) if isinstance(tickers, dict) else None
        if not t:
            dropped_no_data.append(sym)
            continue
        qv = _quote_volume(t)
        if qv is None:
            dropped_no_data.append(sym)
            continue
        if qv < min_vol:
            dropped_low_volume.append((sym, qv))
            continue
        spread = _spread_bps(t)
        if spread is None:
            # Volume passes but no bid/ask — keep (rare; usually a Binance
            # ticker quirk on low-activity moments). Better than dropping.
            kept.append(sym)
            continue
        if spread > max_spread:
            dropped_wide_spread.append((sym, spread))
            continue
        kept.append(sym)

    _CACHE = _CacheEntry(
        monotonic_ts=now,
        kept=kept,
        dropped_low_volume=dropped_low_volume,
        dropped_wide_spread=dropped_wide_spread,
        dropped_no_data=dropped_no_data,
    )

    log.info(
        "liquidity: %d kept / %d low-vol / %d wide-spread / %d no-data "
        "(min_vol=$%.0fM, max_spread=%.1f bps)",
        len(kept), len(dropped_low_volume), len(dropped_wide_spread),
        len(dropped_no_data), min_vol / 1e6, max_spread,
    )
    if dropped_low_volume:
        worst = sorted(dropped_low_volume, key=lambda x: x[1])[:5]
        log.debug("liquidity: lowest-volume drops: %s", worst)
    if dropped_wide_spread:
        worst = sorted(dropped_wide_spread, key=lambda x: -x[1])[:5]
        log.debug("liquidity: widest-spread drops: %s", worst)

    return kept
