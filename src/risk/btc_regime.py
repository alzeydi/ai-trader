"""Global BTC regime gate (EMA slope + price-distance, combined).

Hard pre-veto filter that blocks long candidates when BTC's H1 trend is
down and short candidates when it is up. Applies uniformly to all entry
types (A / B / C / D / E).

Two independent signals, OR'd together:

  1. EMA slope. ema = EMA(period, H1 closes); slope_pct = (ema_now −
     ema_lookback_ago) / ema_lookback_ago × 100.
  2. Price distance from EMA. dist_pct = (close_now − ema_now) / ema_now
     × 100.

EMA50 is a LAGGING filter: in a sharp move the spot price decouples from
the EMA before the EMA's slope catches up. The price-distance signal
covers that gap. If either signal crosses its threshold, the gate fires;
on disagreement, price wins (it leads).

Why a separate module from `data/context.py`:
  • `data/context.py:_compute_btc_direction` feeds the LLM as informational
    context (the LLM sees `btc_direction`).
  • This module is a HARD GATE that runs BEFORE the LLM call so vetoed
    candidates don't burn tokens. It also caches the BTC fetch so the gate
    fires once per cycle, not once per symbol.

Cache lifetime: 60 s — matches `loop_interval_sec`. The H1 EMA evolves
slowly enough that a single fetch per cycle is plenty.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Literal, TypedDict

from src.config import settings

log = logging.getLogger(__name__)


Regime = Literal["up", "down", "flat"]

_BTC_REFERENCE = "BTC/USDT:USDT"
_CACHE_TTL_SEC = 60.0


class BtcRegimeSnapshot(TypedDict):
    regime: Regime
    ema_now: float
    ema_past: float
    slope_pct: float
    slope_signal: Regime
    price_now: float
    price_dist_pct: float
    price_signal: Regime
    period: int
    lookback: int
    slope_threshold_pct: float
    price_dist_threshold_pct: float
    endpoint: str


_lock = threading.Lock()
_cached: BtcRegimeSnapshot | None = None
_cached_at: float = 0.0


def _signal(value: float, threshold: float) -> Regime:
    if value > threshold:
        return "up"
    if value < -threshold:
        return "down"
    return "flat"


def _fetch_snapshot(client) -> BtcRegimeSnapshot | None:  # noqa: ANN001 — avoid import cycle
    period = int(settings.btc_regime_ema_period)
    lookback = int(settings.btc_regime_slope_lookback)
    slope_threshold = float(settings.btc_regime_slope_threshold_pct)
    price_dist_threshold = float(settings.btc_regime_price_dist_threshold_pct)

    # 2× period buffer for a stable EMA seed; +lookback for the slope window;
    # +5 padding so we never run off the start of the array.
    fetch_n = period * 2 + lookback + 5
    try:
        df = client.fetch_ohlcv(_BTC_REFERENCE, timeframe="1h", limit=fetch_n)
    except Exception as exc:  # noqa: BLE001
        log.warning("btc_regime fetch failed: %s", exc)
        return None
    if df is None or df.empty or len(df) < period + lookback + 1:
        log.warning(
            "btc_regime: insufficient bars (%d, need ≥%d)",
            0 if df is None else len(df), period + lookback + 1,
        )
        return None

    ema = df["close"].ewm(span=period, adjust=False).mean()
    ema_now = float(ema.iloc[-1])
    ema_past = float(ema.iloc[-1 - lookback])
    price_now = float(df["close"].iloc[-1])
    if ema_past <= 0 or ema_now <= 0:
        return None
    slope_pct = (ema_now - ema_past) / ema_past * 100.0
    price_dist_pct = (price_now - ema_now) / ema_now * 100.0

    slope_signal = _signal(slope_pct, slope_threshold)
    price_signal = _signal(price_dist_pct, price_dist_threshold)

    # Combine: any non-flat signal fires; on disagreement, price wins
    # (leading indicator vs lagging EMA slope).
    if price_signal != "flat":
        regime: Regime = price_signal
    elif slope_signal != "flat":
        regime = slope_signal
    else:
        regime = "flat"

    endpoint = "demo-fapi" if settings.binance_testnet else "fapi"
    return BtcRegimeSnapshot(
        regime=regime,
        ema_now=ema_now,
        ema_past=ema_past,
        slope_pct=slope_pct,
        slope_signal=slope_signal,
        price_now=price_now,
        price_dist_pct=price_dist_pct,
        price_signal=price_signal,
        period=period,
        lookback=lookback,
        slope_threshold_pct=slope_threshold,
        price_dist_threshold_pct=price_dist_threshold,
        endpoint=endpoint,
    )


def get_btc_regime_snapshot(client) -> BtcRegimeSnapshot | None:  # noqa: ANN001
    """Return the cached snapshot or refresh if stale.

    Returns None on fetch failure — callers MUST treat that as "no signal,
    do not block" (fail-open on data errors; the veto layer still runs).
    """
    global _cached, _cached_at
    with _lock:
        if time.monotonic() - _cached_at < _CACHE_TTL_SEC and _cached is not None:
            return _cached
        snap = _fetch_snapshot(client)
        if snap is not None:
            _cached = snap
            _cached_at = time.monotonic()
        return snap


def get_btc_regime(client) -> Regime | None:  # noqa: ANN001
    """Compatibility wrapper — returns just the regime label."""
    snap = get_btc_regime_snapshot(client)
    return None if snap is None else snap["regime"]


def is_blocked(regime: Regime | None, side: str) -> bool:
    """True iff `side` is blocked under `regime`.

    Rules:
      regime=up   → only longs allowed (block shorts)
      regime=down → only shorts allowed (block longs)
      regime=flat → both allowed
      regime=None → fail-open (do not block)
    """
    if regime is None or regime == "flat":
        return False
    if regime == "up" and side == "short":
        return True
    if regime == "down" and side == "long":
        return True
    return False


def reset_cache() -> None:
    """Test hook — clear the module-level cache."""
    global _cached, _cached_at
    with _lock:
        _cached = None
        _cached_at = 0.0
