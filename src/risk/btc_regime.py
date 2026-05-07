"""Global BTC regime gate (EMA-slope based).

Hard pre-veto filter that blocks long candidates when BTC's H1 EMA is
sloping down and short candidates when it is sloping up. Applies uniformly
to all entry types (A / B / C / D / E).

Why EMA-slope and not last-bar delta:
  • Old logic compared `close[-1]` (in-progress) with `close[-2]` (last
    closed) — a one-bar window. While BTC was visibly dumping over 4-5
    hours, the gate would read "flat" because the very last bar's delta
    happened to be small.
  • EMA(50)-on-H1 with a multi-bar slope captures the actual trend: the
    line you see drawn on a TradingView H1 chart. Slope is measured as
    `(ema_now − ema_lookback_ago) / ema_lookback_ago × 100` — i.e. how
    much the EMA itself has moved over the lookback window.

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
    period: int
    lookback: int
    threshold_pct: float
    endpoint: str


_lock = threading.Lock()
_cached: BtcRegimeSnapshot | None = None
_cached_at: float = 0.0


def _fetch_snapshot(client) -> BtcRegimeSnapshot | None:  # noqa: ANN001 — avoid import cycle
    period = int(settings.btc_regime_ema_period)
    lookback = int(settings.btc_regime_slope_lookback)
    threshold = float(settings.btc_regime_slope_threshold_pct)

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
    if ema_past <= 0:
        return None
    slope_pct = (ema_now - ema_past) / ema_past * 100.0

    if slope_pct > threshold:
        regime: Regime = "up"
    elif slope_pct < -threshold:
        regime = "down"
    else:
        regime = "flat"

    endpoint = "demo-fapi" if settings.binance_testnet else "fapi"
    return BtcRegimeSnapshot(
        regime=regime,
        ema_now=ema_now,
        ema_past=ema_past,
        slope_pct=slope_pct,
        period=period,
        lookback=lookback,
        threshold_pct=threshold,
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
