"""Global BTC regime gate.

Hard pre-veto filter that blocks long candidates when BTC is falling and
short candidates when BTC is rising. Applies uniformly to all entry types
(A / B / C / D / E) — see CLAUDE.md, decision recorded with the user on
adding this layer.

Why a separate module from `data/context.py`:
  • `data/context.py:_compute_btc_direction` feeds the LLM veto as
    informational context (Claude sees "btc_direction": up/down/flat).
  • This module is a HARD GATE that runs BEFORE the LLM call, so vetoed
    candidates don't burn tokens. It also caches the BTC fetch across
    symbols inside one cycle (the LLM-context path re-fetches per symbol).

Cache lifetime: 60 s — matches `loop_interval_sec`. The 1h BTC bar evolves
slowly enough that a single fetch per cycle is plenty.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Literal

from src.config import settings

log = logging.getLogger(__name__)


Regime = Literal["up", "down", "flat"]

_BTC_REFERENCE = "BTC/USDT:USDT"
_CACHE_TTL_SEC = 60.0

_lock = threading.Lock()
_cached_regime: Regime | None = None
_cached_at: float = 0.0


def _fetch_regime(client) -> Regime | None:  # noqa: ANN001 — BinanceClient (avoid import cycle)
    try:
        df = client.fetch_ohlcv(_BTC_REFERENCE, timeframe="1h", limit=3)
    except Exception as exc:  # noqa: BLE001
        log.warning("btc_regime fetch failed: %s", exc)
        return None
    if df.empty or len(df) < 2:
        return None
    last = float(df["close"].iloc[-1])
    prev = float(df["close"].iloc[-2])
    if prev <= 0:
        return None
    delta_pct = (last - prev) / prev * 100.0
    threshold = settings.btc_regime_threshold_pct
    if delta_pct > threshold:
        return "up"
    if delta_pct < -threshold:
        return "down"
    return "flat"


def get_btc_regime(client) -> Regime | None:  # noqa: ANN001
    """Return cached BTC 1h regime or refresh if stale.

    Returns None on fetch failure — callers MUST treat that as "no signal,
    do not block" (fail-open on data errors; the veto layer still runs).
    """
    global _cached_regime, _cached_at
    with _lock:
        if time.monotonic() - _cached_at < _CACHE_TTL_SEC and _cached_regime is not None:
            return _cached_regime
        regime = _fetch_regime(client)
        if regime is not None:
            _cached_regime = regime
            _cached_at = time.monotonic()
        return regime


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
    global _cached_regime, _cached_at
    with _lock:
        _cached_regime = None
        _cached_at = 0.0
