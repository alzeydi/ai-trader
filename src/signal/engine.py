"""Layer 1 — rule-based signal orchestrator.

Pulls multi-timeframe data, runs trend → structure → execution checks, and
emits a `CandidateSignal` when all three align. The LLM veto layer (Layer 2)
gets to approve/reject before anything is sized in Layer 3.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.config import settings
from src.data.binance_client import BinanceClient
from src.data.indicators import add_atr
from src.signal.execution import find_trigger
from src.signal.structure import find_structure
from src.signal.trend import detect_bias
from src.signal.types import CandidateSignal, Side, SignalType

log = logging.getLogger(__name__)


@dataclass
class SignalEngine:
    client: BinanceClient

    def evaluate(self, symbol: str) -> CandidateSignal | None:
        df_4h = self.client.fetch_ohlcv(symbol, settings.timeframe_trend, limit=300)
        df_1h = self.client.fetch_ohlcv(symbol, settings.timeframe_structure, limit=300)
        df_15 = self.client.fetch_ohlcv(symbol, settings.timeframe_execution, limit=300)

        bias = detect_bias(df_4h)
        structure = find_structure(df_1h)
        fired, reason = find_trigger(df_15, bias, structure)
        if not fired:
            log.debug("%s: no signal (bias=%s, reason=%s)", symbol, bias, reason)
            return None

        feat = add_atr(df_15)
        atr = float(feat["atr_14"].iloc[-1])
        entry = float(df_15["close"].iloc[-1])
        side = Side.LONG if bias == "long" else Side.SHORT

        if side is Side.LONG:
            stop = entry - settings.atr_stop_multiplier * atr
            take = entry + settings.atr_take_multiplier * atr
        else:
            stop = entry + settings.atr_stop_multiplier * atr
            take = entry - settings.atr_take_multiplier * atr

        return CandidateSignal(
            symbol=symbol,
            side=side,
            type=SignalType.A,
            entry=entry,
            stop=stop,
            take_profit=take,
            confidence=0.7,
            rationale=reason,
            features={"atr": atr, "bias": bias},
        )
