"""Thin order placement helpers around ccxt.

`place_bracket` opens a market entry and registers a reduce-only stop and
take-profit. `dry_run=True` short-circuits to logging.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from src.config import settings
from src.data.binance_client import BinanceClient
from src.signal.types import CandidateSignal, Side

log = logging.getLogger(__name__)


@dataclass
class BracketOrders:
    entry_id: str | None
    stop_id: str | None
    take_id: str | None


def place_bracket(
    client: BinanceClient,
    signal: CandidateSignal,
    quantity: float,
    dry_run: bool | None = None,
) -> BracketOrders:
    dry = settings.dry_run if dry_run is None else dry_run
    side_buy = signal.side is Side.LONG
    entry_side = "buy" if side_buy else "sell"
    exit_side = "sell" if side_buy else "buy"

    if dry:
        log.info(
            "DRY-RUN bracket: %s %s qty=%.6f entry=%.4f stop=%.4f tp=%.4f",
            signal.symbol,
            entry_side,
            quantity,
            signal.entry,
            signal.stop,
            signal.take_profit,
        )
        return BracketOrders(entry_id=None, stop_id=None, take_id=None)

    ex = client.exchange
    entry = ex.create_order(signal.symbol, "market", entry_side, quantity)
    stop_params: dict[str, Any] = {"stopPrice": signal.stop, "reduceOnly": True}
    take_params: dict[str, Any] = {"stopPrice": signal.take_profit, "reduceOnly": True}
    stop = ex.create_order(signal.symbol, "STOP_MARKET", exit_side, quantity, None, stop_params)
    take = ex.create_order(
        signal.symbol, "TAKE_PROFIT_MARKET", exit_side, quantity, None, take_params
    )

    return BracketOrders(entry_id=entry.get("id"), stop_id=stop.get("id"), take_id=take.get("id"))
