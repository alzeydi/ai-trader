"""Order placement helpers — accepts TradeOrder from the execution layer."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.config import settings

if TYPE_CHECKING:
    from src.data.binance_client import BinanceClient
    from src.execution.trader import TradeOrder

log = logging.getLogger(__name__)


@dataclass
class BracketOrders:
    entry_id: str | None
    stop_id: str | None
    take_id: str | None


def place_bracket(
    client: "BinanceClient",
    order: "TradeOrder",
    dry_run: bool | None = None,
) -> BracketOrders:
    dry = settings.dry_run if dry_run is None else dry_run
    entry_side = "buy"  if order.side == "long" else "sell"
    exit_side  = "sell" if order.side == "long" else "buy"

    if dry:
        log.info(
            "DRY-RUN bracket: %s %s qty=%.6f entry=%.4f sl=%.4f tp=%.4f",
            order.symbol, entry_side, order.quantity,
            order.entry_price, order.stop_loss, order.take_profit,
        )
        return BracketOrders(entry_id=None, stop_id=None, take_id=None)

    ex = client.exchange
    entry_resp = ex.create_order(order.symbol, "market", entry_side, order.quantity)
    stop_params: dict[str, Any]  = {"stopPrice": order.stop_loss,   "reduceOnly": True}
    take_params: dict[str, Any]  = {"stopPrice": order.take_profit,  "reduceOnly": True}
    stop_resp = ex.create_order(
        order.symbol, "STOP_MARKET", exit_side, order.quantity, None, stop_params
    )
    take_resp = ex.create_order(
        order.symbol, "TAKE_PROFIT_MARKET", exit_side, order.quantity, None, take_params
    )
    return BracketOrders(
        entry_id=entry_resp.get("id"),
        stop_id=stop_resp.get("id"),
        take_id=take_resp.get("id"),
    )
