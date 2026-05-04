"""Open-position monitoring.

Polls the exchange for current positions and unrealized PnL. Used by the main
loop and the dashboard.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from src.data.binance_client import BinanceClient

log = logging.getLogger(__name__)


@dataclass
class PositionSnapshot:
    symbol: str
    side: str
    quantity: float
    entry_price: float
    mark_price: float
    unrealized_pnl: float


def fetch_open_positions(client: BinanceClient) -> list[PositionSnapshot]:
    out: list[PositionSnapshot] = []
    try:
        positions: list[dict[str, Any]] = client.exchange.fetch_positions()  # type: ignore[assignment]
    except Exception as exc:  # noqa: BLE001
        log.warning("monitor.fetch_positions failed: %s", exc)
        return out

    for p in positions:
        contracts = float(p.get("contracts") or 0.0)
        if contracts == 0:
            continue
        out.append(
            PositionSnapshot(
                symbol=str(p.get("symbol")),
                side=str(p.get("side") or "").upper(),
                quantity=contracts,
                entry_price=float(p.get("entryPrice") or 0.0),
                mark_price=float(p.get("markPrice") or 0.0),
                unrealized_pnl=float(p.get("unrealizedPnl") or 0.0),
            )
        )
    return out
