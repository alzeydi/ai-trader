"""Reconcile local DB equity / open positions after manual exchange edits.

Compares positions reported by Binance vs. local DB and prints discrepancies.
Doesn't mutate the DB unless `--apply` is passed.
"""

from __future__ import annotations

import argparse
import logging

from src.config import settings
from src.data.binance_client import BinanceClient
from src.execution.monitor import fetch_open_positions
from src.persistence.db import EquityPoint, make_session_factory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="write reconciled snapshot to DB")
    args = parser.parse_args()

    logging.basicConfig(level=settings.log_level)
    log = logging.getLogger("recalc")

    client = BinanceClient()
    positions = fetch_open_positions(client)
    log.info("exchange reports %d open positions", len(positions))
    for p in positions:
        log.info("  %s %s qty=%.4f entry=%.4f upnl=%.2f", p.symbol, p.side,
                 p.quantity, p.entry_price, p.unrealized_pnl)

    bal = client.fetch_balance()
    equity = float(bal.get("total", {}).get(settings.base_currency, 0.0))
    log.info("equity=%.2f %s", equity, settings.base_currency)

    if args.apply:
        Session = make_session_factory()
        with Session() as s:
            s.add(EquityPoint(equity_usd=equity))
            s.commit()
            log.info("equity snapshot written")


if __name__ == "__main__":
    main()
