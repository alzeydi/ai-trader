"""Send a tiny ($2 notional) order on testnet to validate plumbing.

Usage:
    python scripts/smoke_test.py --symbol BTC/USDT:USDT
"""

from __future__ import annotations

import argparse
import logging

from src.config import settings
from src.data.binance_client import BinanceClient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTC/USDT:USDT")
    parser.add_argument("--notional", type=float, default=2.0)
    args = parser.parse_args()

    logging.basicConfig(level=settings.log_level)
    log = logging.getLogger("smoke")

    if not settings.binance_testnet:
        raise SystemExit("smoke test refuses to run on mainnet — set BINANCE_TESTNET=true")

    client = BinanceClient()
    ticker = client.fetch_ticker(args.symbol)
    last = float(ticker["last"])
    qty = max(args.notional / last, 0.001)
    log.info("placing micro-buy: %s qty=%.6f @ ~%.2f", args.symbol, qty, last)
    if settings.dry_run:
        log.info("DRY_RUN=true — not sending")
        return
    order = client.exchange.create_order(args.symbol, "market", "buy", qty)
    log.info("order id=%s", order.get("id"))


if __name__ == "__main__":
    main()
