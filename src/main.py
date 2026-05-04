"""Live-bot entry point.

Wires the four runtime components together:
    SignalEngine     — multi-timeframe candidate generation (Layer 1)
    veto_candidate   — pre-flight + LLM veto                 (Layer 2)
    Trader           — risk + sizing + execution             (Layer 3)
    PositionMonitor  — open-position lifecycle (parallel)
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import yaml

from src.config import settings
from src.data.binance_client import BinanceClient
from src.execution.monitor import PositionMonitor
from src.execution.orders import OrderExecutor
from src.execution.trader import Trader
from src.llm.client import ClaudeClient
from src.notify.telegram import TelegramNotifier
from src.persistence.csv_writer import append_equity
from src.persistence.db import make_session_factory
from src.risk.safety_mode import SafetyMode
from src.signal.engine import SignalEngine


def _configure_logging() -> None:
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _load_symbols() -> list[str]:
    path = Path(settings.symbols_yaml)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [
        item["symbol"]
        for item in data.get("symbols", [])
        if item.get("enabled") and not item.get("is_reference")
    ]


def main() -> None:
    _configure_logging()
    log = logging.getLogger("main")
    log.info(
        "starting ai-trader (mode=%s, paper=%s, testnet=%s)",
        settings.trading_mode, settings.paper_trading, settings.binance_testnet,
    )

    session_factory = make_session_factory()
    binance = BinanceClient()
    engine = SignalEngine(client=binance)
    llm_client = ClaudeClient()
    notifier = TelegramNotifier()
    safety = SafetyMode(session_factory=session_factory, notifier=notifier)
    executor = OrderExecutor(
        client=binance, paper=settings.paper_trading, session_factory=session_factory
    )
    trader = Trader(
        binance=binance,
        engine=engine,
        llm_client=llm_client,
        executor=executor,
        safety=safety,
        notifier=notifier,
        session_factory=session_factory,
        equity_usd=settings.equity_usd,
    )
    monitor = PositionMonitor(
        client=binance,
        executor=executor,
        safety=safety,
        llm_client=llm_client,
        notifier=notifier,
        session_factory=session_factory,
    )

    monitor_thread = threading.Thread(
        target=monitor.run_forever,
        kwargs={"interval_sec": settings.monitor_interval_sec},
        daemon=True,
        name="position-monitor",
    )
    monitor_thread.start()

    symbols = _load_symbols()
    log.info("trading universe: %s", symbols)

    while True:
        append_equity(equity_usd=settings.equity_usd)
        results = trader.run_cycle(symbols)
        accepted = sum(1 for r in results if r.accepted)
        log.info("cycle complete: %d processed, %d accepted", len(results), accepted)
        time.sleep(settings.loop_interval_sec)


if __name__ == "__main__":
    main()
