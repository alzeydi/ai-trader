"""Live-bot entry point.

Wires the four runtime components together:
    SignalEngine     — multi-timeframe candidate generation (Layer 1)
    veto_candidate   — pre-flight + LLM veto                 (Layer 2)
    Trader           — risk + sizing + execution             (Layer 3)
    PositionMonitor  — open-position lifecycle (parallel)
"""

from __future__ import annotations

import logging
import os
import threading
import time
from datetime import datetime, timezone
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
from src.persistence.db import (
    Trade,
    append_equity_snapshot,
    daily_realized_pnl,
    make_session_factory,
)
from src.risk.safety_mode import SafetyMode
from src.signal.engine import SignalEngine


def _configure_logging() -> None:
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _notify_startup(notifier: TelegramNotifier) -> None:
    ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    deploy_id = os.getenv("RAILWAY_DEPLOYMENT_ID") or os.getenv("RAILWAY_GIT_COMMIT_SHA", "")
    deploy_suffix = f"\nDeploy: `{deploy_id[:12]}`" if deploy_id else ""
    notifier.send(
        "🚀 *ai-trader started*\n"
        f"`{ts}`\n"
        f"Mode: `{settings.trading_mode}` | Paper: `{settings.paper_trading}` | "
        f"Testnet: `{settings.binance_testnet}` | Dry-run: `{settings.dry_run}`"
        f"{deploy_suffix}"
    )


def _close_stale_paper_trades(session_factory) -> int:
    """When switching from paper to live, paper trades persisted in SQLite
    keep counting toward the open-positions cap and block real entries.
    Mark them closed at startup so the live run starts from a clean slate.
    """
    if settings.paper_trading:
        return 0
    log = logging.getLogger("main")
    now = datetime.now(tz=timezone.utc)
    with session_factory() as s:
        stale = s.query(Trade).filter(
            Trade.closed_at.is_(None), Trade.paper.is_(True)
        ).all()
        for t in stale:
            t.closed_at = now
            t.close_price = t.entry
            t.close_reason = "mode_switch_paper_to_live"
        if stale:
            s.commit()
            log.info(
                "startup: closed %d stale paper trade(s) on switch to live",
                len(stale),
            )
    return len(stale)


def _filter_known_symbols(binance, symbols, log) -> list[str]:
    """Drop symbols the exchange does not list, warn loudly. Avoids spamming
    BadSymbol errors every cycle for typos or pairs that exist on mainnet
    but not on demo-fapi.
    """
    try:
        binance.exchange.load_markets()
    except Exception as exc:  # noqa: BLE001
        log.warning("could not preload markets for validation: %s", exc)
        return symbols
    known = set(binance.exchange.markets.keys())
    kept, dropped = [], []
    for s in symbols:
        (kept if s in known else dropped).append(s)
    if dropped:
        log.warning("dropping %d unlisted symbol(s): %s", len(dropped), dropped)
    return kept


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
    _close_stale_paper_trades(session_factory)
    binance = BinanceClient()
    engine = SignalEngine(client=binance)
    llm_client = ClaudeClient()
    notifier = TelegramNotifier()
    _notify_startup(notifier)
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
    symbols = _filter_known_symbols(binance, symbols, log)
    log.info("trading universe (%d): %s", len(symbols), symbols)

    while True:
        realized = daily_realized_pnl(session_factory)
        append_equity(equity_usd=settings.equity_usd, realized=realized)
        append_equity_snapshot(
            session_factory,
            equity_usd=settings.equity_usd,
            realized_pnl=realized,
        )
        results = trader.run_cycle(symbols)
        accepted = sum(1 for r in results if r.accepted)
        log.info("cycle complete: %d processed, %d accepted", len(results), accepted)
        time.sleep(settings.loop_interval_sec)


if __name__ == "__main__":
    main()
