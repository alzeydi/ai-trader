"""Live-bot entry point.

Wires the three layers together:
    Layer 1: SignalEngine        — rule-based candidate generation
    Layer 2: VetoAgent (Claude)  — discretionary override
    Layer 3: Trader              — sizing, risk gates, order placement
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import yaml

from src.config import settings
from src.data.binance_client import BinanceClient
from src.data.context import fetch_context
from src.execution.monitor import fetch_open_positions
from src.execution.trader import Trader
from src.llm.client import ClaudeClient
from src.llm.veto import VetoAgent
from src.notify.telegram import TelegramNotifier
from src.persistence.csv_writer import append_decision, append_equity
from src.risk.limits import RiskState
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


def run_once(
    symbols: list[str],
    client: BinanceClient,
    engine: SignalEngine,
    veto: VetoAgent,
    trader: Trader,
    notifier: TelegramNotifier,
) -> None:
    log = logging.getLogger("main")
    open_positions = fetch_open_positions(client)
    risk_state = RiskState(
        equity_usd=settings.equity_usd,
        open_positions=len(open_positions),
        daily_pnl_pct=0.0,
        drawdown_pct=0.0,
    )
    append_equity(equity_usd=risk_state.equity_usd)

    for symbol in symbols:
        try:
            candidate = engine.evaluate(symbol)
            if candidate is None:
                continue
            ctx = fetch_context(client, symbol).as_dict()
            equity_payload = {
                "balance_usdt": risk_state.equity_usd,
                "open_positions": risk_state.open_positions,
                "consecutive_losses": trader.safety.consecutive_losses,
            }
            decision = veto.vet(candidate, context=ctx, equity=equity_payload)
            append_decision(symbol, "ALLOW" if decision.allow else "REJECT",
                            decision.confidence, decision.reason)
            if not decision.allow:
                log.info("%s: VETOED — %s", symbol, decision.reason)
                continue
            outcome = trader.execute(candidate, risk_state)
            if outcome.accepted:
                notifier.entry(symbol, candidate.side.value, outcome.quantity, candidate.entry)
            else:
                log.info("%s: not executed — %s", symbol, outcome.reason)
        except Exception as exc:  # noqa: BLE001
            log.exception("error processing %s", symbol)
            notifier.error(f"{symbol}: {exc}")


def main() -> None:
    _configure_logging()
    log = logging.getLogger("main")
    log.info("starting ai-trader (mode=%s, dry_run=%s, testnet=%s)",
             settings.trading_mode, settings.dry_run, settings.binance_testnet)

    client = BinanceClient()
    engine = SignalEngine(client=client)
    veto = VetoAgent(client=ClaudeClient())
    safety = SafetyMode()
    trader = Trader(client=client, safety=safety)
    notifier = TelegramNotifier()

    symbols = _load_symbols()
    log.info("trading universe: %s", symbols)

    while True:
        run_once(symbols, client, engine, veto, trader, notifier)
        time.sleep(settings.loop_interval_sec)


if __name__ == "__main__":
    main()
