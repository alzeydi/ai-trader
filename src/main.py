"""Live-bot entry point.

Wires the three layers together:
    Layer 1: SignalEngine        — multi-timeframe candidate generation
    Layer 2: veto_candidate (Claude)  — LLM override (TAKE / SKIP)
    Layer 3: Trader              — sizing, risk gates, order placement
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import yaml

from src.config import settings
from src.data.binance_client import BinanceClient
from src.data.context import get_market_context
from src.execution.monitor import fetch_open_positions
from src.execution.trader import Trader
from src.llm.client import ClaudeClient
from src.llm.veto import veto_candidate
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
    llm_client: ClaudeClient,
    trader: Trader,
    notifier: TelegramNotifier,
    equity_usd: float,
) -> None:
    log = logging.getLogger("main")
    open_positions = fetch_open_positions(client)
    risk_state = RiskState(
        equity_usd=equity_usd,
        open_positions=len(open_positions),
        daily_pnl_pct=0.0,
        drawdown_pct=0.0,
    )
    append_equity(equity_usd=equity_usd)

    for symbol in symbols:
        try:
            signal = engine.generate_signals(symbol)
            if signal is None:
                continue

            mc = get_market_context(symbol, client)
            funding_pct = (mc.funding_rate or 0.0) * 100.0
            liq_total = mc.liquidations_4h_usd or 0.0
            # Until we split the liquidation feed by side, attribute the full
            # 4h notional to the side facing the squeeze (worst case for us).
            ctx_payload = {
                "funding_rate_now": funding_pct,
                "funding_rate_8h_avg": funding_pct,
                "oi_delta_1h_pct": mc.open_interest_delta_1h,
                "liq_long": liq_total if signal.side == "long" else 0.0,
                "liq_short": liq_total if signal.side == "short" else 0.0,
                "btc_dominance": mc.btc_dominance,
                "btc_direction": mc.btc_1h_direction,
            }
            account_payload = {
                "open_positions": risk_state.open_positions,
                "daily_pnl_pct": risk_state.daily_pnl_pct,
                "losses_streak": trader.safety.consecutive_losses,
            }
            veto_resp = veto_candidate(
                signal,
                context=ctx_payload,
                account=account_payload,
                client=llm_client,
            )
            append_decision(
                symbol,
                veto_resp.decision,
                veto_resp.confidence,
                veto_resp.reasoning,
            )

            if veto_resp.decision == "SKIP":
                log.info("%s: SKIPPED — %s", symbol, veto_resp.reasoning)
                continue

            outcome = trader.execute(signal, veto_resp, risk_state, equity_usd)
            if outcome.accepted and outcome.order is not None:
                notifier.entry(
                    symbol,
                    outcome.order.side,
                    outcome.order.quantity,
                    outcome.order.entry_price,
                )
            else:
                log.info("%s: not executed — %s", symbol, outcome.reason)

        except Exception as exc:  # noqa: BLE001
            log.exception("error processing %s", symbol)
            notifier.error(f"{symbol}: {exc}")


def main() -> None:
    _configure_logging()
    log = logging.getLogger("main")
    log.info(
        "starting ai-trader (mode=%s, dry_run=%s, testnet=%s)",
        settings.trading_mode,
        settings.dry_run,
        settings.binance_testnet,
    )

    client = BinanceClient()
    engine = SignalEngine(client=client)
    llm_client = ClaudeClient()
    safety = SafetyMode()
    trader = Trader(client=client, safety=safety)
    notifier = TelegramNotifier()

    symbols = _load_symbols()
    log.info("trading universe: %s", symbols)

    while True:
        run_once(symbols, client, engine, llm_client, trader, notifier, settings.equity_usd)
        time.sleep(settings.loop_interval_sec)


if __name__ == "__main__":
    main()
