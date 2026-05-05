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
from sqlalchemy import text

from src.config import settings
from src.data.binance_client import BinanceClient
from src.execution.monitor import PositionMonitor, fetch_open_positions
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


def _reconcile_open_trades(session_factory, binance, log) -> None:
    """At startup, sync the DB's open Trade rows with the exchange.

    Two classes of zombie can survive a restart:
      1) Multiple open Trade rows for the same symbol — left over from
         pre-idempotency-fix duplicate opens. Keep one most-recent per
         symbol that has an exchange position; close the rest.
      2) Open Trade rows for symbols the exchange has NO position on —
         the position was closed (SL/TP filled) but our process died
         before _close_live could mark the row closed.

    Both cases are closed with close_price = entry (no PnL info), reason
    "reconciled_*". The next monitor tick will re-evaluate the survivors
    against the exchange and close them properly if needed.
    """
    if settings.paper_trading:
        return
    try:
        snaps = fetch_open_positions(binance)
    except Exception as exc:  # noqa: BLE001
        log.warning("reconcile: fetch_open_positions failed, skipping: %s", exc)
        return
    exchange_symbols = {s.symbol for s in snaps if s.quantity > 0}

    now = datetime.now(tz=timezone.utc)
    closed_reconciled = 0
    closed_orphans = 0
    with session_factory() as s:
        rows = s.query(Trade).filter(Trade.closed_at.is_(None)).order_by(
            Trade.symbol, Trade.opened_at.desc()
        ).all()
        seen_symbols: set[str] = set()
        for row in rows:
            sym = row.symbol
            if sym not in exchange_symbols:
                row.closed_at = now
                row.close_price = row.entry
                row.close_reason = "reconciled_no_exchange_position"
                closed_orphans += 1
                continue
            if sym in seen_symbols:
                row.closed_at = now
                row.close_price = row.entry
                row.close_reason = "reconciled_duplicate"
                closed_reconciled += 1
                continue
            seen_symbols.add(sym)
        if closed_reconciled or closed_orphans:
            s.commit()
            log.info(
                "startup: reconciled DB with exchange — %d duplicates, "
                "%d orphans (no exchange position) closed",
                closed_reconciled, closed_orphans,
            )


def _adopt_orphan_exchange_positions(session_factory, binance, log) -> int:
    """Insert DB rows for live exchange positions we don't know about.

    Symmetric counterpart to `_reconcile_open_trades`: handles the case where
    the exchange holds a position but we have no matching open Trade row.
    Without this, the monitor cannot trail/close such positions and they
    silently linger on the wallet. Triggers most often after a manual SQL
    wipe (RESET_DB_ON_START) but also after disk loss / DB rollback.

    For each orphan position we:
      * look at open reduce-only orders to recover SL/TP IDs and prices that
        were placed by a previous container instance;
      * if no SL exists, place a fresh STOP_MARKET reduce-only at
        max(min_stop_pct, 0.8 %) of entry — the position must not stay naked;
      * insert a Trade row so the monitor adopts it on the next tick.

    Returns the number of positions adopted.
    """
    if settings.paper_trading:
        return 0
    try:
        snaps = fetch_open_positions(binance)
    except Exception as exc:  # noqa: BLE001
        log.warning("adopt: fetch_open_positions failed, skipping: %s", exc)
        return 0
    if not snaps:
        return 0

    with session_factory() as s:
        known = {
            row.symbol for row in
            s.query(Trade.symbol).filter(Trade.closed_at.is_(None)).all()
        }

    ex = binance.exchange
    adopted = 0
    for snap in snaps:
        if snap.quantity <= 0 or snap.symbol in known:
            continue

        side = "long" if snap.side.lower().startswith("l") else "short"
        entry = float(snap.entry_price) or float(snap.mark_price)
        if entry <= 0:
            log.warning("adopt: %s skipped, no entry/mark price", snap.symbol)
            continue

        sl_oid = ""
        tp_oid = ""
        sl_price: float | None = None
        tp_price: float | None = None
        try:
            open_orders = ex.fetch_open_orders(snap.symbol)
        except Exception as exc:  # noqa: BLE001
            log.warning("adopt: fetch_open_orders %s failed: %s", snap.symbol, exc)
            open_orders = []
        for o in open_orders:
            info = o.get("info") or {}
            if not (o.get("reduceOnly") or info.get("reduceOnly") in (True, "true")):
                continue
            otype = (o.get("type") or info.get("type") or "").upper()
            stop_price = o.get("stopPrice") or info.get("stopPrice")
            try:
                stop_price_f = float(stop_price) if stop_price is not None else None
            except (TypeError, ValueError):
                stop_price_f = None
            if "STOP" in otype and "PROFIT" not in otype and not sl_oid:
                sl_oid = str(o.get("id") or "")
                sl_price = stop_price_f
            elif "TAKE_PROFIT" in otype and not tp_oid:
                tp_oid = str(o.get("id") or "")
                tp_price = stop_price_f

        # Position must not stay naked. Place a defensive SL if none exists.
        if sl_price is None:
            sl_dist = entry * float(settings.min_stop_pct)
            sl_price = entry - sl_dist if side == "long" else entry + sl_dist
            exit_side = "sell" if side == "long" else "buy"
            try:
                price_str = ex.price_to_precision(snap.symbol, sl_price)
                sl_price_priced = float(price_str)
                resp = ex.create_order(
                    snap.symbol, "STOP_MARKET", exit_side, snap.quantity, None,
                    {"stopPrice": sl_price_priced, "reduceOnly": True},
                )
                sl_oid = str(resp.get("id") or "")
                sl_price = sl_price_priced
                log.info(
                    "adopt: placed defensive SL on %s @ %.6f (orphan was naked)",
                    snap.symbol, sl_price,
                )
            except Exception as exc:  # noqa: BLE001
                log.error(
                    "adopt: SL placement FAILED for %s — position remains naked: %s",
                    snap.symbol, exc,
                )

        # TP is best-effort; SL is the hard requirement.
        if tp_price is None:
            sl_dist_actual = abs(entry - sl_price) if sl_price else entry * 0.008
            tp_dist = sl_dist_actual * (
                float(settings.atr_take_multiplier) / float(settings.atr_stop_multiplier)
            )
            tp_price = entry + tp_dist if side == "long" else entry - tp_dist

        risk_usd = abs(entry - sl_price) * snap.quantity if sl_price else 0.0

        with session_factory() as s:
            s.add(Trade(
                symbol=snap.symbol,
                side=side,
                type="adopted",
                entry=entry,
                stop=float(sl_price) if sl_price else entry,
                original_stop=float(sl_price) if sl_price else entry,
                take_profit=float(tp_price),
                quantity=snap.quantity,
                leverage=int(settings.leverage),
                risk_usd=float(risk_usd),
                pnl_usd=0.0,
                paper=False,
                entry_order_id=None,
                stop_order_id=sl_oid or None,
                take_order_id=tp_oid or None,
                invalidation_signal=None,
                llm_confidence=0.0,
            ))
            s.commit()
        adopted += 1
        log.info(
            "adopt: %s %s qty=%.6f entry=%.6f sl=%.6f tp=%.6f (sl_oid=%s tp_oid=%s)",
            snap.symbol, side, snap.quantity, entry,
            sl_price or 0.0, tp_price or 0.0, sl_oid or "-", tp_oid or "-",
        )

    if adopted:
        log.warning(
            "startup: adopted %d orphan exchange position(s) into DB; "
            "monitor will now manage them",
            adopted,
        )
    return adopted


def _live_unrealized_pnl(binance) -> float:
    """Sum unrealized PnL across all open positions reported by the exchange.

    Returns 0.0 on any failure (network, auth) so the equity curve still
    advances with realized PnL even if the snapshot is briefly stale.
    """
    try:
        snaps = fetch_open_positions(binance)
    except Exception:  # noqa: BLE001
        return 0.0
    return float(sum(s.unrealized_pnl for s in snaps))


def _wipe_history(session_factory) -> bool:
    """One-shot reset of all stat-bearing tables, gated by RESET_DB_ON_START.

    Set RESET_DB_ON_START=true in Railway Variables, redeploy once, then
    flip it back to false. The flag is intentionally an env var (not a
    setting) so it can't be left on accidentally across many deploys.
    """
    flag = os.getenv("RESET_DB_ON_START", "").strip().lower()
    if flag not in ("1", "true", "yes", "on"):
        return False
    log = logging.getLogger("main")
    tables = (
        "trades",
        "ai_decisions",
        "equity_snapshots",
        "positions",
        "safety_state",
    )
    with session_factory() as s:
        for tbl in tables:
            try:
                s.execute(text(f"DELETE FROM {tbl}"))
            except Exception as exc:  # noqa: BLE001
                log.warning("wipe %s skipped: %s", tbl, exc)
        s.commit()
    log.warning(
        "RESET_DB_ON_START active — wiped %s. UNSET the env var "
        "after this deploy to avoid wiping again on the next restart.",
        ", ".join(tables),
    )
    return True


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
    markets = binance.exchange.markets
    kept, dropped_unlisted, dropped_inactive = [], [], []
    for s in symbols:
        m = markets.get(s)
        if m is None:
            dropped_unlisted.append(s)
            continue
        # Some symbols are listed but in pre-trading / maintenance / delisted
        # state. Calling fetch_ohlcv on them returns -1122 "Invalid symbol
        # status" and pollutes every cycle with tracebacks.
        if m.get("active") is False:
            dropped_inactive.append(s)
            continue
        kept.append(s)
    if dropped_unlisted:
        log.warning("dropping %d unlisted symbol(s): %s",
                    len(dropped_unlisted), dropped_unlisted)
    if dropped_inactive:
        log.warning("dropping %d inactive symbol(s): %s",
                    len(dropped_inactive), dropped_inactive)
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
    wiped = _wipe_history(session_factory)
    _close_stale_paper_trades(session_factory)
    binance = BinanceClient()
    if not wiped:
        _reconcile_open_trades(session_factory, binance, log)
    # Adoption must run even after a wipe — that's exactly when the DB is
    # empty but the exchange still holds open positions from the previous
    # container instance.
    _adopt_orphan_exchange_positions(session_factory, binance, log)
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
        unrealized = _live_unrealized_pnl(binance) if not settings.paper_trading else 0.0
        equity = float(settings.equity_usd) + realized + unrealized
        append_equity(equity_usd=equity, realized=realized, unrealized=unrealized)
        append_equity_snapshot(
            session_factory,
            equity_usd=equity,
            balance_usd=float(settings.equity_usd) + realized,
            realized_pnl=realized,
            unrealized_pnl=unrealized,
        )
        results = trader.run_cycle(symbols)
        accepted = sum(1 for r in results if r.accepted)
        log.info("cycle complete: %d processed, %d accepted", len(results), accepted)
        time.sleep(settings.loop_interval_sec)


if __name__ == "__main__":
    main()
