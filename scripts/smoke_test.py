"""End-to-end smoke test on Binance USDT-M Futures **testnet**.

Validates that the execution layer can really round-trip a position:

    1. Connect to the testnet, fetch ticker + market info.
    2. Set leverage = 1×, margin mode = ISOLATED.
    3. Place a tiny (~$2 notional) market BUY on BTC/USDT:USDT.
    4. Wait 30 s.
    5. Close with a market SELL (reduceOnly = true).
    6. Print a structured report: latencies, fill prices, slippage vs mid,
       fees, gross + net PnL, and any errors collected along the way.

Run with:

    python scripts/smoke_test.py [--symbol BTC/USDT:USDT] \\
                                 [--notional 2] [--hold-seconds 30]

Refuses to run unless `BINANCE_TESTNET=true`.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Allow `python scripts/smoke_test.py` from project root.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import settings  # noqa: E402
from src.data.binance_client import BinanceClient  # noqa: E402

log = logging.getLogger("smoke")


# ---------------------------------------------------------------------------
# Report containers
# ---------------------------------------------------------------------------

@dataclass
class LegReport:
    side: str                       # "buy" or "sell"
    qty: float = 0.0
    order_id: str | None = None
    status: str | None = None
    fill_price: float | None = None
    mid_at_submit: float | None = None
    slippage_pct: float | None = None  # signed: + = trader paid worse
    fee_usd: float = 0.0
    placed_at: datetime | None = None
    filled_at: datetime | None = None
    latency_ms: int | None = None
    error: str | None = None


@dataclass
class SmokeReport:
    symbol: str
    notional_target_usd: float
    leverage: int
    margin_mode: str
    testnet: bool
    started_at: datetime
    finished_at: datetime | None = None
    connect_latency_ms: int | None = None
    entry: LegReport = field(default_factory=lambda: LegReport(side="buy"))
    exit: LegReport = field(default_factory=lambda: LegReport(side="sell"))
    total_fees_usd: float = 0.0
    pnl_gross_usd: float = 0.0
    pnl_net_usd: float = 0.0
    duration_s: float = 0.0
    errors: list[str] = field(default_factory=list)
    succeeded: bool = False

    def render(self) -> str:
        def _fmt(x: Any, nd: int = 4) -> str:
            if x is None:
                return "—"
            if isinstance(x, datetime):
                return x.strftime("%Y-%m-%d %H:%M:%S UTC")
            if isinstance(x, (int, float)):
                return f"{x:.{nd}f}"
            return str(x)

        lines: list[str] = []
        ok = "PASSED" if self.succeeded else "FAILED"
        lines.append("══════════════════════════════════════════════════════════")
        lines.append(f"  ai-trader smoke test  →  {ok}")
        lines.append("══════════════════════════════════════════════════════════")
        lines.append(f"  Symbol           : {self.symbol}")
        lines.append(f"  Network          : {'TESTNET' if self.testnet else 'MAINNET (!!!)'}")
        lines.append(f"  Leverage         : {self.leverage}×   margin={self.margin_mode}")
        lines.append(f"  Notional target  : ${self.notional_target_usd:.2f}")
        lines.append(f"  Started          : {_fmt(self.started_at)}")
        lines.append(f"  Finished         : {_fmt(self.finished_at)}")
        lines.append(f"  Total duration   : {self.duration_s:.2f} s")
        lines.append(f"  Connect latency  : {_fmt(self.connect_latency_ms, 0)} ms")
        lines.append("")
        for label, leg in (("ENTRY (buy) ", self.entry), ("EXIT  (sell)", self.exit)):
            lines.append(f"── {label} ────────────────────────────────────────────────")
            lines.append(f"    order_id       : {leg.order_id or '—'}")
            lines.append(f"    status         : {leg.status or '—'}")
            lines.append(f"    qty            : {_fmt(leg.qty, 6)}")
            lines.append(f"    mid @ submit   : {_fmt(leg.mid_at_submit, 2)}")
            lines.append(f"    fill price     : {_fmt(leg.fill_price, 2)}")
            lines.append(f"    slippage       : {_fmt(leg.slippage_pct, 4)} %")
            lines.append(f"    fee            : ${_fmt(leg.fee_usd, 4)}")
            lines.append(f"    placed_at      : {_fmt(leg.placed_at)}")
            lines.append(f"    filled_at      : {_fmt(leg.filled_at)}")
            lines.append(f"    latency        : {_fmt(leg.latency_ms, 0)} ms")
            if leg.error:
                lines.append(f"    ERROR          : {leg.error}")
        lines.append("── PnL ───────────────────────────────────────────────────")
        lines.append(f"    gross          : ${self.pnl_gross_usd:+.4f}")
        lines.append(f"    fees           : ${self.total_fees_usd:.4f}")
        lines.append(f"    NET            : ${self.pnl_net_usd:+.4f}")
        if self.errors:
            lines.append("── Errors ───────────────────────────────────────────────")
            for err in self.errors:
                lines.append(f"    • {err}")
        lines.append("══════════════════════════════════════════════════════════")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _ts_to_dt(ms: int | float | None) -> datetime | None:
    if ms is None:
        return None
    try:
        return datetime.fromtimestamp(float(ms) / 1000.0, tz=timezone.utc)
    except (TypeError, ValueError):
        return None


def _round_to_step(value: float, step: float) -> float:
    if step <= 0:
        return value
    return round(value / step) * step


def _fee_in_usd(order: dict[str, Any], mark_price: float | None) -> float:
    """Sum every fee leg, converting non-USDT fees at mark price."""
    fees = order.get("fees") or ([order["fee"]] if order.get("fee") else [])
    total = 0.0
    for f in fees:
        if not f:
            continue
        cost = float(f.get("cost") or 0.0)
        currency = (f.get("currency") or "USDT").upper()
        if currency in ("USDT", "USDC", "BUSD"):
            total += cost
        elif mark_price:
            total += cost * mark_price
        else:
            total += cost  # best effort
    return total


def _ensure_filled(
    exchange: Any, symbol: str, order: dict[str, Any], timeout_s: float = 10.0
) -> dict[str, Any]:
    """Re-fetch the order until it's closed (filled) or we time out.

    Market orders on Binance Futures usually return `closed` in the first
    response, but in case the SDK handed us a still-open snapshot we poll.
    """
    if str(order.get("status") or "").lower() == "closed":
        return order
    order_id = str(order.get("id") or "")
    if not order_id:
        return order
    deadline = time.monotonic() + timeout_s
    last = order
    while time.monotonic() < deadline:
        try:
            last = exchange.fetch_order(order_id, symbol)
        except Exception as exc:  # noqa: BLE001
            log.debug("fetch_order failed: %s", exc)
        if str(last.get("status") or "").lower() == "closed":
            return last
        time.sleep(0.5)
    return last


def _market_qty(
    exchange: Any, symbol: str, notional_usd: float, mid: float
) -> tuple[float, float]:
    """Return (qty, min_notional_floor). Bumps qty up to exchange minimums."""
    try:
        market = exchange.market(symbol)
    except Exception:  # noqa: BLE001
        return notional_usd / mid, notional_usd

    limits = (market or {}).get("limits") or {}
    precision = (market or {}).get("precision") or {}
    min_amount = float((limits.get("amount") or {}).get("min") or 0.0)
    min_cost = float((limits.get("cost") or {}).get("min") or 0.0)

    qty = notional_usd / mid
    if min_amount and qty < min_amount:
        qty = min_amount
    if min_cost and qty * mid < min_cost:
        qty = min_cost / mid

    step = precision.get("amount")
    if isinstance(step, (int, float)):
        # ccxt sometimes encodes precision as decimal places (int) or step (float)
        if step >= 1 and float(step).is_integer():
            qty = round(qty, int(step))
        else:
            qty = _round_to_step(qty, float(step))
    return qty, min_cost


# ---------------------------------------------------------------------------
# Core test
# ---------------------------------------------------------------------------

def run_smoke_test(
    symbol: str,
    notional_usd: float,
    hold_seconds: float,
) -> SmokeReport:
    if not settings.binance_testnet:
        raise SystemExit("smoke test refuses to run on mainnet — set BINANCE_TESTNET=true")

    report = SmokeReport(
        symbol=symbol,
        notional_target_usd=notional_usd,
        leverage=1,
        margin_mode="ISOLATED",
        testnet=True,
        started_at=_now(),
    )

    # ── connect ────────────────────────────────────────────────────────────
    t0 = time.monotonic()
    try:
        client = BinanceClient()
        ex = client.exchange
        ex.load_markets()
        report.connect_latency_ms = int((time.monotonic() - t0) * 1000)
        log.info("connected to testnet in %d ms", report.connect_latency_ms)
    except Exception as exc:  # noqa: BLE001
        report.errors.append(f"connect: {exc}")
        report.finished_at = _now()
        report.duration_s = (report.finished_at - report.started_at).total_seconds()
        return report

    # ── leverage + margin mode ─────────────────────────────────────────────
    try:
        ex.set_leverage(1, symbol)
        log.info("leverage set to 1× for %s", symbol)
    except Exception as exc:  # noqa: BLE001
        report.errors.append(f"set_leverage: {exc}")

    try:
        # Some endpoints raise if mode is already what we want; that's fine.
        if hasattr(ex, "set_margin_mode"):
            ex.set_margin_mode("ISOLATED", symbol)
            log.info("margin mode set to ISOLATED for %s", symbol)
    except Exception as exc:  # noqa: BLE001
        msg = str(exc).lower()
        if "no need to change" in msg or "isolated" in msg:
            log.debug("margin mode already isolated")
        else:
            report.errors.append(f"set_margin_mode: {exc}")

    # ── pre-trade ticker + sizing ─────────────────────────────────────────
    try:
        ticker = client.fetch_ticker(symbol)
        mid = float(ticker.get("last") or ticker.get("close") or 0.0)
        if mid <= 0:
            raise RuntimeError(f"ticker has no usable price: {ticker}")
        log.info("ticker %s: last=%.2f", symbol, mid)
    except Exception as exc:  # noqa: BLE001
        report.errors.append(f"fetch_ticker: {exc}")
        report.finished_at = _now()
        report.duration_s = (report.finished_at - report.started_at).total_seconds()
        return report

    qty, min_cost = _market_qty(ex, symbol, notional_usd, mid)
    if min_cost and qty * mid > notional_usd:
        log.warning(
            "exchange min_notional=%.2f USDT > requested %.2f — bumped qty to %.6f (~$%.2f)",
            min_cost, notional_usd, qty, qty * mid,
        )
    if qty <= 0:
        report.errors.append("computed qty is zero — check market info / mid price")
        report.finished_at = _now()
        report.duration_s = (report.finished_at - report.started_at).total_seconds()
        return report

    # ── ENTRY: market BUY ─────────────────────────────────────────────────
    entry = report.entry
    entry.qty = qty
    entry.mid_at_submit = mid
    entry.placed_at = _now()
    t1 = time.monotonic()
    try:
        log.info("submitting BUY %s qty=%.6f @ mid=%.2f", symbol, qty, mid)
        resp = ex.create_order(symbol, "market", "buy", qty, None, {"reduceOnly": False})
        resp = _ensure_filled(ex, symbol, resp)
        entry.order_id = str(resp.get("id") or "")
        entry.status = str(resp.get("status") or "")
        fill = resp.get("average") or resp.get("price")
        entry.fill_price = float(fill) if fill else None
        entry.fee_usd = _fee_in_usd(resp, mid)
        entry.filled_at = _ts_to_dt(resp.get("timestamp")) or _now()
        entry.latency_ms = int((time.monotonic() - t1) * 1000)
        if entry.fill_price:
            # +ve = bad (paid more than mid for a buy)
            entry.slippage_pct = (entry.fill_price - mid) / mid * 100.0
        log.info(
            "entry filled: id=%s price=%.4f slippage=%.4f%% fee=$%.4f latency=%d ms",
            entry.order_id, entry.fill_price or 0.0,
            entry.slippage_pct or 0.0, entry.fee_usd, entry.latency_ms,
        )
    except Exception as exc:  # noqa: BLE001
        entry.error = str(exc)
        report.errors.append(f"entry: {exc}")
        log.error("entry failed: %s\n%s", exc, traceback.format_exc())
        report.finished_at = _now()
        report.duration_s = (report.finished_at - report.started_at).total_seconds()
        return report

    # ── HOLD ──────────────────────────────────────────────────────────────
    log.info("position open, holding for %.1f s …", hold_seconds)
    time.sleep(hold_seconds)

    # ── EXIT: market SELL (reduceOnly) ────────────────────────────────────
    exit_leg = report.exit
    exit_leg.qty = qty
    try:
        ticker2 = client.fetch_ticker(symbol)
        mid2 = float(ticker2.get("last") or ticker2.get("close") or 0.0) or mid
    except Exception as exc:  # noqa: BLE001
        report.errors.append(f"pre-exit ticker: {exc}")
        mid2 = mid
    exit_leg.mid_at_submit = mid2
    exit_leg.placed_at = _now()

    t2 = time.monotonic()
    try:
        log.info("submitting SELL %s qty=%.6f @ mid=%.2f (reduceOnly)", symbol, qty, mid2)
        resp = ex.create_order(symbol, "market", "sell", qty, None, {"reduceOnly": True})
        resp = _ensure_filled(ex, symbol, resp)
        exit_leg.order_id = str(resp.get("id") or "")
        exit_leg.status = str(resp.get("status") or "")
        fill = resp.get("average") or resp.get("price")
        exit_leg.fill_price = float(fill) if fill else None
        exit_leg.fee_usd = _fee_in_usd(resp, mid2)
        exit_leg.filled_at = _ts_to_dt(resp.get("timestamp")) or _now()
        exit_leg.latency_ms = int((time.monotonic() - t2) * 1000)
        if exit_leg.fill_price:
            # +ve = bad (received less than mid for a sell)
            exit_leg.slippage_pct = (mid2 - exit_leg.fill_price) / mid2 * 100.0
        log.info(
            "exit filled: id=%s price=%.4f slippage=%.4f%% fee=$%.4f latency=%d ms",
            exit_leg.order_id, exit_leg.fill_price or 0.0,
            exit_leg.slippage_pct or 0.0, exit_leg.fee_usd, exit_leg.latency_ms,
        )
    except Exception as exc:  # noqa: BLE001
        exit_leg.error = str(exc)
        report.errors.append(f"exit: {exc}")
        log.error("exit failed: %s\n%s", exc, traceback.format_exc())

    # ── PnL + summary ─────────────────────────────────────────────────────
    if entry.fill_price and exit_leg.fill_price:
        report.pnl_gross_usd = (exit_leg.fill_price - entry.fill_price) * qty
    report.total_fees_usd = entry.fee_usd + exit_leg.fee_usd
    report.pnl_net_usd = report.pnl_gross_usd - report.total_fees_usd
    report.finished_at = _now()
    report.duration_s = (report.finished_at - report.started_at).total_seconds()

    # Success = both legs filled + no fatal errors collected.
    report.succeeded = bool(
        entry.fill_price
        and exit_leg.fill_price
        and not entry.error
        and not exit_leg.error
    )
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Round-trip smoke test on Binance USDT-M Futures testnet",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--symbol", default="BTC/USDT:USDT")
    parser.add_argument("--notional", type=float, default=2.0,
                        help="Target notional in USDT (will bump up to exchange minimum)")
    parser.add_argument("--hold-seconds", type=float, default=30.0,
                        help="How long to keep the position open before closing")
    parser.add_argument("--log-level", default=settings.log_level)
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    report = run_smoke_test(
        symbol=args.symbol,
        notional_usd=args.notional,
        hold_seconds=args.hold_seconds,
    )
    print(report.render())
    return 0 if report.succeeded else 1


if __name__ == "__main__":
    sys.exit(main())
