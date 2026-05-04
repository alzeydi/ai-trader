"""Order execution: paper or live.

`OrderExecutor` is the only place that talks to the Binance REST API for
order placement. In paper mode it simulates a fill at `entry_price` and
records the trade in the DB; live mode also records the trade after sending
the bracket (entry market + reduce-only stop + reduce-only take-profit).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from sqlalchemy import select

from src.config import settings
from src.execution.types import ExecutionResult, TradeOrder
from src.persistence.db import SessionFactory, Trade, make_session_factory

if TYPE_CHECKING:
    from src.data.binance_client import BinanceClient

log = logging.getLogger(__name__)


@dataclass
class OrderExecutor:
    client: "BinanceClient | None" = None
    paper: bool = field(default_factory=lambda: settings.paper_trading)
    session_factory: SessionFactory = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.session_factory is None:
            self.session_factory = make_session_factory()

    # ------------------------------------------------------------------
    def open_position(self, order: TradeOrder) -> ExecutionResult:
        if self.paper:
            return self._open_paper(order)
        return self._open_live(order)

    def close_position(self, symbol: str, reason: str) -> ExecutionResult:
        if self.paper:
            return self._close_paper(symbol, reason)
        return self._close_live(symbol, reason)

    # --- paper -------------------------------------------------------
    def _open_paper(self, order: TradeOrder) -> ExecutionResult:
        trade_id = self._record_open(order, paper=True)
        log.info(
            "PAPER OPEN %s %s qty=%.6f entry=%.4f sl=%.4f tp=%.4f trade_id=%s",
            order.symbol, order.side, order.quantity,
            order.entry_price, order.stop_loss, order.take_profit, trade_id,
        )
        return ExecutionResult(
            success=True,
            paper=True,
            trade_id=trade_id,
            fill_price=order.entry_price,
            reason="paper-fill",
        )

    def _close_paper(self, symbol: str, reason: str) -> ExecutionResult:
        with self.session_factory() as s:
            row = s.execute(
                select(Trade)
                .where(Trade.symbol == symbol)
                .where(Trade.closed_at.is_(None))
                .order_by(Trade.opened_at.desc())
            ).scalar_one_or_none()
            if row is None:
                return ExecutionResult(
                    success=False, paper=True, reason=f"no open paper trade for {symbol}"
                )

            close_price = self._mark_price(symbol) or row.entry
            pnl = self._compute_pnl(
                side=row.side, qty=row.quantity, entry=row.entry, exit_=close_price
            )
            row.closed_at = datetime.now(tz=timezone.utc)
            row.close_price = close_price
            row.close_reason = reason
            row.pnl_usd = pnl
            s.commit()
            trade_id = row.id

        log.info(
            "PAPER CLOSE %s reason=%s exit=%.4f pnl=%.2f trade_id=%s",
            symbol, reason, close_price, pnl, trade_id,
        )
        return ExecutionResult(
            success=True,
            paper=True,
            trade_id=trade_id,
            fill_price=close_price,
            pnl_usd=pnl,
            reason=reason,
        )

    # --- live --------------------------------------------------------
    def _open_live(self, order: TradeOrder) -> ExecutionResult:
        if self.client is None:
            raise RuntimeError("OrderExecutor in live mode requires a BinanceClient")
        ex = self.client.exchange

        try:
            ex.set_leverage(order.leverage, order.symbol)
        except Exception as exc:  # noqa: BLE001
            log.warning("set_leverage failed for %s: %s", order.symbol, exc)

        entry_side = "buy" if order.side == "long" else "sell"
        exit_side = "sell" if order.side == "long" else "buy"

        entry_resp = ex.create_order(
            order.symbol, "market", entry_side, order.quantity,
            None, {"reduceOnly": False},
        )
        sl_params: dict[str, Any] = {"stopPrice": order.stop_loss, "reduceOnly": True}
        tp_params: dict[str, Any] = {"stopPrice": order.take_profit, "reduceOnly": True}
        sl_resp = ex.create_order(
            order.symbol, "STOP_MARKET", exit_side, order.quantity, None, sl_params
        )
        tp_resp = ex.create_order(
            order.symbol, "TAKE_PROFIT_MARKET", exit_side, order.quantity, None, tp_params
        )

        trade_id = self._record_open(
            order,
            paper=False,
            entry_order_id=str(entry_resp.get("id") or ""),
            stop_order_id=str(sl_resp.get("id") or ""),
            take_order_id=str(tp_resp.get("id") or ""),
        )
        return ExecutionResult(
            success=True,
            paper=False,
            trade_id=trade_id,
            entry_order_id=str(entry_resp.get("id") or ""),
            stop_order_id=str(sl_resp.get("id") or ""),
            take_order_id=str(tp_resp.get("id") or ""),
            fill_price=float(entry_resp.get("average") or order.entry_price),
            reason="live-fill",
        )

    def _close_live(self, symbol: str, reason: str) -> ExecutionResult:
        if self.client is None:
            raise RuntimeError("OrderExecutor in live mode requires a BinanceClient")
        ex = self.client.exchange

        with self.session_factory() as s:
            row = s.execute(
                select(Trade)
                .where(Trade.symbol == symbol)
                .where(Trade.closed_at.is_(None))
                .order_by(Trade.opened_at.desc())
            ).scalar_one_or_none()
            if row is None:
                return ExecutionResult(
                    success=False, paper=False, reason=f"no open trade for {symbol}"
                )
            row_id = row.id
            qty = float(row.quantity)
            side = row.side

        # Cancel resting brackets, then market-exit.
        for oid in (row.stop_order_id, row.take_order_id):
            if not oid:
                continue
            try:
                ex.cancel_order(oid, symbol)
            except Exception as exc:  # noqa: BLE001
                log.warning("cancel_order %s failed: %s", oid, exc)

        exit_side = "sell" if side == "long" else "buy"
        try:
            close_resp = ex.create_order(
                symbol, "market", exit_side, qty, None, {"reduceOnly": True}
            )
        except Exception as exc:  # noqa: BLE001
            log.error("close_position market exit failed for %s: %s", symbol, exc)
            return ExecutionResult(success=False, paper=False, trade_id=row_id, reason=str(exc))

        fill = float(close_resp.get("average") or close_resp.get("price") or 0.0)
        if fill <= 0:
            fill = self._mark_price(symbol) or 0.0
        pnl = self._compute_pnl(side=side, qty=qty, entry=row.entry, exit_=fill)

        with self.session_factory() as s:
            row = s.execute(select(Trade).where(Trade.id == row_id)).scalar_one()
            row.closed_at = datetime.now(tz=timezone.utc)
            row.close_price = fill
            row.close_reason = reason
            row.pnl_usd = pnl
            s.commit()

        return ExecutionResult(
            success=True,
            paper=False,
            trade_id=row_id,
            fill_price=fill,
            pnl_usd=pnl,
            reason=reason,
        )

    # --- shared helpers ----------------------------------------------
    def _record_open(
        self,
        order: TradeOrder,
        *,
        paper: bool,
        entry_order_id: str | None = None,
        stop_order_id: str | None = None,
        take_order_id: str | None = None,
    ) -> int:
        with self.session_factory() as s:
            row = Trade(
                symbol=order.symbol,
                side=order.side,
                type=order.entry_type,
                entry=order.entry_price,
                stop=order.stop_loss,
                take_profit=order.take_profit,
                quantity=order.quantity,
                leverage=order.leverage,
                risk_usd=order.risk_usd,
                pnl_usd=0.0,
                paper=paper,
                entry_order_id=entry_order_id,
                stop_order_id=stop_order_id,
                take_order_id=take_order_id,
                invalidation_signal=order.invalidation_signal,
                llm_confidence=order.llm_confidence,
            )
            s.add(row)
            s.commit()
            return int(row.id)

    @staticmethod
    def _compute_pnl(*, side: str, qty: float, entry: float, exit_: float) -> float:
        if side == "long":
            return float(qty) * (float(exit_) - float(entry))
        return float(qty) * (float(entry) - float(exit_))

    def _mark_price(self, symbol: str) -> float | None:
        if self.client is None:
            return None
        try:
            ticker = self.client.fetch_ticker(symbol)
        except Exception as exc:  # noqa: BLE001
            log.debug("mark_price fetch failed for %s: %s", symbol, exc)
            return None
        last = ticker.get("last")
        try:
            return float(last) if last is not None else None
        except (TypeError, ValueError):
            return None


# --- legacy bracket helper (kept so backtest/tests don't break) ----------

@dataclass
class BracketOrders:
    entry_id: str | None
    stop_id: str | None
    take_id: str | None


def place_bracket(client, order, dry_run: bool | None = None) -> BracketOrders:
    """Thin wrapper around `OrderExecutor.open_position` for older callers."""
    paper = settings.paper_trading if dry_run is None else dry_run
    executor = OrderExecutor(client=client, paper=paper)
    res = executor.open_position(order)
    return BracketOrders(
        entry_id=res.entry_order_id,
        stop_id=res.stop_order_id,
        take_id=res.take_order_id,
    )
