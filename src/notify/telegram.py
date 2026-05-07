"""Telegram alerts (Markdown + emoji) for entries, closes, summaries, errors.

Falls back to logging when the bot is disabled or credentials are missing.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable

from src.config import settings

log = logging.getLogger(__name__)


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:+.2f}%"


def _fmt_money(value: float | None, decimals: int = 2) -> str:
    if value is None:
        return "—"
    sign = "+" if value > 0 else ("" if value == 0 else "")
    return f"{sign}{value:,.{decimals}f}"


def _side_emoji(side: str) -> str:
    return "🟢" if side.lower() == "long" else "🔴"


def _close_emoji(pnl_usd: float | None) -> str:
    if pnl_usd is None:
        return "⚪️"
    return "✅" if pnl_usd >= 0 else "❌"


def _get(obj: Any, name: str, default: Any = None) -> Any:
    """Read `name` from obj as attribute or mapping key."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


@dataclass
class TelegramNotifier:
    enabled: bool = settings.telegram_enabled
    chat_id: str = settings.telegram_chat_id

    def __post_init__(self) -> None:
        self._bot: Any | None = None
        token = settings.telegram_bot_token.get_secret_value()
        if self.enabled and token and self.chat_id:
            try:
                from telegram import Bot

                self._bot = Bot(token=token)
            except Exception as exc:  # noqa: BLE001
                log.warning("telegram disabled: failed to init bot: %s", exc)
                self._bot = None

    # ------------------------------------------------------------------
    # Low-level send
    # ------------------------------------------------------------------
    def send(self, text: str) -> None:
        if self._bot is None:
            log.info("[telegram-disabled] %s", text)
            return
        try:
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            loop.run_until_complete(
                self._bot.send_message(
                    chat_id=self.chat_id,
                    text=text,
                    parse_mode="Markdown",
                    disable_web_page_preview=True,
                )
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("telegram send failed: %s", exc)

    # ------------------------------------------------------------------
    # Iteration / cycle summary
    # ------------------------------------------------------------------
    def send_iteration_summary(
        self,
        *,
        positions: Iterable[Any] | None = None,
        equity: float | None = None,
        balance: float | None = None,
        unrealized_pnl: float | None = None,
        decisions: Iterable[Any] | None = None,
        cycle_index: int | None = None,
    ) -> None:
        positions = list(positions or [])
        decisions = list(decisions or [])
        accepted = sum(1 for d in decisions if _get(d, "took") or _get(d, "accepted"))

        lines: list[str] = []
        header = "🔄 *Cycle summary*"
        if cycle_index is not None:
            header += f" #{cycle_index}"
        lines.append(header)
        lines.append(
            f"`{datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}`"
        )

        if equity is not None:
            lines.append(f"💰 Equity: `{_fmt_money(equity)}` USDT")
        if balance is not None:
            lines.append(f"🏦 Balance: `{_fmt_money(balance)}` USDT")
        if unrealized_pnl is not None:
            lines.append(f"📈 Unrealized: `{_fmt_money(unrealized_pnl)}` USDT")

        lines.append(f"📊 Open positions: *{len(positions)}*")
        for p in positions[:10]:
            sym = _get(p, "symbol", "?")
            side = _get(p, "side", "?")
            qty = _get(p, "quantity", _get(p, "qty", 0.0)) or 0.0
            upnl = _get(p, "unrealized_pnl", 0.0) or 0.0
            lines.append(
                f"  {_side_emoji(str(side))} `{sym}` qty=`{qty:.4f}` PnL=`{_fmt_money(upnl)}`"
            )

        lines.append(
            f"🤖 Decisions: *{len(decisions)}* (accepted: *{accepted}*)"
        )
        self.send("\n".join(lines))

    # ------------------------------------------------------------------
    # Entry signal
    # ------------------------------------------------------------------
    def send_entry_signal(
        self,
        order: Any,
        *,
        margin_usd: float | None = None,
        fees_usd: float | None = None,
        ai_reasoning: str | None = None,
    ) -> None:
        symbol = _get(order, "symbol", "?")
        side = str(_get(order, "side", "long"))
        entry_type = str(_get(order, "entry_type", "") or "")
        leverage = _get(order, "leverage", 1) or 1
        entry = float(_get(order, "entry_price", _get(order, "entry", 0.0)) or 0.0)
        qty = float(_get(order, "quantity", _get(order, "qty", 0.0)) or 0.0)
        sl = float(_get(order, "stop_loss", _get(order, "stop", 0.0)) or 0.0)
        tp = float(_get(order, "take_profit", 0.0) or 0.0)
        risk_usd = float(_get(order, "risk_usd", 0.0) or 0.0)
        confidence = float(_get(order, "llm_confidence", 0.0) or 0.0)

        if margin_usd is None:
            margin_usd = (entry * qty / leverage) if leverage else entry * qty

        # R:R relative to entry: |TP-entry| / |entry-SL|.
        risk_dist = abs(entry - sl) if entry and sl else 0.0
        reward_dist = abs(tp - entry) if entry and tp else 0.0
        rr = (reward_dist / risk_dist) if risk_dist > 0 else 0.0

        emoji = _side_emoji(side)
        type_tag = f" [Type {entry_type}]" if entry_type else ""
        lines = [
            f"{emoji} *ENTRY* `{symbol}` *{side.upper()}* x{leverage}{type_tag}",
            f"Entry: `{entry:.4f}`",
            f"Qty: `{qty:.6f}`",
            f"Margin: `{margin_usd:,.2f}` USDT",
            f"Risk: `{risk_usd:,.2f}` USDT",
            f"TP: `{tp:.4f}` | SL: `{sl:.4f}` | R:R `{rr:.2f}`",
            f"Confidence: *{confidence * 100:.0f}%*",
        ]
        if fees_usd is not None:
            lines.append(f"Fees (est): `{fees_usd:,.4f}` USDT")
        if ai_reasoning:
            text = ai_reasoning.strip()
            if len(text) > 600:
                text = text[:597] + "..."
            lines.append(f"\n_AI:_ {text}")
        self.send("\n".join(lines))

    # ------------------------------------------------------------------
    # Close signal
    # ------------------------------------------------------------------
    def send_close_signal(
        self,
        trade: Any,
        *,
        balance_usd: float | None = None,
        fees_usd: float | None = None,
    ) -> None:
        symbol = _get(trade, "symbol", "?")
        side = str(_get(trade, "side", ""))
        entry = float(_get(trade, "entry", 0.0) or 0.0)
        exit_ = _get(trade, "exit", _get(trade, "close_price"))
        exit_ = float(exit_ or 0.0)
        qty = float(_get(trade, "quantity", _get(trade, "qty", 0.0)) or 0.0)
        gross = float(_get(trade, "pnl_usd", 0.0) or 0.0)
        fees = float(fees_usd if fees_usd is not None else (_get(trade, "fees", 0.0) or 0.0))
        net = gross - fees
        reason = _get(trade, "close_reason", "")
        leverage = float(_get(trade, "leverage", 1) or 1)

        change_pct = 0.0
        if entry > 0 and exit_ > 0:
            raw = (exit_ - entry) / entry * 100.0
            change_pct = raw if side.lower() == "long" else -raw

        notional = entry * qty
        margin = notional / leverage if leverage else notional
        roi_pct = (net / margin * 100.0) if margin > 0 else 0.0

        emoji = _close_emoji(net)
        lines = [
            f"{emoji} *CLOSE* `{symbol}` *{side.upper()}*",
            f"Entry → Exit: `{entry:.4f}` → `{exit_:.4f}` ({_fmt_pct(change_pct)})",
            f"Gross PnL: `{_fmt_money(gross)}` USDT",
            f"Fees: `{fees:,.4f}` USDT",
            f"Net PnL: `{_fmt_money(net)}` USDT",
            f"ROI: *{_fmt_pct(roi_pct)}*",
        ]
        if reason:
            lines.append(f"Reason: `{reason}`")
        if balance_usd is not None:
            lines.append(f"Balance: `{_fmt_money(balance_usd)}` USDT")
        self.send("\n".join(lines))

    # ------------------------------------------------------------------
    # Daily report (scheduled at 21:00 Asia/Dubai by main.py)
    # ------------------------------------------------------------------
    def send_daily_report(self, session_factory: Any) -> None:
        """Build and send the daily Dubai-day summary.

        Imported lazily so the notify module stays importable in tests
        that mock out the DB layer entirely.
        """
        from src.notify.daily_report import (
            build_daily_report,
            format_daily_report_telegram,
        )

        try:
            report = build_daily_report(session_factory)
        except Exception as exc:  # noqa: BLE001
            log.exception("daily_report: build failed: %s", exc)
            self.send_critical(f"daily_report build failed: {exc}")
            return
        try:
            self.send(format_daily_report_telegram(report))
        except Exception as exc:  # noqa: BLE001
            log.exception("daily_report: send failed: %s", exc)

    # ------------------------------------------------------------------
    # Critical / errors
    # ------------------------------------------------------------------
    def send_critical(self, text: str) -> None:
        self.send(f"🚨 *CRITICAL*\n{text}")

    # ------------------------------------------------------------------
    # Back-compat shims for older call sites.
    # ------------------------------------------------------------------
    def entry(self, symbol: str, side: str, qty: float, entry: float) -> None:
        emoji = _side_emoji(side)
        self.send(
            f"{emoji} *ENTRY* `{symbol}` *{side.upper()}* qty=`{qty:.4f}` @ `{entry:.4f}`"
        )

    def close(self, symbol: str, pnl_usd: float) -> None:
        emoji = _close_emoji(pnl_usd)
        self.send(
            f"{emoji} *CLOSE* `{symbol}` PnL `{_fmt_money(pnl_usd)}` USDT"
        )

    def error(self, message: str) -> None:
        self.send_critical(message)
