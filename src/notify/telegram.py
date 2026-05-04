"""Telegram alerts (Markdown) for entries, closes, and errors.

Falls back to logging when the bot is disabled or credentials are missing.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from src.config import settings

log = logging.getLogger(__name__)


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

    def send(self, text: str) -> None:
        if self._bot is None:
            log.info("[telegram-disabled] %s", text)
            return
        try:
            asyncio.get_event_loop().run_until_complete(
                self._bot.send_message(chat_id=self.chat_id, text=text, parse_mode="Markdown")
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("telegram send failed: %s", exc)

    def entry(self, symbol: str, side: str, qty: float, entry: float) -> None:
        self.send(f"*ENTRY* `{symbol}` {side} qty=`{qty:.4f}` @ `{entry:.4f}`")

    def close(self, symbol: str, pnl_usd: float) -> None:
        emoji = "+" if pnl_usd >= 0 else ""
        self.send(f"*CLOSE* `{symbol}` PnL `{emoji}{pnl_usd:.2f}` USDT")

    def error(self, message: str) -> None:
        self.send(f"*ERROR* {message}")
