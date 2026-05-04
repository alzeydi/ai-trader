"""Anthropic Claude wrapper with retry + small-but-strict typing surface."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from anthropic import Anthropic
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings

log = logging.getLogger(__name__)


@dataclass
class ClaudeClient:
    model: str = settings.claude_model
    max_tokens: int = settings.claude_max_tokens
    temperature: float = settings.claude_temperature
    _client: Anthropic | None = None

    def __post_init__(self) -> None:
        api_key = settings.anthropic_api_key.get_secret_value()
        self._client = Anthropic(api_key=api_key) if api_key else None

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    def complete(self, system: str, user: str) -> str:
        if self._client is None:
            raise RuntimeError("ANTHROPIC_API_KEY is not configured")
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        parts: list[Any] = list(msg.content or [])
        text_parts = [getattr(p, "text", "") for p in parts]
        return "".join(text_parts).strip()
