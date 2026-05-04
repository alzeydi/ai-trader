"""Layer 2 — LLM veto over a rule-based candidate signal."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from src.config import settings
from src.llm.client import ClaudeClient
from src.signal.types import CandidateSignal

log = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


@dataclass
class VetoDecision:
    allow: bool
    confidence: float
    reason: str
    raw: str


@dataclass
class VetoAgent:
    client: ClaudeClient

    def __post_init__(self) -> None:
        self._env = Environment(
            loader=FileSystemLoader(str(PROMPTS_DIR)),
            autoescape=select_autoescape(),
            keep_trailing_newline=True,
        )
        self._system_prompt = (PROMPTS_DIR / "system.txt").read_text(encoding="utf-8")

    def vet(
        self,
        signal: CandidateSignal,
        context: dict[str, Any],
        equity: dict[str, Any],
    ) -> VetoDecision:
        template = self._env.get_template("user.j2")
        user = template.render(signal=signal, context=context, equity=equity)
        raw = self.client.complete(system=self._system_prompt, user=user)

        try:
            payload = json.loads(raw)
            decision = str(payload.get("decision", "REJECT")).upper()
            return VetoDecision(
                allow=decision == "ALLOW",
                confidence=float(payload.get("confidence", 0.0)),
                reason=str(payload.get("reason", "")),
                raw=raw,
            )
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            log.warning("veto: failed to parse LLM response: %s | raw=%s", exc, raw[:200])
            if settings.dry_run:
                return VetoDecision(allow=False, confidence=0.0, reason="parse_error", raw=raw)
            raise
