"""Layer 2 — LLM veto over a rule-based candidate signal."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from jinja2 import Environment, FileSystemLoader, select_autoescape

from src.config import settings
from src.llm.client import ClaudeClient
from src.signal.types import CandidateSignal

log = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


class VetoResponse(BaseModel):
    decision: Literal["TAKE", "SKIP"]
    confidence: float = Field(ge=0.0, le=1.0)
    invalidation_signal: str
    reasoning: str


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
    ) -> VetoResponse:
        template = self._env.get_template("user.j2")
        user = template.render(signal=signal, context=context, equity=equity)
        raw = self.client.complete(system=self._system_prompt, user=user)

        try:
            payload = json.loads(raw)
            decision = str(payload.get("decision", "SKIP")).upper()
            if decision not in {"TAKE", "SKIP"}:
                decision = "SKIP"
            return VetoResponse(
                decision=decision,
                confidence=float(payload.get("confidence", 0.0)),
                invalidation_signal=str(payload.get("invalidation_signal", "")),
                reasoning=str(payload.get("reasoning", payload.get("reason", ""))),
            )
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            log.warning("veto: failed to parse LLM response: %s | raw=%s", exc, raw[:200])
            if settings.dry_run:
                return VetoResponse(
                    decision="SKIP",
                    confidence=0.0,
                    invalidation_signal="parse_error",
                    reasoning="LLM response could not be parsed",
                )
            raise
