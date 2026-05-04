"""Anthropic Claude wrapper for the veto layer.

Responsibilities:
- Render the veto prompts (system.txt + user.j2) for a `CandidateSignal`.
- Call the Anthropic API with a 30s timeout, retrying transient errors and
  unparseable responses up to `max_retries` times with exponential backoff.
- Parse the strict JSON response into a `VetoResponse`.
- Track input/output tokens and an estimated USD cost per call.
- Persist every decision (request + raw response + cost) to `ai_decisions`.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from anthropic import Anthropic, APIError, APITimeoutError
from jinja2 import Environment, FileSystemLoader, select_autoescape
from pydantic import ValidationError

from src.config import settings
from src.llm.types import VetoAccount, VetoContext, VetoResponse
from src.persistence.db import AIDecision, make_session_factory
from src.signal.types import CandidateSignal

log = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

# Approximate Anthropic list prices in USD per million tokens.
# Refresh when Anthropic publishes new rates.
_PRICING_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-opus-4-7": (15.0, 75.0),
    "claude-opus-4-6": (15.0, 75.0),
    "claude-opus-4-5": (15.0, 75.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-sonnet-4-5": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}
_DEFAULT_PRICE = (15.0, 75.0)


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    in_rate, out_rate = _DEFAULT_PRICE
    for prefix, rates in _PRICING_USD_PER_MTOK.items():
        if model.startswith(prefix):
            in_rate, out_rate = rates
            break
    return (input_tokens / 1_000_000) * in_rate + (output_tokens / 1_000_000) * out_rate


def _extract_json(text: str) -> dict[str, Any]:
    """Best-effort JSON extraction; raises `json.JSONDecodeError` on failure."""
    text = text.strip()
    if text.startswith("```"):
        # The model occasionally wraps output in a fenced code block.
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


@dataclass
class ClaudeClient:
    model: str = field(default_factory=lambda: settings.claude_model)
    max_tokens: int = field(default_factory=lambda: settings.claude_max_tokens)
    temperature: float = field(default_factory=lambda: settings.claude_temperature)
    timeout_s: float = 30.0
    max_retries: int = 3
    log_to_db: bool = True

    _client: Anthropic | None = field(default=None, init=False, repr=False)
    _env: Environment = field(init=False, repr=False)
    _system_prompt: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        api_key = settings.anthropic_api_key.get_secret_value()
        # We drive retries ourselves so we can also re-prompt on JSON parse
        # failures; disable the SDK's built-in retry layer.
        self._client = (
            Anthropic(api_key=api_key, timeout=self.timeout_s, max_retries=0)
            if api_key
            else None
        )
        self._env = Environment(
            loader=FileSystemLoader(str(PROMPTS_DIR)),
            autoescape=select_autoescape(),
            keep_trailing_newline=True,
        )
        self._system_prompt = (PROMPTS_DIR / "system.txt").read_text(encoding="utf-8")

    # ------------------------------------------------------------------
    # Hook overridden by tests to fake the Anthropic call.
    # Returns (response_text, input_tokens, output_tokens).
    # ------------------------------------------------------------------
    def _call(self, system: str, user: str) -> tuple[str, int, int]:
        if self._client is None:
            raise RuntimeError("ANTHROPIC_API_KEY is not configured")
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text_parts = [getattr(p, "text", "") for p in (msg.content or [])]
        text = "".join(text_parts).strip()
        usage = getattr(msg, "usage", None)
        in_tok = int(getattr(usage, "input_tokens", 0) or 0)
        out_tok = int(getattr(usage, "output_tokens", 0) or 0)
        return text, in_tok, out_tok

    def render_user_prompt(
        self,
        candidate: CandidateSignal,
        context: VetoContext,
        account: VetoAccount,
    ) -> str:
        template = self._env.get_template("user.j2")
        return template.render(candidate=candidate, context=context, account=account)

    def veto(
        self,
        candidate: CandidateSignal,
        context: VetoContext | dict[str, Any] | None = None,
        account: VetoAccount | dict[str, Any] | None = None,
    ) -> VetoResponse:
        """Run the LLM veto, retrying transient errors and bad JSON."""
        ctx = context if isinstance(context, VetoContext) else VetoContext.model_validate(context or {})
        acct = account if isinstance(account, VetoAccount) else VetoAccount.model_validate(account or {})
        user_prompt = self.render_user_prompt(candidate, ctx, acct)

        started = time.monotonic()
        in_tok = out_tok = 0
        raw_response = ""
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                raw_response, in_tok, out_tok = self._call(
                    self._system_prompt, user_prompt
                )
                payload = _extract_json(raw_response)
                response = VetoResponse.model_validate(payload)
                duration_ms = int((time.monotonic() - started) * 1000)
                self._persist(
                    candidate=candidate,
                    response=response,
                    user_prompt=user_prompt,
                    raw_response=raw_response,
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    duration_ms=duration_ms,
                    attempts=attempt,
                )
                return response
            except (json.JSONDecodeError, ValidationError, ValueError, TypeError) as exc:
                last_error = exc
                log.warning(
                    "veto: invalid response on attempt %d/%d: %s | raw=%s",
                    attempt, self.max_retries, exc, raw_response[:200],
                )
            except (APITimeoutError, APIError) as exc:
                last_error = exc
                log.warning(
                    "veto: api error on attempt %d/%d: %s",
                    attempt, self.max_retries, exc,
                )

            if attempt < self.max_retries:
                time.sleep(2 ** (attempt - 1))  # 1s, 2s, 4s

        # All retries exhausted — fall back to a safe SKIP and record the failure.
        duration_ms = int((time.monotonic() - started) * 1000)
        log.error(
            "veto: giving up for %s after %d attempts; last_error=%s",
            candidate.symbol, self.max_retries, last_error,
        )
        fallback = VetoResponse(
            decision="SKIP",
            confidence=0.0,
            invalidation_signal="parse_error",
            reasoning=f"LLM call failed after {self.max_retries} attempts: {last_error}",
        )
        self._persist(
            candidate=candidate,
            response=fallback,
            user_prompt=user_prompt,
            raw_response=raw_response,
            input_tokens=in_tok,
            output_tokens=out_tok,
            duration_ms=duration_ms,
            attempts=self.max_retries,
        )
        return fallback

    # ------------------------------------------------------------------
    def _persist(
        self,
        *,
        candidate: CandidateSignal,
        response: VetoResponse,
        user_prompt: str,
        raw_response: str,
        input_tokens: int,
        output_tokens: int,
        duration_ms: int,
        attempts: int,
    ) -> None:
        if not self.log_to_db:
            return
        try:
            session_factory = make_session_factory()
            with session_factory() as s:
                s.add(
                    AIDecision(
                        symbol=candidate.symbol,
                        side=candidate.side,
                        entry_type=candidate.entry_type,
                        signal_strength=candidate.signal_strength,
                        decision=response.decision,
                        confidence=response.confidence,
                        invalidation_signal=response.invalidation_signal,
                        reasoning=response.reasoning,
                        skipped_by_preflight=False,
                        preflight_rule=None,
                        request_user=user_prompt,
                        response_raw=raw_response,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        cost_usd=estimate_cost_usd(self.model, input_tokens, output_tokens),
                        duration_ms=duration_ms,
                        attempts=attempts,
                        model=self.model,
                    )
                )
                s.commit()
        except Exception as exc:  # noqa: BLE001 - DB failure must never sink a trade decision
            log.warning("veto: failed to persist ai_decision: %s", exc)
