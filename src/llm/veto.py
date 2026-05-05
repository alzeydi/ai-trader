"""Layer 2 — pre-flight hard rules + LLM veto.

Hard rules duplicate the constraints in `prompts/system.txt`; we evaluate them
locally first to avoid spending Anthropic credits on calls that the prompt
would auto-reject anyway. Anything that passes pre-flight is delegated to
`ClaudeClient.veto()`.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from src.config import settings
from src.llm.client import ClaudeClient
from src.llm.types import VetoAccount, VetoContext, VetoResponse
from src.persistence.db import AIDecision, make_session_factory
from src.signal.types import CandidateSignal

log = logging.getLogger(__name__)

# Thresholds mirror the HARD RULES section of prompts/system.txt.
# Funding rate is expressed in percent (matches user.j2 rendering).
MAX_OPEN_POSITIONS = 2
MAX_DAILY_LOSS_PCT = -3.0
MAX_LOSSES_STREAK = 3
FUNDING_OVERHEATED_PCT = 0.05
LIQUIDATION_SQUEEZE_USD = 50_000_000.0

# Local cooldown to avoid paying Anthropic for identical decisions every cycle.
LLM_VETO_COOLDOWN_SEC = 600
# Minimum number of non-null context fields required before we even ask the LLM.
# With fewer than this, an LLM call is wasted money — pre-skip deterministically.
MIN_CONTEXT_FIELDS = 2


# Re-export for callers that still import VetoResponse from this module.
__all__ = [
    "VetoResponse",
    "VetoContext",
    "VetoAccount",
    "veto_candidate",
    "preflight_check",
]


# In-process cache: (symbol, side, entry_type) -> (decision, ts, response).
_DECISION_CACHE: dict[tuple[str, str, str], tuple[float, VetoResponse]] = {}


def _cache_key(c: CandidateSignal) -> tuple[str, str, str]:
    return (c.symbol, c.side, c.entry_type)


def _cooldown_hit(c: CandidateSignal) -> VetoResponse | None:
    entry = _DECISION_CACHE.get(_cache_key(c))
    if entry is None:
        return None
    ts, resp = entry
    if time.monotonic() - ts > LLM_VETO_COOLDOWN_SEC:
        return None
    return resp


def _cache_decision(c: CandidateSignal, resp: VetoResponse) -> None:
    _DECISION_CACHE[_cache_key(c)] = (time.monotonic(), resp)


def _count_available_context(ctx: VetoContext) -> int:
    fields = (
        ctx.funding_rate_now,
        ctx.oi_delta_1h_pct,
        ctx.liq_long,
        ctx.liq_short,
        ctx.btc_dominance,
        ctx.btc_direction,
    )
    return sum(1 for v in fields if v is not None)


def preflight_check(
    candidate: CandidateSignal,
    context: VetoContext,
    account: VetoAccount,
) -> str | None:
    """Return the name of the violated hard rule, or `None` if all pass."""
    if account.open_positions >= MAX_OPEN_POSITIONS:
        return "open_positions"
    if account.daily_pnl_pct <= MAX_DAILY_LOSS_PCT:
        return "daily_pnl"
    if account.losses_streak >= MAX_LOSSES_STREAK:
        return "losses_streak"
    if candidate.signal_strength < settings.signal_min_confidence:
        return "signal_strength_below_floor"
    funding = context.funding_rate_now
    if funding is not None:
        if candidate.side == "long" and funding > FUNDING_OVERHEATED_PCT:
            return "funding_long_overheated"
        if candidate.side == "short" and funding < -FUNDING_OVERHEATED_PCT:
            return "funding_short_oversold"
    if (
        candidate.side == "long"
        and context.liq_long is not None
        and context.liq_long > LIQUIDATION_SQUEEZE_USD
    ):
        return "long_liquidation_squeeze"
    if (
        candidate.side == "short"
        and context.liq_short is not None
        and context.liq_short > LIQUIDATION_SQUEEZE_USD
    ):
        return "short_liquidation_squeeze"
    if _count_available_context(context) < MIN_CONTEXT_FIELDS:
        return "insufficient_context"
    return None


def _coerce_context(context: Any) -> VetoContext:
    if isinstance(context, VetoContext):
        return context
    return VetoContext.model_validate(context or {})


def _coerce_account(account: Any) -> VetoAccount:
    if isinstance(account, VetoAccount):
        return account
    return VetoAccount.model_validate(account or {})


def _log_preflight(
    candidate: CandidateSignal,
    rule: str,
    response: VetoResponse,
) -> None:
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
                    skipped_by_preflight=True,
                    preflight_rule=rule,
                    request_user=None,
                    response_raw=None,
                    input_tokens=0,
                    output_tokens=0,
                    cost_usd=0.0,
                    duration_ms=0,
                    attempts=0,
                    model=None,
                )
            )
            s.commit()
    except Exception as exc:  # noqa: BLE001
        log.warning("veto: failed to persist preflight decision: %s", exc)


def veto_candidate(
    candidate: CandidateSignal,
    context: VetoContext | dict[str, Any] | None = None,
    account: VetoAccount | dict[str, Any] | None = None,
    *,
    client: ClaudeClient | None = None,
) -> VetoResponse:
    """Apply hard rules + cooldown first, then defer to the LLM."""
    ctx = _coerce_context(context)
    acct = _coerce_account(account)

    rule = preflight_check(candidate, ctx, acct)
    if rule is not None:
        log.info("veto: pre-flight SKIP for %s (%s)", candidate.symbol, rule)
        response = VetoResponse(
            decision="SKIP",
            confidence=1.0,
            invalidation_signal="hard rule violated; do not enter",
            reasoning=f"pre-flight hard rule failed: {rule}",
        )
        _log_preflight(candidate, rule, response)
        return response

    cached = _cooldown_hit(candidate)
    if cached is not None:
        log.info(
            "veto: cooldown hit for %s — reusing %s (saves LLM call)",
            candidate.symbol, cached.decision,
        )
        return cached

    llm = client if client is not None else ClaudeClient()
    response = llm.veto(candidate=candidate, context=ctx, account=acct)
    _cache_decision(candidate, response)
    return response
