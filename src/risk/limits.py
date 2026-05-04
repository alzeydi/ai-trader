"""Hard risk gates evaluated *after* the LLM veto and *before* sizing."""

from __future__ import annotations

from dataclasses import dataclass

from src.config import settings
from src.llm.types import VetoResponse
from src.risk.types import AccountState
from src.signal.types import CandidateSignal


__all__ = ["AccountState", "RiskState", "can_take_signal", "check"]


# Backwards-compat: a few older call-sites (and tests) still use RiskState.
@dataclass
class RiskState:
    equity_usd: float
    open_positions: int
    daily_pnl_pct: float
    drawdown_pct: float = 0.0


def check(state: RiskState) -> tuple[bool, str]:
    """Legacy helper kept for the old `Trader.execute` path."""
    if state.open_positions >= settings.max_open_positions:
        return False, f"max_open_positions={settings.max_open_positions} reached"
    if state.daily_pnl_pct <= -settings.max_daily_loss_pct:
        return False, f"daily loss limit hit: {state.daily_pnl_pct:.2f}%"
    if state.drawdown_pct >= settings.max_drawdown_pct:
        return False, f"max drawdown breached: {state.drawdown_pct:.2f}%"
    if state.equity_usd <= 0:
        return False, "equity is zero or negative"
    return True, "ok"


def can_take_signal(
    candidate: CandidateSignal,  # noqa: ARG001 - kept for symmetry with the spec
    veto: VetoResponse,
    account: AccountState,
    safety,  # SafetyMode — typed loosely to avoid circular import
) -> tuple[bool, str]:
    """Return `(can_take, reason)`.

    Composes the LLM veto outcome with hard, deterministic account-level
    guards. The order matters: cheaper checks first, telegram-worthy ones last.
    """
    if veto.decision != "TAKE":
        return False, f"LLM veto: {veto.reasoning}"
    if veto.confidence < settings.min_confidence:
        return False, f"Low confidence: {veto.confidence:.2f}"
    if account.open_positions >= settings.max_open_positions:
        return False, "Position limit reached"
    if account.daily_pnl_pct <= -settings.max_daily_loss_pct:
        return False, f"Daily DD hit: {account.daily_pnl_pct:.2f}%"
    if safety.is_active():
        return False, f"Safety mode active until {safety.until}"
    return True, "OK"
