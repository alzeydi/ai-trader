"""Shared Pydantic types for the LLM veto layer.

Lives in its own module so `client.py` and `veto.py` can both import without
creating a circular dependency.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class VetoContext(BaseModel):
    """Market context passed to the veto LLM.

    Numbers are expressed in their natural units as referenced by the system
    prompt: `funding_rate_*` in percent (0.05 == 0.05 %), liquidations in raw
    USD, OI delta in percent.
    """

    model_config = ConfigDict(extra="allow")

    funding_rate_now: float | None = None
    funding_rate_8h_avg: float | None = None
    oi_delta_1h_pct: float | None = None
    liq_long: float | None = None
    liq_short: float | None = None
    btc_dominance: float | None = None
    btc_direction: Literal["up", "down", "flat"] | None = None


class VetoAccount(BaseModel):
    """Account-level state used by the hard-rule pre-flight."""

    model_config = ConfigDict(extra="allow")

    open_positions: int = 0
    daily_pnl_pct: float = 0.0
    losses_streak: int = 0


class VetoResponse(BaseModel):
    decision: Literal["TAKE", "SKIP"]
    confidence: float = Field(ge=0.0, le=1.0)
    invalidation_signal: str
    reasoning: str
