"""Shared types for the execution layer (orders, results)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class TradeOrder(BaseModel):
    """Fully-specified order produced by the risk layer.

    Fields are immutable once constructed; the executor reads them verbatim.
    """

    symbol: str
    side: Literal["long", "short"]
    entry_type: Literal["A", "B", "C", "D", "E"]
    quantity: float = Field(gt=0)
    leverage: int = Field(gt=0)
    entry_price: float = Field(gt=0)
    stop_loss: float = Field(gt=0)
    take_profit: float = Field(gt=0)
    risk_usd: float = Field(ge=0)
    invalidation_signal: str
    llm_confidence: float = Field(ge=0.0, le=1.0)


class ExecutionResult(BaseModel):
    """Outcome of an executor call (open / close)."""

    success: bool
    paper: bool
    trade_id: int | None = None
    entry_order_id: str | None = None
    stop_order_id: str | None = None
    take_order_id: str | None = None
    fill_price: float | None = None
    pnl_usd: float | None = None
    reason: str = ""
