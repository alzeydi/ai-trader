"""Smoke checks for the signal layer (no network, no LLM)."""

from __future__ import annotations


def test_signal_types_importable() -> None:
    from src.signal.types import CandidateSignal, Side, SignalType

    s = CandidateSignal(
        symbol="BTC/USDT:USDT",
        side=Side.LONG,
        type=SignalType.A,
        entry=100.0,
        stop=95.0,
        take_profit=115.0,
        confidence=0.7,
        rationale="test",
    )
    assert s.risk_reward() == 3.0
