"""Veto layer with a stubbed LLM response."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class _StubClient:
    payload: str

    def complete(self, system: str, user: str) -> str:  # noqa: ARG002
        return self.payload


def test_veto_parses_allow() -> None:
    from src.llm.veto import VetoAgent
    from src.signal.types import CandidateSignal, Side, SignalType

    agent = VetoAgent(client=_StubClient(  # type: ignore[arg-type]
        payload='{"decision":"ALLOW","confidence":0.8,"reason":"clean trend"}'
    ))
    sig = CandidateSignal(
        symbol="ETH/USDT:USDT",
        side=Side.LONG,
        type=SignalType.A,
        entry=2000.0,
        stop=1960.0,
        take_profit=2120.0,
        confidence=0.7,
        rationale="break-and-retest",
    )
    decision = agent.vet(
        signal=sig,
        context={"last_price": 2000.0, "funding_rate": 0.0001,
                 "open_interest": None, "btc_dominance": None},
        equity={"balance_usdt": 1000.0, "open_positions": 0, "consecutive_losses": 0},
    )
    assert decision.allow is True
    assert decision.confidence == 0.8
