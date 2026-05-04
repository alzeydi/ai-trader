"""Veto layer tests with a stubbed LLM client."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class _StubClient:
    payload: str

    def complete(self, system: str, user: str) -> str:  # noqa: ARG002
        return self.payload


def _make_signal():
    from src.signal.types import CandidateSignal
    return CandidateSignal(
        symbol="ETH/USDT:USDT",
        side="long",
        entry_type="A",
        signal_strength=0.72,
        entry_price_ref=2000.0,
        atr_14=45.0,
        swing_high_1h=2120.0,
        swing_low_1h=1960.0,
    )


def test_veto_parses_take():
    from src.llm.veto import VetoAgent, VetoResponse

    agent = VetoAgent(client=_StubClient(  # type: ignore[arg-type]
        payload=(
            '{"decision":"TAKE","confidence":0.85,'
            '"invalidation_signal":"Close below 1960 swing low",'
            '"reasoning":"clean trend continuation with volume"}'
        )
    ))
    resp = agent.vet(
        signal=_make_signal(),
        context={"last_price": 2000.0, "funding_rate": 0.0001,
                 "open_interest": None, "btc_dominance": None},
        equity={"balance_usdt": 1000.0, "open_positions": 0, "consecutive_losses": 0},
    )
    assert isinstance(resp, VetoResponse)
    assert resp.decision == "TAKE"
    assert resp.confidence == 0.85
    assert resp.invalidation_signal


def test_veto_parses_skip():
    from src.llm.veto import VetoAgent

    agent = VetoAgent(client=_StubClient(  # type: ignore[arg-type]
        payload=(
            '{"decision":"SKIP","confidence":0.9,'
            '"invalidation_signal":"N/A",'
            '"reasoning":"extreme funding rate against trade"}'
        )
    ))
    resp = agent.vet(
        signal=_make_signal(),
        context={"last_price": 2000.0, "funding_rate": 0.005,
                 "open_interest": None, "btc_dominance": None},
        equity={"balance_usdt": 1000.0, "open_positions": 2, "consecutive_losses": 1},
    )
    assert resp.decision == "SKIP"


def test_veto_falls_back_on_parse_error_in_dry_run():
    from src.llm.veto import VetoAgent

    agent = VetoAgent(client=_StubClient(  # type: ignore[arg-type]
        payload="not valid json at all"
    ))
    resp = agent.vet(
        signal=_make_signal(),
        context={}, equity={},
    )
    # In dry_run mode (default True) a parse error yields a safe SKIP.
    assert resp.decision == "SKIP"
    assert resp.invalidation_signal == "parse_error"
