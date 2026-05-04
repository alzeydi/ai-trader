"""Veto layer tests: pre-flight hard rules + ClaudeClient retry/parse."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.llm.client import ClaudeClient
from src.llm.types import VetoResponse
from src.llm.veto import veto_candidate
from src.signal.types import CandidateSignal


def _make_signal(side: str = "long") -> CandidateSignal:
    return CandidateSignal(
        symbol="ETH/USDT:USDT",
        side=side,
        entry_type="A",
        signal_strength=0.72,
        entry_price_ref=2000.0,
        atr_14=45.0,
        swing_high_1h=2120.0,
        swing_low_1h=1960.0,
    )


@pytest.fixture
def fast_sleep(monkeypatch):
    monkeypatch.setattr("src.llm.client.time.sleep", lambda *_: None)


# ---------------------------------------------------------------------------
# Pre-flight hard rules: skip BEFORE the LLM is called.
# ---------------------------------------------------------------------------

def test_preflight_open_positions_skip_without_llm():
    client = MagicMock(spec=ClaudeClient)
    resp = veto_candidate(
        _make_signal(),
        context={"funding_rate_now": 0.01},
        account={"open_positions": 2, "daily_pnl_pct": 0.0, "losses_streak": 0},
        client=client,
    )
    assert resp.decision == "SKIP"
    assert "open_positions" in resp.reasoning
    client.veto.assert_not_called()


def test_preflight_daily_loss_skip_without_llm():
    client = MagicMock(spec=ClaudeClient)
    resp = veto_candidate(
        _make_signal(),
        context={"funding_rate_now": 0.0},
        account={"open_positions": 0, "daily_pnl_pct": -3.5, "losses_streak": 0},
        client=client,
    )
    assert resp.decision == "SKIP"
    assert "daily_pnl" in resp.reasoning
    client.veto.assert_not_called()


def test_preflight_losses_streak_skip_without_llm():
    client = MagicMock(spec=ClaudeClient)
    resp = veto_candidate(
        _make_signal(),
        context={"funding_rate_now": 0.0},
        account={"open_positions": 0, "daily_pnl_pct": 0.0, "losses_streak": 3},
        client=client,
    )
    assert resp.decision == "SKIP"
    assert "losses_streak" in resp.reasoning
    client.veto.assert_not_called()


def test_preflight_funding_overheated_long_skip_without_llm():
    client = MagicMock(spec=ClaudeClient)
    resp = veto_candidate(
        _make_signal("long"),
        context={"funding_rate_now": 0.06},  # > 0.05 %
        account={"open_positions": 0, "daily_pnl_pct": 0.0, "losses_streak": 0},
        client=client,
    )
    assert resp.decision == "SKIP"
    assert "funding_long_overheated" in resp.reasoning
    client.veto.assert_not_called()


def test_preflight_funding_oversold_short_skip_without_llm():
    client = MagicMock(spec=ClaudeClient)
    resp = veto_candidate(
        _make_signal("short"),
        context={"funding_rate_now": -0.07},  # < -0.05 %
        account={"open_positions": 0, "daily_pnl_pct": 0.0, "losses_streak": 0},
        client=client,
    )
    assert resp.decision == "SKIP"
    assert "funding_short_oversold" in resp.reasoning
    client.veto.assert_not_called()


def test_preflight_liquidation_squeeze_long_skip_without_llm():
    client = MagicMock(spec=ClaudeClient)
    resp = veto_candidate(
        _make_signal("long"),
        context={"funding_rate_now": 0.0, "liq_long": 60_000_000.0},
        account={"open_positions": 0, "daily_pnl_pct": 0.0, "losses_streak": 0},
        client=client,
    )
    assert resp.decision == "SKIP"
    assert "long_liquidation_squeeze" in resp.reasoning
    client.veto.assert_not_called()


def test_preflight_passes_to_llm_when_context_is_clean():
    client = MagicMock(spec=ClaudeClient)
    expected = VetoResponse(
        decision="TAKE",
        confidence=0.85,
        invalidation_signal="BTC breaks below 4h EMA50",
        reasoning="clean trend continuation, neutral funding",
    )
    client.veto.return_value = expected

    resp = veto_candidate(
        _make_signal("long"),
        context={"funding_rate_now": 0.01, "liq_long": 1_000_000.0},
        account={"open_positions": 0, "daily_pnl_pct": -1.0, "losses_streak": 1},
        client=client,
    )
    assert resp is expected
    client.veto.assert_called_once()


# ---------------------------------------------------------------------------
# ClaudeClient: retry on invalid JSON, parse valid TAKE.
# ---------------------------------------------------------------------------

class _ScriptedClient(ClaudeClient):
    """Replay a queue of `(text, input_tokens, output_tokens)` tuples."""

    def __init__(self, responses):
        super().__init__(log_to_db=False)
        self._responses = list(responses)
        self.calls = 0

    def _call(self, system, user):  # noqa: ARG002
        self.calls += 1
        return self._responses.pop(0)


def test_client_retries_on_invalid_json_then_succeeds(fast_sleep):
    valid = (
        '{"decision":"TAKE","confidence":0.85,'
        '"invalidation_signal":"close below 1960 swing low",'
        '"reasoning":"clean trend continuation"}'
    )
    client = _ScriptedClient([
        ("not json at all", 50, 0),
        # Parses but fails Pydantic validation (missing required fields).
        ("{}", 50, 5),
        (valid, 50, 30),
    ])
    resp = client.veto(
        _make_signal("long"),
        context={"funding_rate_now": 0.01},
        account={"open_positions": 0, "daily_pnl_pct": 0.0, "losses_streak": 0},
    )
    assert resp.decision == "TAKE"
    assert resp.confidence == 0.85
    assert client.calls == 3


def test_client_returns_valid_take_with_confidence(fast_sleep):
    valid = (
        '{"decision":"TAKE","confidence":0.85,'
        '"invalidation_signal":"BTC breaks below 4h EMA50",'
        '"reasoning":"clean continuation, neutral funding"}'
    )
    client = _ScriptedClient([(valid, 100, 50)])
    resp = client.veto(
        _make_signal("long"),
        context={"funding_rate_now": 0.01},
        account={"open_positions": 0, "daily_pnl_pct": 0.0, "losses_streak": 0},
    )
    assert resp.decision == "TAKE"
    assert resp.confidence == 0.85
    assert resp.invalidation_signal == "BTC breaks below 4h EMA50"
    assert client.calls == 1


def test_client_handles_fenced_json(fast_sleep):
    valid = (
        "```json\n"
        '{"decision":"SKIP","confidence":0.9,'
        '"invalidation_signal":"N/A",'
        '"reasoning":"funding rate against the trade"}\n'
        "```"
    )
    client = _ScriptedClient([(valid, 80, 20)])
    resp = client.veto(
        _make_signal("long"),
        context={"funding_rate_now": 0.01},
        account={"open_positions": 0, "daily_pnl_pct": 0.0, "losses_streak": 0},
    )
    assert resp.decision == "SKIP"
    assert resp.confidence == 0.9


def test_client_falls_back_to_skip_after_max_retries(fast_sleep):
    client = _ScriptedClient([
        ("not json", 10, 0),
        ("still not json", 10, 0),
        ("absolutely not json", 10, 0),
    ])
    resp = client.veto(
        _make_signal("long"),
        context={"funding_rate_now": 0.01},
        account={"open_positions": 0, "daily_pnl_pct": 0.0, "losses_streak": 0},
    )
    assert resp.decision == "SKIP"
    assert resp.invalidation_signal == "parse_error"
    assert client.calls == 3


def test_cost_estimate_scales_with_tokens():
    from src.llm.client import estimate_cost_usd

    cheap = estimate_cost_usd("claude-opus-4-7", 100, 50)
    pricey = estimate_cost_usd("claude-opus-4-7", 100_000, 50_000)
    assert cheap > 0
    assert pricey > cheap
