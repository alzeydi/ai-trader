"""Risk layer unit tests."""

from __future__ import annotations


def test_sizing_zero_when_flat_stop() -> None:
    from src.risk.sizing import size_position
    from src.signal.types import CandidateSignal, Side, SignalType

    sig = CandidateSignal(
        symbol="BTC/USDT:USDT",
        side=Side.LONG,
        type=SignalType.A,
        entry=100.0,
        stop=100.0,
        take_profit=110.0,
        confidence=0.5,
        rationale="",
    )
    assert size_position(sig, equity_usd=1000.0).quantity == 0.0


def test_limits_block_on_open_positions() -> None:
    from src.config import settings
    from src.risk.limits import RiskState, check

    state = RiskState(
        equity_usd=1000.0,
        open_positions=settings.max_open_positions,
        daily_pnl_pct=0.0,
        drawdown_pct=0.0,
    )
    ok, _ = check(state)
    assert ok is False
