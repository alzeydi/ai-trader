"""Risk layer unit tests."""

from __future__ import annotations


def test_sizing_zero_when_flat_stop() -> None:
    from src.risk.sizing import size_from_risk

    result = size_from_risk(equity_usd=1000.0, entry_price=100.0, stop_price=100.0)
    assert result.quantity == 0.0


def test_sizing_nonzero_with_valid_stop() -> None:
    from src.risk.sizing import size_from_risk

    result = size_from_risk(equity_usd=1000.0, entry_price=100.0, stop_price=95.0)
    assert result.quantity > 0
    assert result.risk_usd > 0
    assert result.notional == result.quantity * 100.0


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
