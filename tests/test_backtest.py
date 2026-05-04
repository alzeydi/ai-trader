"""Backtest harness sanity checks (no exchange calls)."""

from __future__ import annotations


def test_stats_winrate_handles_zero() -> None:
    from src.backtest import BacktestStats

    s = BacktestStats(trades=0, wins=0, losses=0, pnl_usd=0.0)
    assert s.winrate() == 0.0


def test_stats_winrate_basic() -> None:
    from src.backtest import BacktestStats

    s = BacktestStats(trades=4, wins=3, losses=1, pnl_usd=42.0)
    assert s.winrate() == 0.75
