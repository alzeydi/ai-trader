"""Execution layer tests: OrderExecutor (paper) + PositionMonitor + Trader."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from src.execution.monitor import PositionMonitor, _parse_yes_no_unclear
from src.execution.orders import OrderExecutor
from src.execution.trader import Trader
from src.execution.types import TradeOrder
from src.llm.types import VetoResponse
from src.persistence.db import (
    Trade,
    count_open_trades,
    get_last_n_closed_trades,
    list_open_trades,
    make_session_factory,
)
from src.risk.safety_mode import SafetyMode
from src.signal.types import CandidateSignal


# ---------------------------------------------------------------------------
# Fixtures / builders
# ---------------------------------------------------------------------------

def _candidate(side: str = "long") -> CandidateSignal:
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


def _order(side: str = "long") -> TradeOrder:
    return TradeOrder(
        symbol="ETH/USDT:USDT",
        side=side,
        entry_type="A",
        quantity=0.1,
        leverage=3,
        entry_price=2000.0,
        stop_loss=1955.0 if side == "long" else 2045.0,
        take_profit=2090.0 if side == "long" else 1910.0,
        risk_usd=20.0,
        invalidation_signal="BTC breaks below 4h EMA50",
        llm_confidence=0.85,
    )


# ---------------------------------------------------------------------------
# OrderExecutor — paper mode
# ---------------------------------------------------------------------------

def test_paper_open_records_trade_with_full_metadata():
    sf = make_session_factory()
    executor = OrderExecutor(client=None, paper=True, session_factory=sf)
    result = executor.open_position(_order())
    assert result.success
    assert result.paper is True
    assert result.trade_id is not None

    open_trades = list_open_trades(sf)
    assert len(open_trades) == 1
    t = open_trades[0]
    assert t.symbol == "ETH/USDT:USDT"
    assert t.side == "long"
    assert t.type == "A"
    assert t.entry == 2000.0
    assert t.stop == 1955.0
    assert t.take_profit == 2090.0
    assert t.invalidation_signal == "BTC breaks below 4h EMA50"
    assert t.llm_confidence == 0.85
    assert t.paper is True


def test_paper_close_long_computes_positive_pnl():
    sf = make_session_factory()
    fake_client = MagicMock()
    fake_client.fetch_ticker.return_value = {"last": 2050.0}  # +50 USDT × 0.1 = +5 USDT gross
    executor = OrderExecutor(client=fake_client, paper=True, session_factory=sf)
    executor.open_position(_order(side="long"))

    result = executor.close_position("ETH/USDT:USDT", reason="manual")
    assert result.success
    # Net PnL = gross - round-trip fees (entry + exit legs at taker rate).
    from src.config import settings
    expected_fees = 0.1 * (2000.0 + 2050.0) * float(settings.taker_fee_pct)
    assert result.pnl_usd == pytest.approx(5.0 - expected_fees)

    closed = get_last_n_closed_trades(sf, 1)
    assert len(closed) == 1
    assert closed[0].close_reason == "manual"
    assert closed[0].close_price == pytest.approx(2050.0)


def test_paper_close_populates_pnl_pct_and_fees():
    """Close must persist pnl_pct (% price move) and round-trip fees.

    Regression: prior to 2026-05-07 the close path only wrote `pnl_usd`,
    leaving `pnl_pct` and `fees` at the column defaults (0). This broke
    the dashboard's Sharpe (which reads `pnl_pct`) and the by-strategy
    breakdown's avg-pnl-pct.
    """
    sf = make_session_factory()
    fake_client = MagicMock()
    fake_client.fetch_ticker.return_value = {"last": 2050.0}  # +2.5 % long move
    executor = OrderExecutor(client=fake_client, paper=True, session_factory=sf)
    executor.open_position(_order(side="long"))
    result = executor.close_position("ETH/USDT:USDT", reason="manual")
    assert result.success

    closed = get_last_n_closed_trades(sf, 1)[0]
    # Long 2000 → 2050 = +2.5 % gross (matches backtest's `gross/notional × 100`).
    assert closed.pnl_pct == pytest.approx(2.5, rel=1e-6)
    # Fees = qty × (entry + exit) × taker_pct, both legs.
    expected_fees = 0.1 * (2000.0 + 2050.0) * 0.0004
    assert closed.fees == pytest.approx(expected_fees, rel=1e-6)


def test_paper_close_short_inverts_pnl_sign():
    sf = make_session_factory()
    fake_client = MagicMock()
    fake_client.fetch_ticker.return_value = {"last": 1950.0}  # short profits as price falls
    executor = OrderExecutor(client=fake_client, paper=True, session_factory=sf)
    executor.open_position(_order(side="short"))
    result = executor.close_position("ETH/USDT:USDT", reason="manual")
    from src.config import settings
    gross = 0.1 * (2000.0 - 1950.0)
    expected_fees = 0.1 * (2000.0 + 1950.0) * float(settings.taker_fee_pct)
    assert result.pnl_usd == pytest.approx(gross - expected_fees)


def test_paper_close_no_open_trade_returns_failure():
    sf = make_session_factory()
    executor = OrderExecutor(client=None, paper=True, session_factory=sf)
    result = executor.close_position("BTC/USDT:USDT", reason="manual")
    assert result.success is False
    assert "no open" in result.reason


# ---------------------------------------------------------------------------
# PositionMonitor — max-hold + invalidation + safety hand-off
# ---------------------------------------------------------------------------

def test_monitor_closes_on_max_hold_timeout():
    sf = make_session_factory()
    fake_client = MagicMock()
    fake_client.fetch_ticker.return_value = {"last": 2000.0}
    executor = OrderExecutor(client=fake_client, paper=True, session_factory=sf)
    executor.open_position(_order())

    # Backdate the trade so it's older than max_hold_hours.
    with sf() as s:
        row = s.execute(__import__("sqlalchemy").select(Trade)).scalar_one()
        row.opened_at = datetime.now(tz=timezone.utc) - timedelta(hours=24)
        s.commit()

    safety = SafetyMode(session_factory=sf)
    monitor = PositionMonitor(
        client=fake_client,
        executor=executor,
        safety=safety,
        llm_client=None,
        session_factory=sf,
    )
    closed = monitor.tick()
    assert closed == ["ETH/USDT:USDT"]
    assert count_open_trades(sf) == 0
    assert get_last_n_closed_trades(sf, 1)[0].close_reason == "max_hold_timeout"


def test_monitor_closes_on_llm_invalidation_yes():
    sf = make_session_factory()
    fake_client = MagicMock()
    fake_client.fetch_ticker.return_value = {"last": 2000.0}
    executor = OrderExecutor(client=fake_client, paper=True, session_factory=sf)
    executor.open_position(_order())

    llm = MagicMock()
    llm._call.return_value = ("yes", 30, 1)
    safety = SafetyMode(session_factory=sf)
    monitor = PositionMonitor(
        client=fake_client,
        executor=executor,
        safety=safety,
        llm_client=llm,
        session_factory=sf,
    )
    closed = monitor.tick()
    assert closed == ["ETH/USDT:USDT"]
    llm._call.assert_called_once()
    assert get_last_n_closed_trades(sf, 1)[0].close_reason == "invalidation_signal"


def test_monitor_keeps_trade_when_invalidation_no():
    sf = make_session_factory()
    fake_client = MagicMock()
    fake_client.fetch_ticker.return_value = {"last": 2000.0}
    executor = OrderExecutor(client=fake_client, paper=True, session_factory=sf)
    executor.open_position(_order())

    llm = MagicMock()
    llm._call.return_value = ("no", 30, 1)
    monitor = PositionMonitor(
        client=fake_client,
        executor=executor,
        safety=SafetyMode(session_factory=sf),
        llm_client=llm,
        session_factory=sf,
    )
    assert monitor.tick() == []
    assert count_open_trades(sf) == 1
    # Last invalidation check timestamp was stamped.
    with sf() as s:
        row = s.execute(__import__("sqlalchemy").select(Trade)).scalar_one()
        assert row.last_invalidation_check_at is not None


def test_parse_yes_no_unclear_handles_noise():
    assert _parse_yes_no_unclear("yes.") == "yes"
    assert _parse_yes_no_unclear("  No\n") == "no"
    assert _parse_yes_no_unclear("maybe") == "unclear"
    assert _parse_yes_no_unclear("") == "unclear"


# ---------------------------------------------------------------------------
# Trader.run_cycle — full paper-mode flow
# ---------------------------------------------------------------------------

class _FakeEngine:
    def __init__(self, signal):
        self._signal = signal

    def generate_signals(self, symbol):
        return self._signal


def test_trader_run_cycle_paper_mode_records_trade(monkeypatch):
    sf = make_session_factory()
    cand = _candidate()
    engine = _FakeEngine(cand)

    llm_client = MagicMock()
    llm_client.veto.return_value = VetoResponse(
        decision="TAKE",
        confidence=0.85,
        invalidation_signal="BTC breaks below 4h EMA50",
        reasoning="clean trend",
    )

    # Skip the live market_context fetch; the real one needs a Binance client.
    monkeypatch.setattr(
        "src.execution.trader.get_market_context",
        lambda symbol, client: type("MC", (), {
            "funding_rate": 0.0001,
            "open_interest_delta_1h": None,
            "liquidations_4h_usd": 0.0,
            "btc_dominance": None,
            "btc_1h_direction": None,
        })(),
    )
    # csv_writer.append_decision touches the project filesystem; stub it out.
    monkeypatch.setattr("src.execution.trader.append_decision", lambda *a, **k: None)

    executor = OrderExecutor(client=None, paper=True, session_factory=sf)
    safety = SafetyMode(session_factory=sf)
    notifier = MagicMock()

    trader = Trader(
        binance=MagicMock(),
        engine=engine,
        llm_client=llm_client,
        executor=executor,
        safety=safety,
        notifier=notifier,
        session_factory=sf,
        equity_usd=1000.0,
    )

    results = trader.run_cycle([cand.symbol])
    assert len(results) == 1
    res = results[0]
    assert res.accepted is True
    assert res.order is not None
    assert res.execution is not None and res.execution.paper is True

    # The trade is now persisted as open.
    assert count_open_trades(sf) == 1
    notifier.send_entry_signal.assert_called_once()
    _, kwargs = notifier.send_entry_signal.call_args
    assert "margin_usd" in kwargs and kwargs["margin_usd"] > 0


def _stub_market_context(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.execution.trader.get_market_context",
        lambda symbol, client: type("MC", (), {
            "funding_rate": 0.0001,
            "open_interest_delta_1h": None,
            "liquidations_4h_usd": 0.0,
            "btc_dominance": None,
            "btc_1h_direction": None,
        })(),
    )
    monkeypatch.setattr("src.execution.trader.append_decision", lambda *a, **k: None)


def _make_trader(sf, engine, llm_client) -> Trader:
    return Trader(
        binance=MagicMock(),
        engine=engine,
        llm_client=llm_client,
        executor=OrderExecutor(client=None, paper=True, session_factory=sf),
        safety=SafetyMode(session_factory=sf),
        notifier=None,
        session_factory=sf,
        equity_usd=1000.0,
    )


def test_btc_regime_blocks_a_long_without_self_trend_override(monkeypatch):
    """A LONG under BTC=down with self_trend_override=False must be blocked."""
    sf = make_session_factory()
    cand = _candidate(side="long")
    cand = cand.model_copy(update={"self_trend_override": False})
    engine = _FakeEngine(cand)

    llm_client = MagicMock()
    _stub_market_context(monkeypatch)
    monkeypatch.setattr(
        "src.execution.trader.get_btc_regime", lambda client: "down",
    )

    trader = _make_trader(sf, engine, llm_client)
    results = trader.run_cycle([cand.symbol])
    assert results[0].accepted is False
    assert results[0].reason == "btc_regime_down"
    # The LLM must not have been consulted — gate is pre-veto.
    llm_client.veto.assert_not_called()


def test_btc_regime_lets_a_long_through_when_self_trend_override(monkeypatch):
    """A LONG under BTC=down with self_trend_override=True must reach the LLM."""
    sf = make_session_factory()
    cand = _candidate(side="long").model_copy(update={"self_trend_override": True})
    engine = _FakeEngine(cand)

    llm_client = MagicMock()
    llm_client.veto.return_value = VetoResponse(
        decision="TAKE", confidence=0.85,
        invalidation_signal="BTC reverses", reasoning="own trend",
    )
    _stub_market_context(monkeypatch)
    monkeypatch.setattr(
        "src.execution.trader.get_btc_regime", lambda client: "down",
    )

    trader = _make_trader(sf, engine, llm_client)
    results = trader.run_cycle([cand.symbol])
    assert results[0].accepted is True
    llm_client.veto.assert_called_once()


def test_btc_regime_lets_a_short_through_when_self_trend_override(monkeypatch):
    """Mirror: A SHORT under BTC=up with self_trend_override=True is allowed."""
    sf = make_session_factory()
    cand = _candidate(side="short").model_copy(update={"self_trend_override": True})
    engine = _FakeEngine(cand)

    llm_client = MagicMock()
    llm_client.veto.return_value = VetoResponse(
        decision="TAKE", confidence=0.85,
        invalidation_signal="BTC reverses", reasoning="own trend",
    )
    _stub_market_context(monkeypatch)
    monkeypatch.setattr(
        "src.execution.trader.get_btc_regime", lambda client: "up",
    )

    trader = _make_trader(sf, engine, llm_client)
    results = trader.run_cycle([cand.symbol])
    assert results[0].accepted is True
    llm_client.veto.assert_called_once()


def test_btc_regime_self_trend_override_only_for_type_a(monkeypatch):
    """Override is Type A only; same flag on Type B must NOT bypass the gate."""
    sf = make_session_factory()
    base = _candidate(side="long")
    cand = base.model_copy(update={
        "entry_type": "B",
        "self_trend_override": True,
    })
    engine = _FakeEngine(cand)

    llm_client = MagicMock()
    _stub_market_context(monkeypatch)
    monkeypatch.setattr(
        "src.execution.trader.get_btc_regime", lambda client: "down",
    )

    trader = _make_trader(sf, engine, llm_client)
    results = trader.run_cycle([cand.symbol])
    assert results[0].accepted is False
    assert results[0].reason == "btc_regime_down"
    llm_client.veto.assert_not_called()


def test_trader_run_cycle_rejects_on_low_confidence(monkeypatch):
    sf = make_session_factory()
    cand = _candidate()
    engine = _FakeEngine(cand)

    llm_client = MagicMock()
    llm_client.veto.return_value = VetoResponse(
        decision="TAKE",
        confidence=0.5,  # below settings.min_confidence
        invalidation_signal="N/A",
        reasoning="hedge",
    )
    monkeypatch.setattr(
        "src.execution.trader.get_market_context",
        lambda symbol, client: type("MC", (), {
            "funding_rate": 0.0,
            "open_interest_delta_1h": None,
            "liquidations_4h_usd": 0.0,
            "btc_dominance": None,
            "btc_1h_direction": None,
        })(),
    )
    monkeypatch.setattr("src.execution.trader.append_decision", lambda *a, **k: None)

    trader = Trader(
        binance=MagicMock(),
        engine=engine,
        llm_client=llm_client,
        executor=OrderExecutor(client=None, paper=True, session_factory=sf),
        safety=SafetyMode(session_factory=sf),
        notifier=None,
        session_factory=sf,
        equity_usd=1000.0,
    )
    results = trader.run_cycle([cand.symbol])
    assert results[0].accepted is False
    assert "Low confidence" in results[0].reason
    assert count_open_trades(sf) == 0
