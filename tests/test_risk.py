"""Risk layer unit tests: limits, sizing, safety mode."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.config import settings
from src.execution.types import TradeOrder
from src.llm.types import VetoResponse
from src.persistence.db import Trade, make_session_factory
from src.risk.limits import RiskState, can_take_signal, check
from src.risk.safety_mode import SafetyMode
from src.risk.sizing import size_from_risk, size_position
from src.risk.types import AccountState
from src.signal.types import CandidateSignal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _candidate(side: str = "long", entry_type: str = "A", price: float = 2000.0):
    return CandidateSignal(
        symbol="ETH/USDT:USDT",
        side=side,
        entry_type=entry_type,
        signal_strength=0.72,
        entry_price_ref=price,
        atr_14=45.0,
        swing_high_1h=price * 1.06,
        swing_low_1h=price * 0.97,
    )


def _veto(decision: str = "TAKE", confidence: float = 0.85) -> VetoResponse:
    return VetoResponse(
        decision=decision,
        confidence=confidence,
        invalidation_signal="BTC breaks below 4h EMA50",
        reasoning="ok",
    )


def _account(**overrides) -> AccountState:
    base = {"equity": 1000.0, "open_positions": 0, "daily_pnl_pct": 0.0,
            "consecutive_losses": 0}
    base.update(overrides)
    return AccountState(**base)


def _record_closed_trade(session_factory, *, pnl: float) -> None:
    """Insert a trade with a `closed_at` timestamp slightly in the past."""
    with session_factory() as s:
        # Stagger closed_at by 1 ms per insert so ORDER BY is deterministic.
        offset_ms = len(_record_closed_trade.calls)
        _record_closed_trade.calls.append(offset_ms)
        s.add(
            Trade(
                symbol="ETH/USDT:USDT",
                side="long",
                type="A",
                entry=2000.0,
                stop=1955.0,
                take_profit=2090.0,
                quantity=0.1,
                pnl_usd=pnl,
                closed_at=datetime.now(tz=timezone.utc) + timedelta(milliseconds=offset_ms),
                paper=True,
            )
        )
        s.commit()


_record_closed_trade.calls = []  # type: ignore[attr-defined]


@pytest.fixture(autouse=True)
def _reset_close_counter():
    _record_closed_trade.calls = []  # type: ignore[attr-defined]
    yield


# ---------------------------------------------------------------------------
# Legacy sizing (kept for backtest)
# ---------------------------------------------------------------------------

def test_sizing_zero_when_flat_stop():
    assert size_from_risk(equity_usd=1000.0, entry_price=100.0, stop_price=100.0).quantity == 0.0


def test_sizing_nonzero_with_valid_stop():
    r = size_from_risk(equity_usd=1000.0, entry_price=100.0, stop_price=95.0)
    assert r.quantity > 0
    assert r.risk_usd > 0
    assert r.notional == r.quantity * 100.0


def test_legacy_check_blocks_on_open_positions():
    state = RiskState(
        equity_usd=1000.0,
        open_positions=settings.max_open_positions,
        daily_pnl_pct=0.0,
        drawdown_pct=0.0,
    )
    assert check(state)[0] is False


# ---------------------------------------------------------------------------
# can_take_signal — every hard rule path
# ---------------------------------------------------------------------------

class _StubSafety:
    def __init__(self, active: bool = False, until=None):
        self._active = active
        self.until = until

    def is_active(self) -> bool:
        return self._active


def test_can_take_blocks_on_skip_decision():
    ok, reason = can_take_signal(_candidate(), _veto("SKIP"), _account(), _StubSafety())
    assert ok is False
    assert reason.startswith("LLM veto")


def test_can_take_blocks_on_low_confidence():
    ok, reason = can_take_signal(
        _candidate(), _veto(confidence=0.5), _account(), _StubSafety()
    )
    assert ok is False
    assert "Low confidence" in reason


def test_can_take_blocks_on_open_position_limit():
    ok, reason = can_take_signal(
        _candidate(),
        _veto(),
        _account(open_positions=settings.max_open_positions),
        _StubSafety(),
    )
    assert ok is False
    assert "Position limit" in reason


def test_can_take_blocks_on_daily_dd():
    ok, reason = can_take_signal(
        _candidate(),
        _veto(),
        _account(daily_pnl_pct=-(settings.max_daily_loss_pct + 0.01)),
        _StubSafety(),
    )
    assert ok is False
    assert "Daily DD" in reason


def test_can_take_blocks_on_safety_mode_active():
    ok, reason = can_take_signal(_candidate(), _veto(), _account(), _StubSafety(active=True))
    assert ok is False
    assert "Safety mode" in reason


def test_can_take_passes_when_all_clean():
    ok, reason = can_take_signal(_candidate(), _veto(), _account(), _StubSafety())
    assert ok is True
    assert reason == "OK"


# ---------------------------------------------------------------------------
# size_position
# ---------------------------------------------------------------------------

def test_size_position_long_uses_type_a_risk():
    cand = _candidate(side="long", entry_type="A")
    order = size_position(cand, _account(equity=1000.0), _veto())
    assert isinstance(order, TradeOrder)
    assert order.side == "long"
    assert order.entry_type == "A"
    assert order.risk_usd == pytest.approx(1000.0 * settings.risk_per_trade_type_a, rel=1e-3)
    sl_dist = cand.atr_14 * settings.atr_stop_multiplier
    assert order.stop_loss == pytest.approx(cand.entry_price_ref - sl_dist)
    assert order.take_profit > cand.entry_price_ref
    assert order.invalidation_signal == "BTC breaks below 4h EMA50"
    assert order.llm_confidence == 0.85


def test_size_position_short_flips_sl_and_tp():
    cand = _candidate(side="short", entry_type="B")
    order = size_position(cand, _account(equity=1000.0), _veto())
    assert order is not None
    sl_dist = cand.atr_14 * settings.atr_stop_multiplier
    assert order.stop_loss == pytest.approx(cand.entry_price_ref + sl_dist)
    assert order.take_profit < cand.entry_price_ref
    assert order.risk_usd == pytest.approx(1000.0 * settings.risk_per_trade_type_b, rel=1e-3)


def test_size_position_floors_stop_at_min_stop_pct_on_low_vol():
    """ATR tiny relative to price → SL distance is floored at min_stop_pct of entry."""
    cand = CandidateSignal(
        symbol="BTC/USDT:USDT",
        side="long",
        entry_type="A",
        signal_strength=0.5,
        entry_price_ref=100_000.0,
        atr_14=1.0,  # raw sl_distance would be 1.5 → 0.0015 % of price
        swing_high_1h=101_000.0,
        swing_low_1h=99_000.0,
    )
    order = size_position(cand, _account(), _veto())
    assert order is not None
    expected_sl_dist = 100_000.0 * settings.min_stop_pct
    assert (cand.entry_price_ref - order.stop_loss) == pytest.approx(expected_sl_dist, rel=1e-6)


# ---------------------------------------------------------------------------
# SafetyMode (DB-backed, lazy-loaded)
# ---------------------------------------------------------------------------

def test_safety_mode_does_nothing_with_too_few_trades():
    sf = make_session_factory()
    sm = SafetyMode(session_factory=sf)
    _record_closed_trade(sf, pnl=-10.0)
    sm.check_after_close(-10.0)
    assert sm.is_active() is False


def test_safety_mode_activates_on_n_consecutive_losses():
    sf = make_session_factory()
    sm = SafetyMode(session_factory=sf)
    threshold = settings.safety_max_consecutive_losses
    for _ in range(threshold):
        _record_closed_trade(sf, pnl=-10.0)
    sm.check_after_close(-10.0)
    assert sm.is_active() is True
    assert sm.until is not None
    assert sm.until > datetime.now(tz=timezone.utc)


def test_safety_mode_resets_streak_on_winner():
    sf = make_session_factory()
    sm = SafetyMode(session_factory=sf)
    _record_closed_trade(sf, pnl=-10.0)
    sm.check_after_close(-10.0)
    _record_closed_trade(sf, pnl=20.0)
    sm.check_after_close(20.0)
    assert sm.consecutive_losses == 0
    assert sm.is_active() is False


def test_safety_mode_persists_across_instances():
    """State written by one SafetyMode is visible to a fresh instance."""
    sf = make_session_factory()
    sm1 = SafetyMode(session_factory=sf)
    threshold = settings.safety_max_consecutive_losses
    for _ in range(threshold):
        _record_closed_trade(sf, pnl=-10.0)
    sm1.check_after_close(-10.0)
    assert sm1.is_active() is True

    sm2 = SafetyMode(session_factory=sf)
    assert sm2.is_active() is True
    assert sm2.until is not None


def test_safety_mode_is_inactive_after_pause_expires():
    sf = make_session_factory()
    sm = SafetyMode(session_factory=sf)
    sm.activate(hours=1)
    assert sm.is_active() is True
    sm._paused_until = datetime.now(tz=timezone.utc) - timedelta(seconds=1)
    assert sm.is_active() is False
