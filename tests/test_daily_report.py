"""Tests for the daily-report aggregator and Telegram formatter."""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from src.notify.daily_report import (
    DUBAI_TZ,
    build_daily_report,
    dubai_day_window,
    format_daily_report_telegram,
)
from src.persistence.db import (
    AIDecision,
    EquitySnapshot,
    Trade,
    make_session_factory,
)


def test_dubai_window_covers_full_local_day():
    # Pick a wall-clock moment in Dubai.
    now = datetime(2026, 5, 6, 18, 30, tzinfo=DUBAI_TZ)
    start, end, label = dubai_day_window(now)
    assert label.startswith("2026-05-06")
    # Start of Dubai day == 20:00 UTC of the previous calendar day.
    assert start == datetime(2026, 5, 5, 20, 0, tzinfo=ZoneInfo("UTC"))
    # End is "now" in UTC (== 14:30 UTC).
    assert end == datetime(2026, 5, 6, 14, 30, tzinfo=ZoneInfo("UTC"))


def _make_trade(
    sf,
    *,
    symbol: str,
    side: str,
    type_: str,
    pnl_usd: float,
    opened: datetime,
    closed: datetime | None,
    entry: float = 100.0,
    qty: float = 1.0,
):
    with sf() as s:
        row = Trade(
            symbol=symbol,
            side=side,
            type=type_,
            entry=entry,
            stop=entry * 0.99,
            take_profit=entry * 1.02,
            quantity=qty,
            leverage=3,
            risk_usd=10.0,
            pnl_usd=pnl_usd,
            pnl_pct=pnl_usd / (entry * qty) * 100.0,
            fees=0.5,
            opened_at=opened,
            closed_at=closed,
            close_price=entry * 1.01 if closed else None,
            close_reason="tp" if closed and pnl_usd > 0 else "sl" if closed else None,
            paper=True,
            llm_confidence=0.8,
        )
        s.add(row)
        s.commit()


def test_build_daily_report_aggregates_strategies_and_sides():
    sf = make_session_factory()

    # Window: today in Dubai (we pick a fixed Dubai 20:00 as "now").
    now_dubai = datetime.now(tz=DUBAI_TZ).replace(
        hour=20, minute=0, second=0, microsecond=0
    )
    start_utc, _end_utc, _ = dubai_day_window(now_dubai)
    inside = start_utc + timedelta(hours=2)  # safely within window
    outside = start_utc - timedelta(days=2)  # well before window

    # Type A long winner closed today.
    _make_trade(
        sf, symbol="BTC/USDT:USDT", side="long", type_="A",
        pnl_usd=15.0, opened=inside, closed=inside + timedelta(hours=1),
    )
    # Type A short loser closed today.
    _make_trade(
        sf, symbol="ETH/USDT:USDT", side="short", type_="A",
        pnl_usd=-8.0, opened=inside, closed=inside + timedelta(hours=2),
    )
    # Type B long winner closed today.
    _make_trade(
        sf, symbol="SOL/USDT:USDT", side="long", type_="B",
        pnl_usd=4.0, opened=inside, closed=inside + timedelta(hours=3),
    )
    # Open Type C (no closed_at) — counts as opened, doesn't count in PnL.
    _make_trade(
        sf, symbol="DOGE/USDT:USDT", side="short", type_="C",
        pnl_usd=0.0, opened=inside, closed=None,
    )
    # Trade from a previous day — must be excluded.
    _make_trade(
        sf, symbol="XRP/USDT:USDT", side="long", type_="A",
        pnl_usd=999.0, opened=outside, closed=outside + timedelta(hours=1),
    )

    # AI decisions today.
    with sf() as s:
        s.add(AIDecision(
            created_at=inside, symbol="BTC/USDT:USDT", side="long",
            entry_type="A", decision="TAKE", took=True, cost_usd=0.01,
        ))
        s.add(AIDecision(
            created_at=inside, symbol="ETH/USDT:USDT", side="short",
            entry_type="A", decision="SKIP", took=False, cost_usd=0.005,
        ))
        s.add(EquitySnapshot(
            snapshot_at=start_utc + timedelta(minutes=5),
            equity_usd=1000.0, balance_usd=1000.0,
        ))
        s.add(EquitySnapshot(
            snapshot_at=inside + timedelta(hours=4),
            equity_usd=1011.0, balance_usd=1011.0, unrealized_pnl=0.0,
        ))
        s.commit()

    report = build_daily_report(sf, now=now_dubai)

    assert report.n_opened == 4  # all of today's trades, open or closed
    assert report.n_closed == 3
    assert report.n_long_opened == 2
    assert report.n_short_opened == 2
    assert report.n_open_now == 1
    assert report.realized_pnl_usd == 11.0  # 15 + (-8) + 4
    assert report.wins == 2
    assert report.losses == 1
    assert report.best_symbol == "BTC/USDT:USDT"
    assert report.worst_symbol == "ETH/USDT:USDT"

    by_type = {s.type: s for s in report.by_strategy}
    assert "A" in by_type and "B" in by_type and "C" in by_type
    assert by_type["A"].n_closed == 2
    assert by_type["A"].pnl_usd == 7.0
    assert by_type["A"].wins == 1 and by_type["A"].losses == 1
    assert by_type["B"].n_closed == 1
    assert by_type["B"].pnl_usd == 4.0
    assert by_type["C"].n_closed == 0  # only the open one is in C

    by_side = {ss.side: ss for ss in report.by_side}
    assert by_side["long"].n_closed == 2
    assert by_side["short"].n_closed == 1

    assert report.ai_decisions_total == 2
    assert report.ai_decisions_accepted == 1
    assert abs(report.ai_cost_usd - 0.015) < 1e-9

    assert report.equity_start_usd == 1000.0
    assert report.equity_end_usd == 1011.0


def test_format_daily_report_telegram_renders_all_sections():
    sf = make_session_factory()
    now_dubai = datetime.now(tz=DUBAI_TZ)

    start_utc, _, _ = dubai_day_window(now_dubai)
    _make_trade(
        sf, symbol="BTC/USDT:USDT", side="long", type_="A",
        pnl_usd=12.5, opened=start_utc + timedelta(hours=1),
        closed=start_utc + timedelta(hours=2),
    )
    report = build_daily_report(sf, now=now_dubai)
    msg = format_daily_report_telegram(report)

    assert "Daily report" in msg
    assert "Activity" in msg
    assert "Realised P&L" in msg
    assert "By strategy" in msg
    assert "BTC/USDT:USDT" in msg
    assert "AI veto" in msg


def test_empty_day_reports_zeros_without_crashing():
    sf = make_session_factory()
    report = build_daily_report(sf)
    assert report.n_opened == 0
    assert report.n_closed == 0
    assert report.realized_pnl_usd == 0.0
    assert report.win_rate == 0.0
    # Telegram message must render even with no trades.
    msg = format_daily_report_telegram(report)
    assert "Daily report" in msg
