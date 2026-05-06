"""Daily trading report.

Aggregates trades + AI decisions over the Dubai trading day (00:00 → 21:00
local) and produces a structured payload consumed by both the Streamlit
dashboard and the Telegram nightly summary.

Single source of truth so the dashboard view and the Telegram message can
never drift apart.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, time
from zoneinfo import ZoneInfo

from sqlalchemy import select

from src.persistence.db import (
    AIDecision,
    EquitySnapshot,
    SessionFactory,
    Trade,
    make_session_factory,
)

log = logging.getLogger(__name__)

# Dubai is fixed UTC+4 year-round; ZoneInfo handles it correctly even though
# there is no DST. Using a named zone (rather than a hard-coded offset) keeps
# us future-proof if the policy ever changes.
DUBAI_TZ = ZoneInfo("Asia/Dubai")

# Order matters for display.
STRATEGY_TYPES = ("A", "B", "C", "D")


@dataclass
class StrategyStats:
    type: str
    n_closed: int = 0
    wins: int = 0
    losses: int = 0
    pnl_usd: float = 0.0
    avg_pnl_pct: float = 0.0
    n_long: int = 0
    n_short: int = 0

    @property
    def win_rate(self) -> float:
        decided = self.wins + self.losses
        return (self.wins / decided * 100.0) if decided else 0.0


@dataclass
class SideStats:
    side: str
    n_closed: int = 0
    wins: int = 0
    losses: int = 0
    pnl_usd: float = 0.0

    @property
    def win_rate(self) -> float:
        decided = self.wins + self.losses
        return (self.wins / decided * 100.0) if decided else 0.0


@dataclass
class DailyReport:
    """Snapshot of a single Dubai trading day."""

    day_start_utc: datetime
    day_end_utc: datetime
    day_label: str  # "2026-05-06 (Asia/Dubai)"

    # Activity counts.
    n_opened: int = 0
    n_closed: int = 0
    n_long_opened: int = 0
    n_short_opened: int = 0
    n_open_now: int = 0

    # PnL (realised, on trades closed today).
    realized_pnl_usd: float = 0.0
    fees_usd: float = 0.0
    best_trade_usd: float = 0.0
    worst_trade_usd: float = 0.0
    best_symbol: str = ""
    worst_symbol: str = ""

    # Win-rate (closed only).
    wins: int = 0
    losses: int = 0

    # Equity.
    equity_start_usd: float | None = None
    equity_end_usd: float | None = None
    unrealized_pnl_usd: float | None = None

    # Breakdowns.
    by_strategy: list[StrategyStats] = field(default_factory=list)
    by_side: list[SideStats] = field(default_factory=list)

    # AI / veto stats.
    ai_decisions_total: int = 0
    ai_decisions_accepted: int = 0
    ai_cost_usd: float = 0.0

    @property
    def win_rate(self) -> float:
        decided = self.wins + self.losses
        return (self.wins / decided * 100.0) if decided else 0.0

    @property
    def acceptance_rate(self) -> float:
        if self.ai_decisions_total == 0:
            return 0.0
        return self.ai_decisions_accepted / self.ai_decisions_total * 100.0

    @property
    def equity_change_usd(self) -> float | None:
        if self.equity_start_usd is None or self.equity_end_usd is None:
            return None
        return self.equity_end_usd - self.equity_start_usd

    @property
    def equity_change_pct(self) -> float | None:
        if (
            self.equity_start_usd is None
            or self.equity_end_usd is None
            or self.equity_start_usd <= 0
        ):
            return None
        return (self.equity_end_usd - self.equity_start_usd) / self.equity_start_usd * 100.0


# ---------------------------------------------------------------------------
# Window helpers
# ---------------------------------------------------------------------------

def dubai_day_window(now: datetime | None = None) -> tuple[datetime, datetime, str]:
    """Return (start_utc, end_utc, label) for the Dubai day enclosing `now`.

    The window is [00:00 Dubai, now in Dubai], so a 21:00 Dubai report covers
    the full local day up to send time.
    """
    if now is None:
        now = datetime.now(tz=DUBAI_TZ)
    elif now.tzinfo is None:
        # Treat naive as UTC then convert.
        now = now.replace(tzinfo=ZoneInfo("UTC")).astimezone(DUBAI_TZ)
    else:
        now = now.astimezone(DUBAI_TZ)

    start_local = datetime.combine(now.date(), time.min, tzinfo=DUBAI_TZ)
    end_local = now
    label = f"{now.date().isoformat()} (Asia/Dubai)"
    return (
        start_local.astimezone(ZoneInfo("UTC")),
        end_local.astimezone(ZoneInfo("UTC")),
        label,
    )


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_daily_report(
    session_factory: SessionFactory | None = None,
    *,
    now: datetime | None = None,
) -> DailyReport:
    """Aggregate trades + AI decisions for the current Dubai day."""
    sf = session_factory or make_session_factory()
    start_utc, end_utc, label = dubai_day_window(now)

    report = DailyReport(
        day_start_utc=start_utc, day_end_utc=end_utc, day_label=label
    )

    with sf() as s:
        opened_today: list[Trade] = list(
            s.execute(
                select(Trade)
                .where(Trade.opened_at >= start_utc)
                .where(Trade.opened_at <= end_utc)
            ).scalars().all()
        )
        closed_today: list[Trade] = list(
            s.execute(
                select(Trade)
                .where(Trade.closed_at.isnot(None))
                .where(Trade.closed_at >= start_utc)
                .where(Trade.closed_at <= end_utc)
            ).scalars().all()
        )
        open_now: int = len(
            s.execute(
                select(Trade).where(Trade.closed_at.is_(None))
            ).scalars().all()
        )
        decisions_today: list[AIDecision] = list(
            s.execute(
                select(AIDecision)
                .where(AIDecision.created_at >= start_utc)
                .where(AIDecision.created_at <= end_utc)
            ).scalars().all()
        )
        equity_first = s.execute(
            select(EquitySnapshot)
            .where(EquitySnapshot.snapshot_at >= start_utc)
            .where(EquitySnapshot.snapshot_at <= end_utc)
            .order_by(EquitySnapshot.snapshot_at.asc())
            .limit(1)
        ).scalars().first()
        equity_last = s.execute(
            select(EquitySnapshot)
            .where(EquitySnapshot.snapshot_at <= end_utc)
            .order_by(EquitySnapshot.snapshot_at.desc())
            .limit(1)
        ).scalars().first()

    # Activity counts.
    report.n_opened = len(opened_today)
    report.n_long_opened = sum(1 for t in opened_today if (t.side or "").lower() == "long")
    report.n_short_opened = sum(1 for t in opened_today if (t.side or "").lower() == "short")
    report.n_closed = len(closed_today)
    report.n_open_now = open_now

    # Realised PnL.
    report.realized_pnl_usd = float(sum((t.pnl_usd or 0.0) for t in closed_today))
    report.fees_usd = float(sum((t.fees or 0.0) for t in closed_today))
    report.wins = sum(1 for t in closed_today if (t.pnl_usd or 0.0) > 0)
    report.losses = sum(1 for t in closed_today if (t.pnl_usd or 0.0) < 0)

    if closed_today:
        best = max(closed_today, key=lambda t: (t.pnl_usd or 0.0))
        worst = min(closed_today, key=lambda t: (t.pnl_usd or 0.0))
        report.best_trade_usd = float(best.pnl_usd or 0.0)
        report.worst_trade_usd = float(worst.pnl_usd or 0.0)
        report.best_symbol = best.symbol or ""
        report.worst_symbol = worst.symbol or ""

    # Strategy breakdown — bucket by Trade.type, fall back to "?" for legacy
    # rows so adopted/orphan trades still show somewhere instead of being
    # silently dropped.
    seen_types = sorted({(t.type or "?") for t in closed_today + opened_today})
    bucket_order = list(STRATEGY_TYPES) + [t for t in seen_types if t not in STRATEGY_TYPES]
    for type_code in bucket_order:
        closed_bucket = [t for t in closed_today if (t.type or "?") == type_code]
        opened_bucket = [t for t in opened_today if (t.type or "?") == type_code]
        if not closed_bucket and not opened_bucket:
            continue
        s_stats = StrategyStats(type=type_code)
        s_stats.n_closed = len(closed_bucket)
        s_stats.wins = sum(1 for t in closed_bucket if (t.pnl_usd or 0.0) > 0)
        s_stats.losses = sum(1 for t in closed_bucket if (t.pnl_usd or 0.0) < 0)
        s_stats.pnl_usd = float(sum((t.pnl_usd or 0.0) for t in closed_bucket))
        if closed_bucket:
            s_stats.avg_pnl_pct = float(
                sum((t.pnl_pct or 0.0) for t in closed_bucket) / len(closed_bucket)
            )
        s_stats.n_long = sum(1 for t in opened_bucket if (t.side or "").lower() == "long")
        s_stats.n_short = sum(1 for t in opened_bucket if (t.side or "").lower() == "short")
        report.by_strategy.append(s_stats)

    # Side breakdown over closed trades.
    for side in ("long", "short"):
        bucket = [t for t in closed_today if (t.side or "").lower() == side]
        if not bucket:
            continue
        ss = SideStats(side=side)
        ss.n_closed = len(bucket)
        ss.wins = sum(1 for t in bucket if (t.pnl_usd or 0.0) > 0)
        ss.losses = sum(1 for t in bucket if (t.pnl_usd or 0.0) < 0)
        ss.pnl_usd = float(sum((t.pnl_usd or 0.0) for t in bucket))
        report.by_side.append(ss)

    # AI / veto stats. `took=True` means the candidate cleared the gate; for
    # rows missing that flag (older preflight skips) fall back to decision text.
    report.ai_decisions_total = len(decisions_today)
    report.ai_decisions_accepted = sum(
        1 for d in decisions_today
        if bool(d.took) or (d.decision or "").upper() == "TAKE"
    )
    report.ai_cost_usd = float(sum((d.cost_usd or 0.0) for d in decisions_today))

    # Equity.
    if equity_first is not None:
        report.equity_start_usd = float(equity_first.equity_usd)
    if equity_last is not None:
        report.equity_end_usd = float(equity_last.equity_usd)
        report.unrealized_pnl_usd = float(equity_last.unrealized_pnl or 0.0)

    return report


# ---------------------------------------------------------------------------
# Telegram formatter
# ---------------------------------------------------------------------------

def _fmt_usd(value: float | None) -> str:
    if value is None:
        return "—"
    sign = "+" if value > 0 else ("" if value == 0 else "")
    return f"{sign}{value:,.2f}"


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:+.2f}%"


def _fmt_winrate(wins: int, losses: int) -> str:
    decided = wins + losses
    if decided == 0:
        return "—"
    return f"{wins / decided * 100.0:.0f}% ({wins}W/{losses}L)"


def format_daily_report_telegram(report: DailyReport) -> str:
    """Markdown-formatted Telegram message — mirrors the dashboard view."""
    lines: list[str] = []
    lines.append(f"📅 *Daily report — {report.day_label}*")

    if report.equity_end_usd is not None:
        change = report.equity_change_usd
        change_pct = report.equity_change_pct
        lines.append(
            f"💰 Equity: `{report.equity_end_usd:,.2f}` USDT "
            f"({_fmt_usd(change)} / {_fmt_pct(change_pct)})"
        )
    if report.unrealized_pnl_usd is not None:
        lines.append(f"📈 Unrealized: `{_fmt_usd(report.unrealized_pnl_usd)}` USDT")

    lines.append("")
    lines.append("*Activity*")
    lines.append(
        f"  Opened: *{report.n_opened}* "
        f"(🟢 {report.n_long_opened} long / 🔴 {report.n_short_opened} short)"
    )
    lines.append(f"  Closed: *{report.n_closed}* | Open now: *{report.n_open_now}*")

    lines.append("")
    lines.append("*Realised P&L*")
    lines.append(f"  Net: `{_fmt_usd(report.realized_pnl_usd)}` USDT")
    lines.append(f"  Fees: `{report.fees_usd:,.2f}` USDT")
    lines.append(f"  Win rate: {_fmt_winrate(report.wins, report.losses)}")
    if report.n_closed:
        lines.append(
            f"  Best: `{report.best_symbol}` `{_fmt_usd(report.best_trade_usd)}` | "
            f"Worst: `{report.worst_symbol}` `{_fmt_usd(report.worst_trade_usd)}`"
        )

    if report.by_strategy:
        lines.append("")
        lines.append("*By strategy*")
        for s in report.by_strategy:
            lines.append(
                f"  `{s.type}`: closed *{s.n_closed}*, "
                f"PnL `{_fmt_usd(s.pnl_usd)}`, "
                f"WR {_fmt_winrate(s.wins, s.losses)}, "
                f"opened L/S {s.n_long}/{s.n_short}"
            )

    if report.by_side:
        lines.append("")
        lines.append("*By side (closed)*")
        for ss in report.by_side:
            emoji = "🟢" if ss.side == "long" else "🔴"
            lines.append(
                f"  {emoji} `{ss.side}`: *{ss.n_closed}*, "
                f"PnL `{_fmt_usd(ss.pnl_usd)}`, "
                f"WR {_fmt_winrate(ss.wins, ss.losses)}"
            )

    lines.append("")
    lines.append("*AI veto*")
    lines.append(
        f"  Decisions: *{report.ai_decisions_total}* "
        f"(accepted *{report.ai_decisions_accepted}*, "
        f"acc rate {report.acceptance_rate:.0f}%)"
    )
    lines.append(f"  Spend: `${report.ai_cost_usd:,.4f}`")

    return "\n".join(lines)


__all__ = [
    "DailyReport",
    "StrategyStats",
    "SideStats",
    "DUBAI_TZ",
    "build_daily_report",
    "dubai_day_window",
    "format_daily_report_telegram",
]
