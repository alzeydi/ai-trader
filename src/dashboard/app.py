"""Streamlit dashboard for the ai-trader.

Pages
-----
1. Overview     — equity curve vs BTC buy&hold, Sharpe, Sortino, Max DD
2. Daily Report — today's trades, by strategy / side, AI veto stats
3. Trades       — table of all trades with filters
4. AI Decisions — LLM decision log (with raw JSON) + cost-by-day chart
5. Positions    — currently open positions (live)
6. Data Export  — download tables as CSV or a single zip bundle
7. Settings     — read-only view of `.env` and PAPER_TRADING toggle

Run with `streamlit run src/dashboard/app.py`.
"""

from __future__ import annotations

import io
import json
import os
import zipfile
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from sqlalchemy import select

from src.config import settings
from src.notify.daily_report import (
    build_daily_report,
    format_daily_report_telegram,
)
from src.persistence.db import (
    AIDecision,
    EquitySnapshot,
    Position,
    SafetyState,
    Trade,
    make_session_factory,
)

# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _session_factory():
    return make_session_factory()


def _row_to_dict(row: Any) -> dict:
    return {c.name: getattr(row, c.name) for c in row.__table__.columns}


@st.cache_data(ttl=30, show_spinner=False)
def load_trades() -> pd.DataFrame:
    sf = _session_factory()
    with sf() as s:
        rows = s.execute(select(Trade).order_by(Trade.opened_at.asc())).scalars().all()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([_row_to_dict(r) for r in rows])
    for col in ("opened_at", "closed_at"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    return df


@st.cache_data(ttl=30, show_spinner=False)
def load_equity() -> pd.DataFrame:
    sf = _session_factory()
    with sf() as s:
        rows = s.execute(
            select(EquitySnapshot).order_by(EquitySnapshot.snapshot_at.asc())
        ).scalars().all()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([_row_to_dict(r) for r in rows])
    df["snapshot_at"] = pd.to_datetime(df["snapshot_at"], utc=True, errors="coerce")
    return df.dropna(subset=["snapshot_at"]).sort_values("snapshot_at")


@st.cache_data(ttl=30, show_spinner=False)
def load_decisions() -> pd.DataFrame:
    sf = _session_factory()
    with sf() as s:
        rows = s.execute(
            select(AIDecision).order_by(AIDecision.created_at.desc())
        ).scalars().all()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([_row_to_dict(r) for r in rows])
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    return df


@st.cache_data(ttl=10, show_spinner=False)
def load_positions() -> pd.DataFrame:
    """Open positions = Trade rows with closed_at IS NULL.

    The bot uses Trade as the single source of truth for live positions;
    the legacy Position table is unused.
    """
    sf = _session_factory()
    with sf() as s:
        rows = s.execute(
            select(Trade).where(Trade.closed_at.is_(None))
        ).scalars().all()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([_row_to_dict(r) for r in rows])
    if "entry" in df.columns and "entry_price" not in df.columns:
        df["entry_price"] = df["entry"]
    if "stop" in df.columns and "stop_loss" not in df.columns:
        df["stop_loss"] = df["stop"]
    for col in ("opened_at", "closed_at", "last_invalidation_check_at"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def sharpe_from_trades(trades: pd.DataFrame, periods_per_year: int = 252) -> float:
    """Sharpe over the per-trade PnL series (in % of notional, fallback to USD)."""
    closed = trades.dropna(subset=["closed_at"]) if "closed_at" in trades.columns else trades
    if closed.empty:
        return 0.0
    series = closed["pnl_pct"] if "pnl_pct" in closed and closed["pnl_pct"].abs().sum() > 0 else closed.get("pnl_usd")
    if series is None or len(series) < 2:
        return 0.0
    r = series.astype(float)
    sd = r.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return 0.0
    return float(r.mean() / sd * np.sqrt(periods_per_year))


def sortino_from_equity(equity: pd.Series, periods_per_year: int = 365) -> float:
    """Sortino on equity-curve daily returns, downside std only."""
    if equity is None or len(equity) < 2:
        return 0.0
    daily = equity.resample("1D").last().dropna() if isinstance(equity.index, pd.DatetimeIndex) else equity
    rets = daily.pct_change().dropna()
    if rets.empty:
        return 0.0
    downside = rets[rets < 0]
    if downside.empty or downside.std(ddof=1) == 0:
        return 0.0
    return float(rets.mean() / downside.std(ddof=1) * np.sqrt(periods_per_year))


def max_drawdown_pct(equity: pd.Series) -> float:
    if equity is None or len(equity) == 0:
        return 0.0
    running_peak = equity.cummax()
    dd = (equity - running_peak) / running_peak
    return float(dd.min() * 100.0)


def trade_stats(trades: pd.DataFrame) -> dict[str, float | int]:
    closed = trades.dropna(subset=["closed_at"]) if "closed_at" in trades.columns else pd.DataFrame()
    if closed.empty:
        return {
            "n": 0, "win_rate": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
            "profit_factor": 0.0, "avg_hold_h": 0.0,
        }
    pnl = closed["pnl_usd"].astype(float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    gross_loss = float(losses.abs().sum())
    profit_factor = float(wins.sum() / gross_loss) if gross_loss > 0 else float("inf") if not wins.empty else 0.0
    hold = (closed["closed_at"] - closed["opened_at"]).dt.total_seconds() / 3600.0
    return {
        "n": int(len(closed)),
        "win_rate": float(len(wins) / len(closed) * 100.0),
        "avg_win": float(wins.mean()) if not wins.empty else 0.0,
        "avg_loss": float(losses.mean()) if not losses.empty else 0.0,
        "profit_factor": profit_factor,
        "avg_hold_h": float(hold.mean()) if len(hold) else 0.0,
    }


# ---------------------------------------------------------------------------
# BTC buy & hold
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300, show_spinner=False)
def load_btc_reference(start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Return a BTC close-price series scaled to start at the same value as
    the bot's first equity snapshot.

    Falls back to an empty series if Binance is unreachable; the dashboard
    is designed to render without it.
    """
    try:
        from src.data.binance_client import BinanceClient

        client = BinanceClient()
        df = client.fetch_ohlcv("BTC/USDT:USDT", "1h", limit=1500)
    except Exception:
        return pd.Series(dtype=float)

    if df is None or df.empty or "close" not in df.columns:
        return pd.Series(dtype=float)

    if df.index.tz is None:
        df.index = df.index.tz_localize(timezone.utc)
    mask = (df.index >= start) & (df.index <= end)
    return df.loc[mask, "close"]


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def _render_btc_regime_banner() -> None:
    """Top-of-page banner showing current BTC regime + which sides the gate
    blocks. Calls the same `get_btc_regime_snapshot` the trader uses, so
    dashboard and gate are guaranteed in sync (modulo separate process-level
    caches with the same 60 s TTL)."""
    if not settings.btc_regime_enabled:
        st.caption("BTC regime gate: disabled (BTC_REGIME_ENABLED=false)")
        return

    try:
        from src.data.binance_client import BinanceClient
        from src.risk.btc_regime import get_btc_regime_snapshot

        snap = get_btc_regime_snapshot(BinanceClient())
    except Exception as exc:  # noqa: BLE001
        st.warning(f"BTC regime: fetch error ({exc}) — gate fails open")
        return

    if snap is None:
        st.warning("BTC regime: fetch failed (gate fails open — nothing blocked)")
        return

    regime = snap["regime"]
    slope = snap["slope_pct"]
    slope_thr = snap["slope_threshold_pct"]
    price_dist = snap["price_dist_pct"]
    price_thr = snap["price_dist_threshold_pct"]
    period = snap["period"]
    lookback = snap["lookback"]

    if regime == "up":
        allowed, blocked, color = "LONGS only", "shorts blocked", "🟢"
    elif regime == "down":
        allowed, blocked, color = "SHORTS only", "longs blocked", "🔴"
    else:
        allowed, blocked, color = "BOTH sides allowed", "—", "⚪"

    # Which signal triggered the verdict — helps explain "why".
    if regime != "flat":
        if snap["price_signal"] == regime and snap["slope_signal"] == regime:
            trigger = "slope + price"
        elif snap["price_signal"] == regime:
            trigger = "price (EMA lagging)"
        else:
            trigger = "slope"
    else:
        trigger = "neither signal past threshold"

    c1, c2, c3 = st.columns([1.4, 1.4, 2])
    c1.metric(
        f"{color} BTC regime",
        regime.upper(),
        f"slope {slope:+.3f}%  ·  price {price_dist:+.2f}% off EMA",
    )
    c2.metric(
        "Thresholds",
        f"slope ±{slope_thr:.2f}%/{lookback}h",
        f"price ±{price_thr:.2f}%",
        delta_color="off",
    )
    c3.metric("Gate verdict", allowed, f"{blocked} · trigger: {trigger}",
              delta_color="off")

    # Простое объяснение по-русски — что увидел гейт и почему такой вердикт.
    price_now = snap["price_now"]
    ema_now = snap["ema_now"]
    abs_price = abs(price_dist)
    abs_slope = abs(slope)

    if regime == "flat":
        msg = (
            f"**Explanation.** BTC сейчас {price_now:,.0f}, EMA{period} ≈ {ema_now:,.0f}. "
            f"Цена {'выше' if price_dist >= 0 else 'ниже'} EMA на "
            f"{abs_price:.2f}% (порог ±{price_thr:.2f}%), EMA сдвинулась на "
            f"{slope:+.2f}% за {lookback}ч (порог ±{slope_thr:.2f}%). "
            "Ни один сигнал не пробил порог — считаем рынок боковым, "
            "разрешены обе стороны."
        )
    else:
        direction_word = "растёт" if regime == "up" else "падает"
        side_word = "лонги" if regime == "up" else "шорты"
        if snap["price_signal"] == regime and snap["slope_signal"] == regime:
            msg = (
                f"**Explanation.** BTC уверенно {direction_word}: цена "
                f"{price_now:,.0f} {'выше' if price_dist >= 0 else 'ниже'} "
                f"EMA{period} ({ema_now:,.0f}) на {abs_price:.2f}% (порог "
                f"±{price_thr:.2f}%), и сама EMA сдвинулась на {slope:+.2f}% "
                f"за {lookback}ч (порог ±{slope_thr:.2f}%). Открываем только "
                f"{side_word}."
            )
        elif snap["price_signal"] == regime:
            msg = (
                f"**Explanation.** Цена BTC ({price_now:,.0f}) "
                f"{'выше' if price_dist >= 0 else 'ниже'} EMA{period} "
                f"({ema_now:,.0f}) на {abs_price:.2f}% — это больше порога "
                f"±{price_thr:.2f}%. EMA пока не успела (наклон всего "
                f"{slope:+.2f}% за {lookback}ч), но цена уже ушла — значит "
                f"BTC {direction_word}. Открываем только {side_word}."
            )
        else:  # slope-only
            msg = (
                f"**Explanation.** EMA{period} на BTC сдвинулась на "
                f"{slope:+.2f}% за {lookback}ч — больше порога "
                f"±{slope_thr:.2f}%. Цена пока близко к EMA "
                f"({abs_price:.2f}%, порог ±{price_thr:.2f}%), но тренд по "
                f"наклону — {direction_word}. Открываем только {side_word}."
            )

    st.markdown(msg)


def page_overview() -> None:
    st.header("Overview")
    _render_btc_regime_banner()
    eq = load_equity()
    trades = load_trades()

    if eq.empty:
        st.info("No equity snapshots yet. Start the bot to begin recording.")
        return

    eq_indexed = eq.set_index("snapshot_at")["equity_usd"].astype(float)
    start_val = float(eq_indexed.iloc[0])

    sharpe = sharpe_from_trades(trades)
    sortino = sortino_from_equity(eq_indexed)
    mdd = max_drawdown_pct(eq_indexed)
    stats = trade_stats(trades)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Equity (USDT)", f"{eq_indexed.iloc[-1]:,.2f}",
              f"{(eq_indexed.iloc[-1] - start_val):+,.2f}")
    c2.metric("Sharpe", f"{sharpe:.2f}")
    c3.metric("Sortino", f"{sortino:.2f}")
    c4.metric("Max Drawdown", f"{mdd:.2f}%")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Win rate", f"{stats['win_rate']:.1f}%")
    c6.metric("Avg win", f"{stats['avg_win']:+.2f}")
    c7.metric("Avg loss", f"{stats['avg_loss']:+.2f}")
    pf = stats["profit_factor"]
    c8.metric("Profit factor", "∞" if pf == float("inf") else f"{pf:.2f}")

    st.subheader("Equity curve vs BTC buy & hold")
    chart_df = pd.DataFrame({"strategy": eq_indexed})
    btc = load_btc_reference(eq_indexed.index.min(), eq_indexed.index.max())
    if not btc.empty:
        scaled = btc / float(btc.iloc[0]) * start_val
        chart_df = chart_df.join(scaled.rename("btc_buy_hold"), how="outer").ffill()
    st.line_chart(chart_df)


def page_daily_report() -> None:
    """Today's trading day in one screen, mirroring the 21:00 Telegram digest."""
    st.header("Daily Report")
    st.caption(
        "Window: 00:00 → now (Asia/Dubai). The same numbers are pushed to "
        "Telegram nightly at 21:00 Dubai."
    )

    sf = _session_factory()
    report = build_daily_report(sf)
    st.caption(f"Day: `{report.day_label}`")

    c1, c2, c3, c4 = st.columns(4)
    eq_end = report.equity_end_usd
    eq_change = report.equity_change_usd
    c1.metric(
        "Equity (USDT)",
        f"{eq_end:,.2f}" if eq_end is not None else "—",
        f"{eq_change:+,.2f}" if eq_change is not None else None,
    )
    c2.metric("Realised PnL", f"{report.realized_pnl_usd:+,.2f}")
    c3.metric("Trades closed", str(report.n_closed))
    c4.metric(
        "Win rate",
        f"{report.win_rate:.0f}%" if (report.wins + report.losses) else "—",
        f"{report.wins}W/{report.losses}L",
    )

    c5, c6, c7, c8 = st.columns(4)
    c5.metric(
        "Trades opened",
        str(report.n_opened),
        f"{report.n_long_opened} long / {report.n_short_opened} short",
    )
    c6.metric("Open now", str(report.n_open_now))
    c7.metric("Fees today", f"{report.fees_usd:,.2f}")
    c8.metric(
        "Unrealised PnL",
        f"{report.unrealized_pnl_usd:+,.2f}"
        if report.unrealized_pnl_usd is not None else "—",
    )

    if report.n_closed:
        st.subheader("Best / worst")
        cols = st.columns(2)
        cols[0].success(
            f"🏆 `{report.best_symbol}`  {report.best_trade_usd:+,.2f} USDT"
        )
        cols[1].error(
            f"💀 `{report.worst_symbol}`  {report.worst_trade_usd:+,.2f} USDT"
        )

    if report.by_strategy:
        st.subheader("By strategy")
        rows = [
            {
                "type": s.type,
                "closed": s.n_closed,
                "pnl_usd": s.pnl_usd,
                "win_rate_%": s.win_rate,
                "wins": s.wins,
                "losses": s.losses,
                "avg_pnl_pct": s.avg_pnl_pct,
                "opened_long": s.n_long,
                "opened_short": s.n_short,
            }
            for s in report.by_strategy
        ]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    else:
        st.info("No trades opened or closed today yet.")

    if report.by_side:
        st.subheader("By side (closed)")
        rows = [
            {
                "side": ss.side,
                "closed": ss.n_closed,
                "pnl_usd": ss.pnl_usd,
                "win_rate_%": ss.win_rate,
                "wins": ss.wins,
                "losses": ss.losses,
            }
            for ss in report.by_side
        ]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    st.subheader("AI veto")
    a1, a2, a3 = st.columns(3)
    a1.metric("Decisions", str(report.ai_decisions_total))
    a2.metric(
        "Accepted",
        str(report.ai_decisions_accepted),
        f"{report.acceptance_rate:.0f}%" if report.ai_decisions_total else None,
    )
    a3.metric("LLM cost", f"${report.ai_cost_usd:,.4f}")

    with st.expander("Telegram preview"):
        st.code(format_daily_report_telegram(report))


def page_trades() -> None:
    st.header("Trades")
    trades = load_trades()
    if trades.empty:
        st.info("No trades recorded yet.")
        return

    with st.sidebar:
        st.subheader("Filters")
        symbols = ["(all)"] + sorted(trades["symbol"].dropna().unique().tolist())
        symbol = st.selectbox("Symbol", symbols, key="trades_symbol")
        sides = ["(all)", "long", "short"]
        side = st.selectbox("Side", sides, key="trades_side")
        status = st.selectbox(
            "Status", ["(all)", "open", "closed"], key="trades_status"
        )
        outcome = st.selectbox(
            "Outcome", ["(all)", "winners", "losers"], key="trades_outcome"
        )

    df = trades.copy()
    if symbol != "(all)":
        df = df[df["symbol"] == symbol]
    if side != "(all)":
        df = df[df["side"] == side]
    if status == "open":
        df = df[df["closed_at"].isna()]
    elif status == "closed":
        df = df[df["closed_at"].notna()]
    if outcome == "winners":
        df = df[df["pnl_usd"].astype(float) > 0]
    elif outcome == "losers":
        df = df[df["pnl_usd"].astype(float) < 0]

    st.caption(f"{len(df)} trade(s)")
    cols = [
        c for c in [
            "id", "symbol", "side", "type", "entry", "close_price", "quantity",
            "leverage", "pnl_usd", "pnl_pct", "fees", "close_reason",
            "llm_confidence", "opened_at", "closed_at",
        ] if c in df.columns
    ]
    st.dataframe(df[cols].sort_values("opened_at", ascending=False),
                 width="stretch", hide_index=True)


def page_ai_decisions() -> None:
    st.header("AI Decisions")
    df = load_decisions()
    if df.empty:
        st.info("No AI decisions logged yet.")
        return

    with st.sidebar:
        st.subheader("Filters")
        symbols = ["(all)"] + sorted(df["symbol"].dropna().unique().tolist())
        symbol = st.selectbox("Symbol", symbols, key="dec_symbol")
        decisions = ["(all)"] + sorted(df["decision"].dropna().unique().tolist())
        decision = st.selectbox("Decision", decisions, key="dec_decision")
        only_taken = st.checkbox("Only taken", value=False)

    view = df.copy()
    if symbol != "(all)":
        view = view[view["symbol"] == symbol]
    if decision != "(all)":
        view = view[view["decision"] == decision]
    if only_taken and "took" in view.columns:
        view = view[view["took"].astype(bool)]

    st.subheader("Cost by day (USD)")
    if "cost_usd" in df.columns and "created_at" in df.columns:
        daily = (
            df.assign(day=df["created_at"].dt.tz_convert("UTC").dt.floor("D"))
              .groupby("day")["cost_usd"].sum()
              .astype(float)
        )
        if not daily.empty:
            st.bar_chart(daily)
        st.caption(f"Total spend: ${df['cost_usd'].astype(float).sum():,.4f}")

    st.subheader("Decisions")
    summary_cols = [
        c for c in [
            "created_at", "symbol", "side", "decision", "confidence",
            "took", "cost_usd", "duration_ms", "model",
        ] if c in view.columns
    ]
    st.dataframe(view[summary_cols], width="stretch", hide_index=True)

    st.subheader("Inspect raw")
    if not view.empty:
        idx = st.selectbox(
            "Pick a row",
            view.index.tolist(),
            format_func=lambda i: (
                f"{view.at[i, 'created_at']} | {view.at[i, 'symbol']} | "
                f"{view.at[i, 'decision']}"
            ),
        )
        row = view.loc[idx]
        for label, key in (
            ("Candidate", "candidate_json"),
            ("Veto", "veto_json"),
            ("Reasoning", "reasoning"),
            ("Raw response", "response_raw"),
        ):
            if key in row and pd.notna(row[key]) and row[key]:
                st.markdown(f"**{label}**")
                value = row[key]
                if key.endswith("_json"):
                    try:
                        st.json(json.loads(value))
                        continue
                    except (TypeError, ValueError):
                        pass
                st.code(str(value))


def page_positions() -> None:
    st.header("Open positions (live)")
    df = load_positions()
    if df.empty:
        st.info("No open positions.")
        return
    cols = [
        c for c in [
            "id", "symbol", "side", "type", "quantity", "entry_price",
            "leverage", "stop_loss", "take_profit", "risk_usd",
            "llm_confidence", "paper", "opened_at",
        ] if c in df.columns
    ]
    st.dataframe(df[cols], width="stretch", hide_index=True)


# ---------------------------------------------------------------------------
# Data export
# ---------------------------------------------------------------------------

# Tables exposed on the export page. Order = display order.
# Each entry: (display_name, ORM model, time column for date filter or None)
_EXPORT_TABLES: list[tuple[str, type, str | None]] = [
    ("trades", Trade, "opened_at"),
    ("ai_decisions", AIDecision, "created_at"),
    ("equity_snapshots", EquitySnapshot, "snapshot_at"),
    ("positions", Position, None),
    ("safety_state", SafetyState, None),
]


def _load_table_df(model: type, time_col: str | None, since: datetime | None) -> pd.DataFrame:
    """Pull one table to a DataFrame, optionally filtered to rows after `since`."""
    sf = _session_factory()
    stmt = select(model)
    if since is not None and time_col is not None:
        stmt = stmt.where(getattr(model, time_col) >= since)
    with sf() as s:
        rows = s.execute(stmt).scalars().all()
    if not rows:
        return pd.DataFrame(columns=[c.name for c in model.__table__.columns])
    return pd.DataFrame([_row_to_dict(r) for r in rows])


def _df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """UTF-8 CSV with no index column — what every spreadsheet tool expects."""
    return df.to_csv(index=False).encode("utf-8")


def _build_zip_bundle(since: datetime | None) -> bytes:
    """Pack all tables into a single in-memory zip — one CSV per table."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, model, time_col in _EXPORT_TABLES:
            df = _load_table_df(model, time_col, since)
            zf.writestr(f"{name}.csv", _df_to_csv_bytes(df))
    return buf.getvalue()


def page_data_export() -> None:
    st.header("Data Export")
    st.caption(
        "Download raw DB tables as CSV. Use these as input for offline "
        "analysis (Excel, pandas, the chat AI you're sharing them with)."
    )

    with st.sidebar:
        st.subheader("Export filter")
        # Date filter only narrows the time-stamped tables (`trades`,
        # `ai_decisions`, `equity_snapshots`); `positions` and
        # `safety_state` are always exported in full because they're
        # tiny snapshots, not append-only logs.
        choice = st.radio(
            "Time window",
            ["all", "last 1 day", "last 7 days", "last 30 days"],
            index=0,
            key="export_window",
        )
    since: datetime | None = None
    if choice == "last 1 day":
        since = datetime.now(tz=timezone.utc) - timedelta(days=1)
    elif choice == "last 7 days":
        since = datetime.now(tz=timezone.utc) - timedelta(days=7)
    elif choice == "last 30 days":
        since = datetime.now(tz=timezone.utc) - timedelta(days=30)

    stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")

    # One-click bundle first — that's what most users actually want.
    st.subheader("Bundle")
    st.write(
        "Single zip with one CSV per table. Date filter applies to "
        "trades / ai_decisions / equity_snapshots; positions and safety_state "
        "are exported in full."
    )
    zip_bytes = _build_zip_bundle(since)
    st.download_button(
        label=f"⬇️ Download all tables ({len(zip_bytes) / 1024:.1f} KiB)",
        data=zip_bytes,
        file_name=f"ai_trader_export_{stamp}.zip",
        mime="application/zip",
        key="export_zip",
    )

    st.divider()

    # Per-table buttons + a small preview, in case you only need one.
    st.subheader("Per-table CSV")
    for name, model, time_col in _EXPORT_TABLES:
        df = _load_table_df(model, time_col, since)
        cols = st.columns([3, 1])
        cols[0].markdown(
            f"**`{name}`** — {len(df)} row(s)"
            + (f", filtered by `{time_col}`" if (since and time_col) else "")
        )
        cols[1].download_button(
            label="CSV",
            data=_df_to_csv_bytes(df),
            file_name=f"{name}_{stamp}.csv",
            mime="text/csv",
            key=f"export_csv_{name}",
        )
        if not df.empty:
            with st.expander(f"Preview {name} (first 20 rows)", expanded=False):
                st.dataframe(df.head(20), width="stretch", hide_index=True)


def page_settings() -> None:
    st.header("Settings")
    st.caption("Read-only view of the active configuration. Edit `.env` to change.")

    env_path = settings.project_root / ".env"
    rows: list[tuple[str, str]] = []
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, _, value = stripped.partition("=")
            key = key.strip()
            if any(s in key.upper() for s in ("KEY", "SECRET", "TOKEN", "PASSWORD")):
                value = "••• (hidden)"
            rows.append((key, value.strip()))
        st.dataframe(
            pd.DataFrame(rows, columns=["key", "value"]),
            width="stretch", hide_index=True,
        )
    else:
        st.warning(f"No .env file found at `{env_path}`.")

    st.subheader("Runtime toggles")
    st.write(
        "Toggling `PAPER_TRADING` here updates the in-memory setting for the "
        "dashboard process only. Persist the change by editing `.env` and "
        "restarting the bot."
    )
    paper = st.toggle("PAPER_TRADING", value=bool(settings.paper_trading))
    if paper != bool(settings.paper_trading):
        settings.paper_trading = paper  # type: ignore[misc]
        os.environ["PAPER_TRADING"] = "true" if paper else "false"
        st.success(f"PAPER_TRADING set to {paper} for this session.")

    st.write(f"trading_mode = `{settings.trading_mode}`")
    st.write(f"binance_testnet = `{settings.binance_testnet}`")
    st.write(f"db_path = `{settings.db_abspath}`")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

PAGES = {
    "Overview": page_overview,
    "Daily Report": page_daily_report,
    "Trades": page_trades,
    "AI Decisions": page_ai_decisions,
    "Positions": page_positions,
    "Data Export": page_data_export,
    "Settings": page_settings,
}


def main() -> None:
    st.set_page_config(page_title="ai-trader", layout="wide")
    st.title("ai-trader — live dashboard")
    page = st.sidebar.radio("Page", list(PAGES.keys()))
    if st.sidebar.button("Refresh data"):
        st.cache_data.clear()
        # The BTC regime module has its own thread-locked cache (same 60 s
        # TTL). Without this, Refresh would clear streamlit caches but the
        # banner could still serve up to a minute of stale snapshot.
        try:
            from src.risk.btc_regime import reset_cache as _reset_btc_cache

            _reset_btc_cache()
        except Exception:  # noqa: BLE001
            pass
    PAGES[page]()


if __name__ == "__main__":
    main()
