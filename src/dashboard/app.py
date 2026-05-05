"""Streamlit dashboard for the ai-trader.

Pages
-----
1. Overview     — equity curve vs BTC buy&hold, Sharpe, Sortino, Max DD
2. Trades       — table of all trades with filters
3. AI Decisions — LLM decision log (with raw JSON) + cost-by-day chart
4. Positions    — currently open positions (live)
5. Settings     — read-only view of `.env` and PAPER_TRADING toggle

Run with `streamlit run src/dashboard/app.py`.
"""

from __future__ import annotations

import json
import os
from datetime import timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from sqlalchemy import select

from src.config import settings
from src.persistence.db import (
    AIDecision,
    EquitySnapshot,
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

def page_overview() -> None:
    st.header("Overview")
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
    "Trades": page_trades,
    "AI Decisions": page_ai_decisions,
    "Positions": page_positions,
    "Settings": page_settings,
}


def main() -> None:
    st.set_page_config(page_title="ai-trader", layout="wide")
    st.title("ai-trader — live dashboard")
    page = st.sidebar.radio("Page", list(PAGES.keys()))
    if st.sidebar.button("Refresh data"):
        st.cache_data.clear()
    PAGES[page]()


if __name__ == "__main__":
    main()
