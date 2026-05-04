"""SQLite persistence: trades, AI decisions, equity snapshots, OHLCV cache."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    select,
)
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from src.config import settings


class Base(DeclarativeBase):
    pass


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


class Trade(Base):
    __tablename__ = "trades"
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(32), nullable=False, index=True)
    side = Column(String(8), nullable=False)
    type = Column(String(4), nullable=False)
    entry = Column(Float, nullable=False)
    stop = Column(Float, nullable=False)
    take_profit = Column(Float, nullable=False)
    quantity = Column(Float, nullable=False)
    leverage = Column(Integer, default=1)
    risk_usd = Column(Float, default=0.0)
    pnl_usd = Column(Float, default=0.0)
    opened_at = Column(DateTime(timezone=True), default=_now, index=True)
    closed_at = Column(DateTime(timezone=True), nullable=True, index=True)
    close_price = Column(Float, nullable=True)
    close_reason = Column(String(32), nullable=True)
    notes = Column(Text, nullable=True)
    paper = Column(Boolean, default=True)
    entry_order_id = Column(String(64), nullable=True)
    stop_order_id = Column(String(64), nullable=True)
    take_order_id = Column(String(64), nullable=True)
    invalidation_signal = Column(Text, nullable=True)
    llm_confidence = Column(Float, default=0.0)
    last_invalidation_check_at = Column(DateTime(timezone=True), nullable=True)


class Decision(Base):
    __tablename__ = "decisions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(32), index=True)
    decision = Column(String(8))      # ALLOW / REJECT
    confidence = Column(Float)
    reason = Column(Text)
    raw = Column(Text)
    created_at = Column(DateTime(timezone=True), default=_now)


class AIDecision(Base):
    """Full audit trail for every veto-layer decision (preflight or LLM)."""

    __tablename__ = "ai_decisions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime(timezone=True), default=_now, index=True)
    symbol = Column(String(32), index=True)
    side = Column(String(8))
    entry_type = Column(String(4))
    signal_strength = Column(Float, default=0.0)
    decision = Column(String(8))
    confidence = Column(Float, default=0.0)
    invalidation_signal = Column(Text)
    reasoning = Column(Text)
    skipped_by_preflight = Column(Boolean, default=False)
    preflight_rule = Column(String(64), nullable=True)
    request_user = Column(Text, nullable=True)
    response_raw = Column(Text, nullable=True)
    input_tokens = Column(Integer, default=0)
    output_tokens = Column(Integer, default=0)
    cost_usd = Column(Float, default=0.0)
    duration_ms = Column(Integer, default=0)
    attempts = Column(Integer, default=0)
    model = Column(String(64), nullable=True)


class EquityPoint(Base):
    __tablename__ = "equity"
    id = Column(Integer, primary_key=True, autoincrement=True)
    equity_usd = Column(Float, nullable=False)
    realized_pnl = Column(Float, default=0.0)
    unrealized_pnl = Column(Float, default=0.0)
    snapshot_at = Column(DateTime(timezone=True), default=_now, index=True)


class SafetyState(Base):
    """Single-row table storing the active safety-mode pause."""

    __tablename__ = "safety_state"
    id = Column(Integer, primary_key=True, default=1)
    paused_until = Column(DateTime(timezone=True), nullable=True)
    activated_at = Column(DateTime(timezone=True), nullable=True)
    reason = Column(Text, nullable=True)
    consecutive_losses = Column(Integer, default=0)
    updated_at = Column(DateTime(timezone=True), default=_now, onupdate=_now)


class OhlcvCandle(Base):
    """Persistent cache of historical candles per (symbol, timeframe)."""

    __tablename__ = "ohlcv_candles"
    symbol = Column(String(32), primary_key=True)
    timeframe = Column(String(8), primary_key=True)
    ts = Column(DateTime(timezone=True), primary_key=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)


# ----- Engine / session helpers -----

def make_engine(path: Path | None = None):
    db_path = path or settings.db_abspath
    db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{db_path}", future=True)
    Base.metadata.create_all(engine)
    return engine


SessionFactory = sessionmaker[Session]


def make_session_factory(engine=None) -> SessionFactory:
    eng = engine or make_engine()
    return sessionmaker(bind=eng, expire_on_commit=False, future=True)


# ----- Trade helpers -----

def get_last_n_closed_trades(
    session_factory: SessionFactory, n: int
) -> list["Trade"]:
    """Return the `n` most recently closed trades, newest first."""
    if n <= 0:
        return []
    with session_factory() as s:
        rows = s.execute(
            select(Trade)
            .where(Trade.closed_at.isnot(None))
            .order_by(Trade.closed_at.desc())
            .limit(n)
        ).scalars().all()
    return list(rows)


def count_open_trades(session_factory: SessionFactory) -> int:
    with session_factory() as s:
        rows = s.execute(
            select(Trade).where(Trade.closed_at.is_(None))
        ).scalars().all()
    return len(rows)


def list_open_trades(session_factory: SessionFactory) -> list["Trade"]:
    with session_factory() as s:
        rows = s.execute(
            select(Trade).where(Trade.closed_at.is_(None))
        ).scalars().all()
    return list(rows)


def daily_realized_pnl(session_factory: SessionFactory) -> float:
    """Sum of realised PnL on trades closed since UTC midnight."""
    today = datetime.now(tz=timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    with session_factory() as s:
        rows = s.execute(
            select(Trade)
            .where(Trade.closed_at.isnot(None))
            .where(Trade.closed_at >= today)
        ).scalars().all()
    return float(sum((r.pnl_usd or 0.0) for r in rows))


# ----- Safety-state helpers -----

def load_safety_state(session_factory: SessionFactory) -> "SafetyState | None":
    with session_factory() as s:
        return s.execute(select(SafetyState).where(SafetyState.id == 1)).scalar_one_or_none()


def save_safety_state(
    session_factory: SessionFactory,
    *,
    paused_until: datetime | None,
    consecutive_losses: int,
    reason: str | None,
) -> None:
    with session_factory() as s:
        row = s.execute(select(SafetyState).where(SafetyState.id == 1)).scalar_one_or_none()
        if row is None:
            row = SafetyState(
                id=1,
                paused_until=paused_until,
                activated_at=_now() if paused_until else None,
                reason=reason,
                consecutive_losses=consecutive_losses,
            )
            s.add(row)
        else:
            row.paused_until = paused_until
            row.activated_at = _now() if paused_until else None
            row.reason = reason
            row.consecutive_losses = consecutive_losses
        s.commit()


# ----- OHLCV cache helpers -----

def upsert_candles(
    session_factory: SessionFactory,
    symbol: str,
    timeframe: str,
    df: pd.DataFrame,
) -> int:
    """Insert candles, replacing any existing rows for the same (symbol, tf, ts).

    `df` must have a DatetimeIndex (UTC) and columns open/high/low/close/volume.
    Returns the number of rows written.
    """
    if df.empty:
        return 0
    rows: list[dict] = []
    for ts, row in df.iterrows():
        rows.append(
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "ts": ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts,
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
            }
        )
    stmt = sqlite_insert(OhlcvCandle).values(rows)
    upsert = stmt.on_conflict_do_update(
        index_elements=["symbol", "timeframe", "ts"],
        set_={
            "open": stmt.excluded.open,
            "high": stmt.excluded.high,
            "low": stmt.excluded.low,
            "close": stmt.excluded.close,
            "volume": stmt.excluded.volume,
        },
    )
    with session_factory() as s:
        s.execute(upsert)
        s.commit()
    return len(rows)


def read_cached_candles(
    session_factory: SessionFactory,
    symbol: str,
    timeframe: str,
    limit: int,
) -> pd.DataFrame:
    """Return the most recent `limit` cached candles as an indexed DataFrame."""
    stmt = (
        select(OhlcvCandle)
        .where(OhlcvCandle.symbol == symbol)
        .where(OhlcvCandle.timeframe == timeframe)
        .order_by(OhlcvCandle.ts.desc())
        .limit(limit)
    )
    with session_factory() as s:
        rows = s.execute(stmt).scalars().all()
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    records: Iterable[dict] = (
        {
            "ts": r.ts if r.ts.tzinfo else r.ts.replace(tzinfo=timezone.utc),
            "open": r.open,
            "high": r.high,
            "low": r.low,
            "close": r.close,
            "volume": r.volume,
        }
        for r in rows
    )
    df = pd.DataFrame(records).sort_values("ts").set_index("ts")
    return df
