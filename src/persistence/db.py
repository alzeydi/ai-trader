"""SQLite persistence: trades, AI decisions, equity snapshots, positions, OHLCV cache."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
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
    """Realised trade lifecycle row.

    Spec fields (id, ts_open, ts_close, symbol, side, entry, exit, qty,
    pnl_usd, pnl_pct, fees, close_reason, llm_confidence, invalidation_signal)
    are exposed as attributes; `ts_open`/`ts_close`/`exit`/`qty` are aliased
    to the legacy column names so existing call-sites keep working.
    """

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
    pnl_pct = Column(Float, default=0.0)
    fees = Column(Float, default=0.0)
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
    # Trailing stop. `original_stop` preserves the entry-time SL so we can
    # always recompute R = |entry - original_stop|. `stop` is what's live on
    # the exchange and may be tightened by the trailing logic.
    original_stop = Column(Float, nullable=True)
    trail_armed = Column(Boolean, default=False)
    trail_high_water = Column(Float, nullable=True)
    # Order ID of the currently live trail SL. Kept separate from
    # stop_order_id (original bracket SL) so both orders stay on the
    # exchange as a two-layer backstop.
    trail_stop_order_id = Column(String(64), nullable=True)

    # --- spec aliases ----------------------------------------------------
    @property
    def ts_open(self) -> datetime | None:
        return self.opened_at

    @property
    def ts_close(self) -> datetime | None:
        return self.closed_at

    @property
    def exit(self) -> float | None:
        return self.close_price

    @property
    def qty(self) -> float:
        return float(self.quantity)


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
    """Full audit trail for every veto-layer decision (preflight or LLM).

    Spec-required columns (`candidate_json`, `veto_json`, `took`,
    `cost_usd`, `ts`, `symbol`) are present; `ts` is an alias for
    `created_at`.
    """

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
    candidate_json = Column(Text, nullable=True)
    veto_json = Column(Text, nullable=True)
    took = Column(Boolean, default=False)
    input_tokens = Column(Integer, default=0)
    output_tokens = Column(Integer, default=0)
    cost_usd = Column(Float, default=0.0)
    duration_ms = Column(Integer, default=0)
    attempts = Column(Integer, default=0)
    model = Column(String(64), nullable=True)

    @property
    def ts(self) -> datetime | None:
        return self.created_at


class EquitySnapshot(Base):
    """Per-tick portfolio snapshot used by the equity curve and metrics."""

    __tablename__ = "equity_snapshots"
    id = Column(Integer, primary_key=True, autoincrement=True)
    snapshot_at = Column(DateTime(timezone=True), default=_now, index=True)
    equity_usd = Column(Float, nullable=False)
    balance_usd = Column(Float, nullable=False, default=0.0)
    realized_pnl = Column(Float, default=0.0)
    unrealized_pnl = Column(Float, default=0.0)

    @property
    def ts(self) -> datetime | None:
        return self.snapshot_at

    @property
    def equity(self) -> float:
        return float(self.equity_usd)

    @property
    def balance(self) -> float:
        return float(self.balance_usd)


# Legacy alias so older imports (`EquityPoint`) keep resolving.
EquityPoint = EquitySnapshot


class Position(Base):
    """Live state of an open position.

    Mirrors the exchange snapshot for the dashboard and recovery on restart.
    Updated by the position monitor; deleted (or `closed_at` set) on close.
    """

    __tablename__ = "positions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    trade_id = Column(Integer, nullable=True, index=True)
    symbol = Column(String(32), nullable=False, index=True)
    side = Column(String(8), nullable=False)
    quantity = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)
    mark_price = Column(Float, nullable=True)
    leverage = Column(Integer, default=1)
    margin_usd = Column(Float, default=0.0)
    stop_loss = Column(Float, nullable=True)
    take_profit = Column(Float, nullable=True)
    unrealized_pnl = Column(Float, default=0.0)
    opened_at = Column(DateTime(timezone=True), default=_now, index=True)
    updated_at = Column(DateTime(timezone=True), default=_now, onupdate=_now)


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
    _ensure_trade_columns(engine)
    return engine


def _ensure_trade_columns(engine) -> None:
    """SQLite create_all() adds new tables but not new columns to existing
    ones. Ensure trailing-stop columns exist on the persistent DB.
    """
    new_cols = {
        "original_stop": "REAL",
        "trail_armed": "BOOLEAN DEFAULT 0",
        "trail_high_water": "REAL",
        "trail_stop_order_id": "TEXT",
    }
    with engine.begin() as conn:
        rows = conn.exec_driver_sql("PRAGMA table_info(trades)").fetchall()
        existing = {row[1] for row in rows}
        for col, ddl in new_cols.items():
            if col not in existing:
                conn.exec_driver_sql(f"ALTER TABLE trades ADD COLUMN {col} {ddl}")


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


def has_recent_losing_b_trade(
    session_factory: SessionFactory,
    symbol: str,
    cooldown_hours: int,
) -> bool:
    """True iff a Type-B trade on `symbol` closed in the last
    `cooldown_hours` with negative PnL.

    Drives the per-symbol Type-B cooldown: one losing fade is treated as a
    regime signal, so we stop firing more B's on that symbol until the
    cooldown elapses.
    """
    if cooldown_hours <= 0:
        return False
    cutoff = datetime.now(tz=timezone.utc) - timedelta(hours=cooldown_hours)
    with session_factory() as s:
        row = s.execute(
            select(Trade)
            .where(Trade.symbol == symbol)
            .where(Trade.type == "B")
            .where(Trade.closed_at.isnot(None))
            .where(Trade.closed_at >= cutoff)
            .where(Trade.pnl_usd < 0)
            .order_by(Trade.closed_at.desc())
            .limit(1)
        ).scalars().first()
    return row is not None


def has_recent_type_e_decision(
    session_factory: SessionFactory,
    symbol: str,
    cooldown_hours: int,
) -> bool:
    """True iff a Type-E candidate for `symbol` was recorded in the last
    `cooldown_hours` (regardless of veto verdict).

    Strategy E scans the in-progress H4 bar every loop tick, so without this
    per-symbol dedup the same setup would re-fire every cycle for the rest of
    the 4h bar. Spec §5: each individual symbol fires at most once per 24 h.

    Checked against `ai_decisions` (not `trades`) so the cooldown engages even
    when the LLM vetoes the candidate — otherwise vetoed E's would burn fresh
    LLM calls every minute.
    """
    return _has_recent_type_decision(session_factory, symbol, "E", cooldown_hours)


def has_recent_type_g_decision(
    session_factory: SessionFactory,
    symbol: str,
    cooldown_hours: int,
) -> bool:
    """Same purpose as has_recent_type_e_decision, but for Type G (mirror of E).

    Strategy G also re-evaluates the in-progress H4 bar every loop tick, so
    without per-symbol dedup the bot would re-fire every minute.
    """
    return _has_recent_type_decision(session_factory, symbol, "G", cooldown_hours)


def _has_recent_type_decision(
    session_factory: SessionFactory,
    symbol: str,
    entry_type: str,
    cooldown_hours: int,
) -> bool:
    if cooldown_hours <= 0:
        return False
    cutoff = datetime.now(tz=timezone.utc) - timedelta(hours=cooldown_hours)
    with session_factory() as s:
        row = s.execute(
            select(AIDecision)
            .where(AIDecision.symbol == symbol)
            .where(AIDecision.entry_type == entry_type)
            .where(AIDecision.created_at >= cutoff)
            .order_by(AIDecision.created_at.desc())
            .limit(1)
        ).scalars().first()
    return row is not None


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


# ----- Equity helpers -----

def append_equity_snapshot(
    session_factory: SessionFactory,
    *,
    equity_usd: float,
    balance_usd: float | None = None,
    realized_pnl: float = 0.0,
    unrealized_pnl: float = 0.0,
) -> None:
    with session_factory() as s:
        s.add(
            EquitySnapshot(
                equity_usd=float(equity_usd),
                balance_usd=float(balance_usd if balance_usd is not None else equity_usd),
                realized_pnl=float(realized_pnl),
                unrealized_pnl=float(unrealized_pnl),
            )
        )
        s.commit()


# ----- Position helpers -----

def upsert_position(
    session_factory: SessionFactory,
    *,
    symbol: str,
    side: str,
    quantity: float,
    entry_price: float,
    mark_price: float | None = None,
    leverage: int = 1,
    margin_usd: float = 0.0,
    stop_loss: float | None = None,
    take_profit: float | None = None,
    unrealized_pnl: float = 0.0,
    trade_id: int | None = None,
) -> int:
    """Create or update the live `Position` row for `symbol`."""
    with session_factory() as s:
        row = s.execute(
            select(Position).where(Position.symbol == symbol)
        ).scalar_one_or_none()
        if row is None:
            row = Position(
                symbol=symbol,
                side=side,
                quantity=float(quantity),
                entry_price=float(entry_price),
                mark_price=mark_price,
                leverage=int(leverage),
                margin_usd=float(margin_usd),
                stop_loss=stop_loss,
                take_profit=take_profit,
                unrealized_pnl=float(unrealized_pnl),
                trade_id=trade_id,
            )
            s.add(row)
        else:
            row.side = side
            row.quantity = float(quantity)
            row.entry_price = float(entry_price)
            row.mark_price = mark_price
            row.leverage = int(leverage)
            row.margin_usd = float(margin_usd)
            row.stop_loss = stop_loss
            row.take_profit = take_profit
            row.unrealized_pnl = float(unrealized_pnl)
            if trade_id is not None:
                row.trade_id = trade_id
        s.commit()
        return int(row.id)


def remove_position(session_factory: SessionFactory, symbol: str) -> None:
    with session_factory() as s:
        row = s.execute(
            select(Position).where(Position.symbol == symbol)
        ).scalar_one_or_none()
        if row is not None:
            s.delete(row)
            s.commit()


def list_positions(session_factory: SessionFactory) -> list["Position"]:
    with session_factory() as s:
        rows = s.execute(select(Position)).scalars().all()
    return list(rows)


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


# ----- CSV export (dashboard / backup compatibility) -----

_EXPORT_TABLES: dict[str, type[Base]] = {
    "trades": Trade,
    "ai_decisions": AIDecision,
    "decisions": Decision,
    "equity_snapshots": EquitySnapshot,
    "positions": Position,
    "safety_state": SafetyState,
}


def _row_to_dict(row: Base) -> dict:
    out: dict = {}
    for col in row.__table__.columns:  # type: ignore[attr-defined]
        out[col.name] = getattr(row, col.name)
    return out


def export_to_csv(
    out_dir: Path | str | None = None,
    session_factory: SessionFactory | None = None,
    tables: Iterable[str] | None = None,
) -> dict[str, Path]:
    """Dump selected tables to CSV files.

    Mirrors the legacy CSV layout the dashboard reads as a fallback and
    doubles as a one-shot backup. Returns a mapping `table_name -> path`
    of files actually written.
    """
    sf = session_factory or make_session_factory()
    target_dir = Path(out_dir) if out_dir is not None else (settings.project_root / "data" / "exports")
    target_dir.mkdir(parents=True, exist_ok=True)

    selected = list(tables) if tables is not None else list(_EXPORT_TABLES.keys())
    written: dict[str, Path] = {}

    with sf() as s:
        for name in selected:
            model = _EXPORT_TABLES.get(name)
            if model is None:
                continue
            rows = s.execute(select(model)).scalars().all()
            df = pd.DataFrame([_row_to_dict(r) for r in rows]) if rows else pd.DataFrame(
                columns=[c.name for c in model.__table__.columns]  # type: ignore[attr-defined]
            )
            path = target_dir / f"{name}.csv"
            df.to_csv(path, index=False)
            written[name] = path
    return written
