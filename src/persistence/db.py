"""SQLite persistence (trades, decisions, equity snapshots)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
)
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
    pnl_usd = Column(Float, default=0.0)
    opened_at = Column(DateTime(timezone=True), default=_now)
    closed_at = Column(DateTime(timezone=True), nullable=True)
    notes = Column(Text, nullable=True)


class Decision(Base):
    __tablename__ = "decisions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(32), index=True)
    decision = Column(String(8))      # ALLOW / REJECT
    confidence = Column(Float)
    reason = Column(Text)
    raw = Column(Text)
    created_at = Column(DateTime(timezone=True), default=_now)


class EquityPoint(Base):
    __tablename__ = "equity"
    id = Column(Integer, primary_key=True, autoincrement=True)
    equity_usd = Column(Float, nullable=False)
    realized_pnl = Column(Float, default=0.0)
    unrealized_pnl = Column(Float, default=0.0)
    snapshot_at = Column(DateTime(timezone=True), default=_now, index=True)


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
