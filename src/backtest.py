"""Full-fidelity historical replay through the live pipeline.

Layers used — identical to live trading:
  SignalEngine._evaluate  → Layer 1 (deterministic, no I/O)
  preflight_check + LLM   → Layer 2 veto (LLM stubbed unless --use-llm)
  size_position           → Layer 3 sizing
  BacktestEngine          → fake fill with slippage + fee model

CLI
---
python -m src.backtest \\
    --symbols ETH/USDT:USDT SOL/USDT:USDT \\
    --start 2025-01-01 --end 2025-03-31 \\
    [--equity 10000] [--slippage-pct 0.0005] \\
    [--use-llm] [--system-prompt prompts/custom.txt] \\
    [--mc-runs 1000] [--wf-folds 4] \\
    [--out-dir data-backtest]

Output layout (same as live data/):
    <out-dir>/run-<ts>/
        <SYMBOL>/
            trades.csv
            equity_curve.csv
            ai_decisions.csv
        summary.json
        config.json
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field, asdict, replace
from datetime import datetime, timedelta, timezone, date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import settings
from src.llm.client import ClaudeClient, estimate_cost_usd
from src.llm.types import VetoAccount, VetoContext, VetoResponse
from src.llm.veto import preflight_check
from src.risk.sizing import size_position
from src.risk.types import AccountState
from src.signal.engine import SignalEngine, _MIN_4H, _MIN_1H, _MIN_15M
from src.signal.types import CandidateSignal

log = logging.getLogger(__name__)

# Extra bars fetched before `start` so indicators are warm at replay begin.
_WARMUP_4H_BARS = _MIN_4H + 30    # ≈ 38 days of 4h
_WARMUP_1H_BARS = _MIN_1H + 20    # ≈ 3.1 days of 1h
_WARMUP_15M_BARS = _MIN_15M + 20  # ≈ 13 h of 15m

# Shared (mock) engine instance — _evaluate uses no instance state.
_ENGINE: SignalEngine | None = None


def _get_engine() -> SignalEngine:
    global _ENGINE
    if _ENGINE is None:
        from unittest.mock import MagicMock
        _ENGINE = SignalEngine(client=MagicMock())
    return _ENGINE


def _tf_to_ms(tf: str) -> int:
    n = int(tf[:-1])
    return n * {"m": 60_000, "h": 3_600_000, "d": 86_400_000}[tf[-1]]


# ---------------------------------------------------------------------------
# Config / result containers
# ---------------------------------------------------------------------------

@dataclass
class BacktestConfig:
    symbols: list[str]
    start: datetime
    end: datetime
    equity_usd: float = 10_000.0
    slippage_pct: float = 0.0005
    taker_fee_pct: float = field(default_factory=lambda: settings.taker_fee_pct)
    leverage: int = field(default_factory=lambda: settings.leverage)
    use_llm: bool = False
    system_prompt_path: Path | None = None
    out_dir: Path | None = None
    mc_runs: int = 1000
    wf_folds: int = 0

    def to_json(self) -> dict:
        d = asdict(self)
        d["start"] = self.start.isoformat()
        d["end"] = self.end.isoformat()
        d["system_prompt_path"] = str(self.system_prompt_path) if self.system_prompt_path else None
        d["out_dir"] = str(self.out_dir) if self.out_dir else None
        return d


@dataclass
class BtTrade:
    symbol: str
    side: str
    entry_type: str
    entry_price: float
    stop_loss: float
    take_profit: float
    quantity: float
    leverage: int
    risk_usd: float
    opened_at: datetime
    closed_at: datetime | None = None
    close_price: float | None = None
    close_reason: str | None = None
    pnl_usd: float = 0.0
    pnl_pct: float = 0.0
    fees_usd: float = 0.0
    llm_confidence: float = 0.0
    invalidation_signal: str = ""

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "entry_type": self.entry_type,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "quantity": self.quantity,
            "leverage": self.leverage,
            "risk_usd": self.risk_usd,
            "opened_at": self.opened_at.isoformat(),
            "closed_at": self.closed_at.isoformat() if self.closed_at else None,
            "close_price": self.close_price,
            "close_reason": self.close_reason,
            "pnl_usd": self.pnl_usd,
            "pnl_pct": self.pnl_pct,
            "fees_usd": self.fees_usd,
            "llm_confidence": self.llm_confidence,
            "invalidation_signal": self.invalidation_signal,
        }


@dataclass
class BacktestResult:
    run_id: str
    config: BacktestConfig
    trades: list[BtTrade]
    equity_curve: pd.Series
    decisions: list[dict]
    out_dir: Path
    stats: dict
    monte_carlo: dict
    walk_forward: list[dict]


# ---------------------------------------------------------------------------
# Data loading — paginated Binance fetch into SQLite cache
# ---------------------------------------------------------------------------

def _load_ohlcv_range(
    symbol: str,
    timeframe: str,
    since_dt: datetime,
    until_dt: datetime,
    session_factory=None,
) -> pd.DataFrame:
    """Paginated historical OHLCV fetch with SQLite cache upsert.

    Uses ccxt's `since` parameter to walk forward in time, fetching up to
    1000 candles per request. Cache is populated on every fetch so subsequent
    runs of the same window skip the network.
    """
    from src.data.binance_client import BinanceClient
    from src.persistence.db import upsert_candles, read_cached_candles, make_session_factory

    sf = session_factory or make_session_factory()

    # Check if cache already covers the full range.
    cached = read_cached_candles(sf, symbol, timeframe, limit=50_000)
    if not cached.empty:
        if cached.index.tz is None:
            cached.index = cached.index.tz_localize(timezone.utc)
        mask = (cached.index >= since_dt) & (cached.index <= until_dt)
        if mask.sum() >= 50 and cached.index[mask][0] <= since_dt + timedelta(hours=8):
            log.debug("cache hit for %s %s [%s … %s]", symbol, timeframe, since_dt, until_dt)
            return cached[mask]

    client = BinanceClient(session_factory=sf)
    exchange = client.exchange

    since_ms = int(since_dt.timestamp() * 1000)
    until_ms = int(until_dt.timestamp() * 1000)
    tf_ms = _tf_to_ms(timeframe)
    batch = 1000

    all_rows: list[list] = []
    cursor = since_ms

    while cursor < until_ms:
        log.debug("fetching %s %s since=%s", symbol, timeframe,
                  datetime.utcfromtimestamp(cursor / 1000).strftime("%Y-%m-%d %H:%M"))
        try:
            candles = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=batch)
        except Exception as exc:  # noqa: BLE001
            log.warning("fetch_ohlcv failed for %s %s: %s — retrying in 5s", symbol, timeframe, exc)
            time.sleep(5)
            try:
                candles = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=batch)
            except Exception as exc2:  # noqa: BLE001
                log.error("fetch_ohlcv failed twice for %s %s: %s", symbol, timeframe, exc2)
                break

        if not candles:
            break
        filtered = [c for c in candles if c[0] < until_ms]
        all_rows.extend(filtered)
        last_ts = int(candles[-1][0])
        if last_ts <= cursor or len(candles) < batch:
            break
        cursor = last_ts + tf_ms
        time.sleep(0.05)  # be polite to the rate limiter

    if not all_rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    df = pd.DataFrame(all_rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.drop_duplicates("ts").sort_values("ts").set_index("ts")

    try:
        upsert_candles(sf, symbol, timeframe, df)
    except Exception as exc:  # noqa: BLE001
        log.warning("cache upsert failed: %s", exc)

    return df


# ---------------------------------------------------------------------------
# Slippage + fee helpers
# ---------------------------------------------------------------------------

def _apply_slippage(price: float, side: str, is_entry: bool, slippage_pct: float) -> float:
    """Slippage always hurts the trader: entry fills worse, exit fills worse."""
    if side == "long":
        factor = 1 + slippage_pct if is_entry else 1 - slippage_pct
    else:
        factor = 1 - slippage_pct if is_entry else 1 + slippage_pct
    return price * factor


def _compute_fees(notional: float, taker_fee_pct: float) -> float:
    return notional * taker_fee_pct


def _compute_pnl(side: str, qty: float, entry: float, exit_: float) -> float:
    if side == "long":
        return qty * (exit_ - entry)
    return qty * (entry - exit_)


# ---------------------------------------------------------------------------
# Veto wrapper — pre-flight always runs; LLM only when use_llm=True
# ---------------------------------------------------------------------------

class _BacktestLLMClient(ClaudeClient):
    """ClaudeClient variant that overrides the system prompt and disables DB logging."""

    def __init__(self, system_prompt_path: Path | None, **kwargs: Any) -> None:
        super().__init__(log_to_db=False, **kwargs)
        if system_prompt_path is not None and system_prompt_path.exists():
            self._system_prompt = system_prompt_path.read_text(encoding="utf-8")


def _bt_veto(
    candidate: CandidateSignal,
    open_positions: int,
    daily_pnl_pct: float,
    consecutive_losses: int,
    config: BacktestConfig,
    llm_client: _BacktestLLMClient | None,
    decisions_out: list[dict],
) -> tuple[VetoResponse, float]:
    """Return (VetoResponse, llm_cost_usd)."""
    ctx = VetoContext()  # neutral context — historical funding/OI not available
    account = VetoAccount(
        open_positions=open_positions,
        daily_pnl_pct=daily_pnl_pct,
        losses_streak=consecutive_losses,
    )

    cost_usd = 0.0
    rule = preflight_check(candidate, ctx, account)
    if rule is not None:
        resp = VetoResponse(
            decision="SKIP",
            confidence=1.0,
            invalidation_signal="hard rule violated",
            reasoning=f"preflight: {rule}",
        )
    elif config.use_llm and llm_client is not None:
        resp = llm_client.veto(candidate=candidate, context=ctx, account=account)
        cost_usd = estimate_cost_usd(
            llm_client.model,
            getattr(resp, "_in_tok", 0),
            getattr(resp, "_out_tok", 0),
        )
    else:
        # Stub: always TAKE so the sizing + risk layers make the real decision.
        resp = VetoResponse(
            decision="TAKE",
            confidence=0.75,
            invalidation_signal="backtest stub — no live signal available",
            reasoning="stub",
        )

    decisions_out.append({
        "ts": candidate.timestamp.isoformat(),
        "symbol": candidate.symbol,
        "side": candidate.side,
        "entry_type": candidate.entry_type,
        "signal_strength": candidate.signal_strength,
        "decision": resp.decision,
        "confidence": resp.confidence,
        "preflight_skip": rule is not None,
        "preflight_rule": rule,
        "cost_usd": cost_usd,
        "reasoning": resp.reasoning,
    })
    return resp, cost_usd


# ---------------------------------------------------------------------------
# Per-symbol replay engine
# ---------------------------------------------------------------------------

class BacktestEngine:
    """Stateful bar-by-bar replay for one symbol."""

    def __init__(
        self,
        symbol: str,
        df_4h: pd.DataFrame,
        df_1h: pd.DataFrame,
        df_15m: pd.DataFrame,
        config: BacktestConfig,
        initial_equity: float,
        llm_client: _BacktestLLMClient | None,
    ) -> None:
        self.symbol = symbol
        self.df_4h = df_4h
        self.df_1h = df_1h
        self.df_15m = df_15m
        self.config = config
        self.equity = initial_equity
        self.llm_client = llm_client

        self.trades: list[BtTrade] = []
        self.open_trade: BtTrade | None = None
        self.decisions: list[dict] = []
        self.equity_points: list[tuple[datetime, float]] = []
        self.consecutive_losses = 0
        self._daily_pnl: dict[date, float] = {}

        # Precompute sorted numpy timestamp arrays for O(log n) slicing.
        self._idx_4h = df_4h.index.values.astype("datetime64[ns]")
        self._idx_1h = df_1h.index.values.astype("datetime64[ns]")

    # ------------------------------------------------------------------

    def run(self) -> tuple[list[BtTrade], pd.Series, list[dict]]:
        start, end = self.config.start, self.config.end

        bars_mask = (self.df_15m.index >= start) & (self.df_15m.index <= end)
        bars = self.df_15m.loc[bars_mask]
        bars_arr = bars.index.values.astype("datetime64[ns]")

        for i, (ts, bar) in enumerate(bars.iterrows()):
            ts_ns = bars_arr[i]
            self._tick(ts, bar, ts_ns, bars, i)
            self.equity_points.append((ts, self.equity))

        # Force-close any open position at end-of-period.
        if self.open_trade is not None and not bars.empty:
            last_ts = bars.index[-1]
            last_close = float(bars.iloc[-1]["close"])
            self._close(last_ts, last_close, "end_of_backtest")

        equity_series = pd.Series(
            [eq for _, eq in self.equity_points],
            index=pd.DatetimeIndex([ts for ts, _ in self.equity_points], tz=timezone.utc),
            name="equity_usd",
        )
        return self.trades, equity_series, self.decisions

    # ------------------------------------------------------------------

    def _tick(
        self,
        ts: datetime,
        bar: pd.Series,
        ts_ns,
        bars: pd.DataFrame,
        bar_idx: int,
    ) -> None:
        # --- check / manage open position ---
        if self.open_trade is not None:
            hold_h = (ts - self.open_trade.opened_at).total_seconds() / 3600
            if hold_h >= settings.max_hold_hours:
                self._close(ts, float(bar["close"]), "max_hold")
                return

            hi, lo = float(bar["high"]), float(bar["low"])
            t = self.open_trade
            if t.side == "long":
                if lo <= t.stop_loss:
                    self._close(ts, t.stop_loss, "stop_loss")
                elif hi >= t.take_profit:
                    self._close(ts, t.take_profit, "take_profit")
            else:
                if hi >= t.stop_loss:
                    self._close(ts, t.stop_loss, "stop_loss")
                elif lo <= t.take_profit:
                    self._close(ts, t.take_profit, "take_profit")
            return  # one position per symbol at a time

        # --- slice historical data up to current bar ---
        n_4h = int(np.searchsorted(self._idx_4h, ts_ns, side="right"))
        n_1h = int(np.searchsorted(self._idx_1h, ts_ns, side="right"))
        n_15m = bar_idx + 1

        if n_4h < _MIN_4H or n_1h < _MIN_1H or n_15m < _MIN_15M:
            return

        slice_4h = self.df_4h.iloc[:n_4h]
        slice_1h = self.df_1h.iloc[:n_1h]
        slice_15m = bars.iloc[:n_15m]

        # --- Layer 1: signal ---
        engine = _get_engine()
        try:
            signal = engine._evaluate(self.symbol, slice_4h, slice_1h, slice_15m)
        except Exception as exc:  # noqa: BLE001
            log.debug("signal engine error at %s: %s", ts, exc)
            return
        if signal is None:
            return

        # Override timestamp so it reflects replay time, not now().
        signal = signal.model_copy(update={"timestamp": ts})

        # --- Layer 2: veto ---
        daily_pnl_pct = self._daily_pnl_pct(ts.date())
        veto, cost = _bt_veto(
            candidate=signal,
            open_positions=1 if self.open_trade else 0,
            daily_pnl_pct=daily_pnl_pct,
            consecutive_losses=self.consecutive_losses,
            config=self.config,
            llm_client=self.llm_client,
            decisions_out=self.decisions,
        )
        if veto.decision != "TAKE":
            return

        # --- Layer 3: sizing ---
        account = AccountState(
            equity=self.equity,
            open_positions=0,
            daily_pnl_pct=daily_pnl_pct,
            consecutive_losses=self.consecutive_losses,
        )
        order = size_position(signal, account, veto)
        if order is None:
            return

        # --- fill with slippage + fee ---
        fill_entry = _apply_slippage(
            order.entry_price, order.side, is_entry=True,
            slippage_pct=self.config.slippage_pct,
        )
        notional = fill_entry * order.quantity
        fee_open = _compute_fees(notional, self.config.taker_fee_pct)
        self.equity -= fee_open

        self.open_trade = BtTrade(
            symbol=self.symbol,
            side=order.side,
            entry_type=order.entry_type,
            entry_price=fill_entry,
            stop_loss=order.stop_loss,
            take_profit=order.take_profit,
            quantity=order.quantity,
            leverage=order.leverage,
            risk_usd=order.risk_usd,
            opened_at=ts,
            fees_usd=fee_open,
            llm_confidence=veto.confidence,
            invalidation_signal=veto.invalidation_signal,
        )
        log.debug("OPEN %s %s @ %.4f  qty=%.6f", self.symbol, order.side, fill_entry, order.quantity)

    # ------------------------------------------------------------------

    def _close(self, ts: datetime, raw_close_price: float, reason: str) -> None:
        t = self.open_trade
        if t is None:
            return

        fill_exit = _apply_slippage(
            raw_close_price, t.side, is_entry=False,
            slippage_pct=self.config.slippage_pct,
        )
        notional_exit = fill_exit * t.quantity
        fee_close = _compute_fees(notional_exit, self.config.taker_fee_pct)
        gross_pnl = _compute_pnl(t.side, t.quantity, t.entry_price, fill_exit)
        net_pnl = gross_pnl - fee_close
        total_fees = t.fees_usd + fee_close

        entry_notional = t.entry_price * t.quantity
        pnl_pct = gross_pnl / entry_notional * 100.0 if entry_notional > 0 else 0.0

        self.equity += net_pnl - t.fees_usd  # entry fee already deducted; add net now

        t.closed_at = ts
        t.close_price = fill_exit
        t.close_reason = reason
        t.pnl_usd = net_pnl
        t.pnl_pct = pnl_pct
        t.fees_usd = total_fees

        self.trades.append(t)
        self.open_trade = None

        if net_pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        d = ts.date()
        self._daily_pnl[d] = self._daily_pnl.get(d, 0.0) + net_pnl
        log.debug("CLOSE %s %s reason=%s exit=%.4f pnl=%.2f", self.symbol, t.side, reason, fill_exit, net_pnl)

    def _daily_pnl_pct(self, today: date) -> float:
        pnl = self._daily_pnl.get(today, 0.0)
        return (pnl / self.equity * 100.0) if self.equity > 0 else 0.0


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _sharpe(returns: np.ndarray, periods_per_year: float = 252.0) -> float:
    if len(returns) < 2:
        return 0.0
    sd = float(np.std(returns, ddof=1))
    return float(np.mean(returns) / sd * np.sqrt(periods_per_year)) if sd > 0 else 0.0


def _sortino(equity: pd.Series, periods_per_year: float = 365.0) -> float:
    if len(equity) < 2:
        return 0.0
    daily = equity.resample("1D").last().dropna()
    rets = daily.pct_change().dropna().values
    if len(rets) < 2:
        return 0.0
    downside = rets[rets < 0]
    if len(downside) == 0:
        return float("inf")
    dsd = float(np.std(downside, ddof=1))
    return float(np.mean(rets) / dsd * np.sqrt(periods_per_year)) if dsd > 0 else 0.0


def _max_dd_pct(equity: pd.Series) -> float:
    if len(equity) == 0:
        return 0.0
    peak = equity.cummax()
    dd = (equity - peak) / peak
    return float(dd.min() * 100.0)


def compute_stats(
    trades: list[BtTrade],
    equity: pd.Series,
    initial_equity: float,
    decisions: list[dict],
) -> dict:
    closed = [t for t in trades if t.closed_at is not None]
    n = len(closed)
    wins = [t for t in closed if t.pnl_usd > 0]
    losses = [t for t in closed if t.pnl_usd <= 0]

    total_pnl = sum(t.pnl_usd for t in closed)
    total_fees = sum(t.fees_usd for t in closed)
    total_notional = sum(t.entry_price * t.quantity for t in closed)
    final_equity = float(equity.iloc[-1]) if not equity.empty else initial_equity

    gross_wins = sum(t.pnl_usd for t in wins)
    gross_losses = abs(sum(t.pnl_usd for t in losses))
    profit_factor = (gross_wins / gross_losses) if gross_losses > 0 else float("inf")

    pnl_arr = np.array([t.pnl_usd for t in closed], dtype=float)
    pnl_pct_arr = np.array([t.pnl_pct for t in closed], dtype=float)

    hold_hours = [
        (t.closed_at - t.opened_at).total_seconds() / 3600
        for t in closed
        if t.closed_at and t.opened_at
    ]

    llm_cost = sum(d.get("cost_usd", 0.0) for d in decisions)
    llm_calls = sum(1 for d in decisions if not d.get("preflight_skip", False))

    return {
        "n_trades": n,
        "n_wins": len(wins),
        "n_losses": len(losses),
        "win_rate_pct": len(wins) / n * 100 if n else 0.0,
        "total_pnl_usd": round(total_pnl, 4),
        "total_return_pct": round((final_equity - initial_equity) / initial_equity * 100, 4),
        "final_equity_usd": round(final_equity, 2),
        "sharpe": round(_sharpe(pnl_arr), 4),
        "sortino": round(_sortino(equity), 4),
        "max_dd_pct": round(_max_dd_pct(equity), 4),
        "profit_factor": round(profit_factor, 4) if profit_factor != float("inf") else None,
        "avg_win_usd": round(float(np.mean([t.pnl_usd for t in wins])), 4) if wins else 0.0,
        "avg_loss_usd": round(float(np.mean([t.pnl_usd for t in losses])), 4) if losses else 0.0,
        "avg_hold_hours": round(float(np.mean(hold_hours)), 2) if hold_hours else 0.0,
        "total_fees_usd": round(total_fees, 4),
        "fees_pct": round(total_fees / total_notional * 100, 4) if total_notional > 0 else 0.0,
        "llm_cost_usd": round(llm_cost, 6),
        "llm_calls": llm_calls,
    }


# ---------------------------------------------------------------------------
# Monte Carlo — resample trade returns N times
# ---------------------------------------------------------------------------

def monte_carlo(
    trades: list[BtTrade],
    initial_equity: float,
    n_runs: int = 1000,
) -> dict:
    closed = [t for t in trades if t.closed_at is not None]
    if len(closed) < 2 or n_runs == 0:
        return {}

    pnl_arr = np.array([t.pnl_usd for t in closed], dtype=float)
    n = len(pnl_arr)

    total_returns: list[float] = []
    sharpes: list[float] = []
    max_dds: list[float] = []

    rng = np.random.default_rng(42)
    for _ in range(n_runs):
        sample = rng.choice(pnl_arr, size=n, replace=True)
        cum = np.cumsum(sample) + initial_equity
        eq = pd.Series(cum)
        total_returns.append((float(cum[-1]) - initial_equity) / initial_equity * 100)
        sharpes.append(_sharpe(sample))
        peak = eq.cummax()
        dd = ((eq - peak) / peak).min() * 100
        max_dds.append(float(dd))

    def _pct(arr: list[float], q: int) -> float:
        return round(float(np.percentile(arr, q)), 4)

    return {
        "n_runs": n_runs,
        "return_p5": _pct(total_returns, 5),
        "return_p50": _pct(total_returns, 50),
        "return_p95": _pct(total_returns, 95),
        "sharpe_p5": _pct(sharpes, 5),
        "sharpe_p50": _pct(sharpes, 50),
        "sharpe_p95": _pct(sharpes, 95),
        "max_dd_p50": _pct(max_dds, 50),
        "max_dd_p95": _pct(max_dds, 95),
    }


# ---------------------------------------------------------------------------
# Walk-forward — run independent folds over sub-periods
# ---------------------------------------------------------------------------

def walk_forward(
    ohlcv_data: dict[str, dict[str, pd.DataFrame]],
    config: BacktestConfig,
    llm_client: _BacktestLLMClient | None,
) -> list[dict]:
    if config.wf_folds < 2:
        return []

    duration = config.end - config.start
    fold_len = duration / config.wf_folds
    results: list[dict] = []

    for k in range(config.wf_folds):
        fold_start = config.start + k * fold_len
        fold_end = fold_start + fold_len
        fold_cfg = replace(config, start=fold_start, end=fold_end, wf_folds=0, mc_runs=0)

        fold_trades: list[BtTrade] = []
        fold_equity_pts: list[float] = [fold_cfg.equity_usd]
        fold_decisions: list[dict] = []

        for sym in config.symbols:
            if sym not in ohlcv_data:
                continue
            d = ohlcv_data[sym]
            eng = BacktestEngine(
                symbol=sym,
                df_4h=d["4h"],
                df_1h=d["1h"],
                df_15m=d["15m"],
                config=fold_cfg,
                initial_equity=fold_cfg.equity_usd,
                llm_client=llm_client,
            )
            t, eq, dec = eng.run()
            fold_trades.extend(t)
            fold_decisions.extend(dec)
            if not eq.empty:
                fold_equity_pts.append(float(eq.iloc[-1]))

        fold_eq = pd.Series(fold_equity_pts)
        stats = compute_stats(fold_trades, fold_eq, fold_cfg.equity_usd, fold_decisions)
        results.append({
            "fold": k,
            "start": fold_start.isoformat(),
            "end": fold_end.isoformat(),
            **stats,
        })
        log.info(
            "walk-forward fold %d/%d [%s … %s]: n=%d return=%.2f%%",
            k + 1, config.wf_folds,
            fold_start.strftime("%Y-%m-%d"), fold_end.strftime("%Y-%m-%d"),
            stats["n_trades"], stats["total_return_pct"],
        )

    return results


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _save_symbol(
    sym: str,
    trades: list[BtTrade],
    equity: pd.Series,
    decisions: list[dict],
    out_dir: Path,
) -> None:
    safe_sym = sym.replace("/", "-").replace(":", "-")
    sym_dir = out_dir / safe_sym
    sym_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([t.to_dict() for t in trades]).to_csv(sym_dir / "trades.csv", index=False)
    equity.reset_index().rename(columns={"index": "ts", 0: "equity_usd"}).to_csv(
        sym_dir / "equity_curve.csv", index=False
    )
    pd.DataFrame(decisions).to_csv(sym_dir / "ai_decisions.csv", index=False)


def _make_run_dir(base: Path) -> tuple[Path, str]:
    run_id = datetime.now(tz=timezone.utc).strftime("run-%Y%m%d-%H%M%S")
    run_dir = base / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, run_id


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_backtest(config: BacktestConfig) -> BacktestResult:
    """Execute full backtest; return result with saved files."""
    base_dir = config.out_dir or (settings.project_root / "data-backtest")
    run_dir, run_id = _make_run_dir(base_dir)
    log.info("backtest run_id=%s  run_dir=%s", run_id, run_dir)

    llm_client: _BacktestLLMClient | None = None
    if config.use_llm:
        llm_client = _BacktestLLMClient(system_prompt_path=config.system_prompt_path)

    # --- Load OHLCV for all symbols (+ warmup) ---
    warmup_delta = timedelta(days=40)  # enough for _WARMUP_4H_BARS × 4h
    fetch_since = config.start - warmup_delta

    ohlcv_data: dict[str, dict[str, pd.DataFrame]] = {}
    for sym in config.symbols:
        log.info("loading OHLCV for %s …", sym)
        ohlcv_data[sym] = {
            "4h": _load_ohlcv_range(sym, "4h", fetch_since, config.end),
            "1h": _load_ohlcv_range(sym, "1h", fetch_since, config.end),
            "15m": _load_ohlcv_range(sym, "15m", fetch_since, config.end),
        }
        for tf, df in ohlcv_data[sym].items():
            log.info("  %s %s: %d bars", sym, tf, len(df))

    # --- Replay ---
    all_trades: list[BtTrade] = []
    all_decisions: list[dict] = []
    equity_by_sym: dict[str, pd.Series] = {}
    equity = config.equity_usd

    for sym in config.symbols:
        d = ohlcv_data[sym]
        eng = BacktestEngine(
            symbol=sym,
            df_4h=d["4h"],
            df_1h=d["1h"],
            df_15m=d["15m"],
            config=config,
            initial_equity=equity,
            llm_client=llm_client,
        )
        t, eq, dec = eng.run()
        all_trades.extend(t)
        all_decisions.extend(dec)
        equity_by_sym[sym] = eq
        if not eq.empty:
            equity = float(eq.iloc[-1])
        _save_symbol(sym, t, eq, dec, run_dir)

    # Aggregate equity curve: concatenate and sort.
    all_eq_parts = [eq for eq in equity_by_sym.values() if not eq.empty]
    if all_eq_parts:
        aggregate_equity = pd.concat(all_eq_parts).sort_index()
    else:
        aggregate_equity = pd.Series(
            [config.equity_usd],
            index=pd.DatetimeIndex([config.start], tz=timezone.utc),
            name="equity_usd",
        )

    # --- Stats ---
    stats = compute_stats(all_trades, aggregate_equity, config.equity_usd, all_decisions)
    mc = monte_carlo(all_trades, config.equity_usd, config.mc_runs)
    wf = walk_forward(ohlcv_data, config, llm_client) if config.wf_folds >= 2 else []

    # --- Save summary ---
    summary = {
        "run_id": run_id,
        "aggregate_stats": stats,
        "monte_carlo": mc,
        "walk_forward": wf,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (run_dir / "config.json").write_text(json.dumps(config.to_json(), indent=2), encoding="utf-8")

    log.info(
        "backtest done: n_trades=%d  return=%.2f%%  sharpe=%.2f  max_dd=%.2f%%  fees=$%.2f  llm_cost=$%.4f",
        stats["n_trades"], stats["total_return_pct"],
        stats["sharpe"], stats["max_dd_pct"],
        stats["total_fees_usd"], stats["llm_cost_usd"],
    )

    return BacktestResult(
        run_id=run_id,
        config=config,
        trades=all_trades,
        equity_curve=aggregate_equity,
        decisions=all_decisions,
        out_dir=run_dir,
        stats=stats,
        monte_carlo=mc,
        walk_forward=wf,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_dt(s: str) -> datetime:
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    raise argparse.ArgumentTypeError(f"Cannot parse date: {s}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ai-trader backtest — replay live pipeline on historical data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--symbols", nargs="+", required=True, metavar="SYM",
                        help="e.g. ETH/USDT:USDT SOL/USDT:USDT")
    parser.add_argument("--start", type=_parse_dt, required=True, metavar="YYYY-MM-DD")
    parser.add_argument("--end", type=_parse_dt, required=True, metavar="YYYY-MM-DD")
    parser.add_argument("--equity", type=float, default=10_000.0)
    parser.add_argument("--slippage-pct", type=float, default=0.0005,
                        help="Fraction applied per fill leg (0.0005 = 0.05%%)")
    parser.add_argument("--use-llm", action="store_true",
                        help="Enable real Claude API calls for veto (costs money)")
    parser.add_argument("--system-prompt", type=Path, default=None, metavar="FILE",
                        help="Override system prompt file")
    parser.add_argument("--mc-runs", type=int, default=1000,
                        help="Monte Carlo resampling runs (0 to skip)")
    parser.add_argument("--wf-folds", type=int, default=0,
                        help="Walk-forward folds (0 to skip)")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Base output directory (default: data-backtest/)")
    parser.add_argument("--log-level", default=settings.log_level)
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = BacktestConfig(
        symbols=args.symbols,
        start=args.start,
        end=args.end,
        equity_usd=args.equity,
        slippage_pct=args.slippage_pct,
        use_llm=args.use_llm,
        system_prompt_path=args.system_prompt,
        out_dir=args.out_dir,
        mc_runs=args.mc_runs,
        wf_folds=args.wf_folds,
    )

    result = run_backtest(config)

    print("\n── Backtest results ──────────────────────────────────")
    print(f"Run ID   : {result.run_id}")
    print(f"Output   : {result.out_dir}")
    for k, v in result.stats.items():
        print(f"  {k:<25} {v}")
    if result.monte_carlo:
        print("\n── Monte Carlo (trade-level resample) ───────────────")
        for k, v in result.monte_carlo.items():
            print(f"  {k:<25} {v}")
    if result.walk_forward:
        print("\n── Walk-forward folds ────────────────────────────────")
        for fold in result.walk_forward:
            print(f"  fold={fold['fold']}  [{fold['start'][:10]} … {fold['end'][:10]}]"
                  f"  n={fold['n_trades']}  ret={fold['total_return_pct']:.2f}%")


if __name__ == "__main__":
    main()
