"""Backtest harness.

Replays historical OHLCV through the rule layer (Layer 1) and an optional
LLM veto stub (Layer 2). Sizes, fills and PnL are computed deterministically
so the same data + same code produces identical results.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from src.config import settings
from src.data.binance_client import BinanceClient
from src.data.indicators import atr14
from src.signal.execution import find_trigger
from src.signal.structure import find_structure
from src.signal.trend import detect_bias
from src.signal.types import Side


@dataclass
class BacktestStats:
    trades: int
    wins: int
    losses: int
    pnl_usd: float

    def winrate(self) -> float:
        return self.wins / self.trades if self.trades else 0.0


def _iter_windows(df: pd.DataFrame, warmup: int) -> Iterable[int]:
    for i in range(warmup, len(df)):
        yield i


def run_backtest(symbol: str, lookback_bars: int = 1500) -> BacktestStats:
    client = BinanceClient()
    df_4h = client.fetch_ohlcv(symbol, settings.timeframe_trend, limit=lookback_bars // 8)
    df_1h = client.fetch_ohlcv(symbol, settings.timeframe_structure, limit=lookback_bars // 4)
    df_15 = client.fetch_ohlcv(symbol, settings.timeframe_execution, limit=lookback_bars)
    atr_series = atr14(df_15)

    stats = BacktestStats(0, 0, 0, 0.0)
    in_pos = False
    side: Side | None = None
    entry = stop = take = 0.0

    for i in _iter_windows(df_15, warmup=220):
        slice_4h = df_4h[df_4h.index <= df_15.index[i]]
        slice_1h = df_1h[df_1h.index <= df_15.index[i]]
        slice_15 = df_15.iloc[: i + 1]

        if in_pos and side is not None:
            row = df_15.iloc[i]
            hit_stop = row["low"] <= stop if side is Side.LONG else row["high"] >= stop
            hit_take = row["high"] >= take if side is Side.LONG else row["low"] <= take
            if hit_stop or hit_take:
                pnl = (take - entry) if hit_take else (stop - entry)
                pnl = pnl if side is Side.LONG else -pnl
                stats.pnl_usd += pnl
                stats.trades += 1
                stats.wins += int(pnl > 0)
                stats.losses += int(pnl <= 0)
                in_pos = False
                side = None
            continue

        bias = detect_bias(slice_4h)
        structure = find_structure(slice_1h)
        fired, _ = find_trigger(slice_15, bias, structure)
        if not fired:
            continue
        atr = float(atr_series.iloc[i])
        if pd.isna(atr) or atr <= 0:
            continue
        entry = float(slice_15["close"].iloc[-1])
        side = Side.LONG if bias == "long" else Side.SHORT
        if side is Side.LONG:
            stop = entry - settings.atr_stop_multiplier * atr
            take = entry + settings.atr_take_multiplier * atr
        else:
            stop = entry + settings.atr_stop_multiplier * atr
            take = entry - settings.atr_take_multiplier * atr
        in_pos = True

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="ai-trader backtest")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--bars", type=int, default=1500)
    args = parser.parse_args()

    logging.basicConfig(level=settings.log_level)
    stats = run_backtest(args.symbol, args.bars)
    print(
        f"symbol={args.symbol} trades={stats.trades} "
        f"winrate={stats.winrate():.2%} pnl_usd={stats.pnl_usd:.2f}"
    )


if __name__ == "__main__":
    main()
