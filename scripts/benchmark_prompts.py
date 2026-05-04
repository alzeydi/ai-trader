"""Benchmark multiple system prompts over the same historical window.

For each prompt file the script runs a full backtest on the specified window
and collects the key metrics. Results are written to a CSV and visualised
as a matplotlib comparison chart.

Usage
-----
python scripts/benchmark_prompts.py \\
    --prompts src/llm/prompts/system.txt prompts/aggressive.txt \\
    --symbols ETH/USDT:USDT SOL/USDT:USDT \\
    --start 2025-01-01 --end 2025-03-31 \\
    [--equity 10000] [--slippage-pct 0.0005] [--use-llm] \\
    [--mc-runs 500] [--out-dir results]

Output
------
results/benchmark_<ts>.csv
results/benchmark_<ts>.png
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.backtest import (
    BacktestConfig,
    BacktestEngine,
    _BacktestLLMClient,
    _load_ohlcv_range,
    compute_stats,
    monte_carlo,
)
from src.config import settings

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data pre-loading — shared across all prompt variants
# ---------------------------------------------------------------------------

def _preload_ohlcv(
    symbols: list[str],
    start: datetime,
    end: datetime,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Fetch historical OHLCV once and reuse across all prompt variants."""
    since = start - timedelta(days=40)
    data: dict[str, dict[str, pd.DataFrame]] = {}
    for sym in symbols:
        log.info("pre-loading %s …", sym)
        data[sym] = {
            "4h": _load_ohlcv_range(sym, "4h", since, end),
            "1h": _load_ohlcv_range(sym, "1h", since, end),
            "15m": _load_ohlcv_range(sym, "15m", since, end),
        }
        for tf, df in data[sym].items():
            log.info("  %s %s: %d bars", sym, tf, len(df))
    return data


# ---------------------------------------------------------------------------
# Run one prompt variant
# ---------------------------------------------------------------------------

def _run_variant(
    prompt_path: Path | None,
    label: str,
    ohlcv: dict[str, dict[str, pd.DataFrame]],
    config: BacktestConfig,
) -> dict[str, Any]:
    """Run all symbols for one prompt variant; return aggregated stats dict."""
    t0 = time.monotonic()

    llm_client: _BacktestLLMClient | None = None
    if config.use_llm:
        llm_client = _BacktestLLMClient(system_prompt_path=prompt_path)

    all_trades = []
    all_decisions: list[dict] = []
    equity_parts: list[pd.Series] = []
    equity = config.equity_usd

    for sym in config.symbols:
        d = ohlcv[sym]
        eng = BacktestEngine(
            symbol=sym,
            df_4h=d["4h"],
            df_1h=d["1h"],
            df_15m=d["15m"],
            config=config,
            initial_equity=equity,
            llm_client=llm_client,
        )
        trades, eq, decisions = eng.run()
        all_trades.extend(trades)
        all_decisions.extend(decisions)
        if not eq.empty:
            equity_parts.append(eq)
            equity = float(eq.iloc[-1])

    agg_eq: pd.Series
    if equity_parts:
        agg_eq = pd.concat(equity_parts).sort_index()
    else:
        agg_eq = pd.Series(
            [config.equity_usd],
            index=pd.DatetimeIndex([config.start], tz=timezone.utc),
            name="equity_usd",
        )

    stats = compute_stats(all_trades, agg_eq, config.equity_usd, all_decisions)
    mc = monte_carlo(all_trades, config.equity_usd, config.mc_runs) if config.mc_runs else {}
    elapsed = time.monotonic() - t0

    log.info(
        "variant %-30s → n=%d  ret=%.2f%%  sharpe=%.2f  max_dd=%.2f%%  "
        "fees=$%.2f  llm_cost=$%.4f  %.1fs",
        label, stats["n_trades"], stats["total_return_pct"],
        stats["sharpe"], stats["max_dd_pct"],
        stats["total_fees_usd"], stats["llm_cost_usd"], elapsed,
    )

    return {
        "label": label,
        "prompt_path": str(prompt_path) if prompt_path else "(stub)",
        **stats,
        "mc_return_p5": mc.get("return_p5"),
        "mc_return_p50": mc.get("return_p50"),
        "mc_return_p95": mc.get("return_p95"),
        "mc_sharpe_p50": mc.get("sharpe_p50"),
        "mc_max_dd_p95": mc.get("max_dd_p95"),
        "elapsed_s": round(elapsed, 1),
        "_equity": agg_eq,   # kept for chart; stripped from CSV
    }


# ---------------------------------------------------------------------------
# Comparison chart
# ---------------------------------------------------------------------------

def _build_chart(rows: list[dict], out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        log.warning("matplotlib not installed — skipping chart")
        return

    bar_metrics = [
        ("total_return_pct", "Total return (%)"),
        ("sharpe", "Sharpe"),
        ("max_dd_pct", "Max DD % ↓"),
        ("n_trades", "# Trades"),
        ("win_rate_pct", "Win rate (%)"),
        ("profit_factor", "Profit factor"),
        ("llm_cost_usd", "LLM cost ($)"),
        ("total_fees_usd", "Fees ($)"),
    ]

    labels = [r["label"] for r in rows]
    n_bar = len(bar_metrics)
    n_bar_cols = 4
    n_bar_rows = (n_bar + n_bar_cols - 1) // n_bar_cols

    fig = plt.figure(figsize=(6 * n_bar_cols, 4 + 3 * n_bar_rows))
    fig.suptitle("Prompt benchmark comparison", fontsize=14, fontweight="bold", y=1.01)
    gs = gridspec.GridSpec(1 + n_bar_rows, n_bar_cols, figure=fig, hspace=0.6, wspace=0.4)

    # --- equity curves row ---
    ax_eq = fig.add_subplot(gs[0, :])
    for r in rows:
        eq: pd.Series = r.get("_equity", pd.Series(dtype=float))
        if not eq.empty:
            norm = eq / float(eq.iloc[0]) * 100
            ax_eq.plot(norm.index, norm.values, label=r["label"], linewidth=1.5)
    ax_eq.set_title("Normalised equity curves (start = 100)", fontsize=10)
    ax_eq.set_ylabel("Equity index")
    ax_eq.legend(fontsize=8, ncol=min(len(rows), 4))
    ax_eq.grid(alpha=0.3)

    # --- metric bar charts ---
    for mi, (col, title) in enumerate(bar_metrics):
        ri = 1 + mi // n_bar_cols
        ci = mi % n_bar_cols
        ax = fig.add_subplot(gs[ri, ci])
        vals = [float(r.get(col) or 0.0) for r in rows]
        colours = ["#2ecc71" if v >= 0 else "#e74c3c" for v in vals]
        ax.bar(range(len(labels)), vals, color=colours, edgecolor="white", linewidth=0.4)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=7, rotation=30, ha="right")
        ax.set_title(title, fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    log.info("chart saved → %s", out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_dt(s: str) -> datetime:
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    raise argparse.ArgumentTypeError(f"Cannot parse date: {s!r}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark system prompts on the same historical backtest window",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--prompts", nargs="+", required=True, type=Path, metavar="FILE",
                        help="System prompt .txt files to compare")
    parser.add_argument("--symbols", nargs="+", required=True, metavar="SYM")
    parser.add_argument("--start", type=_parse_dt, required=True, metavar="YYYY-MM-DD")
    parser.add_argument("--end",   type=_parse_dt, required=True, metavar="YYYY-MM-DD")
    parser.add_argument("--equity", type=float, default=10_000.0)
    parser.add_argument("--slippage-pct", type=float, default=0.0005)
    parser.add_argument("--use-llm", action="store_true",
                        help="Use Claude API for veto (compares actual prompt behaviour)")
    parser.add_argument("--mc-runs", type=int, default=500,
                        help="Monte Carlo resampling runs per variant (0=skip)")
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d-%H%M%S")

    base_config = BacktestConfig(
        symbols=args.symbols,
        start=args.start,
        end=args.end,
        equity_usd=args.equity,
        slippage_pct=args.slippage_pct,
        use_llm=args.use_llm,
        mc_runs=args.mc_runs,
    )

    # Load data once — reused by all variants.
    log.info("pre-loading OHLCV (shared across all variants) …")
    ohlcv = _preload_ohlcv(args.symbols, args.start, args.end)

    rows: list[dict] = []

    # Baseline: stub veto (signal + risk layers govern entirely).
    stub_config = BacktestConfig(
        symbols=args.symbols,
        start=args.start,
        end=args.end,
        equity_usd=args.equity,
        slippage_pct=args.slippage_pct,
        use_llm=False,
        mc_runs=args.mc_runs,
    )
    log.info("running baseline (stub veto, no LLM) …")
    rows.append(_run_variant(None, "baseline (stub)", ohlcv, stub_config))

    # One run per prompt file.
    for prompt_path in args.prompts:
        if not prompt_path.exists():
            log.warning("prompt file not found: %s — skipping", prompt_path)
            continue
        log.info("running variant: %s …", prompt_path.name)
        rows.append(_run_variant(prompt_path, prompt_path.stem, ohlcv, base_config))

    # --- Build comparison table (strip non-serialisable _equity key) ---
    output_cols = [
        "label", "prompt_path",
        "n_trades", "n_wins", "n_losses", "win_rate_pct",
        "total_pnl_usd", "total_return_pct", "final_equity_usd",
        "sharpe", "sortino", "max_dd_pct",
        "profit_factor", "avg_win_usd", "avg_loss_usd", "avg_hold_hours",
        "total_fees_usd", "fees_pct", "llm_cost_usd", "llm_calls",
        "mc_return_p5", "mc_return_p50", "mc_return_p95",
        "mc_sharpe_p50", "mc_max_dd_p95", "elapsed_s",
    ]
    table = [{c: r.get(c) for c in output_cols} for r in rows]
    df = pd.DataFrame(table, columns=output_cols)

    csv_path = args.out_dir / f"benchmark_{ts}.csv"
    df.to_csv(csv_path, index=False)
    log.info("results table saved → %s", csv_path)

    # --- Print summary ---
    print("\n── Benchmark results ──────────────────────────────────────────────")
    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)
    summary_cols = [
        "label", "n_trades", "win_rate_pct", "total_return_pct",
        "sharpe", "sortino", "max_dd_pct", "profit_factor",
        "total_fees_usd", "llm_cost_usd", "mc_return_p50",
    ]
    print(df[summary_cols].to_string(index=False))
    print(f"\nFull CSV  → {csv_path}")

    # --- Chart ---
    png_path = args.out_dir / f"benchmark_{ts}.png"
    _build_chart(rows, png_path)
    print(f"Chart     → {png_path}")


if __name__ == "__main__":
    main()
