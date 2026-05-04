# ai-trader

AI-augmented trading agent for **Binance USDT-M Perpetual Futures**.
A deterministic three-layer pipeline:

1. A rule engine produces candidate trades.
2. A Claude LLM acts as a veto layer (TAKE / SKIP, never picks the trade).
3. A risk + execution layer applies hard limits and places bracket orders.

> **Status: skeleton / scaffold.** Fit for backtests, dry-run, and testnet
> experimentation. Not production-tested. Read the [disclaimer](#disclaimer)
> before pointing it at real funds.

---

## Architecture

```
                   ┌────────────────────────────────────────────────┐
                   │           Binance USDT-M Futures               │
                   │   (mainnet / testnet — selected by .env flag)  │
                   └───────────────┬────────────────────────────────┘
                                   │  ccxt + tenacity retry
                                   ▼
            ┌─────────────────────────────────────────────────┐
            │  data/binance_client.py     — OHLCV cache (DB)  │
            │  data/indicators.py         — EMA / RSI / ATR / │
            │                               MACD / swings     │
            │  data/context.py            — funding, OI,      │
            │                               liquidations      │
            └───────────────┬─────────────────────────────────┘
                            │
            ┌───────────────▼─────────────────────────────────┐
            │  Layer 1 · SignalEngine  (rules, deterministic) │
            │  4h trend  →  1h structure  →  15m trigger      │
            │  Types A / B / C           →  CandidateSignal   │
            └───────────────┬─────────────────────────────────┘
                            │
            ┌───────────────▼─────────────────────────────────┐
            │  Layer 2 · Veto                                 │
            │    a) preflight_check  (hard rules locally)     │
            │    b) ClaudeClient.veto  → TAKE / SKIP + JSON   │
            │  Every call is logged to ai_decisions table.    │
            └───────────────┬─────────────────────────────────┘
                            │ TAKE
            ┌───────────────▼─────────────────────────────────┐
            │  Layer 3 · Risk + Execution                     │
            │    risk.limits.can_take_signal  (gates)         │
            │    risk.sizing.size_position    (ATR-based)     │
            │    risk.safety_mode             (NoFx pattern)  │
            │    execution.orders             (paper / live)  │
            │    execution.monitor            (SL/TP/timeout) │
            └───────────────┬─────────────────────────────────┘
                            │
            ┌───────────────▼─────────────────────────────────┐
            │  Persistence (SQLAlchemy / SQLite)              │
            │    trades · ai_decisions · equity_snapshots ·   │
            │    positions · safety_state · ohlcv_candles     │
            └───────────────┬─────────────────────────────────┘
                            │
                ┌───────────┴───────────┐
                ▼                       ▼
    ┌───────────────────┐   ┌─────────────────────────┐
    │  Telegram alerts  │   │  Streamlit dashboard    │
    │  ENTRY / CLOSE /  │   │  Overview · Trades ·    │
    │  iter summary /   │   │  AI Decisions ·         │
    │  CRITICAL         │   │  Positions · Settings   │
    └───────────────────┘   └─────────────────────────┘
```

### Modules

| Path              | Responsibility                                                   |
| ----------------- | ---------------------------------------------------------------- |
| `src/config/`     | typed settings (pydantic-settings), trading universe (YAML)      |
| `src/data/`       | ccxt wrapper with cache, indicators, market context              |
| `src/signal/`     | rule engine (trend, structure, execution, types)                 |
| `src/llm/`        | Claude client, prompt templates, pre-flight + veto layer         |
| `src/risk/`       | sizing (ATR), hard limits, NoFx safety mode                      |
| `src/execution/`  | paper / live order placement, bracket logic, position monitor    |
| `src/persistence/`| SQLAlchemy models, CSV mirrors, `export_to_csv()`                |
| `src/notify/`     | Telegram alerts (entry / close / iteration summary / critical)   |
| `src/dashboard/`  | Streamlit multi-page dashboard                                   |
| `src/backtest.py` | full-fidelity historical replay through the same pipeline        |
| `tests/`          | pytest unit tests                                                |
| `scripts/`        | smoke test, prompt benchmark, state recalculation                |

---

## Quickstart

Requires **Python 3.12** and [Poetry](https://python-poetry.org/).

```bash
git clone <repo-url> ai-trader && cd ai-trader
poetry install
cp .env.example .env
# edit .env — at minimum set BINANCE_API_KEY/SECRET (testnet keys are fine)
#                            ANTHROPIC_API_KEY
#                            TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID (optional)
```

Verify the install:

```bash
poetry run pytest
poetry run python -c "from src.config.settings import settings; print(settings.trading_mode)"
```

Or via Docker:

```bash
docker build -t ai-trader .
docker run --rm --env-file .env -v $(pwd)/data:/app/data ai-trader
```

The defaults in `.env.example` are paper-mode + testnet + dry-run, so a fresh
clone is harmless to run.

---

## Backtest workflow

A backtest replays historical OHLCV through the **exact same code paths** as
live trading — only the data source and the order executor are swapped out.

```bash
# 1. Run a backtest over a chosen window
poetry run python -m src.backtest \
    --symbols ETH/USDT:USDT SOL/USDT:USDT \
    --start 2025-10-01 --end 2025-10-08 \
    --equity 10000 \
    --slippage-pct 0.0005 \
    --mc-runs 1000 \
    --wf-folds 4

# 2. Output lands in data-backtest/run-<ts>/
#    ├── BTC-USDT-USDT/
#    │   ├── trades.csv
#    │   ├── equity_curve.csv
#    │   └── ai_decisions.csv
#    ├── summary.json     ← aggregate stats + Monte Carlo + walk-forward
#    └── config.json
```

Useful flags:

- `--use-llm` — call the real Claude API for veto (costs money; otherwise the
  veto is stubbed to TAKE so the rule engine + risk layer drive the run).
- `--system-prompt path/to/file.txt` — override the default veto prompt.
- `--mc-runs 0` / `--wf-folds 0` — skip Monte Carlo / walk-forward.

### Comparing prompt variants

```bash
poetry run python scripts/benchmark_prompts.py \
    --prompts src/llm/prompts/system.txt prompts/aggressive.txt prompts/conservative.txt \
    --symbols ETH/USDT:USDT \
    --start 2025-10-01 --end 2025-10-08 \
    --use-llm \
    --mc-runs 500
```

Each variant runs over the same pre-loaded data; results are written to
`results/benchmark_<ts>.csv` plus a comparison chart `benchmark_<ts>.png`
(equity curves + bar charts of return / Sharpe / Max DD / fees / LLM cost).

A **baseline (stub veto)** run is always included so you can isolate the
contribution of the LLM from the rule layer.

### Stats reported

- `total_return_pct`, `final_equity_usd`
- `sharpe` (per-trade), `sortino` (equity-curve, downside only)
- `max_dd_pct` (peak-to-trough)
- `win_rate_pct`, `profit_factor`, `avg_win_usd`, `avg_loss_usd`
- `avg_hold_hours`, `total_fees_usd`, `fees_pct`
- `llm_cost_usd`, `llm_calls`
- Monte Carlo: `return_p5 / p50 / p95`, `sharpe_p5 / p50 / p95`, `max_dd_p95`
- Walk-forward: same metrics for each independent fold

---

## Live deployment

### 1. Smoke test on testnet (~$2 round-trip)

Before pointing the bot at any equity, run the end-to-end smoke test. It
opens a tiny BTC long at 1× leverage, holds for 30 s, closes with a
reduce-only market order, and prints latencies / slippage / fees / errors.

```bash
# .env: BINANCE_TESTNET=true, BINANCE_API_KEY/SECRET = testnet creds
poetry run python scripts/smoke_test.py --symbol BTC/USDT:USDT --notional 2 --hold-seconds 30
```

The script refuses to start unless `BINANCE_TESTNET=true`. It exits non-zero
on any leg failure, so it's safe to wire into CI.

### 2. Paper trading (24h burn-in)

```bash
# .env: TRADING_MODE=paper, PAPER_TRADING=true, DRY_RUN=true
poetry run python -m src.main
```

In paper mode the executor records trades to SQLite without sending anything
to Binance. Run it for 24 h and verify:

- `data/trades.db` keeps growing (open the dashboard or query directly)
- Telegram messages arrive on entry / close / iteration summary
- No `ERROR` lines in `LOG_LEVEL=INFO` output
- No safety-mode triggers from spurious losses

### 3. Live (with real money)

```bash
# .env: TRADING_MODE=live, PAPER_TRADING=false, BINANCE_TESTNET=false
#       EQUITY_USD set to your actual deposit, leverage / risk knobs reviewed
poetry run python -m src.main
```

Recommendations:

- Run inside `tmux` / `systemd` / `docker run --restart=always`.
- Monitor `data/trades.db` via `streamlit run src/dashboard/app.py`.
- Subscribe to `TELEGRAM_ENABLED=true` alerts.
- Keep `MAX_DAILY_LOSS_PCT` and `SAFETY_MAX_CONSECUTIVE_LOSSES` conservative.

### 4. Dashboard (read-only operator UI)

```bash
poetry run streamlit run src/dashboard/app.py
```

Five pages: equity curve vs BTC buy & hold (Sharpe / Sortino / Max DD),
trades table with filters, AI decision log with raw JSON viewer + cost-by-day
chart, live positions, read-only settings.

---

## Configuration

All knobs live in `.env` (see `.env.example`). Key defaults:

| Variable                          | Default      | Purpose                                  |
| --------------------------------- | ------------ | ---------------------------------------- |
| `TRADING_MODE`                    | `paper`      | `paper` \| `live`                        |
| `BINANCE_TESTNET`                 | `true`       | use Binance testnet                      |
| `PAPER_TRADING`                   | `true`       | simulate fills locally                   |
| `DRY_RUN`                         | `true`       | refuse real-money side effects           |
| `EQUITY_USD`                      | `1000`       | starting equity / sizing reference       |
| `LEVERAGE`                        | `3`          | requested per-position leverage          |
| `RISK_PER_TRADE_TYPE_A/B/C`       | `0.02 / 0.01 / 0.01` | risk per trade by setup type    |
| `MAX_OPEN_POSITIONS`              | `2`          | hard cap on concurrent positions         |
| `MAX_DAILY_LOSS_PCT`              | `3.0`        | shut off the bot for the day             |
| `SAFETY_MAX_CONSECUTIVE_LOSSES`   | `3`          | NoFx pause threshold                     |
| `SAFETY_PAUSE_HOURS`              | `24`         | how long the pause lasts                 |
| `MIN_CONFIDENCE`                  | `0.7`        | reject veto < this confidence            |
| `TAKER_FEE_PCT`                   | `0.0004`     | matched to Binance USDT-M takers (0.04 %)|

The trading universe lives in `src/config/symbols.yaml`.

---

## Disclaimer

This software is provided **for educational and research purposes only**.
Trading cryptocurrency derivatives carries substantial risk of loss.
**Past performance — backtested or live — is not indicative of future
results.** The authors accept no liability for losses incurred through use of
this code.

You are responsible for:

- Securing your API keys. Use **trade scope only** — *never* enable the
  withdrawal scope.
- Validating every code change on **testnet** before pointing it at real funds.
- Understanding the rules, the LLM prompts, and the risk limits that govern
  every order this bot may submit on your behalf.
- The legal status of automated futures trading in your jurisdiction.

If you do not fully understand what a line of this code does, **do not run it
with real money**.
