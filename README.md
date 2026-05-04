# ai-trader

AI-augmented trading agent for **Binance USDT-M Perpetual Futures**. A
deterministic rule engine produces candidate trades; a Claude LLM acts as a
veto layer; a third execution layer applies hard risk limits and places
bracket orders.

> **Status: skeleton / scaffold.** Suitable for backtests and dry-run /
> testnet experimentation. Not production-tested. Read the disclaimer.

## Architecture

```
┌──────────────────────────┐
│  Layer 1: SignalEngine   │  4h trend → 1h structure → 15m trigger
│  (rules, deterministic)  │  → CandidateSignal {entry, stop, take}
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│  Layer 2: VetoAgent      │  Claude reviews candidate + market context
│  (LLM, JSON-only)        │  → ALLOW / REJECT
└────────────┬─────────────┘
             │ ALLOW
             ▼
┌──────────────────────────┐
│  Layer 3: Trader         │  ATR-sized position, hard risk gates,
│  (sizing + execution)    │  safety mode (NoFx pattern), bracket orders
└──────────────────────────┘
```

### Modules

| Path | Responsibility |
|---|---|
| `src/config/`        | typed settings (pydantic), trading universe |
| `src/data/`          | ccxt wrapper, indicators, market context |
| `src/signal/`        | rule layer (trend, structure, execution, types) |
| `src/llm/`           | Claude client, prompt templates, veto agent |
| `src/risk/`          | sizing, hard limits, NoFx safety mode |
| `src/execution/`     | order placement, bracket logic, position monitor |
| `src/persistence/`   | SQLite (trades, decisions, equity) + CSV mirrors |
| `src/notify/`        | Telegram alerts |
| `src/dashboard/`     | Streamlit UI |
| `tests/`             | pytest unit tests |
| `scripts/`           | smoke test, reconciliation, prompt benchmark |

## Setup

Requires **Python 3.12** and [Poetry](https://python-poetry.org/).

```bash
poetry install
cp .env.example .env
# fill in BINANCE_API_KEY, BINANCE_API_SECRET, ANTHROPIC_API_KEY
```

Verify the install:

```bash
poetry run python -c "from src.config.settings import settings; print(settings.trading_mode)"
poetry run pytest
```

Or via Docker:

```bash
docker build -t ai-trader .
docker run --rm --env-file .env -v $(pwd)/data:/app/data ai-trader
```

## Usage

**Live / paper loop:**

```bash
poetry run python -m src.main
```

**Backtest a symbol:**

```bash
poetry run python -m src.backtest --symbol BTC/USDT:USDT --bars 1500
```

**Dashboard:**

```bash
poetry run streamlit run src/dashboard/app.py
```

**Testnet smoke test (~$2 micro-order):**

```bash
poetry run python scripts/smoke_test.py --symbol BTC/USDT:USDT
```

## Configuration

All knobs live in `.env` (see `.env.example`). Key defaults:

- `TRADING_MODE=paper`, `DRY_RUN=true`, `BINANCE_TESTNET=true` — safe by default.
- `RISK_PER_TRADE_PCT=0.5`, `MAX_OPEN_POSITIONS=3`, `MAX_DAILY_LOSS_PCT=3.0`.
- `SAFETY_MAX_CONSECUTIVE_LOSSES=3`, `SAFETY_PAUSE_HOURS=12` — NoFx pattern.

The trading universe lives in `src/config/symbols.yaml`.

## Disclaimer

This software is provided **for educational and research purposes only**.
Trading cryptocurrency derivatives carries substantial risk of loss. Past
performance — backtested or live — is not indicative of future results. The
authors accept no liability for losses incurred through use of this code.

You are responsible for:

- Securing your API keys (use **read + trade** scopes only; never enable
  withdrawal scope).
- Validating every code change on **testnet** before pointing it at real funds.
- Understanding the rules, the LLM prompts, and the risk limits that govern
  every trade this bot may submit on your behalf.

If you don't fully understand what a line of this code does, do not run it
with real money.
