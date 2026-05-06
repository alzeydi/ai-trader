# BACKTEST.md

Детали backtest и benchmark. Подгружай при работе с `src/backtest.py`, `scripts/benchmark_prompts.py`, или когда пользователь упомянул backtest / benchmark / метрики / Monte Carlo / walk-forward.

## Запуск backtest

```bash
poetry run python -m src.backtest \
    --symbols ETH/USDT:USDT SOL/USDT:USDT \
    --start 2025-10-01 --end 2025-10-08 \
    --equity 10000 \
    --slippage-pct 0.0005 \
    --mc-runs 1000 \
    --wf-folds 4
```

### Флаги

- `--use-llm` — реальный Claude API для veto (стоит денег). Без флага veto заглушено в TAKE — прогон демонстрирует вклад rule engine + risk без LLM.
- `--system-prompt path/to/file.txt` — переопределить veto-prompt (по умолчанию `src/llm/prompts/system.txt`).
- `--slippage-pct` — слиппедж в долях (0.0005 = 0.05%).
- `--mc-runs N` — число Monte Carlo прогонов; `0` — пропустить.
- `--wf-folds N` — число walk-forward fold'ов; `0` — пропустить.

### Структура output

```
data-backtest/run-<ts>/
├── <SYMBOL>/
│   ├── trades.csv
│   ├── equity_curve.csv
│   └── ai_decisions.csv
├── summary.json     ← агрегат + Monte Carlo + walk-forward
└── config.json      ← дамп всех CLI/ENV параметров
```

### Метрики (в `summary.json`)

**Базовые:**
- `total_return_pct`, `final_equity_usd`
- `sharpe` (per-trade), `sortino` (equity-curve, только downside)
- `max_dd_pct` (peak-to-trough)
- `win_rate_pct`, `profit_factor`, `avg_win_usd`, `avg_loss_usd`
- `avg_hold_hours`, `total_fees_usd`, `fees_pct`
- `llm_cost_usd`, `llm_calls`

**Monte Carlo** (рандомизация порядка сделок, оценка хвостов):
- `return_p5 / p50 / p95`
- `sharpe_p5 / p50 / p95`
- `max_dd_p95`

**Walk-forward** (последовательные out-of-sample fold'ы):
- те же метрики на каждый fold отдельно

## Benchmark промптов

Сравнение нескольких veto-промптов на одних и тех же исторических данных.

```bash
poetry run python scripts/benchmark_prompts.py \
    --prompts src/llm/prompts/system.txt prompts/aggressive.txt prompts/conservative.txt \
    --symbols ETH/USDT:USDT \
    --start 2025-10-01 --end 2025-10-08 \
    --use-llm \
    --mc-runs 500
```

### Output

- `results/benchmark_<ts>.csv` — таблица метрик по каждому варианту.
- `results/benchmark_<ts>.png` — equity curves + bar charts: return, Sharpe, Max DD, fees, LLM cost.

**Baseline** (stub-veto = всегда TAKE) всегда включается как референс — изолировать вклад LLM от rule engine.
