# DEPLOY.md

Лестница деплоя. Подгружай при подготовке к запуску smoke / paper / live, а также при работе с `scripts/start.sh`, Railway, tmux/systemd.

## Лестница (от безопасного к опасному)

### 1. Backtest

Прогон на исторических данных. Реальные деньги не двигаются. См. `docs/BACKTEST.md`.

### 2. Smoke-тест на testnet (~$2 round-trip)

Проверка end-to-end интеграции с Binance API без рисков для денег.

```bash
# .env: BINANCE_TESTNET=true, BINANCE_API_KEY/SECRET = testnet creds
poetry run python scripts/smoke_test.py --symbol BTC/USDT:USDT --notional 2 --hold-seconds 30
```

Что делает: открывает BTC long на 1× leverage, держит 30 сек, закрывает reduce-only market. Печатает latencies / slippage / fees / errors.

`smoke_test.py` отказывается стартовать без `BINANCE_TESTNET=true`. Ненулевой exit на любой ошибке — можно вешать в CI.

### 3. Paper burn-in (24 часа)

```bash
# .env: TRADING_MODE=paper, PAPER_TRADING=true, DRY_RUN=true
poetry run python -m src.main
```

В paper-режиме executor пишет сделки в SQLite, ничего не отправляя на Binance.

**Что проверять за 24 часа:**
- `data/trades.db` растёт (через дашборд или `sqlite3`).
- Telegram присылает entry / close / итерационную сводку.
- Нет строк `ERROR` при `LOG_LEVEL=INFO`.
- Нет ложных срабатываний safety-mode из-за случайных серий убытков.

### 4. Live (реальные деньги)

```bash
# .env: TRADING_MODE=live, PAPER_TRADING=false, BINANCE_TESTNET=false
#       EQUITY_USD = реальный депозит, leverage / risk-параметры пересмотрены
poetry run python -m src.main
```

**Рекомендации:**
- Запускать внутри `tmux` / `systemd` / `docker run --restart=always` — иначе SIGHUP убьёт бота.
- Мониторить `data/trades.db` через `streamlit run src/dashboard/app.py`.
- Включить Telegram (`TELEGRAM_ENABLED=true`).
- Держать `MAX_DAILY_LOSS_PCT` и `SAFETY_MAX_CONSECUTIVE_LOSSES` консервативно.
- API-ключи Binance: **только trade scope**, никогда withdrawal scope.

## Railway

`scripts/start.sh` запускает торговый цикл (фоном) + Streamlit-дашборд (foreground на `$PORT`). Оба процесса используют один SQLite `/app/data/trades.db`. Конфиг — `railway.toml`.

**Внимание:** `railway.toml` по умолчанию ставит `TRADING_MODE=live`, `PAPER_TRADING=false`, `BINANCE_TESTNET=false` со снятыми лимитами риска (`MAX_OPEN_POSITIONS=10000`, `MAX_DAILY_LOSS_PCT=100`, `SAFETY_MAX_CONSECUTIVE_LOSSES=10000`) — для сбора статистики. Перед деплоем проверить, что это действительно нужно.

## Docker (локально)

```bash
docker build -t ai-trader .
docker run --rm --env-file .env -v $(pwd)/data:/app/data ai-trader
```
