# ENV_REFERENCE.md

Справочник всех настроек `.env`. Подгружай при правке `src/config/settings.py`, `.env`, `railway.toml`, или при тюнинге риск-параметров.

Источник правды по типам и default'ам — `src/config/settings.py` (pydantic-settings). Шаблон значений — `.env.example`. Этот файл — навигационная карта.

## Safety-флаги (4 слоя защиты)

| Variable | Default | Назначение |
|---|---|---|
| `TRADING_MODE` | `paper` | `paper` \| `live` |
| `BINANCE_TESTNET` | `true` | использовать Binance testnet |
| `PAPER_TRADING` | `true` | симулировать fills локально (не отправлять на биржу) |
| `DRY_RUN` | `true` | отказывать в любых действиях с реальными деньгами |

Дефолты в `.env.example` — paper + testnet + dry-run, безопасный fresh-clone.

## Капитал и плечо

| Variable | Default | Назначение |
|---|---|---|
| `EQUITY_USD` | `1000` | стартовый капитал / референс для сайзинга |
| `LEVERAGE` | `3` | запрашиваемое плечо на позицию |
| `MARGIN_MODE` | `ISOLATED` | режим маржи на Binance |
| `BASE_CURRENCY` | `USDT` | валюта баланса |

## Риск

| Variable | Default | Назначение |
|---|---|---|
| `RISK_PER_TRADE_TYPE_A` | `0.02` | риск на сделку типа A (доля капитала) |
| `RISK_PER_TRADE_TYPE_B` | `0.01` | риск на сделку типа B |
| `RISK_PER_TRADE_TYPE_C` | `0.01` | риск на сделку типа C |
| `MAX_OPEN_POSITIONS` | `2` | жёсткий лимит на открытые позиции |
| `MAX_DAILY_LOSS_PCT` | `3.0` | дневной убыток → бот выключается до завтра |
| `MAX_DRAWDOWN_PCT` | — | глобальный drawdown гейт |
| `ATR_STOP_MULTIPLIER` | — | множитель ATR для стоп-лосса |
| `ATR_TAKE_MULTIPLIER` | — | множитель ATR для тейк-профита |

## Safety-mode (NoFx pattern)

| Variable | Default | Назначение |
|---|---|---|
| `SAFETY_MAX_CONSECUTIVE_LOSSES` | `3` | сколько подряд убытков → пауза |
| `SAFETY_PAUSE_HOURS` | `24` | длительность паузы |

## Сигналы

| Variable | Default | Назначение |
|---|---|---|
| `TIMEFRAME_TREND` | `4h` | таймфрейм для трендового слоя |
| `TIMEFRAME_STRUCTURE` | `1h` | для структурного слоя |
| `TIMEFRAME_EXECUTION` | `15m` | для триггера |
| `SIGNAL_MIN_CONFIDENCE` | `0.6` | минимум confidence от rule engine **до** veto (preflight-фильтр) |
| `MIN_CONFIDENCE` | `0.7` | минимум confidence от Claude **после** veto (если ниже — SKIP) |
| `INVALIDATION_CHECK_INTERVAL_MIN` | `15` | как часто PositionMonitor дёргает Claude для проверки открытой позиции (см. `docs/ARCHITECTURE.md` Layer 3) |

**Важно про два порога confidence:** `SIGNAL_MIN_CONFIDENCE` режет **сигналы** до того, как они попадут к Claude (экономия LLM-вызовов). `MIN_CONFIDENCE` режет **ответы Claude** с низкой уверенностью. Это разные шкалы — крутить независимо.

## Trailing stop

| Variable | Default | Назначение |
|---|---|---|
| `TRAIL_ENABLED` | `true` | включить breakeven trail |
| `BREAKEVEN_ACTIVATE_USD` | `5.0` | при какой прибыли двигать стоп в безубыток |
| `BREAKEVEN_LOCK_USD` | `2.5` | сколько фиксировать после активации |
| `TRAIL_DISTANCE_USD` | `2.5` | расстояние трейлинга |

## Комиссии

| Variable | Default | Назначение |
|---|---|---|
| `TAKER_FEE_PCT` | `0.0004` | taker-комиссия Binance USDT-M (0.04%) |

## API-ключи

| Variable | Назначение |
|---|---|
| `BINANCE_API_KEY` / `BINANCE_API_SECRET` | Binance creds. **Только trade scope**, никогда withdrawal. |
| `ANTHROPIC_API_KEY` | Claude API |
| `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` | Telegram (опционально) |
| `TELEGRAM_ENABLED` | `true`/`false` |
| `CLAUDE_MODEL` | модель Claude (по умолчанию `claude-opus-4-7`) |
| `CLAUDE_MAX_TOKENS` | лимит ответа |

## Persistence

| Variable | Default |
|---|---|
| `DB_PATH` | `data/trades.db` |
| `EQUITY_CSV` | `data/equity_curve.csv` |
| `DECISIONS_CSV` | `data/ai_decisions.csv` |

## Прочее

| Variable | Default |
|---|---|
| `LOG_LEVEL` | `INFO` |
| `LOOP_INTERVAL_SEC` | `60` |
| `PYTHONPATH` | `/app` (Docker/Railway) |

## Railway-overrides

`railway.toml` переопределяет дефолты для production:
- `TRADING_MODE=live`, `PAPER_TRADING=false`, `BINANCE_TESTNET=false` — реальные деньги.
- `MAX_OPEN_POSITIONS=10000`, `MAX_DAILY_LOSS_PCT=100`, `SAFETY_MAX_CONSECUTIVE_LOSSES=10000`, `SAFETY_PAUSE_HOURS=0` — лимиты сняты для сбора статистики.

Перед деплоем проверить, что эти override'ы действительно нужны.

## Торговая вселенная

Список инструментов (символов) — `src/config/symbols.yaml`, не `.env`.
