# CLAUDE.md

Этот файл задаёт правила для Claude Code (claude.ai/code) при работе с этим репозиторием.

## Язык общения

**Всегда отвечай на русском.** Объяснения в чате — на русском, в продюсерских терминах (см. глобальный CLAUDE.md §2). Технические термины и идентификаторы — как есть. Сообщения коммитов / PR / комментарии в коде — на английском (стандарт проекта).

---

## Документация по разделам

Подгружай эти файлы когда работаешь с соответствующими темами:

- **`docs/ARCHITECTURE.md`** — ОБЯЗАТЕЛЬНО подгрузи при правке любого слоя (`src/signal/`, `src/risk/`, `src/execution/`, `src/data/`, `src/persistence/`, `src/notify/`, `src/dashboard/`). Содержит детали слоёв, имена функций, race-условия, lifecycle позиции, схему таблиц SQLite.

- **`docs/ENV_REFERENCE.md`** — ОБЯЗАТЕЛЬНО подгрузи при правке `src/config/settings.py`, `.env`, `.env.example`, `railway.toml`, или при тюнинге риск-параметров. Полный справочник всех переменных окружения.

- **`docs/BACKTEST.md`** — подгрузи при работе с `src/backtest.py`, `scripts/benchmark_prompts.py`, или если пользователь упомянул backtest / benchmark / метрики / Monte Carlo / walk-forward. Содержит структуру output, флаги, метрики.

- **`docs/DEPLOY.md`** — подгрузи при подготовке к запуску smoke / paper / live, при работе с `scripts/start.sh`, Railway, tmux/systemd. Содержит лестницу деплоя.

- **`docs/DISCLAIMER.md`** — подгрузи при работе с live-режимом, API-ключами, безопасностью, юридическими вопросами.

---

## 1. Суть проекта

Бот, который сам торгует криптой на бирже Binance с плечом (берёт у биржи в долг, чтобы заработать или потерять больше на том же движении цены). Торгует фьючерсами — ставка «вырастет или упадёт» с конкретной ценой выхода.

**Текущий статус.** Каркас (skeleton/scaffold), не готовый продукт. Подходит для экспериментов на исторических данных и testnet, не для реальных денег.

### Как принимается решение войти в сделку — три фильтра подряд

1. **Робот-аналитик** (`src/signal/`). Программа смотрит на графики (4ч/1ч/15м) и по жёстким правилам ищет момент. Без AI, чистая математика. Если совпало — выдаёт «кандидата».
2. **AI-цензор Claude** (`src/llm/`). Кандидат уходит к Claude. Claude может только сказать «бери» или «пропусти» — выбрать сделку сам не может. Каждое решение пишется в `ai_decisions`.
3. **Риск-менеджер** (`src/risk/`, `src/execution/`). Если оба согласны — расчёт размера, стоп-лосс, тейк-профит. Гейты: число открытых сделок, дневной убыток, drawdown.

**Защита от потерь.** N последовательных проигрышей → пауза на сутки (NoFx pattern). Лимит дневного убытка → стоп до завтра.

**Где результаты.** Streamlit-дашборд (5 страниц) + Telegram-уведомления.

---

## 2. Архитектура (короткая версия)

Трёхслойный детерминированный конвейер. Один и тот же код в live, paper, backtest — меняются только источник данных и order executor.

- **Слой 1 · Signal** (`src/signal/engine.py`) → `CandidateSignal` типа A/B/C
- **Слой 2 · Veto** (`src/llm/veto.py`, `src/llm/client.py`) → TAKE/SKIP
- **Слой 3 · Risk + Execution** (`src/risk/`, `src/execution/`) → bracket-ордер

**Точки входа:**
- `src/main.py` — главный цикл live/paper.
- `src/backtest.py` — replay через тот же конвейер.

Детали по слоям, race-условия, lifecycle позиции — `docs/ARCHITECTURE.md`.

---

## 3. Указатели на ключевые файлы

| Тема | Файл |
|---|---|
| Все настройки + типы (pydantic-settings) | `src/config/settings.py` |
| Торговая вселенная | `src/config/symbols.yaml` |
| Главный цикл live/paper | `src/main.py` |
| Backtest engine | `src/backtest.py` |
| Сигналы (rule engine) | `src/signal/engine.py` |
| Veto-промпты | `src/llm/prompts/` |
| Risk-гейты + safety mode | `src/risk/limits.py`, `src/risk/safety_mode.py` |
| Сайзинг ATR | `src/risk/sizing.py` |
| Order placement | `src/execution/orders.py` |
| Position lifecycle | `src/execution/monitor.py` |
| SQLAlchemy-модели | `src/persistence/db.py` |
| Дашборд | `src/dashboard/app.py` |
| Railway start-script | `scripts/start.sh` |

---

## 4. Safety-флаги (4 слоя защиты)

`TRADING_MODE` (paper|live) × `PAPER_TRADING` (симулировать fills локально) × `DRY_RUN` (запретить реальные действия) × `BINANCE_TESTNET`.

Дефолты в `.env.example` — `paper + testnet + dry-run`, свежий клон запускать безопасно.

**Внимание:** `railway.toml` переопределяет на `live + non-testnet` со снятыми лимитами (`MAX_OPEN_POSITIONS=10000`, `MAX_DAILY_LOSS_PCT=100`, `SAFETY_MAX_CONSECUTIVE_LOSSES=10000`) — проверять перед деплоем.

API-ключи Binance: **только trade scope**, никогда withdrawal scope.

Все остальные параметры — `docs/ENV_REFERENCE.md`.

---

## 5. Базовые команды

```bash
poetry install                                      # установка
poetry run pytest                                   # все тесты
poetry run pytest tests/test_signal.py              # один файл
poetry run pytest tests/test_risk.py::test_name     # один тест
poetry run ruff check .
poetry run mypy src
poetry run python -m src.main                       # торговый цикл live/paper
poetry run streamlit run src/dashboard/app.py       # дашборд
```

Backtest, benchmark, smoke-test, Docker, Railway — см. `docs/BACKTEST.md` и `docs/DEPLOY.md`.

---

## 6. Правила для Claude Code

- **Не коммитить без явного одобрения.** Не запускать необратимые операции (force push, реальный live-ордер, drop таблиц SQLite) без подтверждения.

- **Один источник правды для настроек.** Все тюнинговые параметры — в `.env` (через `src/config/settings.py`) и в `src/config/symbols.yaml`. Прежде чем зашить число/флаг/строку в `.py`, проверить, есть ли ключ в `settings.py`. Если нет — добавить поле в `Settings` и читать через `settings.<field>`. Хардкод параметров торговли в коде запрещён.

- **Магические числа в торговой логике без объяснения — запрещены.** Любой числовой литерал в `src/signal/`, `src/risk/`, `src/execution/` (порог индикатора, множитель ATR, окно EMA, таймаут, slippage) должен либо браться из settings, либо иметь комментарий: что подбиралось, на каких данных, почему именно столько.

- **Safety-флаги не ослаблять без подтверждения.** `DRY_RUN`, `PAPER_TRADING`, `BINANCE_TESTNET`, `MAX_DAILY_LOSS_PCT`, `SAFETY_MAX_CONSECUTIVE_LOSSES` — последняя линия защиты. Любое предложение их изменить выносить пользователю явно: «это снимает защиту X, ок?».

- **Cross-cutting фикс — пройтись по всем трём executor'ам.** Если поправил баг в `execution.orders` для paper — проверь live-ветку и backtest-executor в `src/backtest.py`. Один конвейер на три режима — баг сайзинга/округления/fees обычно живёт во всех трёх.

- **Любой вопрос по Binance API — сначала в документацию, потом в код.** Всё, что касается Binance USDⓈ-M Futures (endpoints, параметры, типы ордеров, error codes, поведение при rate-limit, формат id, миграции, conditional/algo-разделение) — обязательно сверять с актуальной [Binance Derivatives API doc](https://developers.binance.com/docs/derivatives/) **до** правки кода или построения гипотезы. Не полагаться на ccxt-абстракции и память: Binance ломает совместимость без предупреждения (пример: 2025-12-09 conditional orders переехали из `/fapi/v1/order` в `/fapi/v1/algoOrder` — все наши cleanup-пути молча сломались, потому что мы ориентировались на «как было», а не на доку). Если фикс затрагивает endpoint — прикладывать ссылку на конкретный раздел доки в коммите/PR.

- **Edge-кейсы для торгового кода — обязательны.** Любая логика с балансом / сайзингом / переходами состояний / таймерами / синхронизацией с внешним API сопровождается блоком «Edge-кейсы, которые я учёл/не учёл» (см. глобальный CLAUDE.md §4). Особо: двойные fills, потеря сети между placeOrder и подтверждением, race между PositionMonitor и main loop, манипуляции системным временем для cooldown'ов.

- **При признаках застоя** (3 итерации без прогресса) — остановиться и спросить пользователя про саморефлексию (см. глобальный CLAUDE.md §8).

- **При обнаружении пробела в документации** — молча запомнить, перечислить пользователю при `/session-end`.

- **Точечные правки больших файлов.** `src/main.py`, `src/backtest.py`, `src/dashboard/app.py` — большие. Менять через Edit. Полная перезапись — только при изменении > 50% содержимого.

- **Не вводить новые зависимости** (`poetry add ...`) без согласования.

- **Сразу мерджить feature-ветки в `main`.** После того как изменения закоммичены и запушены в feature-ветку, сразу создать PR и слить его в `main` (squash или merge — по обстоятельствам), не ждать отдельного «давай мерджи». Подтверждение нужно только если правка затрагивает safety-флаги, удаление файлов, force-push или изменение `main`-ветки напрямую без PR.

---

## 7. Конфиг тулинга

- `ruff`: line 100, py312, selects E/F/I/B/UP/N/SIM/C4, ignore E501
- `mypy`: py312, non-strict, плагин pydantic
- `pytest`: `asyncio_mode=auto`, testpaths=`tests`

---

*Размер файла — целевой бюджет ≤ 8k символов. Если превышен — выносить разделы в `docs/` и обновлять эту строку.*
