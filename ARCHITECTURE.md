# ARCHITECTURE.md

Глубокая архитектура. Подгружай при правке любого из слоёв (signal / risk / execution / data / persistence / monitor / notify), а также если нужны имена функций, потоки данных, lifecycle позиции, race-условия.

## Принцип

Один и тот же код используется в **live**, **paper** и **backtest**. Подменяются только два компонента: источник данных (`BinanceClient` vs `HistoricalDataSource` в backtest) и executor ордеров (`OrderExecutor` paper/live vs backtest-executor).

Это значит: баг в signal/risk-логике одинаково проявится во всех трёх режимах. Бэктест валиден как доказательство корректности логики (но не латентности и не реальных fills).

## Слой 1 · Signal (`src/signal/`)

Rule engine, детерминированный. Без LLM, без рандома.

**Конвейер:** 4h trend → 1h structure → 15m trigger.

- `trend.py` — определяет тренд на 4h (EMA-классификация, MACD-направление).
- `structure.py` — на 1h ищет структуру цены: pullback / breakout / range.
- `execution.py` — на 15m ищет конкретный триггер входа.
- `engine.py` — `SignalEngine` связывает все три. Точка входа: **`SignalEngine.generate_signals(symbol)`** (возвращает `CandidateSignal | None`).

**Output:** `CandidateSignal` (см. `types.py`) — direction, entry, SL, TP, тип (A/B/C), confidence.

**Типы сделок:**
- A — самый качественный сетап (выше risk-per-trade).
- B / C — сетапы с меньшей частотой / меньшим качеством.

**Scoring (взвешенная сумма компонентов):**
- A: `_score_a(ema_gap_pct, rsi_delta, vol_ratio)` — веса 0.40 / 0.35 / 0.25.
- B: `_score_b(rsi_4h, side, hist_delta)` — веса 0.50 / 0.50.
- C: `_score_c(proximity, rsi_delta)` — веса 0.50 / 0.50.

Strength обрезается на 0.99. Веса и пороги (`0.08` ema-gap, `0.03` близость к EMA50, `0.20`/`0.80` позиция в диапазоне) — магические числа в `engine.py:36, 126, 204-220`. **Не задокументировано, на каких данных подбирались.** Тюнинговать осторожно.

## Слой 2 · Veto (`src/llm/`)

Двухстадийный фильтр.

**(а) Локальный preflight (`veto.preflight_check`)** — жёсткие правила, не зависящие от LLM. Например: проверка спреда, проверка свежести данных, минимальный объём. Если preflight отклонил — Claude не вызывается.

**(б) Claude veto (`ClaudeClient.veto`)** — отправляет описание сделки + рыночный контекст. Claude возвращает строгий JSON: `{action: "TAKE"|"SKIP", confidence: float, reason: str}`. LLM **не** генерирует сделку — только утверждает или отклоняет существующего кандидата.

Промпты лежат в `src/llm/prompts/` (system prompt + шаблоны).

**Логирование:** каждый вызов (включая стоимость в USD, латентность, raw response) пишется в таблицу `ai_decisions`. Дашборд показывает cost-by-day.

**Cooldown-кэш (важно):** настраивается через `LLM_VETO_COOLDOWN_SEC` (по умолчанию 1800 с / 30 мин, см. `settings.llm_veto_cooldown_sec`). Если идентичный кандидат `(symbol, side, entry_type)` повторяется в течение этого окна — Claude **не вызывается**, переиспользуется предыдущий ответ. Кэш in-process (теряется при перезапуске). Экономит деньги, но: одинаковые кандидаты автоматически получают одинаковый вердикт без нового анализа.

## Слой 3 · Risk + Execution

### `src/risk/`

- `limits.can_take_signal(ctx)` — гейты: число открытых позиций (`MAX_OPEN_POSITIONS`), дневной убыток (`MAX_DAILY_LOSS_PCT`), глобальный drawdown (`MAX_DRAWDOWN_PCT`).
- `sizing.size_position(signal, equity, atr)` — расчёт размера позиции на базе ATR и `RISK_PER_TRADE_TYPE_*`.
- `safety_mode.SafetyMode` — паттерн NoFx. Считает последовательные убытки. После N подряд (`SAFETY_MAX_CONSECUTIVE_LOSSES`) ставит бот на паузу на `SAFETY_PAUSE_HOURS`. Состояние персистится в `safety_state`.

### `src/execution/`

- `orders.OrderExecutor` — единый интерфейс для placeOrder. Реализации: paper (запись в БД) или live (через ccxt + tenacity retry). Ставит bracket-ордера (entry + SL + TP).
- `monitor.PositionMonitor` — параллельный поток. Цикл:
  1. Опросить открытые позиции через ccxt (`fetch_open_positions`).
  2. Сверить с локальной БД (`positions` таблица).
  3. Если SL/TP сработал на бирже — закрыть локальную запись, посчитать realized PnL, обновить equity, обновить safety-mode counter.
  4. Применить breakeven trail если включён (`TRAIL_ENABLED`).
  5. Применить timeout если позиция висит дольше лимита.
  6. **LLM invalidation poll** — раз в `INVALIDATION_CHECK_INTERVAL_MIN` (дефолт 15 мин) для каждой открытой позиции дёргается Claude с вопросом «всё ещё валидна?». Если invalidation_signal сработал — позиция закрывается принудительно. Это **дополнительные LLM-вызовы** (не только при входе) — затраты на API растут с числом и длительностью открытых позиций. См. `src/execution/monitor.py:71` (промпт) и `:259-312` (логика).
- `trader.Trader` — связывает risk-гейты + sizing + executor.

**Race-условия (важно):**
- Двойные fills: если main-loop успел вызвать placeOrder дважды до получения ack — возможен дубль. Защита: idempotency key.
- Race между monitor и main-loop: оба могут параллельно решить, что позиция закрыта. Защита: optimistic lock через статус в `positions`.
- Потеря сети между placeOrder и подтверждением: tenacity retry + reconcile при перезапуске (`_reconcile_open_trades` в `main.py:60-107`).

**Reconciliation при старте (`_reconcile_open_trades`):**
1. Получить открытые позиции с биржи.
2. Сравнить с локальными `positions` со статусом OPEN.
3. Закрыть «зомби»: позиции, которые есть локально, но нет на бирже (например, закрыты вручную в Binance UI).
4. Подобрать «orphans»: позиции, которые есть на бирже, но не в БД (например, placeOrder прошёл, но ack потерялся при сбое сети).
5. Удалить дубли по `(symbol, side, entry_time)`.

Без этой процедуры при перезапуске бот может: либо думать, что позиция открыта, когда её нет; либо открыть вторую позицию поверх существующей; либо «забыть» позицию и не закрыть её по SL/TP.

## Data layer (`src/data/`)

- `binance_client.BinanceClient` — обёртка над ccxt. Кэширует OHLCV в SQLite (`ohlcv_candles`), чтобы не долбить API при одних и тех же вычислениях (особенно в backtest).
- `indicators.py` — EMA / RSI / ATR / MACD / swing highs/lows. На pandas / pandas-ta.
- `context.py` — рыночный контекст: funding rate, open interest, недавние liquidations. Для veto-промпта.

## Persistence (`src/persistence/db.py`)

SQLAlchemy + SQLite. Файл `data/trades.db`.

**Таблицы:**
- `trades` — сделка как факт: вход, выход, PnL, fees.
- `ai_decisions` — каждый вызов Claude veto: input/output, cost, latency. Модель — `AIDecision` (`db.py:106`).
- `equity_snapshots` — снимок капитала с шагом `LOOP_INTERVAL_SEC`. Модель — `EquitySnapshot`. Есть legacy alias `EquityPoint = EquitySnapshot` (`db.py:169`) — используется в `scripts/recalculate_state.py`. Не плодить новые ссылки на `EquityPoint`.
- `positions` — открытые позиции (live + paper). Source of truth для PositionMonitor.
- `safety_state` — состояние SafetyMode (счётчик losses, время паузы).
- `ohlcv_candles` — кэш свечей.

**Deprecated:** класс `Decision` (`db.py:95`) — старая таблица решений, **не используется**. Текущий код пишет только в `AIDecision`. При правке persistence не путать.

CSV-зеркала через `csv_writer.py` — для удобства внешнего анализа (Excel, R).

## Notify (`src/notify/`)

Telegram-алерты через `python-telegram-bot`. Типы:
- ENTRY — открыта позиция.
- CLOSE — закрыта позиция (с PnL).
- ITER — итерационная сводка раз в N циклов (equity, открытые, daily PnL).
- CRITICAL — ошибка, требующая внимания (например, сбой API).

## Dashboard (`src/dashboard/app.py`)

Streamlit, multi-page. 5 страниц:
1. **Overview** — equity curve vs BTC buy & hold, Sharpe / Sortino / Max DD.
2. **Trades** — таблица сделок с фильтрами.
3. **AI Decisions** — лог veto, raw JSON viewer, cost-by-day chart.
4. **Positions** — открытые позиции в реальном времени.
5. **Settings** — read-only снимок настроек.

Дашборд читает ту же `data/trades.db`, что и торговый цикл — синхронизация через **WAL-режим SQLite** (Write-Ahead Logging). Без WAL Streamlit-чтения блокировали бы запись в торговом цикле. Если меняешь `db.py`, режим WAL включается на старте — не отключать.

Страница **Settings** — read-only снимок текущих значений из `settings.py`. Не для редактирования. Изменения настроек только через `.env` + перезапуск процесса.

## Точки входа

- `src/main.py` — главный цикл. На каждой итерации (`LOOP_INTERVAL_SEC`):
  1. Reconcile открытых позиций (на старте).
  2. Проверка safety-mode.
  3. Для каждого символа: `SignalEngine.scan` → veto → `Trader.execute`.
  4. Equity snapshot + Telegram iter-сводка раз в N циклов.
  - `PositionMonitor` крутится в параллельном threading.
- `src/backtest.py` — детерминированный реплей через тот же конвейер. См. `docs/BACKTEST.md`.
