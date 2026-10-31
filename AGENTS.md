# AI_trader/ivan-repo — Codex

Перед работой прочитай `CLAUDE.md` этого репозитория целиком и архитектуру. Обязательные профильные документы находятся внутри репозитория: архитектура — `docs/ARCHITECTURE.md` или `ARCHITECTURE.md`; backtest/метрики — `docs/BACKTEST.md` или `BACKTEST.md`; торговля/ключи — `docs/DISCLAIMER.md` или `DISCLAIMER.md` и `docs/DEPLOY.md` или `DEPLOY.md`; настройки — `docs/ENV_REFERENCE.md` или `ENV_REFERENCE.md`. Сначала используй вариант в `docs/`, если он существует, иначе одноименный файл в корне. Так поддерживаются рабочая ветка с каталогом docs и основной репозиторий с документами в корне. Эти документы обязательны также в отдельном клоне.

При работе из обертки BRAIN_2/Telegram/AI_trader дополнительно можно прочитать `../AGENTS.md`, `../docs/CODEX.md` и `../docs/CODEX_ARCHITECTURE.md`, если они существуют. Обертка необязательна; в отдельном клоне ее отсутствие не блокирует работу. Используй реально доступные инструменты Codex; настройки и разрешения Claude не переносятся автоматически. Hooks требуют штатного доверия `/hooks`.

Конвейер rule signal → Claude TAKE/SKIP veto → risk/execution сохраняется во всех трех режимах. Не ослабляй DRY_RUN/PAPER_TRADING/BINANCE_TESTNET и риск-лимиты без прямого поручения. Деплой railway может задавать опасные live defaults: не запускай его для проверки. Не запускай main loop, smoke/benchmark с сетью, Telegram-уведомления или paid LLM ради миграции.

Параметры только через Settings/symbols.yaml; сквозное исправление проверять в live/paper/backtest executor. Для Binance endpoint сначала актуальная официальная документация. Чат на русском; комментарии, коммиты/PR на английском по локальному правилу. Коммит/PR/merge не разрешены одним наличием старой инструкции об auto-merge.
