#!/usr/bin/env bash
# Railway unified start script.
# Starts the trading loop in the background and the Streamlit dashboard in the
# foreground (bound to $PORT so Railway's router can reach it).
set -euo pipefail

PORT="${PORT:-8501}"

echo "==> ai-trader startup"
echo "    TRADING_MODE=${TRADING_MODE:-paper}"
echo "    PAPER_TRADING=${PAPER_TRADING:-true}"
echo "    DRY_RUN=${DRY_RUN:-true}"
echo "    BINANCE_TESTNET=${BINANCE_TESTNET:-true}"
echo "    DB_PATH=${DB_PATH:-data/trades.db}"
echo "    PORT=$PORT"

# Ensure data directory exists (in case no volume is mounted).
mkdir -p /app/data

# ── Trading loop ─────────────────────────────────────────────────────────────
# Runs in the background; shares the same /app/data SQLite DB as the dashboard.
python -m src.main &
TRADER_PID=$!
echo "==> trading loop started (pid $TRADER_PID)"

# ── Dashboard ─────────────────────────────────────────────────────────────────
echo "==> starting dashboard on port $PORT"
exec streamlit run src/dashboard/app.py \
    --server.port "$PORT" \
    --server.address "0.0.0.0" \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false
