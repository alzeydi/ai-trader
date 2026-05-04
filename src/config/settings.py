"""Application settings loaded from .env via pydantic-settings.

All knobs declared in `.env.example` are surfaced here as typed attributes.
Importing `settings` triggers validation; a misconfigured environment fails fast.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ----- Binance -----
    binance_api_key: SecretStr = Field(default=SecretStr(""))
    binance_api_secret: SecretStr = Field(default=SecretStr(""))
    binance_testnet: bool = True

    # ----- Anthropic / Claude -----
    anthropic_api_key: SecretStr = Field(default=SecretStr(""))
    claude_model: str = "claude-opus-4-7"
    claude_max_tokens: int = 2048
    claude_temperature: float = 0.0

    # ----- Telegram -----
    telegram_bot_token: SecretStr = Field(default=SecretStr(""))
    telegram_chat_id: str = ""
    telegram_enabled: bool = False

    # ----- Trading core -----
    trading_mode: Literal["paper", "live"] = "paper"
    base_currency: str = "USDT"
    equity_usd: float = 1000.0
    leverage: int = 3
    margin_mode: Literal["ISOLATED", "CROSSED"] = "ISOLATED"

    # ----- Risk -----
    risk_per_trade_pct: float = 0.5
    max_open_positions: int = 3
    max_daily_loss_pct: float = 3.0
    max_drawdown_pct: float = 15.0
    atr_stop_multiplier: float = 1.5
    atr_take_multiplier: float = 3.0

    # ----- Safety mode -----
    safety_max_consecutive_losses: int = 3
    safety_pause_hours: int = 12

    # ----- Signal engine -----
    timeframe_trend: str = "4h"
    timeframe_structure: str = "1h"
    timeframe_execution: str = "15m"
    signal_min_confidence: float = 0.6

    # ----- Persistence -----
    db_path: str = "data/trades.db"
    equity_csv: str = "data/equity_curve.csv"
    decisions_csv: str = "data/ai_decisions.csv"

    # ----- Runtime -----
    log_level: str = "INFO"
    loop_interval_sec: int = 60
    dry_run: bool = True

    # ----- Derived helpers -----
    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT

    @property
    def symbols_yaml(self) -> Path:
        return PROJECT_ROOT / "src" / "config" / "symbols.yaml"

    @property
    def db_abspath(self) -> Path:
        path = Path(self.db_path)
        return path if path.is_absolute() else PROJECT_ROOT / path


@lru_cache(maxsize=1)
def _load() -> Settings:
    return Settings()


settings: Settings = _load()
