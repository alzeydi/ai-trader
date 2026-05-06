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
    risk_per_trade_type_a: float = 0.02  # ratio, e.g. 0.02 == 2 % of equity
    risk_per_trade_type_b: float = 0.01
    risk_per_trade_type_c: float = 0.01
    # Per-trade margin policy cap. Without this the risk-USD-based sizing
    # produces wildly different notionals across symbols (e.g. ZIL at $0.004
    # gets a $9.6k notional vs DOGE at $0.11 with $90), because qty =
    # risk / sl_distance explodes when sl_distance is tiny in absolute
    # terms. Capping by margin makes positions comparable in size
    # regardless of price.
    max_margin_per_trade_pct: float = 0.10  # 10 % of equity per single trade
    max_open_positions: int = 2
    max_daily_loss_pct: float = 3.0
    max_drawdown_pct: float = 15.0
    atr_stop_multiplier: float = 1.5
    atr_take_multiplier: float = 3.0
    # Hard floor on stop distance as a fraction of entry price. Calm symbols
    # have ATRs that translate to <0.5 % of price, which is inside typical
    # spread/noise — anything tighter gets stopped out by random ticks.
    min_stop_pct: float = 0.008  # 0.8 %
    trail_enabled: bool = True
    # USD-denominated breakeven trailing. When unrealized NET profit (after
    # round-trip taker fees) reaches breakeven_activate_usd, the stop is
    # moved to lock in at least breakeven_lock_usd of NET profit. After
    # that the stop continues to follow the high-water profit, never
    # giving back more than trail_distance_usd, and never loosening below
    # the locked-in floor.
    breakeven_activate_usd: float = 5.0
    breakeven_lock_usd: float = 2.5
    trail_distance_usd: float = 2.5
    # ----- Type B counter-trend gates -----
    # Type B fires on extreme 4h RSI + 15m reversal. Without a 1h trend
    # filter it keeps fading every minor pullback on a strongly trending
    # coin (see ZEC short-B series 2026-05-05/06: 18 trades, −154 USD,
    # against a +32 % 36-hour uptrend). Block B when EMA50(1h) has drifted
    # against the trade direction by more than `b_trend_block_pct` over
    # the last `b_trend_lookback_bars` hours.
    b_trend_block_pct: float = 0.02      # 2 % drift across the window
    b_trend_lookback_bars: int = 12      # 12 hours of 1h closes
    # After a losing Type-B trade closes on a symbol, suppress new Type-B
    # entries on that symbol for this many hours. One bad fade is usually a
    # regime signal; let the bot stop re-arming it against the same coin.
    b_loss_cooldown_hours: int = 6
    min_confidence: float = 0.7
    taker_fee_pct: float = 0.0004  # 0.04 % per leg on Binance USDT-M
    # Binance USDT-M futures rejects any non-reduce-only order whose notional
    # is below ~5 USDT (error -4164). Sizing rejects below this floor.
    min_notional_usd: float = 5.0
    max_hold_hours: int = 8

    # ----- Safety mode -----
    safety_max_consecutive_losses: int = 3
    safety_pause_hours: int = 24

    # ----- Signal engine -----
    timeframe_trend: str = "4h"
    timeframe_structure: str = "1h"
    timeframe_execution: str = "15m"
    signal_min_confidence: float = 0.6
    invalidation_check_interval_min: int = 15

    # ----- Persistence -----
    db_path: str = "data/trades.db"
    equity_csv: str = "data/equity_curve.csv"
    decisions_csv: str = "data/ai_decisions.csv"

    # ----- Runtime -----
    log_level: str = "INFO"
    loop_interval_sec: int = 60
    monitor_interval_sec: int = 10
    dry_run: bool = True
    paper_trading: bool = True

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
