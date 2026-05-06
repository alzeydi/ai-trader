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
    claude_model: str = "claude-sonnet-4-6"
    claude_max_tokens: int = 2048
    # Per-(symbol, side, entry_type) cooldown for the veto LLM. While a
    # decision is cached, repeated cycles reuse it instead of paying for an
    # identical call. Bumped to 1800s after the prompt rework: the new
    # SKIP-criteria prompt is more deterministic on the same input, so a
    # longer cache is safe and meaningfully cuts spend.
    llm_veto_cooldown_sec: int = 1800

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
    risk_per_trade_type_d: float = 0.02  # breakouts share A's conviction tier
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
    # Notional-percent breakeven trailing. Thresholds are expressed as a
    # fraction of entry notional (qty × entry), so they self-scale across
    # equity, leverage and per-trade margin without re-tuning.
    #
    # Lifecycle:
    #   1) When unrealised NET profit (after round-trip taker fees) reaches
    #      `breakeven_activate_pct` of notional, the SL is moved to lock in
    #      at least `breakeven_lock_pct` of notional in NET profit.
    #   2) After arming, the SL follows the high-water profit, never giving
    #      back more than `trail_distance_pct` of notional, and never
    #      loosening below the locked-in floor.
    #
    # Defaults sized for the existing universe (~1 % round-trip slippage
    # budget) — bump higher for chop, lower for trending alts.
    breakeven_activate_pct: float = 0.010   # 1.0 % of notional
    breakeven_lock_pct: float = 0.0075      # 0.75 % of notional
    trail_distance_pct: float = 0.009       # 0.9 % of notional
    # ----- Type B counter-trend gates -----
    # Type B fires on extreme 4h RSI + 15m reversal. Without a 1h trend
    # filter it keeps fading every minor pullback on a strongly trending
    # coin (see ZEC short-B series 2026-05-05/06: 18 trades, −154 USD,
    # against a +32 % 36-hour uptrend). Block B when EMA50(1h) has drifted
    # against the trade direction by more than `b_trend_block_pct` over
    # the last `b_trend_lookback_bars` hours.
    #
    # Tuning history: started at 2 %, but that turned out to be far too
    # tight — IO/USDT 2026-05-06 pumped +47 % in 4 hours without any news,
    # the textbook Type-B short setup, and the gate killed it at slope
    # 9.01 %. The slope-as-proxy-for-fundamentals heuristic is too noisy:
    # a sentiment-driven move and a one-shot squeeze look identical at the
    # EMA50(1h) level. Raised to 10 % so the gate only blocks truly
    # extreme parabolas (>10 % drift in 12 h on EMA50, where another leg
    # is genuinely likely); the veto layer decides «is there a catalyst»
    # using funding / OI / liquidations, which actually distinguish
    # news-driven moves from clean blowoffs.
    # Tightened from 10 % → 5 % after the BNB/USDT case (2026-05-06):
    # B SHORT fired on a sustained alt pump where 12 h-1h-EMA50 drift was
    # ~3-4 % — well under the 10 % gate but still firmly in the trend.
    # 5 % keeps the gate permissive enough for genuine ranges while
    # excluding mid-pump fades.
    b_trend_block_pct: float = 0.05
    b_trend_lookback_bars: int = 12      # 12 hours of 1h closes
    # HTF gate for Type B: don't fade mid-trend. Block B SHORT when 4h
    # close is more than `b_htf_extension_pct` ABOVE 4h-EMA200, and B
    # LONG when 4h close is more than that BELOW. Catches sustained
    # multi-day moves that the 12 h-1h-slope window cannot see.
    b_htf_extension_pct: float = 0.05
    # After a losing Type-B trade closes on a symbol, suppress new Type-B
    # entries on that symbol for this many hours. One bad fade is usually a
    # regime signal; let the bot stop re-arming it against the same coin.
    b_loss_cooldown_hours: int = 6
    # ----- Type D breakout -----
    # 1h close above (or below) the high (or low) of the prior
    # `d_breakout_lookback_bars` 1h bars, with 15m volume confirmation.
    # `d_max_excess_pct` filters out stale breakouts where price has
    # already run too far past the level — chasing a breakout 5 % past
    # invalidation has negative edge.
    d_breakout_lookback_bars: int = 20
    d_min_vol_ratio: float = 1.3
    d_max_excess_pct: float = 0.02
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

    # ----- Liquidity filter -----
    # Per-cycle screen that drops thin pairs from `symbols.yaml` before the
    # signal engine sees them. Was added after a STX/USDT close slipped 8
    # ticks past the SL (~$5 PnL error) — root cause was a thin book at
    # the trigger level, not the bot logic.
    min_quote_volume_24h_usdt: float = 50_000_000.0
    max_spread_bps: float = 5.0
    # `fetch_tickers()` is one REST call for all symbols on Binance, so
    # per-cycle (60s) refresh is cheap. Cache adds robustness if the
    # ticker call transiently fails.
    liquidity_filter_cache_sec: int = 300

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
