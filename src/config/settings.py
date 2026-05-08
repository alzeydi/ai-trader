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
    trail_distance_pct: float = 0.00675     # 0.675 % of notional (was 0.9 %, −25 %)
    # Break-even protection: pre-trail guard for positions that have shown a
    # profit peak but haven't yet armed the full trail.
    #
    # Lifecycle (phase 0.5):
    #   Once unrealised NET profit HWM reaches `be_protect_arm_pct`, ANY tick
    #   where the position starts retracing (current net < HWM) causes the SL
    #   to be moved to a price that locks in `be_protect_lock_pct` net profit.
    #   Fires only once per trade (the tighter-stop check prevents re-firing).
    #   After firing, the regular trail takes over if price recovers to
    #   `breakeven_activate_pct`.
    be_protect_arm_pct: float = 0.005    # 0.5 % of notional — arm threshold
    be_protect_lock_pct: float = 0.005   # 0.5 % of notional locked at BE-stop
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
    # `d_max_ema_dist_pct` guards against entering a breakout when price
    # has already extended too far from 1h-EMA50: the breakout should
    # originate from a consolidation near the MAs, not from mid-air after
    # a large runup. Tuned at 3 % — blocks entries like XRPUSDT 2026-05-07
    # where price was 2.4 %+ above MA25(1h) before the signal fired.
    d_breakout_lookback_bars: int = 20
    d_min_vol_ratio: float = 1.3
    d_max_excess_pct: float = 0.02
    d_max_ema_dist_pct: float = 0.03  # 3 % from EMA50(1h)
    min_confidence: float = 0.7
    taker_fee_pct: float = 0.0004  # 0.04 % per leg on Binance USDT-M
    # Binance USDT-M futures rejects any non-reduce-only order whose notional
    # is below ~5 USDT (error -4164). Sizing rejects below this floor.
    min_notional_usd: float = 5.0
    # "Meaningful trade" floor as a fraction of the policy cap notional
    # (`equity × max_margin_per_trade_pct × leverage`). When the free-margin
    # cap forces qty below this, sizing returns None instead of opening a
    # micro-position. Default 0.30 means: if the trade can only get 30 % of
    # its intended size, skip it. Set to 0.0 to disable.
    #
    # Tuning history: introduced 2026-05-07 after observing a cluster of
    # $5–$15-notional trades on Railway demo (#102–#125) that opened when
    # ~10 parallel positions had drained the wallet's free margin. Each
    # micro-position was barely larger than fees and added noise to stats
    # without contributing any signal. 30 % is the smallest fraction that
    # still leaves meaningful R:R after a typical 2 × taker round-trip.
    min_notional_pct_of_policy: float = 0.30
    max_hold_hours: int = 8

    # ----- Safety mode -----
    safety_max_consecutive_losses: int = 3
    safety_pause_hours: int = 24

    # Per-trade risk for Type E (MA25 bounce). Between A/D (2 %) and B/C
    # (1 %): the setup is with-trend like A, but fires less often and on a
    # tighter structural reference, so we start conservative until we have
    # live trade statistics. Tune via MA25_RISK_PCT env var once you have
    # ≥30 closed trades.
    risk_per_trade_type_e: float = 0.015  # 1.5 % of equity
    # Type F (BoS short) and G (MA25 breakdown) are the bearish counterparts
    # of D-style structural shorts and E (MA25 bounce). Both ship at the same
    # conviction tier as E until live statistics tell us otherwise.
    risk_per_trade_type_f: float = 0.015  # 1.5 % of equity
    risk_per_trade_type_g: float = 0.015  # 1.5 % of equity

    # ----- MA25 Bounce (Strategy X) -----
    # MA25 must have risen by at least this many % over the 12-bar look-back
    # window to confirm the uptrend is active (not just inertial drift).
    ma25_slope_min_pct: float = 0.5
    # Fraction of the 10 prior bars (candles[-11:-1]) that must have closed
    # above MA25. Separates "pullback in uptrend" from "reclaim from below".
    ma25_prior_above_min: float = 0.7
    # Max % the current close may sit above MA25. Entries past this are
    # chasing a bounce that has already run its course.
    ma25_pct_above_max: float = 4.0
    # Upper bound of the touch zone (as a fraction above MA25). A bar whose
    # lowest wick never reached within 0.5 % of MA25 never bounced off it.
    ma25_touch_max_pct: float = 0.005
    # Lower bound of the touch zone (as a fraction below MA25). Wicks more
    # than 3 % below MA25 indicate a flash-crash / liquidation cascade, not
    # a measured pullback; they have different follow-through statistics.
    ma25_touch_min_pct: float = 0.03
    # Per-symbol cooldown after any Type-E or Type-G candidate fires. The
    # H4-bar-based detectors re-evaluate the in-progress bar every loop tick
    # (default 60 s); a non-zero cooldown prevents the same setup from
    # re-firing for the rest of the 4h bar.
    #
    # Set to 0 (disabled) per 2026-05-08 paper-run review: in markets where
    # the cooldown was binding (regime-induced pause + LLM SKIPs counted by
    # ai_decisions table), it was suppressing 80%+ of valid Type E
    # opportunities even when no actual trade had been opened. Without
    # cooldown, dedup falls back on:
    #   • llm_veto_cooldown_sec (1800 s) — caches veto verdicts per
    #     (symbol, side, entry_type), so the same setup is not re-priced
    #     against the LLM every cycle even though it now reaches the
    #     veto layer;
    #   • the 4h bar boundary itself — once the bar closes the setup is
    #     re-evaluated against the new bar's data, so we cannot enter the
    #     same closed-bar setup twice in any case.
    # Re-enable (set 24) if intra-bar re-firing becomes a problem in live.
    ma25_symbol_cooldown_hours: int = 0

    # ----- MA25 Breakdown (Strategy G — mirror of E) -----
    # Bearish twin of E: MA25 falling (slope ≤ −0.5 %), MA25 < MA99,
    # ≥70 % of the last 10 closed bars closed BELOW MA25, current bar is
    # red, running close is below MA25 by ≤ 4 %, and the upper wick of the
    # last 3 bars touched the MA25 zone from below.
    ma25_breakdown_slope_max_pct: float = -0.5
    ma25_breakdown_prior_below_min: float = 0.7
    ma25_breakdown_pct_below_max: float = 4.0
    ma25_breakdown_touch_max_pct: float = 0.005   # high within 0.5 % under MA25
    ma25_breakdown_touch_min_pct: float = 0.03    # not more than 3 % above

    # ----- Type F: Break-of-Structure short -----
    # 1h structural short. Looks back `f_swing_lookback_bars` closed 1h
    # bars to locate the lowest low (swing low). Between that bar and the
    # current bar we look for the highest high (the lower-high retest).
    # Triggers when:
    #   * current 1h close < swing_low (BoS confirmed)
    #   * the lower-high formed within `f_max_lower_high_age` bars (fresh)
    #   * 15m volume on the breakdown ≥ `f_min_vol_ratio` × 20-bar mean
    #   * close has not run more than `f_max_excess_pct` past swing_low
    # 4h-bias gating mirrors B's `b_htf_extension_pct`: in a bullish 4h
    # stack, F is blocked when close_4h sits more than this fraction
    # ABOVE 4h-EMA200 (mid-trend pump territory). neutral / bearish 4h
    # are always allowed; bullish-but-near-EMA200 (distribution zone) is
    # allowed too.
    f_swing_lookback_bars: int = 20
    f_max_lower_high_age: int = 6
    f_min_vol_ratio: float = 1.2
    f_max_excess_pct: float = 0.02
    f_htf_extension_pct: float = 0.05

    # ----- Global BTC regime gate -----
    # Hard pre-veto filter: blocks longs when BTC is falling and shorts when
    # BTC is rising. Runs BEFORE the LLM call so vetoed candidates don't burn
    # tokens. Applies uniformly to all entry types (A/B/C/D/E).
    btc_regime_enabled: bool = True
    # EMA-on-H1 slope filter. Replaces the previous "1h close vs prev close"
    # comparison, which only saw the LAST hour's delta and would read flat
    # while a multi-hour trend was in progress (e.g. last bar already at the
    # bottom of a multi-bar dump → small Δ vs prev → "flat").
    #
    # Logic: ema = EMA(period, H1_closes); slope_pct = (ema_now − ema_lookback_ago)
    # / ema_lookback_ago × 100. up if slope > +threshold, down if < −threshold,
    # else flat.
    btc_regime_ema_period: int = 50          # H1 bars; EMA50 = standard trend filter
    btc_regime_slope_lookback: int = 6       # H1 bars; "EMA50 changed X% over 6h"
    btc_regime_slope_threshold_pct: float = 0.15  # 0.15 % over 6h ≈ 0.025 %/bar
    # Spot price's distance from EMA50, in %. EMA50 is a lagging filter:
    # during a sharp move the price decouples from the EMA before the EMA's
    # slope catches up. Combined verdict: regime is non-flat if EITHER
    # slope OR price-distance crosses its threshold (price wins on
    # disagreement — it's the leading indicator).
    btc_regime_price_dist_threshold_pct: float = 0.5

    # ----- BTC regime "self-trend" override for Type A -----
    # Hypothesis: when a symbol is moving on its own (decoupled from BTC),
    # the BTC-regime hard gate over-blocks. Specifically: a Type A LONG on
    # an alt that just printed N green 30m candles in a row is much more
    # likely to be a genuine independent uptrend than a BTC-correlated
    # bounce that will get dumped at the next BTC tick. Mirror logic for
    # SHORT (N red 30m candles).
    #
    # The override applies ONLY to Type A (A long under BTC=down, A short
    # under BTC=up). Other entry types keep the standard BTC regime gate.
    # Decisions where the override fires are logged so we can measure WR
    # of override-permitted A trades vs the population.
    a_self_trend_override_enabled: bool = True
    # Number of consecutive 30m candles that must be in-trend (close > open
    # for long, close < open for short). 5 bars = 2.5 hours of own move.
    # 30m candles are aggregated client-side from the existing 15m fetch,
    # so this costs zero additional REST calls.
    a_self_trend_bars: int = 5

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
