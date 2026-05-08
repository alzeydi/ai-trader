"""Layer-3 orchestrator: signal → veto → risk → sizing → execution.

`Trader.run_cycle(symbols)` is the per-tick entry point used by `main.py`.
Each step is logged to the DB through the components it delegates to
(`AIDecision` for veto, `Trade` for execution); the orchestrator itself only
emits structured logs and Telegram notifications.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pydantic import BaseModel

from src.config import settings
from src.data.context import get_market_context
from src.execution.orders import OrderExecutor
from src.execution.types import ExecutionResult, TradeOrder
from src.llm.types import VetoResponse
from src.llm.veto import veto_candidate
from src.persistence.csv_writer import append_decision
from src.persistence.db import (
    SessionFactory,
    count_open_trades,
    daily_realized_pnl,
    has_recent_losing_b_trade,
    has_recent_type_e_decision,
    has_recent_type_g_decision,
    make_session_factory,
)
from src.risk.btc_regime import get_btc_regime, is_blocked as btc_regime_blocks
from src.risk.limits import can_take_signal
from src.risk.sizing import size_position
from src.risk.types import AccountState
from src.signal.types import CandidateSignal

if TYPE_CHECKING:
    from src.data.binance_client import BinanceClient
    from src.llm.client import ClaudeClient
    from src.notify.telegram import TelegramNotifier
    from src.risk.safety_mode import SafetyMode
    from src.signal.engine import SignalEngine


log = logging.getLogger(__name__)


__all__ = ["Trader", "TradeOrder", "CycleResult"]


class CycleResult(BaseModel):
    symbol: str
    accepted: bool
    reason: str
    candidate: CandidateSignal | None = None
    veto: VetoResponse | None = None
    order: TradeOrder | None = None
    execution: ExecutionResult | None = None


@dataclass
class Trader:
    binance: "BinanceClient"
    engine: "SignalEngine"
    llm_client: "ClaudeClient"
    executor: OrderExecutor
    safety: "SafetyMode"
    notifier: "TelegramNotifier | None" = None
    session_factory: SessionFactory = field(default=None)  # type: ignore[assignment]
    equity_usd: float = field(default_factory=lambda: settings.equity_usd)

    def __post_init__(self) -> None:
        if self.session_factory is None:
            self.session_factory = make_session_factory()

    # ------------------------------------------------------------------
    def run_cycle(self, symbols: list[str]) -> list[CycleResult]:
        results: list[CycleResult] = []
        for symbol in symbols:
            try:
                results.append(self._process_symbol(symbol))
            except Exception as exc:  # noqa: BLE001
                log.exception("trader: error processing %s: %s", symbol, exc)
                results.append(
                    CycleResult(symbol=symbol, accepted=False, reason=f"error: {exc}")
                )
                if self.notifier is not None:
                    try:
                        self.notifier.error(f"{symbol}: {exc}")
                    except Exception:  # noqa: BLE001
                        pass
        return results

    # ------------------------------------------------------------------
    def _process_symbol(self, symbol: str) -> CycleResult:
        # 1. Generate candidate.
        signal = self.engine.generate_signals(symbol)
        if signal is None:
            return CycleResult(symbol=symbol, accepted=False, reason="no candidate")

        # 1b. Type-B post-loss cooldown. A single losing fade on a symbol
        # within the cooldown window is treated as a regime signal; suppress
        # new B candidates on that symbol until it expires. Cheaper than a
        # veto call — we never even ask the LLM.
        if signal.entry_type == "B" and has_recent_losing_b_trade(
            self.session_factory,
            symbol,
            settings.b_loss_cooldown_hours,
        ):
            log.info("%s: Type B suppressed by post-loss cooldown", symbol)
            return CycleResult(
                symbol=symbol,
                accepted=False,
                reason="b_loss_cooldown",
                candidate=signal,
            )

        # 1b'. Type-E per-symbol 24h cooldown. Strategy E re-evaluates the
        # in-progress H4 bar every loop tick, so a qualifying setup would
        # otherwise re-fire every 60 s for the rest of the bar. The spec
        # mandates one fire per symbol per 24h. Checked BEFORE the LLM call
        # so vetoed E's don't burn tokens every cycle either.
        if signal.entry_type == "E" and has_recent_type_e_decision(
            self.session_factory,
            symbol,
            settings.ma25_symbol_cooldown_hours,
        ):
            log.info("%s: Type E suppressed by 24h per-symbol cooldown", symbol)
            return CycleResult(
                symbol=symbol,
                accepted=False,
                reason="e_symbol_cooldown",
                candidate=signal,
            )

        # Same dedup for Type G (mirror of E): re-evaluates the live H4 bar
        # every loop tick, so without this it would re-fire every cycle.
        if signal.entry_type == "G" and has_recent_type_g_decision(
            self.session_factory,
            symbol,
            settings.ma25_symbol_cooldown_hours,
        ):
            log.info("%s: Type G suppressed by 24h per-symbol cooldown", symbol)
            return CycleResult(
                symbol=symbol,
                accepted=False,
                reason="g_symbol_cooldown",
                candidate=signal,
            )

        # 1d. Global BTC regime gate. BTC up → longs only; BTC down → shorts
        # only; flat → both allowed. Fail-open on fetch errors (regime=None).
        # Runs BEFORE the LLM call so vetoed candidates don't burn tokens.
        #
        # Self-trend override (Type A only): when the symbol just printed
        # `a_self_trend_bars` consecutive in-trend 30m candles, the gate is
        # skipped. The hypothesis is that an alt moving on its own is more
        # likely on a genuine independent trend than a BTC-correlated
        # bounce. Both directions covered (A long under BTC=down with green
        # 30m run; A short under BTC=up with red 30m run). Logged as a
        # "surrogate decision" so we can measure WR of override-permitted
        # A trades vs the population.
        if settings.btc_regime_enabled:
            regime = get_btc_regime(self.binance)
            if btc_regime_blocks(regime, signal.side):
                if signal.entry_type == "A" and signal.self_trend_override:
                    log.info(
                        "%s: %s A_self_trend_override active — would have "
                        "been blocked by btc_regime=%s, %d in-trend 30m "
                        "candles let it through",
                        symbol, signal.side, regime,
                        int(settings.a_self_trend_bars),
                    )
                else:
                    log.info(
                        "%s: %s suppressed by BTC regime gate (regime=%s)",
                        symbol, signal.side, regime,
                    )
                    return CycleResult(
                        symbol=symbol,
                        accepted=False,
                        reason=f"btc_regime_{regime}",
                        candidate=signal,
                    )

        # 1c. Fast account pre-check — runs BEFORE market-context REST calls
        # and the LLM veto so we don't spend tokens or rate-limit budget when
        # the account cannot physically take another trade.
        account = self._account_state()
        skip, skip_reason = self._pre_trade_check(account)
        if skip:
            log.debug("%s: pre-trade skip — %s", symbol, skip_reason)
            return CycleResult(
                symbol=symbol, accepted=False, reason=skip_reason, candidate=signal,
            )

        # 2. Build veto inputs from market context + DB-derived account state.
        ctx_payload = self._market_context_payload(signal)
        account_payload = {
            "open_positions": account.open_positions,
            "daily_pnl_pct": account.daily_pnl_pct,
            "losses_streak": account.consecutive_losses,
        }

        # 3. LLM veto (with pre-flight hard rules).
        veto = veto_candidate(
            signal,
            context=ctx_payload,
            account=account_payload,
            client=self.llm_client,
        )
        append_decision(symbol, veto.decision, veto.confidence, veto.reasoning)

        # 4. Composite gate (LLM + account + safety mode).
        ok, reason = can_take_signal(signal, veto, account, self.safety)
        if not ok:
            log.info("%s: rejected — %s", symbol, reason)
            return CycleResult(
                symbol=symbol,
                accepted=False,
                reason=reason,
                candidate=signal,
                veto=veto,
            )

        # 5. Sizing.
        order = size_position(
            signal, account, veto,
            market_max_qty=self._market_max_qty(symbol),
        )
        if order is None:
            return CycleResult(
                symbol=symbol,
                accepted=False,
                reason="size rejected (target too small vs fees)",
                candidate=signal,
                veto=veto,
            )

        # 6. Execution.
        exec_result = self.executor.open_position(order)
        if exec_result.success and self.notifier is not None:
            try:
                # Use the rich entry payload so users always see margin used,
                # leverage, R:R and the LLM rationale — not just qty/price.
                margin_usd = (
                    order.entry_price * order.quantity / order.leverage
                    if order.leverage else order.entry_price * order.quantity
                )
                self.notifier.send_entry_signal(
                    order,
                    margin_usd=margin_usd,
                    ai_reasoning=veto.reasoning,
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("notifier entry failed: %s", exc)

        return CycleResult(
            symbol=symbol,
            accepted=exec_result.success,
            reason=exec_result.reason or "ok",
            candidate=signal,
            veto=veto,
            order=order,
            execution=exec_result,
        )

    # ------------------------------------------------------------------
    def _market_context_payload(self, signal: CandidateSignal) -> dict:
        try:
            mc = get_market_context(signal.symbol, self.binance)
        except Exception as exc:  # noqa: BLE001
            log.warning("market_context fetch failed for %s: %s", signal.symbol, exc)
            return {}

        funding_pct = mc.funding_rate * 100.0 if mc.funding_rate is not None else None
        liq_total = mc.liquidations_4h_usd
        # Until we split liquidations by side, attribute the full notional
        # to the trade's own side (worst-case for the squeeze rule); leave
        # the opposite side as None so the LLM treats it as "not measured"
        # rather than "zero".
        return {
            "funding_rate_now": funding_pct,
            "funding_rate_8h_avg": None,
            "oi_delta_1h_pct": mc.open_interest_delta_1h,
            "liq_long": liq_total if signal.side == "long" else None,
            "liq_short": liq_total if signal.side == "short" else None,
            "btc_dominance": mc.btc_dominance,
            "btc_direction": mc.btc_1h_direction,
        }

    def _pre_trade_check(self, account: AccountState) -> tuple[bool, str]:
        """Cheap deterministic gate evaluated before market-context REST calls
        and the LLM veto.  Returns (should_skip, reason).

        Mirrors the hard rules inside preflight_check / can_take_signal but
        runs earlier so we never pay for a Claude API call or funding/OI fetch
        when the account cannot open another trade regardless of signal quality.
        """
        if account.open_positions >= settings.max_open_positions:
            return True, f"max_open_positions={settings.max_open_positions} reached"
        if account.daily_pnl_pct <= -float(settings.max_daily_loss_pct):
            return True, f"daily DD hit: {account.daily_pnl_pct:.2f}%"
        if self.safety.is_active():
            return True, f"safety mode active until {self.safety.until}"
        # Margin floor: if free margin is below the minimum needed for a
        # policy-sized position (equity × max_margin_per_trade_pct), sizing
        # will clamp the order to a micro-notional that doesn't survive the
        # min_notional_usd filter anyway.  Skip early and save the LLM call.
        if account.free_margin_usd is not None and settings.leverage > 0:
            min_margin = account.equity * float(settings.max_margin_per_trade_pct)
            if account.free_margin_usd < min_margin:
                return True, (
                    f"insufficient free margin: "
                    f"${account.free_margin_usd:.2f} < ${min_margin:.2f} required"
                )
        return False, "ok"

    def _market_max_qty(self, symbol: str) -> float | None:
        """Per-order MAX_QTY from the cached ccxt markets table.

        ccxt populates `limits.amount.max` from Binance's MARKET_LOT_SIZE /
        LOT_SIZE filters. Markets are loaded once at startup
        (`_filter_known_symbols`), so this is a dict lookup, not a REST call.
        Returns None on any failure so sizing falls back to "no clamp" rather
        than blocking entries on a metadata hiccup.
        """
        try:
            market = self.binance.exchange.markets.get(symbol) if (
                self.binance is not None and self.binance.exchange.markets
            ) else None
        except Exception as exc:  # noqa: BLE001
            log.warning("market lookup failed for %s: %s", symbol, exc)
            return None
        if not market:
            return None
        amount_limits = (market.get("limits") or {}).get("amount") or {}
        max_amount = amount_limits.get("max")
        try:
            return float(max_amount) if max_amount is not None else None
        except (TypeError, ValueError):
            return None

    def _account_state(self) -> AccountState:
        # Live balance from the exchange takes precedence over the static
        # `settings.equity_usd` fallback. Without this, sizing computes risk
        # against a phantom $1000 even when the demo wallet holds far less,
        # which produces orders the exchange rejects with -2019.
        equity = float(self.equity_usd)
        free_margin: float | None = None
        try:
            bal = self.binance.fetch_balance()
            usdt = (bal.get("USDT") or {}) if isinstance(bal, dict) else {}
            total = usdt.get("total")
            free = usdt.get("free")
            if total is not None:
                equity = float(total)
            if free is not None:
                free_margin = float(free)
        except Exception as exc:  # noqa: BLE001
            log.warning("fetch_balance failed, using static equity_usd: %s", exc)

        open_pos = count_open_trades(self.session_factory)
        realized = daily_realized_pnl(self.session_factory)
        daily_pct = (realized / equity * 100.0) if equity > 0 else 0.0
        return AccountState(
            equity=equity,
            open_positions=int(open_pos),
            daily_pnl_pct=daily_pct,
            consecutive_losses=self.safety.consecutive_losses,
            free_margin_usd=free_margin,
        )

    # ------------------------------------------------------------------
    # Legacy entry point used by the previous main.py wiring. Kept until
    # backtest.py / older scripts stop calling it.
    def execute(
        self,
        signal: CandidateSignal,
        veto: VetoResponse,
        risk_state,  # legacy RiskState
        equity_usd: float,
    ):
        from src.risk.limits import check

        if self.safety.is_active():
            return _LegacyOutcome(False, "safety mode paused")
        ok, reason = check(risk_state)
        if not ok:
            return _LegacyOutcome(False, f"risk gate: {reason}")

        account = AccountState(
            equity=float(equity_usd),
            open_positions=risk_state.open_positions,
            daily_pnl_pct=risk_state.daily_pnl_pct,
            consecutive_losses=self.safety.consecutive_losses,
        )
        order = size_position(
            signal, account, veto,
            market_max_qty=self._market_max_qty(signal.symbol),
        )
        if order is None:
            return _LegacyOutcome(False, "size rejected")
        result = self.executor.open_position(order)
        return _LegacyOutcome(result.success, result.reason or "ok", order, result)


@dataclass
class _LegacyOutcome:
    accepted: bool
    reason: str
    order: TradeOrder | None = None
    execution: ExecutionResult | None = None
