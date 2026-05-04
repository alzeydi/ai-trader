"""Safety mode (NoFx pattern) — DB-backed pause after a losing streak.

The state is loaded lazily on first access from `safety_state` and persisted
on every mutation, so a restart preserves an active pause.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from src.config import settings
from src.persistence.db import (
    SessionFactory,
    get_last_n_closed_trades,
    load_safety_state,
    make_session_factory,
    save_safety_state,
)

log = logging.getLogger(__name__)


class SafetyMode:
    def __init__(
        self,
        session_factory: SessionFactory | None = None,
        notifier=None,
    ) -> None:
        self.session_factory: SessionFactory = session_factory or make_session_factory()
        self.notifier = notifier
        self._loaded = False
        self._paused_until: datetime | None = None
        self._consecutive_losses: int = 0

    # --- lazy load on first access ------------------------------------
    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        row = load_safety_state(self.session_factory)
        if row is not None:
            paused = row.paused_until
            if paused is not None and paused.tzinfo is None:
                paused = paused.replace(tzinfo=timezone.utc)
            self._paused_until = paused
            self._consecutive_losses = int(row.consecutive_losses or 0)
        self._loaded = True

    # --- public API ---------------------------------------------------
    @property
    def consecutive_losses(self) -> int:
        self._ensure_loaded()
        return self._consecutive_losses

    @property
    def until(self) -> datetime | None:
        self._ensure_loaded()
        return self._paused_until

    def is_active(self) -> bool:
        self._ensure_loaded()
        if self._paused_until is None:
            return False
        if datetime.now(tz=timezone.utc) >= self._paused_until:
            self._paused_until = None
            self._consecutive_losses = 0
            self._persist(reason="auto-expired")
            return False
        return True

    # Back-compat shim for older callers.
    def is_paused(self) -> bool:
        return self.is_active()

    def activate(self, hours: int, reason: str = "loss_streak") -> None:
        self._ensure_loaded()
        self._paused_until = datetime.now(tz=timezone.utc) + timedelta(hours=hours)
        self._persist(reason=reason)

    def check_after_close(self, trade_pnl: float) -> bool:
        """Update the streak counter and activate safety mode if tripped.

        Returns `True` iff this call activated safety mode.
        """
        self._ensure_loaded()
        if trade_pnl < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

        threshold = settings.safety_max_consecutive_losses
        recent = get_last_n_closed_trades(self.session_factory, threshold)
        if len(recent) < threshold:
            self._persist(reason=None)
            return False

        if all((t.pnl_usd or 0.0) < 0 for t in recent):
            hours = settings.safety_pause_hours
            self.activate(hours=hours, reason=f"{threshold}_consecutive_losses")
            msg = (
                f"⛔ Safety mode активирован: {threshold} убытков подряд. "
                f"Бот остановлен на {hours} часов."
            )
            log.warning(msg)
            if self.notifier is not None:
                try:
                    self.notifier.error(msg)
                except Exception as exc:  # noqa: BLE001
                    log.warning("safety_mode: failed to notify: %s", exc)
            return True

        self._persist(reason=None)
        return False

    # Legacy adapter for the previous in-memory API used by tests.
    def record_trade(self, pnl_usd: float) -> None:
        self.check_after_close(pnl_usd)

    # --- internals ----------------------------------------------------
    def _persist(self, *, reason: str | None) -> None:
        try:
            save_safety_state(
                self.session_factory,
                paused_until=self._paused_until,
                consecutive_losses=self._consecutive_losses,
                reason=reason,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("safety_mode: failed to persist state: %s", exc)
