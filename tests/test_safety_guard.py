"""Startup safety-guard tests for live mainnet with relaxed gates."""

from __future__ import annotations

import pytest

from src.config import settings
from src.main import _enforce_live_safety_gates


def _make_unsafe(monkeypatch):
    """Live + mainnet + real fills + every gate effectively disabled."""
    monkeypatch.setattr(settings, "trading_mode", "live")
    monkeypatch.setattr(settings, "binance_testnet", False)
    monkeypatch.setattr(settings, "paper_trading", False)
    monkeypatch.setattr(settings, "dry_run", False)
    monkeypatch.setattr(settings, "max_daily_loss_pct", 100.0)
    monkeypatch.setattr(settings, "safety_max_consecutive_losses", 10000)
    monkeypatch.setattr(settings, "max_open_positions", 12)


def test_blocks_live_mainnet_with_disabled_gates_unless_acked(monkeypatch):
    _make_unsafe(monkeypatch)
    monkeypatch.setattr(settings, "allow_unsafe_live", False)
    with pytest.raises(RuntimeError, match="safety gates disabled"):
        _enforce_live_safety_gates()


def test_allows_live_mainnet_with_disabled_gates_when_acked(monkeypatch):
    _make_unsafe(monkeypatch)
    monkeypatch.setattr(settings, "allow_unsafe_live", True)
    # Should return without raising.
    _enforce_live_safety_gates()


def test_allows_paper_mode_regardless(monkeypatch):
    _make_unsafe(monkeypatch)
    monkeypatch.setattr(settings, "paper_trading", True)
    monkeypatch.setattr(settings, "allow_unsafe_live", False)
    _enforce_live_safety_gates()


def test_allows_testnet_regardless(monkeypatch):
    _make_unsafe(monkeypatch)
    monkeypatch.setattr(settings, "binance_testnet", True)
    monkeypatch.setattr(settings, "allow_unsafe_live", False)
    _enforce_live_safety_gates()


def test_allows_dry_run_regardless(monkeypatch):
    _make_unsafe(monkeypatch)
    monkeypatch.setattr(settings, "dry_run", True)
    monkeypatch.setattr(settings, "allow_unsafe_live", False)
    _enforce_live_safety_gates()


def test_allows_live_mainnet_with_sane_gates(monkeypatch):
    _make_unsafe(monkeypatch)
    monkeypatch.setattr(settings, "max_daily_loss_pct", 3.0)
    monkeypatch.setattr(settings, "safety_max_consecutive_losses", 3)
    monkeypatch.setattr(settings, "max_open_positions", 3)
    monkeypatch.setattr(settings, "allow_unsafe_live", False)
    _enforce_live_safety_gates()
