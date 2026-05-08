"""Tests for TelegramNotifier.send_close_signal pnl/fee/ROI accounting.

The `pnl_usd` column in the trades table is NET of round-trip taker fees
(see execution/orders.py). The Telegram close message must therefore treat
that field as net and compute gross by adding fees back — otherwise fees
are double-deducted in the user-facing message.
"""

from __future__ import annotations

from types import SimpleNamespace

from src.notify.telegram import TelegramNotifier


def _trade(**overrides):
    """Build a trade-like object that satisfies what send_close_signal reads
    via _get (attribute access first, dict fallback)."""
    base = {
        "symbol": "QNT/USDT:USDT",
        "side": "short",
        "entry": 73.01,
        "close_price": 72.59,
        "quantity": 17.4,
        "leverage": 3.0,
        "pnl_usd": 6.2946,    # NET — what the DB row actually carries
        "close_reason": "sl_or_tp_hit",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_close_message_does_not_double_deduct_fees(monkeypatch):
    """Net in the message must equal pnl_usd; gross must add fees back."""
    notifier = TelegramNotifier.__new__(TelegramNotifier)
    sent: list[str] = []
    notifier.send = lambda msg: sent.append(msg)

    notifier.send_close_signal(_trade(), fees_usd=1.0134)

    assert len(sent) == 1
    msg = sent[0]
    # Real net == DB pnl_usd, NOT pnl_usd minus fees.
    assert "Net PnL: `+6.29` USDT" in msg, msg
    # Gross is reconstructed = net + fees.
    assert "Gross PnL: `+7.31` USDT" in msg, msg
    assert "Fees: `1.0134` USDT" in msg


def test_close_message_roi_uses_real_net(monkeypatch):
    """ROI = net / margin × 100; with net=6.2946, qty=17.4, entry=73.01,
    leverage=3 → margin = 17.4 × 73.01 / 3 ≈ 423.46 → ROI ≈ 1.49 %."""
    notifier = TelegramNotifier.__new__(TelegramNotifier)
    sent: list[str] = []
    notifier.send = lambda msg: sent.append(msg)

    notifier.send_close_signal(_trade(), fees_usd=1.0134)

    msg = sent[0]
    # 1.49% — within rounding of (6.2946 / 423.458) * 100.
    assert "ROI: *+1.49%*" in msg, msg


def test_close_message_zero_fees_unchanged(monkeypatch):
    """When fees are zero (e.g. backtest harness without taker model), gross
    and net coincide and no double-deduct can happen."""
    notifier = TelegramNotifier.__new__(TelegramNotifier)
    sent: list[str] = []
    notifier.send = lambda msg: sent.append(msg)

    notifier.send_close_signal(_trade(pnl_usd=10.0), fees_usd=0.0)

    msg = sent[0]
    assert "Gross PnL: `+10.00` USDT" in msg
    assert "Net PnL: `+10.00` USDT" in msg
    assert "Fees: `0.0000` USDT" in msg


def test_close_message_losing_trade_keeps_signs(monkeypatch):
    """Negative net stays negative; gross before fees still negative when the
    underlying move was the wrong way."""
    notifier = TelegramNotifier.__new__(TelegramNotifier)
    sent: list[str] = []
    notifier.send = lambda msg: sent.append(msg)

    # Net = -3.0, fees = 1.0 → gross = -2.0 (the price move lost $2, fees
    # took another dollar).
    notifier.send_close_signal(_trade(pnl_usd=-3.0), fees_usd=1.0)

    msg = sent[0]
    assert "Net PnL: `-3.00` USDT" in msg, msg
    assert "Gross PnL: `-2.00` USDT" in msg, msg
