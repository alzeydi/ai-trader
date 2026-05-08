"""Regression tests for the side-based exit-fill fallback.

Background: Binance migrated conditional orders to the algo endpoint on
2025-12-09. SL/TP fills surfaced via `fetchMyTrades` after that change
do not reliably carry `reduceOnly` and surface their order id under
`algoOrderId` rather than `orderId`. The reduceOnly+oid partitioner in
`OrderExecutor._split_fills` therefore classes them as "entry" leftovers,
`vwap_exit` falls to 0, and the live close path falls back on
`_mark_price` — recording the current mark as the close price instead
of the actual fill VWAP.

Observed in QNT #200 on 2026-05-08: SL/TP triggered at VWAP 73.05 but
the trade was recorded with close_price 72.59 (mark at the moment of
detection). The sign of the PnL was inverted in the process.

This test pins down the side-based fallback that recovers the correct
exit fills when both reduceOnly and oid signals fail.
"""

from __future__ import annotations

from src.execution.orders import OrderExecutor


def _trade(*, side: str, price: float, amount: float, oid: str = "x", reduce: bool = False) -> dict:
    """Build a fetchMyTrades-shaped dict that exercises the partitioner."""
    return {
        "side": side,
        "price": price,
        "amount": amount,
        "order": oid,
        "reduceOnly": reduce,
        "info": {"reduceOnly": "true" if reduce else "false", "orderId": oid},
        "fee": {"currency": "USDT", "cost": 0.0},
    }


# ── partition_by_side primitive ─────────────────────────────────────────────

def test_partition_short_position_buys_are_exits():
    """For a short position the closing side is `buy`; sells are entries."""
    trades = [
        _trade(side="sell", price=73.01, amount=17.4),  # short open
        _trade(side="buy",  price=73.05, amount=17.1),  # close leg 1
        _trade(side="buy",  price=73.04, amount=0.3),   # close leg 2
    ]
    exits, entries = OrderExecutor._partition_by_side(trades, exit_side="buy")
    assert len(exits) == 2
    assert len(entries) == 1
    assert all(t["side"] == "buy" for t in exits)


def test_partition_long_position_sells_are_exits():
    trades = [
        _trade(side="buy",  price=100.0, amount=10.0),  # long open
        _trade(side="sell", price=101.5, amount=10.0),  # close
    ]
    exits, entries = OrderExecutor._partition_by_side(trades, exit_side="sell")
    assert len(exits) == 1
    assert len(entries) == 1


def test_partition_drops_trades_without_side():
    """Trades missing a `side` field are dropped — silently misclassifying
    them as either side could cause double-counting."""
    trades = [
        _trade(side="buy", price=100.0, amount=1.0),
        {"price": 100.0, "amount": 1.0},  # no `side` key
    ]
    exits, entries = OrderExecutor._partition_by_side(trades, exit_side="buy")
    assert len(exits) == 1
    assert len(entries) == 0


# ── VWAP recovery on the QNT scenario ───────────────────────────────────────

def test_vwap_recovers_qnt_200_exit_price():
    """Reproduces the QNT #200 close: 17.1 @ 73.05 + 0.3 @ 73.04 → VWAP 73.0498.

    With the previous logic, neither of these fills carried reduceOnly nor
    matched the cached SL/TP order ids (algo-order migration), so they
    were misclassified as entries, vwap_exit was 0, and the close fell
    back on `_mark_price`. The side-based fallback restores the correct
    VWAP."""
    trades = [
        _trade(side="sell", price=73.01, amount=17.4, oid="entry-1", reduce=False),
        _trade(side="buy",  price=73.05, amount=17.1, oid="algo-A",  reduce=False),
        _trade(side="buy",  price=73.04, amount=0.3,  oid="algo-A",  reduce=False),
    ]
    # Primary partitioner sees no reduceOnly and oid mismatch — exits empty.
    exits, _ = OrderExecutor._split_fills(
        trades, entry_order_id="entry-1", exit_order_ids=("stop-old", "take-old"),
    )
    assert exits == []

    # Side-based fallback recovers them.
    fb_exits, fb_entries = OrderExecutor._partition_by_side(trades, exit_side="buy")
    assert len(fb_exits) == 2
    # (73.05 × 17.1 + 73.04 × 0.3) / 17.4 ≈ 73.0498
    assert abs(OrderExecutor._vwap(fb_exits) - 73.0498) < 1e-3


def test_vwap_long_exit_price():
    """Same recovery for a long position: sells are exits."""
    trades = [
        _trade(side="buy",  price=100.0, amount=5.0),
        _trade(side="sell", price=101.0, amount=2.0),
        _trade(side="sell", price=101.5, amount=3.0),
    ]
    fb_exits, _ = OrderExecutor._partition_by_side(trades, exit_side="sell")
    assert len(fb_exits) == 2
    # (101 × 2 + 101.5 × 3) / 5 = (202 + 304.5) / 5 = 506.5 / 5 = 101.3
    assert OrderExecutor._vwap(fb_exits) == 101.3
