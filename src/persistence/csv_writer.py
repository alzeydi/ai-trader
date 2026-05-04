"""CSV mirrors of key tables for the dashboard and external tooling."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import settings


def _path(rel: str) -> Path:
    p = Path(rel)
    if not p.is_absolute():
        p = settings.project_root / p
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _append(path: Path, headers: list[str], row: dict[str, Any]) -> None:
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=headers)
        if new_file:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in headers})


def append_equity(equity_usd: float, realized: float = 0.0, unrealized: float = 0.0) -> None:
    _append(
        _path(settings.equity_csv),
        ["ts", "equity_usd", "realized_pnl", "unrealized_pnl"],
        {
            "ts": datetime.now(tz=timezone.utc).isoformat(),
            "equity_usd": equity_usd,
            "realized_pnl": realized,
            "unrealized_pnl": unrealized,
        },
    )


def append_decision(symbol: str, decision: str, confidence: float, reason: str) -> None:
    _append(
        _path(settings.decisions_csv),
        ["ts", "symbol", "decision", "confidence", "reason"],
        {
            "ts": datetime.now(tz=timezone.utc).isoformat(),
            "symbol": symbol,
            "decision": decision,
            "confidence": confidence,
            "reason": reason,
        },
    )
