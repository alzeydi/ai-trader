"""Shared pytest fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure project root is on sys.path so `src.*` imports work without install.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    """Redirect every DB write to an isolated per-test SQLite file.

    Tests that exercise persistence (veto, risk, execution) would otherwise
    write to the project's real `data/trades.db`.
    """
    monkeypatch.setattr("src.config.settings.db_path", str(tmp_path / "test.db"))
    yield
