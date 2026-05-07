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


@pytest.fixture(autouse=True)
def _clear_veto_cache():
    """Clear the in-process LLM veto decision cache before each test.

    _DECISION_CACHE is module-level and persists between tests in the same
    process. Without this, a TAKE verdict cached by an earlier test leaks
    into a later test that expects a fresh LLM call, producing ghost decisions
    (wrong confidence, wrong reasoning) that break assertions.
    """
    import src.llm.veto as veto_mod
    veto_mod._DECISION_CACHE.clear()
    yield
    veto_mod._DECISION_CACHE.clear()
