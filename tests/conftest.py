"""Shared pytest fixtures and markers for the KABOCS test suite.

The suite is split into two tiers:

* **Pure tests**: exercise modules with only NumPy / pandas / scikit-learn
  dependencies (``kabo.utils``, ``kabo.candidate``, ``kabo.feature_selection``).
  These always run.
* **Integration tests**: exercise the full optimizer pipeline and therefore
  pull in ``torch`` / ``botorch`` / ``gpytorch``.  They are guarded by the
  ``requires_torch`` marker and skipped automatically when the optional
  stack is unavailable.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Dependency probe
# ---------------------------------------------------------------------------
def _has_module(name: str) -> bool:
    try:
        importlib.import_module(name)
    except Exception:
        return False
    return True


HAS_TORCH = _has_module("torch")
HAS_BOTORCH = _has_module("botorch")
HAS_TORCH_STACK = HAS_TORCH and HAS_BOTORCH and _has_module("gpytorch")


# ---------------------------------------------------------------------------
# Auto-skip integration tests when the torch stack is unavailable
# ---------------------------------------------------------------------------
def pytest_collection_modifyitems(config, items):
    if HAS_TORCH_STACK:
        return
    skip_marker = pytest.mark.skip(
        reason="requires torch / botorch / gpytorch (optional extras)",
    )
    for item in items:
        if "requires_torch" in item.keywords or "integration" in item.keywords:
            item.add_marker(skip_marker)


# ---------------------------------------------------------------------------
# Common fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Absolute path to the repository root (one level above ``tests/``)."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def test_data_csv(repo_root: Path) -> Path:
    """Minimal 3-feature / 1-product dataset shipped with the repo."""
    p = repo_root / "data" / "test_data.csv"
    if not p.exists():
        pytest.skip(f"test data file missing: {p}")
    return p


@pytest.fixture(scope="session")
def test_candidates_csv(repo_root: Path) -> Path:
    p = repo_root / "data" / "test_candidates.csv"
    if not p.exists():
        pytest.skip(f"test candidates file missing: {p}")
    return p
