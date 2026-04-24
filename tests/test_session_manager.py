"""Unit tests for ``webui.backend.runner.SessionManager`` (v1.2 registry).

These tests never spin up the real BO pipeline — they monkey-patch the
``SessionRunner.start`` thread launch so ``SessionManager`` can be
exercised without importing torch/botorch or touching the filesystem
beyond ``tmp_path``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# Skip if FastAPI / pydantic aren't available (the webui is optional).
_fastapi = pytest.importorskip("fastapi")


@pytest.fixture
def no_thread_start(monkeypatch, tmp_path: Path):
    """Replace ``SessionRunner.start`` with a no-op that only toggles status.

    Also redirects the registry root to ``tmp_path`` so on-disk archives
    do not pollute the repo.
    """
    from webui.backend import runner as runner_module

    def _fake_start(self):
        self.status = "running"

    monkeypatch.setattr(runner_module.SessionRunner, "start", _fake_start)
    monkeypatch.setattr(runner_module, "RUNS_ROOT", tmp_path / "runs")
    return runner_module


def _make_manager(runner_module):
    hub = runner_module.EventHub()
    return runner_module.SessionManager(hub)


def _basic_cfg(runner_module):
    return runner_module.RunConfig(task="test", interactive=False)


# ---------------------------------------------------------------------------
# Registry basics
# ---------------------------------------------------------------------------
class TestRegistry:
    def test_empty_state(self, no_thread_start):
        mgr = _make_manager(no_thread_start)
        assert mgr.current is None
        assert mgr.list() == []
        assert mgr.list_snapshots() == []

    def test_start_adds_to_registry(self, no_thread_start):
        mgr = _make_manager(no_thread_start)
        runner = mgr.start(_basic_cfg(no_thread_start))
        assert runner.run_id in {r.run_id for r in mgr.list()}
        assert mgr.get(runner.run_id) is runner

    def test_current_prefers_active_over_terminal(self, no_thread_start):
        mgr = _make_manager(no_thread_start)
        first = mgr.start(_basic_cfg(no_thread_start))
        first.status = "done"
        second = mgr.start(_basic_cfg(no_thread_start))
        assert mgr.current is second
        # When active is finished, current falls back to the most recent one.
        second.status = "done"
        assert mgr.current is second

    def test_start_rejects_concurrent_by_default(self, no_thread_start):
        mgr = _make_manager(no_thread_start)
        mgr.start(_basic_cfg(no_thread_start))
        with pytest.raises(RuntimeError, match="already active"):
            mgr.start(_basic_cfg(no_thread_start))

    def test_start_allows_concurrent_when_opted_in(self, no_thread_start):
        mgr = _make_manager(no_thread_start)
        a = mgr.start(_basic_cfg(no_thread_start))
        b = mgr.start(_basic_cfg(no_thread_start), allow_concurrent=True)
        assert a.run_id != b.run_id
        # Both should still be in the registry.
        ids = {r.run_id for r in mgr.list()}
        assert {a.run_id, b.run_id} <= ids


# ---------------------------------------------------------------------------
# Abort / remove semantics
# ---------------------------------------------------------------------------
class TestAbortAndRemove:
    def test_abort_by_id_flips_status(self, no_thread_start):
        mgr = _make_manager(no_thread_start)
        runner = mgr.start(_basic_cfg(no_thread_start))
        # Stub the bridge.abort call to avoid touching the real EventHub.
        runner.bridge.abort = lambda: None  # type: ignore[assignment]
        ok = mgr.abort(runner.run_id)
        assert ok is True
        assert runner.status == "aborted"

    def test_abort_unknown_returns_false(self, no_thread_start):
        mgr = _make_manager(no_thread_start)
        assert mgr.abort("does-not-exist") is False

    def test_abort_with_no_id_targets_current(self, no_thread_start):
        mgr = _make_manager(no_thread_start)
        r = mgr.start(_basic_cfg(no_thread_start))
        r.bridge.abort = lambda: None  # type: ignore[assignment]
        assert mgr.abort() is True
        assert r.status == "aborted"

    def test_remove_refuses_active_sessions(self, no_thread_start):
        mgr = _make_manager(no_thread_start)
        r = mgr.start(_basic_cfg(no_thread_start))
        assert mgr.remove(r.run_id) is False
        # Still in registry.
        assert mgr.get(r.run_id) is r

    def test_remove_drops_terminal_sessions(self, no_thread_start):
        mgr = _make_manager(no_thread_start)
        r = mgr.start(_basic_cfg(no_thread_start))
        r.status = "done"
        assert mgr.remove(r.run_id) is True
        assert mgr.get(r.run_id) is None


# ---------------------------------------------------------------------------
# Registry eviction
# ---------------------------------------------------------------------------
class TestEviction:
    def test_eviction_caps_registry_and_spares_active(self, no_thread_start, monkeypatch):
        monkeypatch.setattr(no_thread_start, "_MAX_REGISTRY_SIZE", 3)
        mgr = _make_manager(no_thread_start)

        # Fill past the cap with terminal sessions + one active one.
        terminals = []
        for _ in range(5):
            r = mgr.start(_basic_cfg(no_thread_start), allow_concurrent=True)
            r.status = "done"
            terminals.append(r)

        active = mgr.start(_basic_cfg(no_thread_start), allow_concurrent=True)

        # Cap enforced on the next insert.
        mgr.start(_basic_cfg(no_thread_start), allow_concurrent=True)

        ids = {r.run_id for r in mgr.list()}
        # Active session must still be present.
        assert active.run_id in ids
        # Registry must not exceed the cap.
        assert len(ids) <= no_thread_start._MAX_REGISTRY_SIZE
