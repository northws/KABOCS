"""
SessionRunner — orchestrates a single WebUI-driven KABO optimization run.

Lifecycle
---------
1. ``start(config)`` validates the config, builds a ``KABOOptimizer``
   from ``kabo.task.get_task(...)``, installs a ``WebUIBridge``, and
   spawns a worker thread that executes ``optimizer.run(...)``.
2. While the worker runs, events (logs, recommendations, prompts)
   are emitted into ``bridge.events`` and fanned out through
   ``EventHub``.
3. On completion, the run's ``output_dir`` is archived into
   ``output/runs/<run_id>/`` for the historical dashboard.
4. ``stop()`` uninstalls the bridge and joins the thread.

Only one run may be active at a time (enforced by :class:`SessionManager`).
"""

from __future__ import annotations

import json
import shutil
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

from webui.backend.event_hub import EventHub
from webui.backend.ui_bridge import WebUIBridge


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output"
RUNS_ROOT = DEFAULT_OUTPUT_ROOT / "runs"


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------
@dataclass
class RunConfig:
    task: str = "co2rr"
    data_path: str = "data/data.csv"
    candidates_path: Optional[str] = "data/candidates.csv"
    target_product: Optional[str] = None
    top_k: int = 10
    beta: float = 2.0
    beta_schedule: str = "fixed"
    beta_delta: float = 0.1
    acq_strategy: str = "ucb"
    qnei_mc_samples: int = 128
    kernel_type: str = "matern"
    h2_penalty_weight: float = 0.0
    skip_feature_selection: bool = False
    strict_training_schema: bool = False
    pre_fill_before_choice: bool = False
    seed: Optional[int] = None
    device: str = "auto"
    iterations: int = 10
    interactive: bool = True
    kabo_mode: bool = False
    lambda_p: float = 1.0
    lambda_k: float = 1.0
    lambda_v: float = 0.0
    expert_prior_file: Optional[str] = None
    diversity_weight: float = 0.5
    pe_budget: int = 0
    generate_candidates_n: int = 1000
    prefer_file_candidates: bool = False
    discrete_strategy: str = "acq"


# ---------------------------------------------------------------------------
# Session runner
# ---------------------------------------------------------------------------
@dataclass
class SessionRunner:
    run_id: str
    config: RunConfig
    output_dir: Path
    hub: EventHub
    bridge: WebUIBridge = field(default_factory=WebUIBridge)
    thread: Optional[threading.Thread] = field(default=None, init=False)
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = field(default=None, init=False)
    status: str = field(default="pending", init=False)  # pending|running|done|error|aborted
    error: Optional[str] = field(default=None, init=False)

    # --------------------------------------------------------------
    def start(self) -> None:
        if self.status != "pending":
            raise RuntimeError(f"Run already in state '{self.status}'.")
        self.status = "running"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.hub.set_bridge(self.bridge)
        self.thread = threading.Thread(
            target=self._worker, name=f"kabo-run-{self.run_id[:8]}", daemon=True,
        )
        self.thread.start()

    # --------------------------------------------------------------
    def _worker(self) -> None:
        """Background worker — builds optimizer and runs the BO pipeline."""
        cfg = self.config
        from kabo.task import get_task
        from kabo.optimizer import KABOOptimizer

        try:
            task = get_task(cfg.task)

            # Resolve candidate path special value "none"
            cand_path = cfg.candidates_path
            if cand_path is not None and str(cand_path).strip().lower() in {
                "", "none", "null"
            }:
                cand_path = None

            data_path = _resolve_project_path(cfg.data_path)
            if cand_path is not None:
                cand_path = _resolve_project_path(cand_path)
            expert_prior_path = (
                _resolve_project_path(cfg.expert_prior_file)
                if cfg.expert_prior_file else None
            )

            optimizer = KABOOptimizer(
                data_path=data_path,
                task=task,
                target_product=cfg.target_product,
                top_k=cfg.top_k,
                beta=cfg.beta,
                beta_schedule=cfg.beta_schedule,
                beta_delta=cfg.beta_delta,
                acq_strategy=cfg.acq_strategy,
                qnei_mc_samples=cfg.qnei_mc_samples,
                kernel_type=cfg.kernel_type,
                h2_penalty_weight=cfg.h2_penalty_weight,
                candidates_path=cand_path,
                skip_feature_selection=cfg.skip_feature_selection,
                strict_training_schema=cfg.strict_training_schema,
                pre_fill_before_choice=cfg.pre_fill_before_choice,
                seed=cfg.seed,
                device=cfg.device,
                output_dir=str(self.output_dir),
                kabo_mode=cfg.kabo_mode,
                lambda_p=cfg.lambda_p,
                lambda_k=cfg.lambda_k,
                lambda_v=cfg.lambda_v,
                expert_prior_file=str(expert_prior_path) if expert_prior_path else None,
                diversity_weight=cfg.diversity_weight,
                pe_budget=cfg.pe_budget,
                generate_candidates_n=cfg.generate_candidates_n,
                prefer_file_candidates=cfg.prefer_file_candidates,
                discrete_strategy=cfg.discrete_strategy,
            )

            # Install bridge (monkey-patches interactive hooks).
            self.bridge.install(
                task=task,
                worker_thread=threading.current_thread(),
                optimizer=optimizer,
            )

            self.bridge.emit(
                "run_started",
                run_id=self.run_id,
                config=asdict(cfg),
                task=task.task_name(),
            )

            optimizer.run(
                n_iterations=cfg.iterations,
                interactive=cfg.interactive,
            )

            self.status = "done"
            self.bridge.emit("run_completed", run_id=self.run_id)

        except Exception as exc:  # noqa: BLE001
            self.status = "error"
            self.error = f"{exc.__class__.__name__}: {exc}"
            tb = traceback.format_exc()
            try:
                self.bridge.emit(
                    "run_failed", run_id=self.run_id, error=self.error, traceback=tb,
                )
            except Exception:
                pass
        finally:
            self.finished_at = time.time()
            try:
                self.bridge.uninstall()
            except Exception:
                pass
            _archive_run(self)

    # --------------------------------------------------------------
    def abort(self) -> None:
        """Ask the run to stop at the next prompt."""
        self.bridge.abort()
        if self.status == "running":
            self.status = "aborted"

    # --------------------------------------------------------------
    def wait(self, timeout: Optional[float] = None) -> bool:
        if self.thread is None:
            return True
        self.thread.join(timeout)
        return not self.thread.is_alive()

    # --------------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        pending = self.bridge.get_pending_prompt()
        return {
            "run_id": self.run_id,
            "status": self.status,
            "error": self.error,
            "config": asdict(self.config),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "output_dir": str(self.output_dir),
            "pending_prompt": pending,
        }


# ---------------------------------------------------------------------------
# Session manager (registry keyed by run_id — v1.2)
# ---------------------------------------------------------------------------
# The legacy design kept a single ``_current`` slot.  v1.2 promotes this to
# an **ordered registry** keyed by ``run_id`` so multiple sessions can be
# tracked simultaneously (status queries, abort-by-id, SSE fan-out).  For
# backward compatibility the ``current`` property still returns "the most
# recently started session that is still running or pending"; when all
# active runs have finished, it returns the most recent terminal one so
# the existing ``/api/runs/current/*`` endpoints keep working.
#
# Concurrency note: the global ``EventHub`` still owns a single
# ``WebUIBridge`` at a time — making events truly multi-tenant requires a
# deeper refactor of the interaction layer.  The ``allow_concurrent``
# knob is therefore opt-in and primarily useful for non-interactive runs.
_MAX_REGISTRY_SIZE = 64
# Active-status set: treated as "not finished" for listing / current.
_ACTIVE_STATES: frozenset[str] = frozenset({"pending", "running"})


class SessionManager:
    """Thread-safe registry of ``SessionRunner`` instances (one per run_id)."""

    def __init__(self, hub: EventHub):
        self.hub = hub
        # OrderedDict so "most recent" semantics fall out naturally.
        from collections import OrderedDict as _OrderedDict
        self._sessions: _OrderedDict[str, SessionRunner] = _OrderedDict()
        self._lock = threading.Lock()

    # --------------------------------------------------------------
    @property
    def current(self) -> Optional[SessionRunner]:
        """Most recently started active session, or the most recent
        terminal one when no session is active.  Never raises."""
        with self._lock:
            return self._pick_current_locked()

    def _pick_current_locked(self) -> Optional[SessionRunner]:
        if not self._sessions:
            return None
        # Prefer active runs (status in _ACTIVE_STATES); else fall back to
        # the most recently inserted session.
        for runner in reversed(self._sessions.values()):
            if runner.status in _ACTIVE_STATES:
                return runner
        return next(reversed(self._sessions.values()))

    # --------------------------------------------------------------
    def start(
        self,
        config: RunConfig,
        allow_concurrent: bool = False,
    ) -> SessionRunner:
        """Create, register, and start a new ``SessionRunner``.

        Parameters
        ----------
        config : RunConfig
            Run parameters.
        allow_concurrent : bool, optional
            When False (default, legacy behaviour) this raises
            ``RuntimeError`` if another run is already active.  When
            True, multiple active sessions can coexist in the registry
            — the caller is responsible for ensuring the interaction
            layer can cope (non-interactive runs are always safe).
        """
        with self._lock:
            active = [
                r for r in self._sessions.values()
                if r.status in _ACTIVE_STATES
            ]
            if active and not allow_concurrent:
                raise RuntimeError(
                    "A run is already active. Stop it before starting a new one, "
                    "or pass allow_concurrent=True."
                )
            run_id = _new_run_id()
            output_dir = RUNS_ROOT / run_id
            runner = SessionRunner(
                run_id=run_id, config=config,
                output_dir=output_dir, hub=self.hub,
            )
            self._sessions[run_id] = runner
            # Bound registry size — drop oldest *terminal* entries first.
            self._evict_old_terminal_locked()
            runner.start()
            return runner

    # --------------------------------------------------------------
    def get(self, run_id: str) -> Optional[SessionRunner]:
        """Return the session with this id or ``None``."""
        with self._lock:
            return self._sessions.get(run_id)

    def list(self) -> list[SessionRunner]:
        """All sessions currently in the registry, oldest first."""
        with self._lock:
            return list(self._sessions.values())

    def list_snapshots(self) -> list[dict[str, Any]]:
        """Convenience wrapper returning ``runner.snapshot()`` for every entry."""
        return [r.snapshot() for r in self.list()]

    # --------------------------------------------------------------
    def abort(self, run_id: Optional[str] = None) -> bool:
        """Abort a session (``run_id=None`` → current).  Returns True on hit."""
        if run_id is None:
            runner = self.current
        else:
            runner = self.get(run_id)
        if runner is None:
            return False
        runner.abort()
        return True

    # --------------------------------------------------------------
    def remove(self, run_id: str) -> bool:
        """Drop a session from the registry.  Silently ignores unknown
        ids and refuses to remove active runs."""
        with self._lock:
            r = self._sessions.get(run_id)
            if r is None:
                return False
            if r.status in _ACTIVE_STATES:
                return False
            del self._sessions[run_id]
            return True

    # --------------------------------------------------------------
    def clear_if_finished(self) -> None:
        """No-op kept for backward compatibility with v1.1 callers."""
        return None

    # --------------------------------------------------------------
    def _evict_old_terminal_locked(self) -> None:
        """Drop the oldest terminal sessions so the registry stays bounded.

        Active sessions (``pending`` / ``running``) are never evicted.
        The registry is capped at :data:`_MAX_REGISTRY_SIZE` — when the
        cap is exceeded, this method removes oldest terminal entries
        until the cap is respected (or only active entries remain).
        """
        while len(self._sessions) > _MAX_REGISTRY_SIZE:
            for key, runner in list(self._sessions.items()):
                if runner.status not in _ACTIVE_STATES:
                    del self._sessions[key]
                    break
            else:
                break  # all remaining are active; stop evicting


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _new_run_id() -> str:
    ts = time.strftime("%Y%m%d-%H%M%S")
    return f"{ts}-{uuid.uuid4().hex[:6]}"


def _resolve_project_path(path_like: str | Path) -> Path:
    """Resolve relative paths against the project root."""
    p = Path(path_like)
    if p.is_absolute():
        return p
    return (PROJECT_ROOT / p).resolve()


def _archive_run(runner: SessionRunner) -> None:
    """Write a small metadata file summarising the run outcome."""
    info = runner.snapshot()
    info["finished_at"] = runner.finished_at
    try:
        summary_path = runner.output_dir / "webui_run.json"
        summary_path.write_text(json.dumps(info, indent=2, ensure_ascii=False))
    except Exception:
        pass


def list_archived_runs() -> list[dict[str, Any]]:
    """Scan output/runs/ and return metadata for each archived run."""
    if not RUNS_ROOT.exists():
        return []
    runs: list[dict[str, Any]] = []
    for p in sorted(RUNS_ROOT.iterdir(), reverse=True):
        if not p.is_dir():
            continue
        info: dict[str, Any] = {"run_id": p.name, "output_dir": str(p)}
        summary = p / "webui_run.json"
        if summary.exists():
            try:
                info.update(json.loads(summary.read_text()))
            except Exception:
                pass
        meta = p / "run_metadata.json"
        if meta.exists():
            try:
                info["metadata"] = json.loads(meta.read_text())
            except Exception:
                pass
        info["has_data_updated"] = (p / "data_updated.csv").exists()
        info["has_feature_importances"] = (p / "feature_importances.png").exists()
        runs.append(info)
    return runs


def delete_archived_run(run_id: str) -> bool:
    target = RUNS_ROOT / run_id
    if not target.exists() or not target.is_dir():
        return False
    shutil.rmtree(target, ignore_errors=True)
    return True
