"""
WebUI Bridge — replaces KABO's CLI interaction with async queue-based I/O.

Strategy (non-invasive):
  * Monkey-patch the prompt / display functions in ``kabo.optimizer``
    (because it did ``from kabo.acquisition import ...`` at module load,
    the references live in ``kabo.optimizer`` and that is what the
    running optimizer will call).
  * Monkey-patch the ``prompt_observation`` method on the ``Task``
    **instance** (bound method swap).
  * Monkey-patch ``builtins.input`` for the Preference-Exploration
    inline loop.
  * Redirect ``sys.stdout`` so stray ``print()`` calls (e.g. the PE
    banner lines) become structured log events.
  * Attach a ``logging.Handler`` so every ``kabo.*`` log line is
    streamed to the frontend.

The bridge keeps the ``kabo`` package 100% untouched.
"""

from __future__ import annotations

import builtins
import io
import logging
import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import torch

import kabo.optimizer as _opt_mod

from kabo.utils import unnormalize_x


# ---------------------------------------------------------------------------
# Event payload helpers
# ---------------------------------------------------------------------------
def _jsonable_number(v: Any) -> Any:
    """Convert numpy / torch scalars into plain Python numbers; NaN -> None."""
    try:
        if v is None:
            return None
        if isinstance(v, (np.floating, np.integer)):
            v = v.item()
        elif isinstance(v, torch.Tensor):
            v = v.item()
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            return None
        return v
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Bridge
# ---------------------------------------------------------------------------
@dataclass
class WebUIBridge:
    """Runtime glue between a background optimizer thread and the web UI.

    Thread model
    ------------
    * The optimizer runs in **one** worker thread.
    * FastAPI request handlers run in the main asyncio loop (and its
      default threadpool).
    * ``events`` is written from the worker thread, read by the event
      hub dispatcher task on the main loop.
    * ``_answer_event`` / ``_current_answer`` synchronise the worker
      thread's blocking ``wait_answer()`` with the HTTP POST that
      submits the answer.
    """

    events: "queue.Queue[dict]" = field(default_factory=queue.Queue)

    _current_answer: Optional[dict] = field(default=None, init=False)
    _answer_event: threading.Event = field(default_factory=threading.Event, init=False)
    _pending_prompt: Optional[dict] = field(default=None, init=False)
    _pending_prompt_id: int = field(default=0, init=False)
    _worker_thread: Optional[threading.Thread] = field(default=None, init=False)

    _originals: dict = field(default_factory=dict, init=False)
    _task_ref: Any = field(default=None, init=False)
    _orig_task_prompt: Optional[Callable] = field(default=None, init=False)
    _orig_task_simulate: Optional[Callable] = field(default=None, init=False)
    _orig_input: Callable = field(default=None, init=False)
    _orig_stdout: Any = field(default=None, init=False)
    _log_handler: Optional[logging.Handler] = field(default=None, init=False)
    _installed: bool = field(default=False, init=False)
    _aborted: bool = field(default=False, init=False)

    # ------------------------------------------------------------------
    # Event emission
    # ------------------------------------------------------------------
    def emit(self, event_type: str, **payload: Any) -> None:
        event = {"type": event_type, "ts": time.time(), **payload}
        if event_type == "prompt":
            self._pending_prompt_id += 1
            event["prompt_id"] = self._pending_prompt_id
            self._pending_prompt = event
        self.events.put(event)

    def emit_log(self, level: str, message: str) -> None:
        self.emit("log", level=level, message=message)

    # ------------------------------------------------------------------
    # Answer synchronisation
    # ------------------------------------------------------------------
    def wait_answer(self) -> dict:
        """Block the worker thread until an answer arrives."""
        while not self._answer_event.wait(timeout=0.5):
            if self._aborted:
                return {"action": "exit"}
        ans = self._current_answer or {}
        self._current_answer = None
        self._answer_event.clear()
        self._pending_prompt = None
        return ans

    def submit_answer(self, answer: dict, prompt_id: Optional[int] = None) -> bool:
        """Called from FastAPI; returns True if accepted."""
        if self._pending_prompt is None:
            return False
        if prompt_id is not None and prompt_id != self._pending_prompt_id:
            return False
        self._current_answer = dict(answer or {})
        self._answer_event.set()
        return True

    def abort(self) -> None:
        """Signal the optimizer to stop at the next interaction."""
        self._aborted = True
        # Unblock any pending wait_answer with an implicit exit
        if self._pending_prompt is not None:
            self._current_answer = {"action": "exit"}
            self._answer_event.set()

    def get_pending_prompt(self) -> Optional[dict]:
        return self._pending_prompt

    # ------------------------------------------------------------------
    # Replacement prompt implementations
    # ------------------------------------------------------------------
    def ask_candidate_choice(
        self, top_indices: list[int], n_total: int
    ) -> Optional[int]:
        self.emit(
            "prompt",
            kind="candidate_choice",
            top_indices=list(top_indices),
            n_total=int(n_total),
        )
        ans = self.wait_answer()
        action = (ans.get("action") or "").lower()
        if action in ("exit", "quit"):
            return None
        if action in ("manual", "override"):
            return -1
        if action in ("tie", "equal"):
            return -2
        rank = int(ans.get("rank", 1) or 1)
        if 1 <= rank <= len(top_indices):
            return int(top_indices[rank - 1])
        return int(top_indices[0])

    def ask_manual_candidate(
        self,
        all_feature_columns: list[str],
        design_bounds: dict[str, tuple[float, float]],
    ):
        if not all_feature_columns:
            return {}, 0, []
        self.emit(
            "prompt",
            kind="manual_candidate",
            features=list(all_feature_columns),
            bounds={k: list(v) for k, v in design_bounds.items()},
        )
        ans = self.wait_answer()
        if (ans.get("action") or "").lower() in ("exit", "quit"):
            return None

        values = ans.get("values", {}) or {}
        oob = int(ans.get("oob_confirmations") or 0)
        results: dict[str, float] = {}
        overridden: list[str] = []
        for feat in all_feature_columns:
            lo, hi = design_bounds.get(feat, (0.0, 1.0))
            mid = (lo + hi) / 2.0
            v = values.get(feat, None)
            if v is None or v == "":
                results[feat] = float(mid)
                continue
            try:
                val = float(v)
            except (TypeError, ValueError):
                results[feat] = float(mid)
                continue
            results[feat] = val
            if not np.isclose(val, mid, rtol=0.0, atol=1e-12):
                overridden.append(feat)
        return results, oob, overridden

    def ask_nonselected_features(
        self,
        nonselected_features: list[str],
        design_bounds: dict[str, tuple[float, float]],
    ):
        if not nonselected_features:
            return {}, 0, []
        self.emit(
            "prompt",
            kind="nonselected_features",
            features=list(nonselected_features),
            bounds={k: list(v) for k, v in design_bounds.items()},
        )
        ans = self.wait_answer()
        values = ans.get("values", {}) or {}
        oob = int(ans.get("oob_confirmations") or 0)
        results: dict[str, float] = {}
        overridden: list[str] = []
        for feat in nonselected_features:
            lo, hi = design_bounds.get(feat, (0.0, 1.0))
            mid = (lo + hi) / 2.0
            v = values.get(feat, None)
            if v is None or v == "":
                results[feat] = float(mid)
                continue
            try:
                val = float(v)
            except (TypeError, ValueError):
                results[feat] = float(mid)
                continue
            results[feat] = val
            if not np.isclose(val, mid, rtol=0.0, atol=1e-12):
                overridden.append(feat)
        return results, oob, overridden

    def ask_product_yields(self, target_column: str):
        """Replacement for ``task.prompt_observation``."""
        task = self._task_ref
        if task is None:
            return None
        product_cols = task.target_columns()
        product_names = task.product_names()
        self.emit(
            "prompt",
            kind="product_yields",
            target_column=target_column,
            target_name=product_names.get(target_column, target_column),
            products=[
                {
                    "short": short,
                    "column": col,
                    "is_target": col == target_column,
                    "display": product_names.get(col, short),
                }
                for short, col in product_cols.items()
            ],
        )
        ans = self.wait_answer()
        if (ans.get("action") or "").lower() in ("exit", "quit"):
            return None
        yields = ans.get("yields", {}) or {}
        results: dict[str, float] = {}
        for _, col in product_cols.items():
            v = yields.get(col, None)
            try:
                results[col] = float(v) if v is not None and v != "" else 0.0
            except (TypeError, ValueError):
                results[col] = 0.0
        return results

    # ------------------------------------------------------------------
    # Recommendations / Best display (structured, replaces prints)
    # ------------------------------------------------------------------
    def _select_top_n(
        self,
        candidates_norm: list[torch.Tensor],
        acq_values: list[float],
        top_n: int,
        diversity_weight: float,
    ) -> list[int]:
        """Replicates the diversity-aware greedy Top-N from acquisition.py."""
        n_cands = len(acq_values)
        acq_arr = np.array(acq_values, dtype=float)
        acq_min, acq_max = acq_arr.min(), acq_arr.max()
        acq_range = acq_max - acq_min if acq_max > acq_min else 1.0
        acq_norm = (acq_arr - acq_min) / acq_range

        cand_vecs = np.array([c.detach().cpu().numpy() for c in candidates_norm])

        selected: list[int] = []
        remaining = set(range(n_cands))

        for slot in range(min(top_n, n_cands)):
            if slot == 0:
                best = int(np.argmax(acq_arr))
            else:
                best_score = -np.inf
                best = -1
                sel_vecs = cand_vecs[selected]
                for i in remaining:
                    dists = np.linalg.norm(sel_vecs - cand_vecs[i], axis=1)
                    min_dist = float(dists.min())
                    score = float(acq_norm[i]) + diversity_weight * min_dist
                    if score > best_score:
                        best_score = score
                        best = i
            selected.append(best)
            remaining.discard(best)
        return selected

    def show_recommendations(
        self,
        candidates_norm,
        acq_values,
        source_labels,
        all_orig_rows,
        discrete_df,
        selected_features,
        all_feature_columns,
        bounds_raw,
        iteration,
        target_column,
        product_names,
        top_n: int = 3,
        continuous_nonselected_values=None,
        diversity_weight: float = 0.5,
    ) -> list[int]:
        top_indices = self._select_top_n(
            candidates_norm, list(acq_values), top_n, diversity_weight,
        )

        recs: list[dict] = []
        for rank, idx in enumerate(top_indices, 1):
            cand_norm = candidates_norm[idx]
            acq_val = float(acq_values[idx])
            source = source_labels[idx]
            orig_row = all_orig_rows[idx]

            cand_raw = unnormalize_x(cand_norm, bounds_raw)
            feat_map: dict[str, dict] = {}

            if orig_row >= 0 and discrete_df is not None:
                row_data = discrete_df.iloc[orig_row]
                for feat in all_feature_columns:
                    val = row_data.get(feat, np.nan)
                    feat_map[feat] = {
                        "value": _jsonable_number(val),
                        "origin": "selected" if feat in selected_features else "fixed",
                    }
            else:
                for feat in all_feature_columns:
                    if feat in selected_features:
                        f_idx = selected_features.index(feat)
                        feat_map[feat] = {
                            "value": _jsonable_number(cand_raw[f_idx]),
                            "origin": "selected",
                        }
                    elif (
                        continuous_nonselected_values is not None
                        and feat in continuous_nonselected_values
                    ):
                        feat_map[feat] = {
                            "value": _jsonable_number(
                                continuous_nonselected_values[feat]
                            ),
                            "origin": "expert",
                        }
                    else:
                        feat_map[feat] = {"value": None, "origin": "pending"}

            recs.append(
                {
                    "rank": rank,
                    "idx": int(idx),
                    "acq_value": acq_val,
                    "source": source,
                    "features": feat_map,
                }
            )

        self.emit(
            "recommendations",
            iteration=int(iteration),
            target_column=target_column,
            target_name=product_names.get(target_column, target_column),
            top_n=int(top_n),
            recommendations=recs,
            selected_features=list(selected_features),
            all_features=list(all_feature_columns),
        )
        return top_indices

    def show_best(
        self,
        df: pd.DataFrame,
        selected_features: list[str],
        target_column: str,
        all_product_columns: list[str],
        product_names: dict[str, str],
    ) -> None:
        if df is None or df.empty or target_column not in df.columns:
            return
        best_idx = df[target_column].idxmax()
        best_row = df.loc[best_idx]
        self.emit(
            "best_found",
            target_column=target_column,
            target_name=product_names.get(target_column, target_column),
            best_value=_jsonable_number(best_row[target_column]),
            products={
                col: _jsonable_number(best_row.get(col, np.nan))
                for col in all_product_columns
                if col in df.columns
            },
            features={
                feat: _jsonable_number(best_row.get(feat, np.nan))
                for feat in selected_features
            },
        )

    # ------------------------------------------------------------------
    # Raw builtins.input fallback (PE loop etc.)
    # ------------------------------------------------------------------
    def patched_input(self, prompt_text: str = "") -> str:
        # Restrict scope: only intercept input() calls on the worker thread.
        # All other threads see the original input() implementation.
        if (
            self._worker_thread is None
            or threading.current_thread() is not self._worker_thread
        ):
            return self._orig_input(prompt_text)
        self.emit("prompt", kind="raw_input", prompt_text=str(prompt_text))
        ans = self.wait_answer()
        return str(ans.get("value", ""))

    # ------------------------------------------------------------------
    # Install / uninstall lifecycle
    # ------------------------------------------------------------------
    def install(self, task: Any, worker_thread: threading.Thread) -> None:
        if self._installed:
            raise RuntimeError("WebUIBridge already installed.")
        self._installed = True
        self._task_ref = task
        self._worker_thread = worker_thread

        self._originals = {
            "prompt_user_candidate_choice": _opt_mod.prompt_user_candidate_choice,
            "prompt_user_manual_candidate": _opt_mod.prompt_user_manual_candidate,
            "prompt_user_nonselected_features": _opt_mod.prompt_user_nonselected_features,
            "print_recommendations": _opt_mod.print_recommendations,
            "print_best_found": _opt_mod.print_best_found,
        }
        _opt_mod.prompt_user_candidate_choice = self.ask_candidate_choice
        _opt_mod.prompt_user_manual_candidate = self.ask_manual_candidate
        _opt_mod.prompt_user_nonselected_features = self.ask_nonselected_features
        _opt_mod.print_recommendations = self.show_recommendations
        _opt_mod.print_best_found = self.show_best

        # Task bound-method swap
        self._orig_task_prompt = getattr(task, "prompt_observation", None)
        task.prompt_observation = self.ask_product_yields  # type: ignore[method-assign]

        # builtins.input
        self._orig_input = builtins.input
        builtins.input = self.patched_input  # type: ignore[assignment]

        # sys.stdout redirect → log events (worker thread only)
        self._orig_stdout = sys.stdout
        sys.stdout = _StdoutToBridge(self, self._orig_stdout, worker_thread)

        # Attach logging handler
        self._log_handler = _BridgeLogHandler(self)
        self._log_handler.setLevel(logging.INFO)
        self._log_handler.setFormatter(
            logging.Formatter("%(message)s")
        )
        root = logging.getLogger("kabo")
        root.addHandler(self._log_handler)

    def uninstall(self) -> None:
        if not self._installed:
            return
        self._installed = False
        _opt_mod.prompt_user_candidate_choice = self._originals["prompt_user_candidate_choice"]
        _opt_mod.prompt_user_manual_candidate = self._originals["prompt_user_manual_candidate"]
        _opt_mod.prompt_user_nonselected_features = self._originals["prompt_user_nonselected_features"]
        _opt_mod.print_recommendations = self._originals["print_recommendations"]
        _opt_mod.print_best_found = self._originals["print_best_found"]

        if self._task_ref is not None and self._orig_task_prompt is not None:
            try:
                self._task_ref.prompt_observation = self._orig_task_prompt  # type: ignore[method-assign]
            except Exception:
                pass

        if self._orig_input is not None:
            builtins.input = self._orig_input  # type: ignore[assignment]

        if self._orig_stdout is not None:
            sys.stdout = self._orig_stdout

        if self._log_handler is not None:
            logging.getLogger("kabo").removeHandler(self._log_handler)
            self._log_handler = None

        self._task_ref = None
        self._worker_thread = None


# ---------------------------------------------------------------------------
# stdout & logging adapters
# ---------------------------------------------------------------------------
class _StdoutToBridge(io.TextIOBase):
    """Capture ``print()`` output from the optimizer worker thread."""

    def __init__(
        self,
        bridge: WebUIBridge,
        fallback: Any,
        worker: threading.Thread,
    ):
        self._bridge = bridge
        self._fallback = fallback
        self._worker = worker
        self._buf = ""

    def writable(self) -> bool:
        return True

    def write(self, s: str) -> int:
        # Pass-through for non-worker threads (e.g. uvicorn logs).
        if threading.current_thread() is not self._worker:
            try:
                return self._fallback.write(s)
            except Exception:
                return len(s)
        if not s:
            return 0
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            line = line.rstrip("\r")
            if line.strip():
                self._bridge.emit_log("stdout", line)
        return len(s)

    def flush(self) -> None:
        if threading.current_thread() is not self._worker:
            try:
                self._fallback.flush()
            except Exception:
                pass
            return
        if self._buf.strip():
            self._bridge.emit_log("stdout", self._buf.rstrip())
        self._buf = ""


class _BridgeLogHandler(logging.Handler):
    """Stream ``kabo.*`` log records to the UI."""

    def __init__(self, bridge: WebUIBridge):
        super().__init__()
        self._bridge = bridge

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
        except Exception:
            msg = record.getMessage()
        self._bridge.emit_log(record.levelname.lower(), msg)
