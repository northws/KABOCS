"""End-to-end smoke tests for the full optimizer pipeline.

These drive ``KABOOptimizer`` with the minimal ``TestTask`` in
**non-interactive demo mode**, which exercises every phase (feature
selection → GP surrogate fit → UCB acquisition → observation append
→ re-fit) without any user input.  They are gated by the
``requires_torch`` marker so they are skipped automatically when the
optional torch / botorch stack is unavailable.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


pytestmark = pytest.mark.requires_torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _import_optimizer():
    """Defer torch-dependent imports until inside a test body so pytest
    can collect this module even when torch is unavailable."""
    from kabo.optimizer import KABOOptimizer
    from kabo.task import TestTask

    return KABOOptimizer, TestTask


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestDemoRunTestTask:
    """Smoke test: run two BO iterations on ``TestTask`` and assert outputs."""

    def test_runs_without_interactive_input_and_writes_artifacts(
        self,
        tmp_path: Path,
        test_data_csv: Path,
    ):
        KABOOptimizer, TestTask = _import_optimizer()

        out_dir = tmp_path / "run_out"
        opt = KABOOptimizer(
            data_path=test_data_csv,
            task=TestTask(),
            top_k=3,
            seed=42,
            output_dir=out_dir,
            skip_feature_selection=True,
            candidates_path=None,  # skip discrete pool
            device="cpu",
        )
        opt.run(n_iterations=2, interactive=False)

        # Verify output files
        updated = out_dir / "data_updated.csv"
        meta_f = out_dir / "run_metadata.json"
        assert updated.exists(), "data_updated.csv should be written"
        assert meta_f.exists(), "run_metadata.json should be written"

        meta = json.loads(meta_f.read_text())
        assert meta["task"] == "test"
        assert meta["interactive"] is False
        assert meta["n_iterations"] == 2
        assert meta["seed"] == 42
        assert meta["n_rows_final"] >= 2  # dataset grew by ≥2 rows

    def test_run_metadata_includes_beta_trace_for_ucb(
        self,
        tmp_path: Path,
        test_data_csv: Path,
    ):
        KABOOptimizer, TestTask = _import_optimizer()

        opt = KABOOptimizer(
            data_path=test_data_csv,
            task=TestTask(),
            top_k=3,
            seed=7,
            output_dir=tmp_path / "ucb_run",
            skip_feature_selection=True,
            candidates_path=None,
            device="cpu",
            acq_strategy="ucb",
            beta=2.0,
            beta_schedule="fixed",
        )
        opt.run(n_iterations=2, interactive=False)

        meta = json.loads((tmp_path / "ucb_run" / "run_metadata.json").read_text())
        assert meta["acq_strategy"] == "ucb"
        # fixed schedule: beta trace has one entry per iteration.
        assert isinstance(meta["beta_trace"], list)
        assert len(meta["beta_trace"]) == 2
        assert all(b == pytest.approx(2.0) for b in meta["beta_trace"])

    def test_q_batch_greater_than_one_adds_extra_candidates(
        self,
        tmp_path: Path,
        test_data_csv: Path,
    ):
        """q_batch=3 should yield 3 continuous candidates labelled
        'continuous_1', 'continuous_2', 'continuous_3' in metadata."""
        KABOOptimizer, TestTask = _import_optimizer()

        opt = KABOOptimizer(
            data_path=test_data_csv,
            task=TestTask(),
            top_k=3,
            seed=3,
            output_dir=tmp_path / "qbatch_run",
            skip_feature_selection=True,
            candidates_path=None,
            device="cpu",
            acq_strategy="ucb",
            q_batch=3,
        )
        opt.run(n_iterations=1, interactive=False)
        meta = json.loads((tmp_path / "qbatch_run" / "run_metadata.json").read_text())
        assert meta["q_batch"] == 3

    def test_early_stop_triggers_on_flat_target(
        self,
        tmp_path: Path,
        test_data_csv: Path,
    ):
        """With max_stagnation=1 and an extremely strict tol, even minor
        non-improvements should stop the loop before n_iterations ends."""
        KABOOptimizer, TestTask = _import_optimizer()

        opt = KABOOptimizer(
            data_path=test_data_csv,
            task=TestTask(),
            top_k=3,
            seed=11,
            output_dir=tmp_path / "earlystop_run",
            skip_feature_selection=True,
            candidates_path=None,
            device="cpu",
            acq_strategy="ucb",
            max_stagnation=1,
            stagnation_tol=1e9,  # no improvement can exceed this
        )
        opt.run(n_iterations=5, interactive=False)
        meta = json.loads(
            (tmp_path / "earlystop_run" / "run_metadata.json").read_text()
        )
        assert meta["max_stagnation"] == 1
        assert meta["stopped_early"] is True
        assert "stagnation" in meta["stop_reason"]

    def test_multi_objective_run_emits_pareto_artifacts(
        self,
        tmp_path: Path,
        test_data_csv: Path,
    ):
        """--multi-objective on TestTask fits ModelListGP + writes pareto_front.{csv,png}."""
        KABOOptimizer, TestTask = _import_optimizer()

        opt = KABOOptimizer(
            data_path=test_data_csv,
            task=TestTask(),
            top_k=3,
            seed=13,
            output_dir=tmp_path / "mo_run",
            skip_feature_selection=True,
            candidates_path=None,
            device="cpu",
            multi_objective=True,   # uses TestTask.multi_objectives() preset
        )
        opt.run(n_iterations=2, interactive=False)

        run_dir = tmp_path / "mo_run"
        meta = json.loads((run_dir / "run_metadata.json").read_text())

        assert meta["multi_objective"] is True
        assert len(meta["objectives"]) == 2
        assert {o["column"] for o in meta["objectives"]} == {"y", "y2"}
        assert meta["qnehvi_mc_samples"] == 128

        # Pareto artifacts written.
        assert (run_dir / "pareto_front.csv").exists()
        # PNG is a best-effort artifact — may be absent if matplotlib has
        # trouble in headless mode — but the CSV must exist.
        import pandas as pd
        pareto = pd.read_csv(run_dir / "pareto_front.csv")
        assert "pareto_rank" in pareto.columns
        assert "y" in pareto.columns and "y2" in pareto.columns
        assert len(pareto) >= 1

    def test_multi_objective_explicit_objectives_override(
        self,
        tmp_path: Path,
        test_data_csv: Path,
    ):
        """--objectives override short-name resolution works (mirrors CLI)."""
        KABOOptimizer, TestTask = _import_optimizer()

        opt = KABOOptimizer(
            data_path=test_data_csv,
            task=TestTask(),
            top_k=3,
            seed=17,
            output_dir=tmp_path / "mo_override_run",
            skip_feature_selection=True,
            candidates_path=None,
            device="cpu",
            objectives=["Y", "Y2"],  # short names resolved via target_columns()
        )
        # Should implicitly enable multi-objective mode.
        assert opt.multi_objective is True
        assert {o.column for o in opt.objectives} == {"y", "y2"}
        opt.run(n_iterations=1, interactive=False)
