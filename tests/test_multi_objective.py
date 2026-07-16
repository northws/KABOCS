"""Unit tests for ``kabo.multi_objective``.

Covers the torch-free surface: ``ObjectiveSpec`` validation, Pareto
front identification (with the numpy fallback when torch is missing),
reference-point inference, and the optimizer-side objective resolution
helper.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kabo.multi_objective import (
    ObjectiveSpec,
    compute_pareto_front,
    infer_ref_point,
)


# =============================================================================
#  ObjectiveSpec
# =============================================================================
class TestObjectiveSpec:
    def test_defaults(self):
        o = ObjectiveSpec(column="y")
        assert o.column == "y"
        assert o.direction == "max"
        assert o.ref_point is None
        assert o.sign == 1.0
        assert o.label == "y"

    def test_min_direction_flips_sign(self):
        o = ObjectiveSpec(column="y_her", direction="min")
        assert o.sign == -1.0

    def test_display_name_overrides_label(self):
        o = ObjectiveSpec(column="Y_CO", display_name="CO yield")
        assert o.label == "CO yield"

    def test_rejects_empty_column(self):
        with pytest.raises(ValueError, match="column"):
            ObjectiveSpec(column="")

    def test_rejects_bad_direction(self):
        with pytest.raises(ValueError, match="direction"):
            ObjectiveSpec(column="y", direction="maximise")


# =============================================================================
#  Pareto front
# =============================================================================
class TestComputeParetoFront:
    def test_two_dim_max_max(self):
        # (1, 1) dominated by (2, 2); (3, 1.5), (1.5, 3), (2, 2) non-dominated
        Y = np.array([
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 1.5],
            [1.5, 3.0],
            [0.5, 0.5],
        ])
        objs = [ObjectiveSpec("a"), ObjectiveSpec("b")]
        mask = compute_pareto_front(Y, objs)
        assert mask.tolist() == [False, True, True, True, False]

    def test_direction_mix(self):
        # Maximise y1, minimise y2.  Signed (maximise-normalised) coords:
        #   row 0 (1.0, 5.0) -> (1.0, -5.0)   dominated by row 1 and row 3
        #   row 1 (2.0, 2.0) -> (2.0, -2.0)   dominated by row 3 (2.5, -1.0)
        #   row 2 (3.0, 4.0) -> (3.0, -4.0)   Pareto (unique max y1)
        #   row 3 (2.5, 1.0) -> (2.5, -1.0)   Pareto (unique min y2)
        Y = np.array([
            [1.0, 5.0],
            [2.0, 2.0],
            [3.0, 4.0],
            [2.5, 1.0],
        ])
        objs = [ObjectiveSpec("y1", "max"), ObjectiveSpec("y2", "min")]
        mask = compute_pareto_front(Y, objs)
        assert mask.tolist() == [False, False, True, True]

    def test_nan_rows_excluded(self):
        Y = np.array([
            [1.0, 1.0],
            [np.nan, 5.0],
            [3.0, 3.0],
        ])
        objs = [ObjectiveSpec("a"), ObjectiveSpec("b")]
        mask = compute_pareto_front(Y, objs)
        # Only row 2 is Pareto; row 1 dropped for NaN.
        assert mask[1] == False  # noqa: E712
        assert mask[2] == True   # noqa: E712

    def test_rejects_wrong_shape(self):
        Y = np.array([[1.0, 1.0, 1.0]])
        with pytest.raises(ValueError, match="does not match"):
            compute_pareto_front(Y, [ObjectiveSpec("a"), ObjectiveSpec("b")])


# =============================================================================
#  Reference-point inference
# =============================================================================
class TestInferRefPoint:
    def test_max_objective_padding(self):
        Y = np.array([[1.0], [3.0], [5.0]])
        ref = infer_ref_point(Y, [ObjectiveSpec("y")], margin=0.1)
        # lo=1.0, hi=5.0, span=4, ref = 1.0 - 0.1 * 4 = 0.6
        assert ref == pytest.approx([0.6])

    def test_min_objective_flips(self):
        Y = np.array([[1.0], [3.0], [5.0]])
        ref = infer_ref_point(Y, [ObjectiveSpec("y", "min")], margin=0.1)
        # direction=min → sign=-1, y_signed = -Y → lo=-5, hi=-1, span=4
        # ref (signed, on maximise scale) = -5 - 0.1 * 4 = -5.4
        assert ref == pytest.approx([-5.4])

    def test_explicit_ref_point_wins(self):
        Y = np.array([[1.0], [5.0]])
        obj = ObjectiveSpec("y", "max", ref_point=-100.0)
        ref = infer_ref_point(Y, [obj])
        assert ref == pytest.approx([-100.0])

    def test_constant_column_still_returns_finite(self):
        Y = np.array([[2.0], [2.0], [2.0]])
        ref = infer_ref_point(Y, [ObjectiveSpec("y")], margin=0.1)
        assert np.isfinite(ref[0])

    def test_margin_must_be_non_negative(self):
        with pytest.raises(ValueError, match="margin"):
            infer_ref_point(np.array([[1.0]]), [ObjectiveSpec("y")], margin=-0.1)


# =============================================================================
#  TaskBase default multi-objective hook
# =============================================================================
class TestTaskBaseMultiObjectives:
    def test_default_returns_empty_list(self):
        from kabo.task import get_task
        task = get_task("test")  # TestTask overrides; check it's a list
        specs = task.multi_objectives()
        assert isinstance(specs, list)
        assert len(specs) >= 2, "TestTask must ship a 2-objective preset"

    def test_build_training_target_multi_default(self):
        from kabo.task import get_task
        task = get_task("test")
        df = pd.DataFrame({"y": [1.0, 2.0], "y2": [3.0, 4.0]})
        Y = task.build_training_target_multi(df, task.multi_objectives())
        assert Y.shape == (2, 2)
        assert Y[0].tolist() == [1.0, 3.0]

    def test_build_training_target_multi_raises_on_missing(self):
        from kabo.task import get_task
        task = get_task("test")
        df = pd.DataFrame({"y": [1.0]})  # y2 missing
        with pytest.raises(KeyError, match="missing"):
            task.build_training_target_multi(df, task.multi_objectives())


# =============================================================================
#  CO2RRTask preset
# =============================================================================
class TestCO2RRPreset:
    def test_mo_preset_is_co_vs_her(self):
        from kabo.task import get_task
        task = get_task("co2rr")
        specs = task.multi_objectives()
        cols = [s.column for s in specs]
        directions = [s.direction for s in specs]
        assert "Y_CO" in cols
        assert "Y_H2" in cols
        # H2 is the HER side reaction — must be minimised.
        assert directions[cols.index("Y_H2")] == "min"


# =============================================================================
#  Discrete candidate pool in MO mode
# =============================================================================
@pytest.mark.requires_torch
class TestEvaluateDiscreteInMOMode:
    """MO mode never fits the single-objective surrogate, so anything that
    reads bounds off ``engine.surrogate`` directly breaks the moment a run
    also has a discrete candidate pool — which is every built-in Task that
    implements ``generate_candidates()``, plus any ``--candidates x.csv``.
    """

    @pytest.fixture
    def fitted_engine(self):
        import torch

        from kabo.engine import KABOEngine

        rng = np.random.default_rng(0)
        features = ["x1", "x2", "x3"]
        bounds = {f: (0.0, 1.0) for f in features}
        X = rng.random((12, 3))
        # Two objectives: one to maximise, one to minimise.
        Y = np.column_stack([X.sum(axis=1), X[:, 0] * 2.0])

        engine = KABOEngine(device=torch.device("cpu"))
        engine.fit_mo_surrogate(
            X, Y, features, design_bounds=bounds,
            objectives=[
                ObjectiveSpec(column="a", direction="max"),
                ObjectiveSpec(column="b", direction="min"),
            ],
        )
        return engine, features, bounds

    def test_single_objective_surrogate_is_never_fit(self, fitted_engine):
        """Guards the premise of this whole test class."""
        engine, _, _ = fitted_engine
        assert engine.is_multi_objective
        assert engine.surrogate.bounds_raw is None

    def test_ranks_a_discrete_pool_without_raising(self, fitted_engine):
        engine, features, bounds = fitted_engine
        acq = engine.build_mo_acquisition(ref_point_signed=[-0.1, -0.1])
        pool = pd.DataFrame(
            np.random.default_rng(1).random((5, 3)), columns=features,
        )

        results = engine.evaluate_discrete(
            acq, pool, features,
            all_feature_columns=features,
            design_bounds=bounds,
        )

        assert len(results) == len(pool)
        for cand, value, orig_idx in results:
            assert cand.shape[-1] == len(features)
            assert np.isfinite(float(value))
            assert 0 <= orig_idx < len(pool)

    def test_thompson_reports_that_it_is_unsupported(self, fitted_engine):
        """Thompson sampling off a ModelListGP is ambiguous; the old code
        surfaced this as a misleading "surrogate must be fit"."""
        engine, features, bounds = fitted_engine
        engine.discrete_strategy = "thompson"
        pool = pd.DataFrame(
            np.random.default_rng(1).random((3, 3)), columns=features,
        )

        with pytest.raises(RuntimeError, match="not supported in"):
            engine.evaluate_discrete(
                None, pool, features,
                all_feature_columns=features,
                design_bounds=bounds,
            )

    def test_still_raises_when_nothing_is_fit(self):
        import torch

        from kabo.engine import KABOEngine

        engine = KABOEngine(device=torch.device("cpu"))
        with pytest.raises(RuntimeError, match="must be fit"):
            engine.evaluate_discrete(
                None, pd.DataFrame({"x1": [0.5]}), ["x1"],
                all_feature_columns=["x1"],
                design_bounds={"x1": (0.0, 1.0)},
            )
