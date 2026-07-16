"""Unit tests for ``kabo.task.eco2rr.ECO2RRTask``.

Covers the torch-free surface: the 19-descriptor schema, ordinal
encoding of the three categorical descriptors, the FE-budget invariant
(products sum to <= 100%), the HER-penalty composite target, and the MO
preset.  ``generate_candidates`` needs ``torch.quasirandom.SobolEngine``
and is therefore marked ``requires_torch``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kabo.constants import ECO2RR_FE_TOTAL_MAX
from kabo.task import TASK_REGISTRY, ECO2RRTask, TaskBase, get_task


@pytest.fixture
def task() -> ECO2RRTask:
    return ECO2RRTask()


# =============================================================================
#  Registration
# =============================================================================
class TestRegistration:
    def test_registered_under_lowercased_name(self):
        assert "eco2rr" in TASK_REGISTRY

    def test_get_task_is_case_insensitive(self):
        assert type(get_task("eco2rr")) is type(get_task("ECO2RR")) is ECO2RRTask

    def test_is_subclass_of_taskbase(self, task):
        assert isinstance(task, TaskBase)

    def test_does_not_shadow_photocatalytic_task(self):
        """The electro task extends the registry — it must not replace co2rr."""
        assert "co2rr" in TASK_REGISTRY
        assert TASK_REGISTRY["co2rr"] is not TASK_REGISTRY["eco2rr"]


# =============================================================================
#  Feature schema
# =============================================================================
class TestFeatureSchema:
    def test_declares_19_descriptors(self, task):
        assert len(task.feature_columns()) == 19

    def test_feature_columns_are_unique(self, task):
        feats = task.feature_columns()
        assert len(set(feats)) == len(feats)

    def test_bounds_cover_every_feature_and_increase(self, task):
        feats = task.feature_columns()
        bounds = task.design_space_bounds()
        assert set(feats) == set(bounds)
        for name, (lo, hi) in bounds.items():
            assert lo < hi, f"bounds for {name!r} must be strictly increasing"

    def test_feature_types_cover_every_feature(self, task):
        types = task.feature_types()
        assert set(types) == set(task.feature_columns())
        assert set(types.values()) <= {"continuous", "integer", "categorical"}

    def test_type_partition(self, task):
        """3 categorical + 2 integer + 14 continuous = 19."""
        types = task.feature_types()
        counts = {t: sum(1 for v in types.values() if v == t) for t in set(types.values())}
        assert counts == {"categorical": 3, "integer": 2, "continuous": 14}

    def test_applied_potential_is_cathodic(self, task):
        """A CO2RR cathode is driven negative vs. RHE; a positive bound
        here would let BO propose a physically meaningless anodic run."""
        lo, hi = task.design_space_bounds()["Applied_potential"]
        assert lo < 0 and hi < 0


# =============================================================================
#  Categorical descriptors
# =============================================================================
class TestCategoricalDescriptors:
    def test_declared_categoricals_match_feature_types(self, task):
        cat_by_type = {
            f for f, t in task.feature_types().items() if t == "categorical"
        }
        assert cat_by_type == set(task.categorical_values())

    def test_bounds_match_ordinal_code_range(self, task):
        """Bounds must be (0, n-1): the surrogate normalizes with these and
        the engine snaps onto the grid they imply."""
        bounds = task.design_space_bounds()
        for feature, values in task.categorical_values().items():
            assert bounds[feature] == (0, len(values) - 1)

    def test_category_values_are_unique(self, task):
        for feature, values in task.categorical_values().items():
            assert len(set(values)) == len(values), feature

    def test_encode_decode_round_trip(self, task):
        for feature, values in task.categorical_values().items():
            for v in values:
                code = task.encode_categorical(feature, v)
                assert task.decode_categorical(feature, code) == v

    def test_decode_tolerates_float_drift(self, task):
        """Codes survive a normalize/unnormalize round-trip, not exact ints."""
        assert task.decode_categorical("Metal_identity", 0.0000001) == "Cu"
        assert task.decode_categorical("Metal_identity", 1.9999997) == "Au"

    def test_encode_rejects_unknown_category(self, task):
        with pytest.raises(ValueError, match="not a valid Metal_identity"):
            task.encode_categorical("Metal_identity", "Unobtainium")

    def test_decode_rejects_out_of_range_code(self, task):
        with pytest.raises(ValueError, match="out of range"):
            task.decode_categorical("Cation", 99)

    def test_encode_rejects_non_categorical_feature(self, task):
        with pytest.raises(ValueError, match="not a categorical feature"):
            task.encode_categorical("Temperature", "hot")

    def test_categorical_values_are_defensive_copies(self, task):
        """Mutating the returned dict must not corrupt the constants."""
        task.categorical_values()["Metal_identity"].append("Fe")
        assert "Fe" not in task.categorical_values()["Metal_identity"]


# =============================================================================
#  Product / target schema
# =============================================================================
class TestProductSchema:
    def test_products_are_faradaic_efficiency_columns(self, task):
        assert all(c.startswith("FE_") for c in task.all_product_columns())

    def test_default_target_resolves_to_column(self, task):
        col = task.resolve_target_column(task.default_target())
        assert col == "FE_CO"
        assert col in task.target_columns().values()

    def test_product_names_inverts_target_columns(self, task):
        assert task.product_names() == {
            v: k for k, v in task.target_columns().items()
        }

    def test_her_product_present(self, task):
        """FE_H2 is load-bearing: the penalty and MO preset both name it."""
        assert "FE_H2" in task.all_product_columns()

    def test_mo_preset_columns_exist_and_oppose(self, task):
        objs = task.multi_objectives()
        cols = set(task.target_columns().values())
        assert [o.column for o in objs] == ["FE_CO", "FE_H2"]
        assert all(o.column in cols for o in objs)
        assert {o.direction for o in objs} == {"max", "min"}


# =============================================================================
#  Training target
# =============================================================================
class TestBuildTrainingTarget:
    @pytest.fixture
    def df(self) -> pd.DataFrame:
        return pd.DataFrame({"FE_CO": [60.0, 20.0], "FE_H2": [10.0, 70.0]})

    def test_without_penalty_returns_raw_target(self, task, df):
        y = task.build_training_target(df, "FE_CO")
        np.testing.assert_allclose(y, [60.0, 20.0])

    def test_penalty_subtracts_scaled_her(self, task, df):
        y = task.build_training_target(df, "FE_CO", h2_penalty_weight=0.5)
        np.testing.assert_allclose(y, [55.0, -15.0])

    def test_penalty_reorders_her_dominant_conditions(self, task, df):
        """The whole point of the penalty: row 0 must beat row 1 by more
        once HER is charged for."""
        raw = task.build_training_target(df, "FE_CO")
        pen = task.build_training_target(df, "FE_CO", h2_penalty_weight=0.5)
        assert (pen[0] - pen[1]) > (raw[0] - raw[1])

    def test_zero_penalty_is_a_no_op(self, task, df):
        np.testing.assert_allclose(
            task.build_training_target(df, "FE_CO", h2_penalty_weight=0.0),
            task.build_training_target(df, "FE_CO"),
        )

    def test_missing_her_column_falls_back(self, task):
        df = pd.DataFrame({"FE_CO": [60.0]})
        y = task.build_training_target(df, "FE_CO", h2_penalty_weight=0.5)
        np.testing.assert_allclose(y, [60.0])

    def test_returns_float64(self, task, df):
        assert task.build_training_target(df, "FE_CO").dtype == np.float64


# =============================================================================
#  Simulated observations
# =============================================================================
class TestSimulateObservation:
    def test_returns_every_product_column(self, task):
        np.random.seed(0)
        out = task.simulate_observation("FE_CO", y_mean=50.0, y_std=5.0)
        assert set(out) == set(task.all_product_columns())

    def test_respects_the_fe_budget(self, task):
        """FE is a share of charge — the vector cannot sum past 100%."""
        for seed in range(50):
            np.random.seed(seed)
            out = task.simulate_observation("FE_CO", y_mean=90.0, y_std=20.0)
            assert sum(out.values()) <= ECO2RR_FE_TOTAL_MAX + 1e-6, f"seed={seed}"

    def test_never_negative(self, task):
        for seed in range(50):
            np.random.seed(seed)
            out = task.simulate_observation("FE_CO", y_mean=1.0, y_std=50.0)
            assert all(v >= 0.0 for v in out.values()), f"seed={seed}"

    def test_values_are_floats(self, task):
        np.random.seed(0)
        out = task.simulate_observation("FE_CO", y_mean=50.0, y_std=5.0)
        assert all(isinstance(v, float) for v in out.values())


# =============================================================================
#  Candidate validation
# =============================================================================
class TestValidateCandidates:
    def test_accepts_a_well_formed_frame(self, task):
        df = pd.DataFrame({f: [0.0] for f in task.feature_columns()})
        out = task.validate_candidates(df)
        assert len(out) == len(df)

    def test_warns_on_out_of_range_categorical_code(self, task, caplog):
        df = pd.DataFrame({f: [0.0] for f in task.feature_columns()})
        df.loc[0, "Metal_identity"] = 99.0
        with caplog.at_level("WARNING"):
            task.validate_candidates(df)
        assert "out-of-range" in caplog.text
        assert "Metal_identity" in caplog.text

    def test_returns_frame_unchanged(self, task):
        df = pd.DataFrame({f: [0.0] for f in task.feature_columns()})
        df.loc[0, "Metal_identity"] = 99.0
        out = task.validate_candidates(df)
        pd.testing.assert_frame_equal(out, df)


# =============================================================================
#  Dynamic candidate pool (needs torch for SobolEngine)
# =============================================================================
@pytest.mark.requires_torch
class TestGenerateCandidates:
    def test_shape_and_column_order(self, task):
        df = task.generate_candidates(n=100, seed=0)
        assert len(df) == 100
        assert list(df.columns) == task.feature_columns()

    def test_all_values_within_design_bounds(self, task):
        df = task.generate_candidates(n=200, seed=1)
        for f, (lo, hi) in task.design_space_bounds().items():
            assert df[f].min() >= lo, f
            assert df[f].max() <= hi, f

    def test_categorical_and_integer_dims_are_whole_numbers(self, task):
        df = task.generate_candidates(n=200, seed=2)
        types = task.feature_types()
        for f, t in types.items():
            if t in ("categorical", "integer"):
                np.testing.assert_array_equal(
                    df[f].values, np.round(df[f].values), err_msg=f,
                )

    def test_every_category_is_represented(self, task):
        """Enumeration, not sampling: no electrode may go unproposed."""
        df = task.generate_candidates(n=200, seed=3)
        for f, values in task.categorical_values().items():
            assert set(df[f].unique()) == set(range(len(values))), f

    def test_covers_the_full_categorical_cross_product(self, task):
        df = task.generate_candidates(n=200, seed=4)
        cats = list(task.categorical_values())
        n_combos = int(np.prod([len(v) for v in task.categorical_values().values()]))
        assert len(df[cats].drop_duplicates()) == n_combos

    def test_is_deterministic_given_a_seed(self, task):
        pd.testing.assert_frame_equal(
            task.generate_candidates(n=50, seed=7),
            task.generate_candidates(n=50, seed=7),
        )

    def test_decodes_back_to_real_categories(self, task):
        df = task.generate_candidates(n=20, seed=8)
        labels = [
            task.decode_categorical("Metal_identity", c)
            for c in df["Metal_identity"]
        ]
        assert set(labels) <= set(task.categorical_values()["Metal_identity"])
