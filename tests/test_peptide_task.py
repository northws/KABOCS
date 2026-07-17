"""Unit tests for ``kabo.task.peptide_eco2rr.PeptideECO2RRTask``.

The load-bearing property here is that the ligand is encoded by CONTINUOUS
descriptors, not a categorical identity — that is the whole reason the task
exists, so most of these tests defend it directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kabo.constants import AMINO_ACID_DESCRIPTORS
from kabo.task import TASK_REGISTRY, PeptideECO2RRTask, TaskBase, get_task
from kabo.task.peptide_eco2rr import featurize_sequence, parse_sequence


@pytest.fixture
def task() -> PeptideECO2RRTask:
    return PeptideECO2RRTask()


# =============================================================================
#  Registration
# =============================================================================
class TestRegistration:
    def test_registered(self):
        assert "peptide" in TASK_REGISTRY

    def test_get_task_case_insensitive(self):
        assert type(get_task("peptide")) is type(get_task("PEPTIDE")) is PeptideECO2RRTask

    def test_is_taskbase(self, task):
        assert isinstance(task, TaskBase)

    def test_does_not_shadow_other_tasks(self):
        for name in ("co2rr", "eco2rr", "test"):
            assert name in TASK_REGISTRY
            assert TASK_REGISTRY[name] is not TASK_REGISTRY["peptide"]


# =============================================================================
#  The encoding contract — this is the point of the task
# =============================================================================
class TestEncodingIsContinuous:
    def test_no_categorical_dims(self, task):
        """A categorical ligand dim would make untested residues unreachable."""
        assert set(task.feature_types().values()) == {"continuous"}

    def test_declares_no_categorical_values(self, task):
        assert task.categorical_values() == {}

    def test_bounds_span_every_canonical_residue(self, task):
        """An untested residue must land INSIDE the design space, or the
        surrogate's normalization would clip it onto another residue."""
        bounds = task.design_space_bounds()
        for residue in AMINO_ACID_DESCRIPTORS:
            feats = task.ligand_features(residue)
            for col, v in feats.items():
                lo, hi = bounds[col]
                assert lo <= v <= hi, f"{residue}.{col}={v} outside ({lo},{hi})"

    def test_schema_is_consistent(self, task):
        feats = task.feature_columns()
        bounds = task.design_space_bounds()
        assert set(feats) == set(bounds)
        assert len(set(feats)) == len(feats)
        for name, (lo, hi) in bounds.items():
            assert lo < hi, name

    def test_potential_is_cathodic(self, task):
        lo, hi = task.design_space_bounds()["Applied_potential"]
        assert lo < 0 and hi < 0


# =============================================================================
#  Sequence parsing / featurization
# =============================================================================
class TestSequenceParsing:
    def test_parses_hyphenated_peptide(self):
        assert parse_sequence("His-Arg-His") == ["His", "Arg", "His"]

    def test_parses_lowercase_single(self):
        assert parse_sequence("met") == ["Met"]
        assert parse_sequence("gln") == ["Gln"]

    def test_parses_spaces_and_underscores(self):
        assert parse_sequence("his arg his") == ["His", "Arg", "His"]
        assert parse_sequence("Gln_Met") == ["Gln", "Met"]

    def test_rejects_unknown_residue(self):
        """A typo must not silently become a shorter peptide."""
        with pytest.raises(ValueError, match="Unknown residue"):
            parse_sequence("His-Xyz-His")

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="Empty ligand"):
            parse_sequence("-")

    def test_homopeptide_equals_its_residue(self):
        """His-His-His averages back to His (up to float error)."""
        tri = featurize_sequence(["His", "His", "His"])
        one = featurize_sequence(["His"])
        assert tri.keys() == one.keys()
        for k in one:
            assert tri[k] == pytest.approx(one[k])

    def test_average_lies_between_constituents(self):
        """His-Arg-His must sit between His and Arg on pI."""
        his = featurize_sequence(["His"])["pI"]
        arg = featurize_sequence(["Arg"])["pI"]
        mix = featurize_sequence(["His", "Arg", "His"])["pI"]
        assert his < mix < arg

    def test_distinguishes_different_peptides(self, task):
        """The whole point of averaging: His-His-His != His-Arg-His."""
        a = task.ligand_features("His-His-His")
        b = task.ligand_features("His-Arg-His")
        assert a != b


# =============================================================================
#  DataFrame featurization
# =============================================================================
class TestAddLigandDescriptors:
    def test_adds_all_descriptor_columns(self, task):
        df = pd.DataFrame({"ligand": ["His-Arg-His", "met"], "x": [1, 2]})
        out = task.add_ligand_descriptors(df)
        for col in task.feature_columns():
            if col != "Applied_potential":
                assert col in out.columns

    def test_does_not_mutate_input(self, task):
        df = pd.DataFrame({"ligand": ["met"]})
        task.add_ligand_descriptors(df)
        assert list(df.columns) == ["ligand"]

    def test_raises_on_missing_column(self, task):
        with pytest.raises(KeyError, match="not in DataFrame"):
            task.add_ligand_descriptors(pd.DataFrame({"a": [1]}))

    def test_values_match_direct_featurization(self, task):
        df = pd.DataFrame({"ligand": ["Gln-Met"]})
        out = task.add_ligand_descriptors(df)
        direct = task.ligand_features("Gln-Met")
        for col, v in direct.items():
            assert out.loc[0, col] == pytest.approx(v)


# =============================================================================
#  nearest_residue — interpreting continuous proposals
# =============================================================================
class TestNearestResidue:
    def test_real_residue_maps_to_itself_at_zero_distance(self, task):
        for residue in ("His", "Arg", "Met", "Gln", "Trp", "Asp"):
            name, d = task.nearest_residue(task.ligand_features(residue))
            assert name == residue
            assert d == pytest.approx(0.0, abs=1e-9)

    def test_chimera_reports_nonzero_distance(self, task):
        """A proposal between two residues is not a real residue, and the
        returned distance must say so rather than pretending it is."""
        a = task.ligand_features("Gly")
        b = task.ligand_features("Trp")
        mid = {k: (a[k] + b[k]) / 2 for k in a}
        _, d = task.nearest_residue(mid)
        assert d > 0.1

    def test_returns_a_known_residue(self, task):
        a = task.ligand_features("His")
        name, _ = task.nearest_residue(a)
        assert name in AMINO_ACID_DESCRIPTORS


# =============================================================================
#  Objectives
# =============================================================================
class TestObjectives:
    @pytest.fixture
    def df(self) -> pd.DataFrame:
        return pd.DataFrame({"FE_CO": [80.0, 20.0], "FE_H2": [5.0, 60.0]})

    def test_no_penalty_returns_raw(self, task, df):
        np.testing.assert_allclose(task.build_training_target(df, "FE_CO"), [80.0, 20.0])

    def test_penalty_subtracts_her(self, task, df):
        np.testing.assert_allclose(
            task.build_training_target(df, "FE_CO", h2_penalty_weight=0.3),
            [78.5, 2.0],
        )

    def test_missing_her_falls_back(self, task):
        d = pd.DataFrame({"FE_CO": [50.0]})
        np.testing.assert_allclose(
            task.build_training_target(d, "FE_CO", h2_penalty_weight=0.3), [50.0]
        )

    def test_default_target_resolves(self, task):
        assert task.resolve_target_column(task.default_target()) == "FE_CO"

    def test_mo_preset(self, task):
        objs = task.multi_objectives()
        assert [o.column for o in objs] == ["FE_CO", "FE_H2"]
        assert {o.direction for o in objs} == {"max", "min"}

    def test_products_are_gas_phase_fe(self, task):
        cols = task.all_product_columns()
        assert all(c.startswith("FE_") for c in cols)
        assert "FE_H2" in cols
        # Liquid products belong to ECO2RRTask, not this GC-based schema.
        assert "FE_HCOOH" not in cols


# =============================================================================
#  Candidate pool — how untested residues actually reach BO
# =============================================================================
@pytest.mark.requires_torch
class TestGenerateCandidates:
    def test_column_order(self, task):
        df = task.generate_candidates(n=100, seed=0)
        assert list(df.columns) == task.feature_columns()

    def test_within_bounds(self, task):
        df = task.generate_candidates(n=200, seed=1)
        for col, (lo, hi) in task.design_space_bounds().items():
            assert df[col].min() >= lo, col
            assert df[col].max() <= hi, col

    def test_covers_all_twenty_residues(self, task):
        """Including residues absent from any training set — this is the
        mechanism by which BO can propose an untested residue at all."""
        df = task.generate_candidates(n=200, seed=2)
        names = {task.nearest_residue(r._asdict() if hasattr(r, "_asdict") else dict(r))[0]
                 for _, r in df[[c for c in task.feature_columns()
                                 if c != "Applied_potential"]].iterrows()}
        assert names == set(AMINO_ACID_DESCRIPTORS)

    def test_every_candidate_is_a_real_residue(self, task):
        """Unlike the acquisition's continuous relaxation, the pool must
        contain only synthesisable residues."""
        df = task.generate_candidates(n=100, seed=3)
        lig_cols = [c for c in task.feature_columns() if c != "Applied_potential"]
        for _, row in df.iterrows():
            _, d = task.nearest_residue(dict(row[lig_cols]))
            assert d == pytest.approx(0.0, abs=1e-9)

    def test_deterministic(self, task):
        pd.testing.assert_frame_equal(
            task.generate_candidates(n=60, seed=7),
            task.generate_candidates(n=60, seed=7),
        )


# =============================================================================
#  Simulation
# =============================================================================
class TestSimulateObservation:
    def test_respects_fe_budget(self, task):
        for seed in range(30):
            np.random.seed(seed)
            out = task.simulate_observation("FE_CO", y_mean=90.0, y_std=20.0)
            assert sum(out.values()) <= 100.0 + 1e-6, f"seed={seed}"

    def test_non_negative(self, task):
        for seed in range(30):
            np.random.seed(seed)
            out = task.simulate_observation("FE_CO", y_mean=1.0, y_std=50.0)
            assert all(v >= 0.0 for v in out.values()), f"seed={seed}"

    def test_returns_all_products(self, task):
        np.random.seed(0)
        out = task.simulate_observation("FE_CO", y_mean=50.0, y_std=5.0)
        assert set(out) == set(task.all_product_columns())
