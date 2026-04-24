"""Unit tests for ``kabo.feature_selection``.

These tests construct a tiny in-memory DataFrame so that the full stack
(RandomForest + importance plotting) can be exercised without any torch
dependency.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kabo.feature_selection import (
    VALID_FS_METHODS,
    load_and_validate_data,
    plot_correlation_heatmap,
    plot_feature_importances,
    rank_features,
    select_top_k_features,
    train_random_forest,
)


# ---------------------------------------------------------------------------
# Synthetic dataset: y depends on x1 + 2*x2 + 0.1*x3
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def synthetic_csv(tmp_path_factory) -> Path:
    rng = np.random.default_rng(42)
    n = 60
    x1 = rng.uniform(0, 1, n)
    x2 = rng.uniform(0, 1, n)
    x3 = rng.uniform(0, 1, n)
    y = 1.0 * x1 + 2.0 * x2 + 0.1 * x3 + rng.normal(0, 0.02, n)

    df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "y": y})
    tmp = tmp_path_factory.mktemp("data")
    csv = tmp / "synthetic.csv"
    df.to_csv(csv, index=False)
    return csv


@pytest.fixture(scope="module")
def synthetic_df(synthetic_csv: Path) -> pd.DataFrame:
    return pd.read_csv(synthetic_csv)


# ---------------------------------------------------------------------------
# load_and_validate_data
# ---------------------------------------------------------------------------
class TestLoadAndValidate:
    def test_happy_path(self, synthetic_csv: Path):
        df = load_and_validate_data(
            synthetic_csv,
            target_column="y",
            all_feature_columns=["x1", "x2", "x3"],
            all_product_columns=["y"],
            product_names={"y": "Y"},
        )
        assert list(df.columns) >= ["x1", "x2", "x3", "y"]
        assert len(df) > 0

    def test_raises_when_target_missing(self, synthetic_csv: Path):
        with pytest.raises(ValueError, match="Target column"):
            load_and_validate_data(
                synthetic_csv,
                target_column="does_not_exist",
                all_feature_columns=["x1", "x2", "x3"],
                all_product_columns=["y"],
                product_names={"y": "Y"},
            )

    def test_raises_when_file_missing(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_and_validate_data(
                tmp_path / "no-such.csv",
                target_column="y",
                all_feature_columns=["x1"],
                all_product_columns=["y"],
                product_names={"y": "Y"},
            )

    def test_strict_mode_rejects_missing_features(self, synthetic_csv: Path):
        with pytest.raises(ValueError, match="Strict"):
            load_and_validate_data(
                synthetic_csv,
                target_column="y",
                all_feature_columns=["x1", "x2", "x3", "x_extra"],
                all_product_columns=["y"],
                product_names={"y": "Y"},
                strict_feature_schema=True,
            )

    def test_non_strict_mode_allows_subset(self, synthetic_csv: Path):
        df = load_and_validate_data(
            synthetic_csv,
            target_column="y",
            all_feature_columns=["x1", "x2", "x3", "x_extra"],
            all_product_columns=["y"],
            product_names={"y": "Y"},
            strict_feature_schema=False,
        )
        assert "x_extra" not in df.columns


# ---------------------------------------------------------------------------
# train_random_forest + select_top_k_features
# ---------------------------------------------------------------------------
class TestRandomForestSelection:
    def test_importances_reflect_generating_weights(self, synthetic_df: pd.DataFrame):
        _, importances, available = train_random_forest(
            synthetic_df,
            target_column="y",
            all_feature_columns=["x1", "x2", "x3"],
            n_estimators=100,
            random_state=0,
        )
        assert available == ["x1", "x2", "x3"]
        assert len(importances) == 3
        assert np.isclose(importances.sum(), 1.0, atol=1e-6)

        # x2 was the highest-weighted generator (×2), so it should be #1 or #2.
        top_two = list(importances.index[:2])
        assert "x2" in top_two

    def test_small_dataset_downshifts_estimators(self, tmp_path: Path):
        """<10 rows path: estimators and min_samples_split are relaxed."""
        df = pd.DataFrame(
            {
                "x1": [0.1, 0.2, 0.5, 0.9, 0.3],
                "x2": [0.9, 0.7, 0.3, 0.1, 0.5],
                "y": [1.0, 1.1, 1.5, 1.9, 1.3],
            }
        )
        _, importances, _ = train_random_forest(
            df,
            target_column="y",
            all_feature_columns=["x1", "x2"],
            n_estimators=200,
            random_state=0,
        )
        assert len(importances) == 2
        assert np.isclose(importances.sum(), 1.0, atol=1e-6)

    def test_select_top_k_trims_and_preserves_order(self):
        s = pd.Series(
            [0.5, 0.3, 0.15, 0.05],
            index=["a", "b", "c", "d"],
        )
        top = select_top_k_features(s, top_k=2)
        assert top == ["a", "b"]

    def test_select_top_k_clamps_when_oversized(self):
        s = pd.Series([0.6, 0.4], index=["a", "b"])
        top = select_top_k_features(s, top_k=10)
        assert top == ["a", "b"]


# ---------------------------------------------------------------------------
# plot_feature_importances — smoke test
# ---------------------------------------------------------------------------
class TestPlotFeatureImportances:
    def test_plot_writes_png(self, tmp_path: Path):
        s = pd.Series([0.55, 0.30, 0.15], index=["x1", "x2", "x3"])
        out = plot_feature_importances(
            s,
            top_k=2,
            output_dir=tmp_path,
            target_column="y",
            product_names={"y": "Y"},
            task_name="TEST",
        )
        assert out.exists()
        assert out.suffix == ".png"
        assert out.stat().st_size > 100  # non-empty file


# ---------------------------------------------------------------------------
# v1.2 — rank_features unified entry point
# ---------------------------------------------------------------------------
class TestRankFeatures:
    def test_random_forest_matches_legacy_path(self, synthetic_df: pd.DataFrame):
        """method='random_forest' must match train_random_forest exactly."""
        imp_new, avail_new = rank_features(
            synthetic_df,
            target_column="y",
            all_feature_columns=["x1", "x2", "x3"],
            method="random_forest",
            n_estimators=100,
            random_state=0,
        )
        _, imp_old, avail_old = train_random_forest(
            synthetic_df,
            target_column="y",
            all_feature_columns=["x1", "x2", "x3"],
            n_estimators=100,
            random_state=0,
        )
        assert avail_new == avail_old
        pd.testing.assert_series_equal(
            imp_new.rename(None), imp_old.rename(None), check_names=False,
        )

    def test_permutation_promotes_dominant_feature(self, synthetic_df: pd.DataFrame):
        imp, _ = rank_features(
            synthetic_df,
            target_column="y",
            all_feature_columns=["x1", "x2", "x3"],
            method="permutation",
            n_estimators=100,
            random_state=0,
            permutation_repeats=3,
        )
        # x2 carries 2x the weight of x1 and 20x of x3 — must land top-2.
        assert "x2" in imp.index[:2].tolist()
        # All importances must be >= 0 after clipping.
        assert (imp >= 0).all()

    def test_mutual_info_returns_nonneg_and_ranks(self, synthetic_df: pd.DataFrame):
        imp, available = rank_features(
            synthetic_df,
            target_column="y",
            all_feature_columns=["x1", "x2", "x3"],
            method="mutual_info",
            random_state=0,
            mutual_info_n_neighbors=3,
        )
        assert set(available) == {"x1", "x2", "x3"}
        # Mutual information is always non-negative.
        assert (imp >= 0).all()
        # Descending order.
        assert list(imp) == sorted(imp.tolist(), reverse=True)

    def test_unknown_method_raises(self, synthetic_df: pd.DataFrame):
        with pytest.raises(ValueError, match="Unknown feature-selection"):
            rank_features(
                synthetic_df,
                target_column="y",
                all_feature_columns=["x1", "x2", "x3"],
                method="deep_kernel",  # type: ignore[arg-type]
            )

    def test_shap_optional_dep(self, synthetic_df: pd.DataFrame):
        """'shap' must either succeed or fail with a clear RuntimeError."""
        from kabo.feature_selection import _HAS_SHAP

        if _HAS_SHAP:
            imp, _ = rank_features(
                synthetic_df,
                target_column="y",
                all_feature_columns=["x1", "x2", "x3"],
                method="shap",
                n_estimators=50,
                random_state=0,
            )
            assert len(imp) == 3
            assert (imp >= 0).all()
            assert "x2" in imp.index[:2].tolist()
        else:
            with pytest.raises(RuntimeError, match="shap"):
                rank_features(
                    synthetic_df,
                    target_column="y",
                    all_feature_columns=["x1", "x2", "x3"],
                    method="shap",
                    n_estimators=50,
                    random_state=0,
                )

    def test_valid_fs_methods_is_tuple_of_known(self):
        assert set(VALID_FS_METHODS) == {
            "random_forest", "permutation", "mutual_info", "shap",
        }


# ---------------------------------------------------------------------------
# v1.2 — correlation heatmap
# ---------------------------------------------------------------------------
class TestCorrelationHeatmap:
    def test_writes_png_for_multi_column_df(
        self, tmp_path: Path, synthetic_df: pd.DataFrame,
    ):
        out = plot_correlation_heatmap(
            synthetic_df,
            feature_columns=["x1", "x2", "x3"],
            output_dir=tmp_path,
            title_suffix="synthetic",
        )
        assert out is not None
        assert out.exists()
        assert out.suffix == ".png"
        assert out.stat().st_size > 100

    def test_skips_when_only_one_column(self, tmp_path: Path):
        df = pd.DataFrame({"x1": [0.1, 0.2, 0.3]})
        out = plot_correlation_heatmap(
            df,
            feature_columns=["x1"],
            output_dir=tmp_path,
        )
        assert out is None

    def test_handles_missing_columns_gracefully(
        self, tmp_path: Path, synthetic_df: pd.DataFrame,
    ):
        # x_extra does not exist — should be silently dropped.
        out = plot_correlation_heatmap(
            synthetic_df,
            feature_columns=["x1", "x2", "x_extra"],
            output_dir=tmp_path,
        )
        assert out is not None and out.exists()
