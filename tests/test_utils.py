"""Unit tests for ``kabo.utils``: normalization, standardization, grid snap."""

from __future__ import annotations

import numpy as np
import pytest

# torch is optional here because unnormalize_x / round_integer_dims_to_grid need it.
torch = pytest.importorskip("torch", reason="kabo.utils.unnormalize_x needs torch tensors")

from kabo.utils import (  # noqa: E402  (import after skip guard)
    categorical_indices_from_types,
    compute_bounds,
    integer_indices_from_types,
    normalize_x,
    round_integer_dims_to_grid,
    standardize_y,
    unnormalize_x,
    unstandardize_y,
)


# ---------------------------------------------------------------------------
# compute_bounds / normalize_x
# ---------------------------------------------------------------------------
class TestNormalization:
    def test_normalize_maps_to_unit_interval(self):
        X = np.array([[0.0, 100.0], [10.0, 200.0], [5.0, 150.0]])
        x_min, x_max, x_range = compute_bounds(X)
        X_norm = normalize_x(X, x_min, x_range)

        assert X_norm.min() == pytest.approx(0.0)
        assert X_norm.max() == pytest.approx(1.0)
        assert X_norm.shape == X.shape

    def test_zero_range_column_is_handled(self):
        """A constant column would cause a divide-by-zero; utils replaces
        range with 1.0 so downstream code stays numerically safe."""
        X = np.array([[1.0, 7.0], [1.0, 8.0], [1.0, 9.0]])
        x_min, x_max, x_range = compute_bounds(X)

        assert x_range[0] == 1.0, "constant column range should be coerced to 1"
        assert x_range[1] == 2.0

        X_norm = normalize_x(X, x_min, x_range)
        assert np.isfinite(X_norm).all()

    def test_unnormalize_is_inverse(self):
        bounds = torch.tensor([[-1.0, 5.0], [3.0, 50.0]], dtype=torch.double)
        x_norm = torch.tensor([0.0, 0.5, 1.0][:2], dtype=torch.double)
        x_raw = unnormalize_x(x_norm, bounds)

        # x_norm=[0.0, 0.5] with bounds col0=(-1,3), col1=(5,50) =>
        #   raw = [-1.0 + 0.0*4, 5.0 + 0.5*45] = [-1.0, 27.5]
        assert np.allclose(x_raw, [-1.0, 27.5])


# ---------------------------------------------------------------------------
# standardize_y / unstandardize_y
# ---------------------------------------------------------------------------
class TestStandardization:
    def test_standardize_yields_zero_mean_unit_var(self):
        Y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        Y_std, mu, sigma = standardize_y(Y)

        assert Y_std.mean() == pytest.approx(0.0, abs=1e-12)
        assert Y_std.std() == pytest.approx(1.0, abs=1e-12)
        assert mu == pytest.approx(3.0)

    def test_standardize_constant_series_does_not_divide_by_zero(self):
        Y = np.array([7.0, 7.0, 7.0])
        Y_std, mu, sigma = standardize_y(Y)

        assert np.isfinite(Y_std).all()
        assert sigma == 1.0  # coerced
        assert mu == 7.0

    def test_unstandardize_is_inverse(self):
        Y = np.array([10.0, 20.0, 30.0])
        Y_std, mu, sigma = standardize_y(Y)

        recovered = [unstandardize_y(v, mu, sigma) for v in Y_std]
        assert np.allclose(recovered, Y)


# ---------------------------------------------------------------------------
# round_integer_dims_to_grid
# ---------------------------------------------------------------------------
class TestIntegerGridSnap:
    def test_no_integer_dims_returns_input(self):
        x_norm = torch.tensor([0.37, 0.12, 0.99], dtype=torch.double)
        bounds = torch.tensor([[0.0, 0.0, 0.0], [10.0, 20.0, 6.0]], dtype=torch.double)
        out = round_integer_dims_to_grid(x_norm, integer_indices=[], bounds_raw=bounds)
        assert torch.equal(out, x_norm)

    def test_snaps_to_nearest_grid_point(self):
        # integer dim with raw span (0, 6) → normalized grid: 0, 1/6, 2/6, …, 1
        x_norm = torch.tensor([0.20], dtype=torch.double)  # closer to 1/6 ≈ 0.1667
        bounds = torch.tensor([[0.0], [6.0]], dtype=torch.double)

        out = round_integer_dims_to_grid(x_norm, integer_indices=[0], bounds_raw=bounds)
        assert out[0].item() == pytest.approx(1.0 / 6, abs=1e-9)

    def test_snap_respects_clamp(self):
        x_norm = torch.tensor([1.1], dtype=torch.double)  # out of range
        bounds = torch.tensor([[0.0], [6.0]], dtype=torch.double)

        out = round_integer_dims_to_grid(x_norm, integer_indices=[0], bounds_raw=bounds)
        assert 0.0 <= out[0].item() <= 1.0

    def test_degenerate_dim_snaps_to_zero(self):
        x_norm = torch.tensor([0.7], dtype=torch.double)
        bounds = torch.tensor([[5.0], [5.0]], dtype=torch.double)  # span 0

        out = round_integer_dims_to_grid(x_norm, integer_indices=[0], bounds_raw=bounds)
        assert out[0].item() == 0.0


# ---------------------------------------------------------------------------
# integer_indices_from_types / categorical_indices_from_types
# ---------------------------------------------------------------------------
class TestFeatureTypeIndices:
    def test_integer_indices_basic(self):
        feats = ["a", "b", "c"]
        types = {"a": "continuous", "b": "integer", "c": "integer"}
        assert integer_indices_from_types(feats, types) == [1, 2]

    def test_categorical_indices_picks_both_categorical_and_ordinal(self):
        feats = ["a", "b", "c", "d"]
        types = {"a": "continuous", "b": "categorical", "c": "ordinal", "d": "integer"}
        assert categorical_indices_from_types(feats, types) == [1, 2]

    def test_none_feature_types_returns_empty(self):
        feats = ["a", "b"]
        assert integer_indices_from_types(feats, None) == []
        assert categorical_indices_from_types(feats, None) == []
