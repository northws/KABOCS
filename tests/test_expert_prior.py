"""Unit tests for ``kabo.knowledge.ExpertPrior`` — distribution coverage.

Requires torch (the log-score evaluators return ``torch.Tensor``).
Skipped automatically when the optional torch stack is unavailable.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

pytestmark = pytest.mark.requires_torch


@pytest.fixture(scope="module")
def torch_mod():
    return pytest.importorskip("torch")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _write_prior(tmp_path: Path, config: dict) -> Path:
    p = tmp_path / "prior.json"
    p.write_text(json.dumps(config))
    return p


def _make_prior(torch, tmp_path: Path, config: dict, features=("x1", "x2")):
    from kabo.knowledge import ExpertPrior

    # Design bounds: x1 ∈ [0, 10], x2 ∈ [0, 10]   (2-row tensor)
    bounds_raw = torch.tensor(
        [[0.0] * len(features), [10.0] * len(features)], dtype=torch.double,
    )
    return ExpertPrior(
        config_path=_write_prior(tmp_path, config),
        selected_features=list(features),
        bounds_raw=bounds_raw,
        device=torch.device("cpu"),
    )


def _norm(torch, value: float, lo: float = 0.0, hi: float = 10.0) -> float:
    """Helper: raw → normalised in [0, 1]."""
    return (value - lo) / (hi - lo)


# ---------------------------------------------------------------------------
# Validator coverage
# ---------------------------------------------------------------------------
class TestValidator:
    def test_rejects_unknown_type(self, torch_mod, tmp_path: Path):
        prior = _make_prior(
            torch_mod, tmp_path,
            {"x1": {"type": "cauchy", "x0": 1.0, "gamma": 0.5}},
        )
        assert prior.priors == {}  # silently ignored

    def test_rejects_nonpositive_gaussian_std(self, torch_mod, tmp_path: Path):
        prior = _make_prior(
            torch_mod, tmp_path,
            {"x1": {"type": "gaussian", "mean": 3.0, "std": 0.0}},
        )
        assert 0 not in prior.priors

    def test_rejects_bad_beta(self, torch_mod, tmp_path: Path):
        prior = _make_prior(
            torch_mod, tmp_path,
            {"x1": {"type": "beta", "alpha": 0, "beta": 2, "min": 0, "max": 1}},
        )
        assert 0 not in prior.priors

    def test_rejects_inverted_beta_bounds(self, torch_mod, tmp_path: Path):
        prior = _make_prior(
            torch_mod, tmp_path,
            {"x1": {"type": "beta", "alpha": 2, "beta": 2, "min": 5, "max": 3}},
        )
        assert 0 not in prior.priors

    def test_rejects_nonpositive_lognormal_sigma(self, torch_mod, tmp_path: Path):
        prior = _make_prior(
            torch_mod, tmp_path,
            {"x1": {"type": "lognormal", "mu": 0.0, "sigma": -0.5}},
        )
        assert 0 not in prior.priors

    def test_rejects_empty_categorical(self, torch_mod, tmp_path: Path):
        prior = _make_prior(
            torch_mod, tmp_path,
            {"x1": {"type": "categorical", "weights": {}}},
        )
        assert 0 not in prior.priors

    def test_accepts_new_types(self, torch_mod, tmp_path: Path):
        prior = _make_prior(
            torch_mod, tmp_path,
            {
                "x1": {"type": "beta", "alpha": 2, "beta": 5, "min": 0, "max": 10},
                "x2": {"type": "lognormal", "mu": 1.0, "sigma": 0.5},
            },
        )
        # Both should survive validation.
        assert 0 in prior.priors
        assert 1 in prior.priors


# ---------------------------------------------------------------------------
# Gaussian / Uniform — regression (legacy behaviour preserved)
# ---------------------------------------------------------------------------
class TestGaussianUniformRegression:
    def test_gaussian_peaks_at_mean(self, torch_mod, tmp_path: Path):
        torch = torch_mod
        prior = _make_prior(
            torch, tmp_path,
            {"x1": {"type": "gaussian", "mean": 5.0, "std": 1.0}},
        )
        X = torch.tensor(
            [[_norm(torch, 5.0), 0.5],
             [_norm(torch, 3.0), 0.5]],
            dtype=torch.double,
        )
        scores = prior.evaluate(X).squeeze(-1)
        assert scores[0].item() > scores[1].item()

    def test_uniform_inside_outside(self, torch_mod, tmp_path: Path):
        torch = torch_mod
        prior = _make_prior(
            torch, tmp_path,
            {"x1": {"type": "uniform", "min": 2.0, "max": 6.0}},
        )
        X = torch.tensor(
            [[_norm(torch, 4.0), 0.0],   # inside
             [_norm(torch, 8.0), 0.0]],  # outside
            dtype=torch.double,
        )
        scores = prior.evaluate(X).squeeze(-1)
        assert scores[0].item() == 0.0
        assert scores[1].item() < 0.0


# ---------------------------------------------------------------------------
# Beta
# ---------------------------------------------------------------------------
class TestBeta:
    def test_beta_peaks_near_mode(self, torch_mod, tmp_path: Path):
        torch = torch_mod
        # Beta(2, 5) on [0, 10] peaks at t = (α-1)/(α+β-2) = 1/5 → x = 2.
        prior = _make_prior(
            torch, tmp_path,
            {"x1": {"type": "beta", "alpha": 2, "beta": 5, "min": 0, "max": 10}},
        )
        X = torch.tensor(
            [[_norm(torch, 2.0), 0.0],  # mode
             [_norm(torch, 9.0), 0.0]], # far tail
            dtype=torch.double,
        )
        scores = prior.evaluate(X).squeeze(-1)
        assert scores[0].item() > scores[1].item()

    def test_beta_outside_bounds_penalised(self, torch_mod, tmp_path: Path):
        torch = torch_mod
        prior = _make_prior(
            torch, tmp_path,
            {"x1": {"type": "beta", "alpha": 2, "beta": 2, "min": 3, "max": 7}},
        )
        # Normalised X=0 means raw=0 — well below min=3, should be strongly
        # penalised.
        X = torch.tensor([[0.0, 0.0]], dtype=torch.double)
        scores = prior.evaluate(X).squeeze(-1)
        assert scores[0].item() < -10.0  # penalty kicked in

    def test_beta_gradient_exists(self, torch_mod, tmp_path: Path):
        """The Beta handler must be autograd-friendly for UCB optimization."""
        torch = torch_mod
        prior = _make_prior(
            torch, tmp_path,
            {"x1": {"type": "beta", "alpha": 3, "beta": 2, "min": 0, "max": 10}},
        )
        X = torch.tensor(
            [[_norm(torch, 4.0), 0.0]],
            dtype=torch.double, requires_grad=True,
        )
        out = prior.evaluate(X).sum()
        out.backward()
        assert X.grad is not None
        assert torch.isfinite(X.grad).all()


# ---------------------------------------------------------------------------
# Lognormal
# ---------------------------------------------------------------------------
class TestLogNormal:
    def test_lognormal_peaks_near_median(self, torch_mod, tmp_path: Path):
        torch = torch_mod
        # LogNormal(μ=ln(3), σ=0.5): median = e^μ = 3.
        prior = _make_prior(
            torch, tmp_path,
            {"x1": {"type": "lognormal", "mu": math.log(3.0), "sigma": 0.5}},
        )
        X = torch.tensor(
            [[_norm(torch, 3.0), 0.0],   # near median
             [_norm(torch, 9.5), 0.0]],  # far tail
            dtype=torch.double,
        )
        scores = prior.evaluate(X).squeeze(-1)
        assert scores[0].item() > scores[1].item()

    def test_lognormal_nonpositive_penalised(self, torch_mod, tmp_path: Path):
        """X <= 0 is outside the log-normal support."""
        torch = torch_mod
        # Build a prior whose normalised 0 maps to raw -1 (below support).
        from kabo.knowledge import ExpertPrior
        bounds_raw = torch.tensor([[-1.0, -1.0], [10.0, 10.0]], dtype=torch.double)
        cfg = _write_prior(
            tmp_path,
            {"x1": {"type": "lognormal", "mu": 0.0, "sigma": 0.5}},
        )
        prior = ExpertPrior(
            config_path=cfg,
            selected_features=["x1", "x2"],
            bounds_raw=bounds_raw,
            device=torch.device("cpu"),
        )
        X = torch.tensor([[0.0, 0.0]], dtype=torch.double)  # raw x1 = -1 < 0
        scores = prior.evaluate(X).squeeze(-1)
        assert scores[0].item() < 0.0


# ---------------------------------------------------------------------------
# Categorical
# ---------------------------------------------------------------------------
class TestCategorical:
    def test_categorical_prefers_higher_weight(self, torch_mod, tmp_path: Path):
        torch = torch_mod
        # weights={0: 0.1, 1: 0.9}  → value 1 preferred.
        # On [0, 10], raw==1 → norm = 0.1; raw==0 → norm = 0.0.
        prior = _make_prior(
            torch, tmp_path,
            {"x1": {"type": "categorical", "weights": {"0": 0.1, "1": 0.9}}},
        )
        X = torch.tensor(
            [[_norm(torch, 1.0), 0.0],
             [_norm(torch, 0.0), 0.0]],
            dtype=torch.double,
        )
        scores = prior.evaluate(X).squeeze(-1)
        assert scores[0].item() > scores[1].item()

    def test_categorical_unlisted_value_penalised(self, torch_mod, tmp_path: Path):
        torch = torch_mod
        prior = _make_prior(
            torch, tmp_path,
            {"x1": {"type": "categorical", "weights": {"0": 1.0, "1": 1.0}}},
        )
        # Normalised 0.5 on [0, 10] → raw 5, not in {0, 1}.
        X_listed = torch.tensor([[_norm(torch, 1.0), 0.0]], dtype=torch.double)
        X_missing = torch.tensor([[_norm(torch, 5.0), 0.0]], dtype=torch.double)
        s_listed = prior.evaluate(X_listed).squeeze(-1).item()
        s_missing = prior.evaluate(X_missing).squeeze(-1).item()
        assert s_listed > s_missing

    def test_categorical_batch_shape_preserved(self, torch_mod, tmp_path: Path):
        torch = torch_mod
        prior = _make_prior(
            torch, tmp_path,
            {"x1": {"type": "categorical", "weights": {"0": 1.0, "1": 2.0}}},
        )
        # (b, q, K) — BoTorch-style batch.
        X = torch.zeros((3, 2, 2), dtype=torch.double)
        out = prior.evaluate(X)
        assert out.shape == (3, 2, 1)


# ---------------------------------------------------------------------------
# Composition: multiple types in a single config
# ---------------------------------------------------------------------------
class TestComposition:
    def test_sum_of_independent_priors(self, torch_mod, tmp_path: Path):
        torch = torch_mod
        prior = _make_prior(
            torch, tmp_path,
            {
                "x1": {"type": "gaussian", "mean": 5.0, "std": 1.0},
                "x2": {"type": "beta", "alpha": 2, "beta": 5, "min": 0, "max": 10},
            },
        )
        # Point at the joint mode: x1=5, x2≈2.
        X = torch.tensor(
            [[_norm(torch, 5.0), _norm(torch, 2.0)],
             [_norm(torch, 0.0), _norm(torch, 9.0)]],
            dtype=torch.double,
        )
        scores = prior.evaluate(X).squeeze(-1)
        assert scores[0].item() > scores[1].item()
