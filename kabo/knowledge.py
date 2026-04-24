"""
KABO Knowledge Encoding: Expert Prior.

Allows the injection of expert domain knowledge as a static prior
distribution over the design space.  The resulting score is referred to
as ``expert_prior_score(x)`` in the KABO acquisition function.

Supported prior types (v1.2):

* ``gaussian``     — ``mean`` + ``std`` in physical units.  Log-probability
                     up to a constant, i.e. ``-0.5 ((x-μ)/σ)²``.
* ``uniform``      — bracketed flat region ``[min, max]`` with a linear
                     outside-penalty so gradients stay stable.
* ``beta``         — concentration ``alpha``, ``beta`` after mapping the
                     feature onto ``[min, max]``.  Natural fit for
                     pH-like ratios and normalised concentrations.
* ``lognormal``    — ``mu``, ``sigma`` on the log scale.  Use for
                     strictly-positive quantities with a long upper tail
                     (mass loadings, reaction rates …).
* ``categorical``  — for integer-indexed features: a dict of
                     ``value → weight`` entries.  Unlisted values get a
                     configurable penalty.

.. note::
   This is **not** a Knowledge Gradient (KG) or Value-of-Information
   (VOI) computation.  It is a deterministic, configuration-driven prior
   penalty/reward that biases the search towards expert-believed regions.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from kabo.utils import unnormalize_x

logger = logging.getLogger(__name__)

SUPPORTED_PRIOR_TYPES: tuple[str, ...] = (
    "gaussian", "uniform", "beta", "lognormal", "categorical",
)


# =============================================================================
#  Per-type config validators
# =============================================================================
def _validate_params(feat: str, params: dict) -> bool:
    """Return True iff the prior config for ``feat`` is self-consistent.

    Raises *nothing* — on failure logs an error and returns False so the
    caller can skip the offending entry.
    """
    ptype = str(params.get("type", "uniform")).lower()
    if ptype not in SUPPORTED_PRIOR_TYPES:
        logger.warning(
            f"Unknown prior type '{ptype}' for feature '{feat}'. "
            f"Supported: {SUPPORTED_PRIOR_TYPES}. Ignored."
        )
        return False

    if ptype == "gaussian":
        std = float(params.get("std", 1.0))
        if std <= 0:
            logger.error(
                f"Expert prior for '{feat}': std must be > 0 (got {std}). "
                "Skipping."
            )
            return False

    elif ptype == "uniform":
        lo = float(params.get("min", -np.inf))
        hi = float(params.get("max", np.inf))
        if lo >= hi:
            logger.error(
                f"Expert prior for '{feat}': min ({lo}) must be < max ({hi}). "
                "Skipping."
            )
            return False

    elif ptype == "beta":
        alpha = float(params.get("alpha", 1.0))
        beta_ = float(params.get("beta", 1.0))
        lo = float(params.get("min", 0.0))
        hi = float(params.get("max", 1.0))
        if alpha <= 0 or beta_ <= 0:
            logger.error(
                f"Expert prior for '{feat}': Beta alpha/beta must be > 0 "
                f"(got alpha={alpha}, beta={beta_}). Skipping."
            )
            return False
        if lo >= hi:
            logger.error(
                f"Expert prior for '{feat}': Beta min ({lo}) must be < "
                f"max ({hi}). Skipping."
            )
            return False

    elif ptype == "lognormal":
        sigma = float(params.get("sigma", 1.0))
        if sigma <= 0:
            logger.error(
                f"Expert prior for '{feat}': lognormal sigma must be > 0 "
                f"(got {sigma}). Skipping."
            )
            return False

    elif ptype == "categorical":
        weights = params.get("weights")
        if not isinstance(weights, dict) or not weights:
            logger.error(
                f"Expert prior for '{feat}': 'weights' must be a non-empty "
                "dict of {value: weight}. Skipping."
            )
            return False
        try:
            _ = [float(v) for v in weights.values()]
        except (TypeError, ValueError):
            logger.error(
                f"Expert prior for '{feat}': categorical weights must be "
                "numeric. Skipping."
            )
            return False

    return True


# =============================================================================
#  Per-type log-score evaluators (vectorised, differentiable)
# =============================================================================
def _score_gaussian(feat_val: torch.Tensor, params: dict) -> torch.Tensor:
    mu = float(params.get("mean", 0.0))
    std = float(params.get("std", 1.0))
    return -0.5 * ((feat_val - mu) / std) ** 2


def _score_uniform(feat_val: torch.Tensor, params: dict) -> torch.Tensor:
    min_val = float(params.get("min", -np.inf))
    max_val = float(params.get("max", np.inf))
    penalty = float(params.get("penalty", 100.0))
    inside = (feat_val >= min_val) & (feat_val <= max_val)
    dist_out = (
        torch.clamp(torch.as_tensor(min_val, dtype=feat_val.dtype,
                                     device=feat_val.device) - feat_val, min=0)
        + torch.clamp(feat_val - max_val, min=0)
    )
    return torch.where(inside, torch.zeros_like(feat_val), -penalty * dist_out)


def _score_beta(feat_val: torch.Tensor, params: dict) -> torch.Tensor:
    """Beta distribution mapped onto ``[min, max]``.

    We evaluate ``(α-1) log(t) + (β-1) log(1-t)`` on the re-scaled
    variable ``t = (x - min) / (max - min) ∈ (0, 1)``.  Outside
    [min, max] we apply a strong linear penalty matching the uniform
    handler's convention.  Constants (normaliser) drop out since only
    relative scores matter.
    """
    alpha = float(params.get("alpha", 1.0))
    beta_ = float(params.get("beta", 1.0))
    lo = float(params.get("min", 0.0))
    hi = float(params.get("max", 1.0))
    penalty = float(params.get("penalty", 100.0))
    eps = 1e-8
    scale = hi - lo

    t = (feat_val - lo) / scale
    inside = (t > 0) & (t < 1)
    # Clamp only for the log() argument so gradient stays defined.
    t_c = torch.clamp(t, min=eps, max=1.0 - eps)
    log_pdf = (alpha - 1.0) * torch.log(t_c) + (beta_ - 1.0) * torch.log1p(-t_c)
    # Outside penalty: reuse uniform-style linear distance.
    dist_out = (
        torch.clamp(torch.as_tensor(lo, dtype=feat_val.dtype,
                                     device=feat_val.device) - feat_val, min=0)
        + torch.clamp(feat_val - hi, min=0)
    )
    return torch.where(inside, log_pdf, -penalty * dist_out)


def _score_lognormal(feat_val: torch.Tensor, params: dict) -> torch.Tensor:
    """Log-normal log-pdf (up to a constant).

    Defined only for ``feat_val > 0``; negative / zero inputs get a large
    linear penalty.  Parameters ``mu`` / ``sigma`` are in **log-space**,
    matching :class:`scipy.stats.lognorm` / :class:`torch.distributions.LogNormal`.
    """
    mu = float(params.get("mu", 0.0))
    sigma = float(params.get("sigma", 1.0))
    penalty = float(params.get("penalty", 100.0))
    eps = 1e-8

    positive = feat_val > eps
    safe = torch.clamp(feat_val, min=eps)
    log_x = torch.log(safe)
    # Log-pdf(x) = -log(x) - 0.5 ((log x - μ)/σ)²  (dropping -log(σ√2π))
    log_pdf = -log_x - 0.5 * ((log_x - mu) / sigma) ** 2

    # Smooth penalty for non-positive inputs: linear distance from 0.
    dist_neg = torch.clamp(-feat_val, min=0)
    return torch.where(positive, log_pdf, -penalty * dist_neg)


def _score_categorical(feat_val: torch.Tensor, params: dict) -> torch.Tensor:
    """Discrete distribution on rounded feature values.

    Given ``weights = {value: weight}``, we round ``feat_val`` to the
    nearest integer and look up its log-weight.  Unlisted values get
    ``missing_penalty`` (default 10).  Weights are treated as
    unnormalised probabilities; ``log(w + eps)`` is returned so that
    zero-weight entries behave like missing values.

    Continuous deviation from an integer is **not** penalised here —
    acquisition-time grid snapping (``round_integer_dims_to_grid``)
    handles that in callers that care.
    """
    weights: dict[Any, Any] = params.get("weights", {})
    missing_penalty = float(params.get("missing_penalty", 10.0))
    eps = 1e-8

    # Build (int_value, log_weight) table on the same device/dtype.
    int_vals = []
    log_weights = []
    for k, v in weights.items():
        try:
            int_vals.append(int(float(k)))
            log_weights.append(math.log(max(float(v), 0.0) + eps))
        except (TypeError, ValueError):
            continue
    if not int_vals:
        return torch.full_like(feat_val, -missing_penalty)

    rounded = torch.round(feat_val).to(torch.int64)
    int_tensor = torch.tensor(
        int_vals, dtype=torch.int64, device=feat_val.device,
    )
    log_weight_tensor = torch.tensor(
        log_weights, dtype=feat_val.dtype, device=feat_val.device,
    )

    # Shape: (..., len(int_vals)) — True where the input matches an entry.
    matches = rounded.unsqueeze(-1) == int_tensor
    any_match = matches.any(dim=-1)
    # Use argmax to grab the first matching weight; fall back to 0 for misses.
    match_idx = matches.to(torch.int64).argmax(dim=-1)
    picked = log_weight_tensor[match_idx]
    return torch.where(
        any_match, picked,
        torch.full_like(picked, -missing_penalty),
    )


_SCORE_TABLE = {
    "gaussian":    _score_gaussian,
    "uniform":     _score_uniform,
    "beta":        _score_beta,
    "lognormal":   _score_lognormal,
    "categorical": _score_categorical,
}


class ExpertPrior:
    """Encodes expert priors over specific features.

    Calculates a prior score (sum of log-probabilities) for a given
    normalized X based on expert-provided distributions.
    """

    def __init__(
        self,
        config_path: Path | str | None,
        selected_features: list[str],
        bounds_raw: torch.Tensor,
        device: torch.device,
    ):
        self.selected_features = selected_features
        self.bounds_raw = bounds_raw
        self.device = device
        self.priors: dict[int, dict] = {}

        if config_path:
            self._load_config(Path(config_path))

    def _load_config(self, config_path: Path) -> None:
        if not config_path.exists():
            logger.warning(f"Expert prior file not found: {config_path}")
            return

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)

            for feat, params in config.items():
                if feat not in self.selected_features:
                    logger.warning(
                        f"Feature '{feat}' in prior config not in selected "
                        "features. Ignored."
                    )
                    continue
                if not isinstance(params, dict):
                    logger.error(
                        f"Expert prior for '{feat}': params must be a dict; "
                        f"got {type(params).__name__}. Skipping."
                    )
                    continue
                if not _validate_params(feat, params):
                    continue

                idx = self.selected_features.index(feat)
                self.priors[idx] = params
                logger.info(f"Loaded expert prior for '{feat}': {params}")
        except Exception as e:
            logger.error(f"Error loading expert prior config: {e}")

    def evaluate(self, X_norm: torch.Tensor) -> torch.Tensor:
        """Evaluate the expert prior score for normalised candidates.

        .. note::
           This is a static prior penalty / reward, not a Knowledge
           Gradient computation.  See module docstring.

        Parameters
        ----------
        X_norm : torch.Tensor
            Normalized candidates, shape (..., K).

        Returns
        -------
        torch.Tensor
            Prior score (sum of log-probabilities), shape (..., 1).
        """
        if not self.priors:
            return torch.zeros(
                X_norm.shape[:-1] + (1,),
                dtype=X_norm.dtype, device=X_norm.device,
            )

        # Unnormalize X to evaluate distributions in original physical space
        bounds = self.bounds_raw.to(X_norm.device)
        x_min = bounds[0]
        x_range = bounds[1] - bounds[0]
        X_raw = X_norm * x_range + x_min

        total_score = torch.zeros(
            X_norm.shape[:-1], dtype=X_norm.dtype, device=X_norm.device,
        )

        for idx, params in self.priors.items():
            feat_val = X_raw[..., idx]
            ptype = str(params.get("type", "uniform")).lower()
            scorer = _SCORE_TABLE.get(ptype)
            if scorer is None:
                # Should never happen — _validate_params already guards.
                continue
            total_score = total_score + scorer(feat_val, params)

        return total_score.unsqueeze(-1)
