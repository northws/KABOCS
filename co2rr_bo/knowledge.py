"""
KABO Knowledge Encoding: Expert Prior.

Allows the injection of expert domain knowledge as a static prior
distribution over the design space.  The resulting score is referred to
as ``expert_prior_score(x)`` in the KABO acquisition function.

.. note::
   This is **not** a Knowledge Gradient (KG) or Value-of-Information
   (VOI) computation.  It is a deterministic, configuration-driven prior
   penalty/reward that biases the search towards expert-believed regions.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch

from co2rr_bo.utils import unnormalize_x

logger = logging.getLogger(__name__)


class ExpertPrior:
    """Encodes expert priors over specific features.

    Calculates a prior score (log-probability) for a given normalized X based
    on expert-provided distributions (e.g., Gaussian, Uniform).
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
                    logger.warning(f"Feature '{feat}' in prior config not in selected features. Ignored.")
                    continue

                idx = self.selected_features.index(feat)
                ptype = params.get("type", "uniform").lower()

                # ---- parameter validation ----
                if ptype == "gaussian":
                    std = float(params.get("std", 1.0))
                    if std <= 0:
                        logger.error(
                            f"Expert prior for '{feat}': std must be > 0 (got {std}). "
                            "Skipping this prior."
                        )
                        continue
                elif ptype == "uniform":
                    lo = float(params.get("min", -np.inf))
                    hi = float(params.get("max", np.inf))
                    if lo >= hi:
                        logger.error(
                            f"Expert prior for '{feat}': min ({lo}) must be < max ({hi}). "
                            "Skipping this prior."
                        )
                        continue
                else:
                    logger.warning(
                        f"Unknown prior type '{ptype}' for feature '{feat}'. Ignored."
                    )
                    continue

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
            return torch.zeros(X_norm.shape[:-1] + (1,), dtype=X_norm.dtype, device=X_norm.device)

        # Unnormalize X to evaluate distributions in original physical space
        # bounds_raw is shape (2, K)
        # X_norm could be (q, K) or (b, q, K), so we use tensor ops
        bounds = self.bounds_raw.to(X_norm.device)
        x_min = bounds[0]
        x_range = bounds[1] - bounds[0]
        X_raw = X_norm * x_range + x_min

        total_score = torch.zeros(X_norm.shape[:-1], dtype=X_norm.dtype, device=X_norm.device)

        for idx, params in self.priors.items():
            feat_val = X_raw[..., idx]
            ptype = params.get("type", "uniform").lower()
            
            if ptype == "gaussian":
                mu = float(params.get("mean", 0.0))
                std = float(params.get("std", 1.0))
                # Log-probability of Gaussian (ignoring constant terms)
                score = -0.5 * ((feat_val - mu) / std) ** 2
                total_score += score
                
            elif ptype == "uniform":
                min_val = float(params.get("min", -np.inf))
                max_val = float(params.get("max", np.inf))
                # Flat score inside, strong penalty outside
                # Using a smooth approximation to keep gradients stable
                penalty = 100.0
                inside = (feat_val >= min_val) & (feat_val <= max_val)
                dist_out = torch.clamp(min_val - feat_val, min=0) + torch.clamp(feat_val - max_val, min=0)
                score = torch.where(inside, torch.zeros_like(feat_val), -penalty * dist_out)
                total_score += score

        return total_score.unsqueeze(-1)
