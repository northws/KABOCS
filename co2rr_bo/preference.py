"""
Phase 2 Knowledge Encoding: Preference Learning.

This module provides a mechanism to learn an expert's preference from their
candidate selections (Human-in-the-Loop choices) using a Pairwise Gaussian Process
(Bradley-Terry model).

If preferences are too sparse or fitting fails, it falls back to a ZeroPreference
baseline.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models.pairwise_gp import PairwiseGP, PairwiseLaplaceMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel

logger = logging.getLogger(__name__)


class PreferenceModel:
    """Encapsulates expert preference modeling via PairwiseGP.

    Extracts comparisons from historical iterations and fits a Bradley-Terry style
    Gaussian Process using BoTorch's PairwiseGP.

    Provides a fallback mechanism if data is too sparse or fitting fails.
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.model: Optional[PairwiseGP] = None
        self.has_valid_model = False
        
        # Internal storage for normalized feature vectors and their comparisons
        self._datapoints: list[torch.Tensor] = []
        self._comparisons: list[Tuple[int, int]] = []  # (winner_idx, loser_idx)

    def _find_or_add(self, point: torch.Tensor, eps: float = 1e-4) -> int:
        """Return the index of a matching existing datapoint, or add a new one.

        Two points are considered duplicates if their L2 distance is < eps.
        This prevents the comparison graph from growing with redundant nodes.
        """
        pt = point.clone().detach().cpu()
        for idx, existing in enumerate(self._datapoints):
            if torch.norm(pt - existing).item() < eps:
                return idx
        new_idx = len(self._datapoints)
        self._datapoints.append(pt)
        return new_idx

    def add_comparisons(
        self,
        winner_norm: torch.Tensor,
        losers_norm: list[torch.Tensor]
    ) -> None:
        """Add pairwise comparisons based on a human choice.

        Datapoints are deduplicated: if a candidate is spatially close
        (L2 < 1e-4) to an existing node in the comparison graph, the
        existing index is reused rather than appending a duplicate.

        Parameters
        ----------
        winner_norm : torch.Tensor
            Normalized features of the chosen candidate, shape (K,).
        losers_norm : list[torch.Tensor]
            List of normalized features of the rejected candidates.
        """
        if not losers_norm:
            return

        winner_idx = self._find_or_add(winner_norm)

        for loser in losers_norm:
            loser_idx = self._find_or_add(loser)
            if winner_idx == loser_idx:
                continue  # skip self-comparison (same point)
            self._comparisons.append((winner_idx, loser_idx))

    def fit(self) -> bool:
        """Fit the PairwiseGP model given the accumulated comparisons.

        Returns
        -------
        bool
            True if model was successfully fitted, False if reverted to fallback.
        """
        self.has_valid_model = False
        self.model = None

        if len(self._comparisons) < 1:
            logger.info("Not enough preference data to fit PairwiseGP.")
            return False

        try:
            # Prepare tensors
            X = torch.stack(self._datapoints).to(torch.double).to(self.device)
            # Add jitter to avoid non-positive definite issues with duplicate points
            X += torch.randn_like(X) * 1e-6
            
            comp_tensor = torch.tensor(
                self._comparisons, dtype=torch.long, device=self.device
            )

            # PairwiseGP naturally uses a ScaleKernel(MaternKernel) under the hood 
            # if we don't specify covar_module, but let's be explicit
            covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=X.shape[-1]))
            
            self.model = PairwiseGP(
                datapoints=X,
                comparisons=comp_tensor,
                covar_module=covar_module,
            ).to(self.device)

            mll = PairwiseLaplaceMarginalLogLikelihood(
                self.model.likelihood, self.model
            )

            # Fit
            logger.info(f"Fitting PairwiseGP with {len(self._comparisons)} comparisons...")
            fit_gpytorch_mll(mll)
            
            self.has_valid_model = True
            logger.info("PairwiseGP fit successfully.")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to fit PairwiseGP (fallback to zero preference): {e}")
            self.has_valid_model = False
            self.model = None
            return False

    def evaluate(
        self,
        X: torch.Tensor,
        n_mc_samples: int = 16,
    ) -> torch.Tensor:
        """Evaluate the latent preference score for new normalized candidates.

        Following Astudillo & Frazier 2019 (EI-UU), we integrate over the
        posterior uncertainty of the preference function rather than using
        a point estimate.  Specifically, we draw ``n_mc_samples`` samples
        from the PairwiseGP posterior and return the sample mean as the
        preference score.

        Parameters
        ----------
        X : torch.Tensor
            Normalized candidate features, shape (..., K).
        n_mc_samples : int, optional
            Number of Monte Carlo samples from the preference posterior
            (default 16).  Higher values reduce variance but increase cost.

        Returns
        -------
        torch.Tensor
            Preference scores, shape (..., 1). Returns 0.0 if fallback.
        """
        if not self.has_valid_model or self.model is None:
            return torch.zeros(X.shape[:-1] + (1,), dtype=X.dtype, device=X.device)

        with torch.no_grad():
            posterior = self.model.posterior(X)
            # MC integration over utility uncertainty (cf. EI-UU)
            samples = posterior.rsample(torch.Size([n_mc_samples]))  # (S, ..., 1)
            pref_score = samples.mean(dim=0)  # (..., 1)
        return pref_score

    # ------------------------------------------------------------------
    # Preference Exploration (PE) query generation — cf. Lin et al. 2022
    # ------------------------------------------------------------------
    def generate_pe_queries(
        self,
        candidates_norm: list[torch.Tensor],
        n_queries: int = 1,
    ) -> list[tuple[int, int]]:
        """Generate informative pairwise queries for Preference Exploration.

        Implements the *uncertainty* strategy from PEBO (Lin et al. 2022):
        select candidate pairs where the preference model is most uncertain
        (highest posterior variance) and the predicted preference difference
        is smallest (hardest to distinguish).

        Parameters
        ----------
        candidates_norm : list[torch.Tensor]
            Pool of normalised candidate vectors.
        n_queries : int
            Number of pairwise queries to generate.

        Returns
        -------
        list[tuple[int, int]]
            List of (idx_a, idx_b) pairs into ``candidates_norm``.
        """
        n = len(candidates_norm)
        if n < 2 or n_queries <= 0:
            return []

        if not self.has_valid_model or self.model is None:
            # No preference model yet — pick random pairs for cold start
            rng = np.random.default_rng()
            pairs = []
            for _ in range(min(n_queries, n * (n - 1) // 2)):
                a, b = rng.choice(n, size=2, replace=False)
                pairs.append((int(a), int(b)))
            return pairs

        # Score all candidates: higher variance = more informative
        X = torch.stack(candidates_norm).to(torch.double).to(self.device)
        with torch.no_grad():
            posterior = self.model.posterior(X)
            means = posterior.mean.squeeze(-1)      # (n,)
            variances = posterior.variance.squeeze(-1)  # (n,)

        # For each pair (i, j), compute an information score:
        #   info(i,j) = var(i) + var(j)  (high uncertainty)
        #             - |mean(i) - mean(j)|  (close predictions = harder)
        scored_pairs: list[tuple[float, int, int]] = []
        for i in range(n):
            for j in range(i + 1, n):
                info = (
                    variances[i].item() + variances[j].item()
                    - abs(means[i].item() - means[j].item())
                )
                scored_pairs.append((info, i, j))

        # Sort descending by info score, take top-n_queries
        scored_pairs.sort(key=lambda x: x[0], reverse=True)
        return [(p[1], p[2]) for p in scored_pairs[:n_queries]]
