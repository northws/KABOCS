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
        max_pool_size: Optional[int] = None,
        strategy: str = "uncertainty",
        random_state: Optional[int] = None,
    ) -> list[tuple[int, int]]:
        """Generate informative pairwise queries for Preference Exploration.

        Implements the *uncertainty* strategy from PEBO (Lin et al. 2022):
        select candidate pairs where the preference model is most uncertain
        (highest posterior variance) and the predicted preference difference
        is smallest (hardest to distinguish).

        v1.2 improvements
        -----------------
        * **Vectorised pair scoring**: the per-pair info score is computed
          via broadcasting on the full ``n x n`` matrix and the top-k
          upper-triangle entries are selected with ``topk``, replacing the
          previous Python double loop (O(n²) Python → O(n²) tensor + O(n·log n)
          selection).  For a pool of 2000 candidates this is > 100x faster.
        * **Candidate pool cap** (``max_pool_size``): when the pool is larger
          than the cap, a uniform random subsample of size ``max_pool_size``
          is used for pair scoring.  Returned indices are remapped back
          into the **original** pool order so callers can look up the same
          candidates they passed in.
        * **Strategy selector**: ``"uncertainty"`` (default) for the
          PEBO score; ``"random"`` returns uniformly sampled distinct
          pairs (useful for ablations and cold-start parity tests).

        Parameters
        ----------
        candidates_norm : list[torch.Tensor]
            Pool of normalised candidate vectors.
        n_queries : int, optional
            Number of pairwise queries to generate (default 1).
        max_pool_size : int or None, optional
            If set, cap the candidate pool at this many points via
            uniform subsampling.  Returned indices are still into
            ``candidates_norm``.
        strategy : {"uncertainty", "random"}, optional
            Query scoring strategy (default ``"uncertainty"``).  The
            random strategy always runs regardless of model validity
            and is deterministic when ``random_state`` is set.
        random_state : int or None, optional
            Seed for both pool subsampling and the random strategy.

        Returns
        -------
        list[tuple[int, int]]
            List of ``(idx_a, idx_b)`` pairs into ``candidates_norm``.
        """
        n = len(candidates_norm)
        if n < 2 or n_queries <= 0:
            return []
        if strategy not in {"uncertainty", "random"}:
            raise ValueError(
                f"Unknown PE strategy '{strategy}'. "
                "Use 'uncertainty' or 'random'."
            )

        rng = np.random.default_rng(random_state)

        # ---- (1) optional pool cap ----
        if max_pool_size is not None and max_pool_size < n:
            pool_idx = rng.choice(n, size=int(max_pool_size), replace=False)
            pool_idx = np.sort(pool_idx)  # deterministic order for pairs
            logger.info(
                "PE query pool capped: %d → %d candidates (random subsample).",
                n, len(pool_idx),
            )
        else:
            pool_idx = np.arange(n)
        m = len(pool_idx)
        if m < 2:
            return []

        # ---- (2) fall back to random when the model is not usable ----
        if (
            strategy == "random"
            or not self.has_valid_model
            or self.model is None
        ):
            max_pairs = m * (m - 1) // 2
            k = min(n_queries, max_pairs)
            # Sample distinct pairs without replacement — use a flat
            # upper-triangular indexing trick: encode pair → linear id.
            flat = rng.choice(max_pairs, size=k, replace=False)
            pairs: list[tuple[int, int]] = []
            # Convert linear id → (row, col) in upper triangle of mxm.
            # Using the closed-form triangle inversion keeps this O(k).
            for fid in flat:
                # i = largest such that T(i) <= fid, where T(i) = i*(2m-i-1)/2
                # Easier & still O(1) in practice: iterate i until cumulative
                # count exceeds fid.  m is small (bounded by max_pool_size).
                remaining = int(fid)
                i = 0
                while True:
                    row_width = m - 1 - i
                    if remaining < row_width:
                        j = i + 1 + remaining
                        break
                    remaining -= row_width
                    i += 1
                pairs.append((int(pool_idx[i]), int(pool_idx[j])))
            return pairs

        # ---- (3) Uncertainty scoring — vectorised ----
        X = torch.stack([candidates_norm[int(i)] for i in pool_idx])
        X = X.to(torch.double).to(self.device)
        with torch.no_grad():
            posterior = self.model.posterior(X)
            means = posterior.mean.squeeze(-1)          # (m,)
            variances = posterior.variance.squeeze(-1)  # (m,)

        # info(i,j) = var_i + var_j − |mean_i − mean_j|, vectorised.
        var_sum = variances.unsqueeze(0) + variances.unsqueeze(1)     # (m, m)
        mean_abs_diff = (means.unsqueeze(0) - means.unsqueeze(1)).abs()  # (m, m)
        info = var_sum - mean_abs_diff                               # (m, m)

        # Keep only the strict upper triangle (i < j).
        iu, ju = torch.triu_indices(m, m, offset=1, device=info.device)
        info_flat = info[iu, ju]                                     # (m*(m-1)/2,)

        k = min(int(n_queries), info_flat.numel())
        if k <= 0:
            return []
        top = torch.topk(info_flat, k=k, largest=True).indices
        top_i = iu[top].cpu().tolist()
        top_j = ju[top].cpu().tolist()
        return [(int(pool_idx[a]), int(pool_idx[b])) for a, b in zip(top_i, top_j)]
