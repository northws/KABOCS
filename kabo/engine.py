"""
KABOEngine — system-agnostic Bayesian Optimization core.

The engine aggregates the three algorithmic components of the KABO
pipeline:

    * ``SurrogateModel``  — GP surrogate fitting
    * ``PreferenceModel`` — PairwiseGP preference learning (KABO mode)
    * ``ExpertPrior``     — JSON-driven expert prior scoring (KABO mode)

and exposes a compact, domain-agnostic API used by the orchestrator
(``KABOOptimizer``).  No system-specific constants, product names, or
feature schemas appear at this layer; domain information flows in from
the caller (the ``Task`` object sits upstream in the orchestrator).

This is stage 2 of the KABO_Engine + Task generalization plan — see
``docs/KABO_Engine_Task_Feasibility_Report.md`` §4.1.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from kabo.acquisition import (
    build_kabo,
    build_qnei,
    build_ucb,
    evaluate_discrete_candidates,
    evaluate_discrete_thompson,
    optimize_continuous,
)
from kabo.knowledge import ExpertPrior
from kabo.preference import PreferenceModel
from kabo.surrogate import SurrogateModel
from kabo.utils import get_logger

logger = get_logger(__name__)


class KABOEngine:
    """System-agnostic Bayesian Optimization engine.

    The engine owns three stateful algorithmic modules:

    * ``self.surrogate`` — GP surrogate (fit after every observation)
    * ``self.preference_model`` — Bradley-Terry PairwiseGP
    * ``self.expert_prior`` — deterministic expert-prior scorer

    and provides grouped entry points for fitting, acquisition construction,
    candidate suggestion, and preference bookkeeping.  The full BO loop
    (which observation to record, whether to run interactively, etc.)
    remains in the orchestrator.

    Parameters
    ----------
    device : torch.device
        Torch device used for all tensors.
    kernel_type : str
        Surrogate kernel type (``"matern"`` or ``"spectral_mixture"``).
    acq_strategy : str
        Acquisition strategy (``"ucb"`` or ``"qnei"``).
    qnei_mc_samples : int
        Monte-Carlo samples for qNEI.
    n_restarts : int
        Random restarts for ``optimize_acqf``.
    raw_samples : int
        Raw sample count for acquisition initialization.
    kabo_mode : bool
        When True, the acquisition function is wrapped by ``KABOAcquisition``
        to combine preference and expert-prior scores.
    lambda_p, lambda_k, lambda_v : float
        KABO blending weights (only used when ``kabo_mode`` is True).
    """

    def __init__(
        self,
        device: torch.device,
        kernel_type: str = "matern",
        acq_strategy: str = "ucb",
        qnei_mc_samples: int = 128,
        n_restarts: int = 10,
        raw_samples: int = 256,
        kabo_mode: bool = False,
        lambda_p: float = 1.0,
        lambda_k: float = 1.0,
        lambda_v: float = 0.0,
        discrete_strategy: str = "acq",
    ) -> None:
        self.device = device
        self.kernel_type = kernel_type
        self.acq_strategy = acq_strategy
        self.qnei_mc_samples = int(qnei_mc_samples)
        self.n_restarts = int(n_restarts)
        self.raw_samples = int(raw_samples)

        self.kabo_mode = bool(kabo_mode)
        self.lambda_p = float(lambda_p)
        self.lambda_k = float(lambda_k)
        self.lambda_v = float(lambda_v)

        # P3 of discrete variables proposal: how to rank candidate pool.
        # ``"acq"``     — score every candidate with the acquisition function
        #                 (current default; identical to legacy behaviour).
        # ``"thompson"`` — draw top_n independent posterior samples and
        #                 return their argmaxes (natural diversity,
        #                 scales to large dynamic pools).
        self.discrete_strategy = str(discrete_strategy).lower()
        if self.discrete_strategy not in {"acq", "thompson"}:
            raise ValueError(
                f"Unsupported discrete_strategy='{discrete_strategy}'. "
                "Use 'acq' or 'thompson'."
            )

        # Stateful algorithmic components.
        self.surrogate = SurrogateModel(device)
        self.preference_model = PreferenceModel(device)
        self.expert_prior: Optional[ExpertPrior] = None

    # ------------------------------------------------------------------
    #  Convenience proxies (read-only)
    # ------------------------------------------------------------------
    @property
    def bounds_raw(self) -> Optional[torch.Tensor]:
        """Design-space bounds tensor ``(2, K)`` used for un-normalization."""
        return self.surrogate.bounds_raw

    @property
    def y_mean(self) -> float:
        return self.surrogate.y_mean

    @property
    def y_std(self) -> float:
        return self.surrogate.y_std

    @property
    def surrogate_model(self):
        """Underlying BoTorch ``SingleTaskGP`` (or ``None`` before fit)."""
        return self.surrogate.model

    # ------------------------------------------------------------------
    #  Surrogate fitting
    # ------------------------------------------------------------------
    def fit_surrogate(
        self,
        X_raw: np.ndarray,
        Y_raw: np.ndarray,
        selected_features: list[str],
        design_bounds: dict[str, tuple[float, float]],
        feature_types: Optional[dict[str, str]] = None,
    ):
        """Fit / refit the GP surrogate on raw (X, y).

        Normalization and standardization are handled internally by
        ``SurrogateModel.fit`` using the supplied design-space bounds.

        Parameters
        ----------
        feature_types : dict[str, str] or None, optional
            Per-feature type labels from ``TaskBase.feature_types()``.
            When non-empty, categorical dims switch the GP to
            ``MixedSingleTaskGP`` and integer dims enable grid-snap in
            subsequent ``suggest_continuous`` calls.  Default ``None``
            preserves legacy all-continuous behaviour.
        """
        return self.surrogate.fit(
            X_raw,
            Y_raw,
            selected_features,
            design_bounds=design_bounds,
            kernel_type=self.kernel_type,
            feature_types=feature_types,
        )

    # ------------------------------------------------------------------
    #  Expert prior
    # ------------------------------------------------------------------
    def init_expert_prior(
        self,
        config_path: Optional[str | Path],
        selected_features: list[str],
    ) -> ExpertPrior:
        """Instantiate the expert-prior scorer from a JSON config.

        The surrogate must already have been fit (so ``bounds_raw`` is
        available) because prior evaluation requires un-normalization.
        """
        if self.surrogate.bounds_raw is None:
            raise RuntimeError(
                "Surrogate must be fit before the expert prior can be "
                "initialized (bounds_raw is unavailable)."
            )
        self.expert_prior = ExpertPrior(
            config_path=config_path,
            selected_features=selected_features,
            bounds_raw=self.surrogate.bounds_raw,
            device=self.device,
        )
        return self.expert_prior

    # ------------------------------------------------------------------
    #  Preference learning
    # ------------------------------------------------------------------
    def add_preference_pair(
        self,
        winner_norm: torch.Tensor,
        losers_norm: list[torch.Tensor],
    ) -> None:
        """Record a winner-vs-losers preference comparison."""
        self.preference_model.add_comparisons(
            winner_norm=winner_norm, losers_norm=losers_norm,
        )

    def refit_preference(self) -> bool:
        """Refit the PairwiseGP model on the current comparison set."""
        return self.preference_model.fit()

    def generate_pe_queries(
        self,
        candidates_norm: list[torch.Tensor],
        n_queries: int,
    ) -> list[tuple[int, int]]:
        """Delegate to ``PreferenceModel.generate_pe_queries``."""
        return self.preference_model.generate_pe_queries(
            candidates_norm, n_queries=n_queries,
        )

    # ------------------------------------------------------------------
    #  Acquisition construction & optimization
    # ------------------------------------------------------------------
    def build_acquisition(self, beta: float):
        """Build the acquisition function for the current surrogate.

        * Starts from ``build_ucb`` or ``build_qnei`` depending on
          ``self.acq_strategy``.
        * If ``self.kabo_mode`` is True, wraps with ``KABOAcquisition``
          to mix in preference and expert-prior scores.  The preference
          model is (re)fit before wrapping so the PairwiseGP reflects
          any comparisons accumulated this iteration.
        """
        if self.surrogate.model is None:
            raise RuntimeError(
                "Cannot build acquisition before the surrogate is fit."
            )

        if self.acq_strategy == "ucb":
            acq_func = build_ucb(self.surrogate.model, float(beta))
        elif self.acq_strategy == "qnei":
            acq_func = build_qnei(
                self.surrogate.model,
                num_mc_samples=self.qnei_mc_samples,
            )
        else:
            raise ValueError(
                f"Unsupported acq_strategy='{self.acq_strategy}'. "
                "Use 'ucb' or 'qnei'."
            )

        if self.kabo_mode:
            self.preference_model.fit()
            acq_func = build_kabo(
                base_acq_func=acq_func,
                preference_model=self.preference_model,
                expert_prior=self.expert_prior,
                lambda_p=self.lambda_p,
                lambda_k=self.lambda_k,
                lambda_v=self.lambda_v,
            )

        return acq_func

    def suggest_continuous(
        self,
        acq_func,
        dim: int,
    ) -> tuple[torch.Tensor, float]:
        """Optimize the acquisition function over ``[0, 1]^dim``.

        When the surrogate was fit with ``feature_types`` declaring
        integer dims, those dims are snapped to their raw integer grid
        after continuous optimization (round-trick) and the acquisition
        value is recomputed on the snapped point.
        """
        return optimize_continuous(
            acq_func, dim, self.device,
            self.n_restarts, self.raw_samples,
            integer_indices=self.surrogate.integer_indices,
            bounds_raw=self.surrogate.bounds_raw,
        )

    def suggest_continuous_batch(
        self,
        acq_func,
        dim: int,
        q: int,
    ) -> list[tuple[torch.Tensor, float]]:
        """Propose ``q`` diverse continuous candidates in one call.

        Thin wrapper around :func:`kabo.acquisition.optimize_continuous_batch`
        that injects the engine's integer-dim / bounds context.  For
        ``q == 1`` the behaviour is identical to ``suggest_continuous``.

        Parameters
        ----------
        acq_func
            Acquisition function built via :meth:`build_acquisition`.
        dim : int
            Selected-feature dimensionality.
        q : int
            Batch size to request.  Must be >= 1.

        Returns
        -------
        list[tuple[torch.Tensor, float]]
            ``[(candidate_norm, acq_value), ...]`` of length ``q``.
        """
        # Late import to avoid a circular dependency at module load.
        from kabo.acquisition import optimize_continuous_batch

        return optimize_continuous_batch(
            acq_func, dim, q, self.device,
            self.n_restarts, self.raw_samples,
            integer_indices=self.surrogate.integer_indices,
            bounds_raw=self.surrogate.bounds_raw,
        )

    def evaluate_discrete(
        self,
        acq_func,
        candidates_df: pd.DataFrame,
        selected_features: list[str],
        all_feature_columns: list[str],
        design_bounds: dict[str, tuple[float, float]],
    ) -> list[tuple[torch.Tensor, float, int]]:
        """Rank the discrete candidate pool under the configured strategy.

        Dispatches based on ``self.discrete_strategy``:

        * ``"acq"``     — uses the supplied acquisition function (default).
        * ``"thompson"`` — ignores ``acq_func`` and draws posterior samples
          directly from the surrogate (see :func:`evaluate_discrete_thompson`).

        ``all_feature_columns`` and ``design_bounds`` must be supplied by
        the orchestrator (originating from the active ``Task``) so that
        the engine itself carries no domain-specific defaults.
        """
        if self.surrogate.bounds_raw is None:
            raise RuntimeError(
                "Surrogate must be fit before evaluating discrete candidates."
            )
        if self.discrete_strategy == "thompson":
            if self.surrogate.model is None:
                raise RuntimeError(
                    "Surrogate model is None; cannot run Thompson sampling."
                )
            return evaluate_discrete_thompson(
                self.surrogate.model,
                candidates_df, selected_features,
                self.surrogate.bounds_raw, self.device,
                all_feature_columns=all_feature_columns,
                design_bounds=design_bounds,
            )
        # Default: acquisition-function scoring.
        return evaluate_discrete_candidates(
            acq_func, candidates_df, selected_features,
            self.surrogate.bounds_raw, self.device,
            all_feature_columns=all_feature_columns,
            design_bounds=design_bounds,
        )
