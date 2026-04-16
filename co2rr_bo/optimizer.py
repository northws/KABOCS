"""
CO2RROptimizer — Main Pipeline Orchestrator.

Coordinates the three-phase Bayesian Optimization workflow:
    Phase 1: Feature selection (Random Forest) — **optional**
    Phase 2: GP surrogate fitting (BoTorch SingleTaskGP + ARD Matérn)
    Phase 3: UCB acquisition + human-in-the-loop optimization

Supports multi-product CO2RR yield optimization — the user selects
which product to optimize (CO, HCOOH, CH₄, C₂H₄, CH₃OH, C₂H₅OH, H₂),
and the human-in-the-loop interaction collects yields for all products.

Key design choices following REVIEW_REPORT corrections:
- Normalization uses explicit design-space bounds, not training data min/max.
- Discrete candidates must contain all selected features (no silent padding).
- New observations preserve all original feature values.
- Feature selection (RF) is an optional engineering heuristic.

Usage::

    from co2rr_bo import CO2RROptimizer

    optimizer = CO2RROptimizer(
        data_path="data/data.csv",
        target_product="CO",
    )
    optimizer.run(n_iterations=10, interactive=True)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from datetime import datetime

import numpy as np
import pandas as pd
import torch

from co2rr_bo.acquisition import (
    build_kabo,
    build_qnei,
    build_ucb,
    evaluate_discrete_candidates,
    load_discrete_candidates,
    optimize_continuous,
    print_best_found,
    print_recommendations,
    prompt_user_candidate_choice,
    prompt_user_input_multiproduct,
    prompt_user_manual_candidate,
    prompt_user_nonselected_features,
    simulate_multiproduct_yields,
)
from co2rr_bo.candidate import CandidateRecord
from co2rr_bo.constants import (
    ALL_FEATURE_COLUMNS,
    ALL_PRODUCT_COLUMNS,
    DEFAULT_TARGET_PRODUCT,
    DESIGN_SPACE_BOUNDS,
    PRODUCT_COLUMNS,
    PRODUCT_NAMES,
    TARGET_COLUMN,
)
from co2rr_bo.feature_selection import (
    load_and_validate_data,
    plot_feature_importances,
    select_top_k_features,
    train_random_forest,
)
from co2rr_bo.surrogate import SurrogateModel
from co2rr_bo.utils import (
    get_logger,
    select_device,
    set_global_seed,
    unnormalize_x,
)
from co2rr_bo.preference import PreferenceModel
from co2rr_bo.knowledge import ExpertPrior

logger = get_logger(__name__)


class CO2RROptimizer:
    """Bayesian Optimization pipeline for photocatalytic CO2RR.

    This class orchestrates the full three-phase workflow:
        1. Feature importance evaluation & top-K selection via Random Forest
           (**optional** — can be skipped with ``skip_feature_selection=True``).
        2. GP surrogate model setup using BoTorch (SingleTaskGP + ARD Matérn).
        3. UCB acquisition function optimization with human-in-the-loop.

    The pipeline supports multi-product CO2RR yield optimization.
    Products include: CO, HCOOH, CH₄, C₂H₄, CH₃OH, C₂H₅OH, H₂.
    The user selects which product to maximize via ``target_product``.

    The methodology follows the GP-UCB algorithm (Algorithm 2) and
    Human-in-the-Loop BO workflow (Algorithm 3) described in
    arXiv:2604.01328v3.

    Parameters
    ----------
    data_path : str or Path
        Path to the CSV dataset with 19 descriptors + product yield columns.
    target_product : str, optional
        Product to optimize. Must be one of:
        ``"CO"``, ``"HCOOH"``, ``"CH4"``, ``"C2H4"``, ``"CH3OH"``,
        ``"C2H5OH"``, ``"H2"`` (default ``"CO"``).
    top_k : int, optional
        Number of top features to select (default 10).
        Ignored when ``skip_feature_selection=True``.
    beta : float, optional
        UCB exploration parameter β (default 2.0).
        Under ``beta_schedule='fixed'``, this is the per-iteration β.
        Under ``beta_schedule='theory'``, this is a scale factor applied
        to the theoretical β_t sequence.
        Under ``beta_schedule='theory_strict'``, this value is ignored.
    beta_schedule : str, optional
        Beta schedule strategy: ``"fixed"``, ``"theory"`` or
        ``"theory_strict"``
        (default ``"fixed"``).
    beta_delta : float, optional
        Confidence parameter δ for theoretical β_t schedule
        (default 0.1).
    candidates_path : str or Path or None, optional
        Path to a CSV of discrete candidate vectors.
    n_restarts : int, optional
        Number of random restarts for ``optimize_acqf`` (default 10).
    raw_samples : int, optional
        Number of raw samples for acquisition initialization (default 256).
    rf_n_estimators : int, optional
        Number of trees in the Random Forest (default 200).
    skip_feature_selection : bool, optional
        If True, skip Phase 1 and use all 19 features directly (default False).
    strict_training_schema : bool, optional
        If True, require all 19 descriptor columns in training data
        before optimization starts (default False).
    pre_fill_before_choice : bool, optional
        If True, ask the expert to pre-fill non-selected feature values
        for the continuous candidate before candidate ranking/selection
        display (default False).
    acq_strategy : str, optional
        Acquisition strategy. ``"ucb"`` uses analytic UCB;
        ``"qnei"`` uses Monte Carlo qNoisyExpectedImprovement
        (default ``"ucb"``).
    qnei_mc_samples : int, optional
        Number of QMC samples for qNEI (default 128).
    kernel_type : str, optional
        Surrogate kernel type: ``"matern"`` or ``"spectral_mixture"``
        (default ``"matern"``).
    h2_penalty_weight : float, optional
        If > 0, optimize a composite target
        ``Y_target - h2_penalty_weight * Y_H2`` to discourage HER-dominant
        conditions (default 0.0, disabled).
    seed : int or None, optional
        Global random seed for NumPy/Torch/Python to improve reproducibility
        (default None).
    device : str, optional
        Torch device string — ``"auto"`` selects CUDA if available.
    output_dir : str or Path, optional
        Directory for output files (default ``"output"``).
    """

    def __init__(
        self,
        data_path: str | Path,
        target_product: str = DEFAULT_TARGET_PRODUCT,
        top_k: int = 10,
        beta: float = 2.0,
        beta_schedule: str = "fixed",
        beta_delta: float = 0.1,
        acq_strategy: str = "ucb",
        qnei_mc_samples: int = 128,
        kernel_type: str = "matern",
        h2_penalty_weight: float = 0.0,
        candidates_path: Optional[str | Path] = None,
        n_restarts: int = 10,
        raw_samples: int = 256,
        rf_n_estimators: int = 200,
        skip_feature_selection: bool = False,
        strict_training_schema: bool = False,
        pre_fill_before_choice: bool = False,
        seed: Optional[int] = None,
        device: str = "auto",
        output_dir: str | Path = "output",
        kabo_mode: bool = False,
        lambda_p: float = 1.0,
        lambda_k: float = 1.0,
        expert_prior_file: Optional[str | Path] = None,
        diversity_weight: float = 0.5,
    ) -> None:
        self.data_path = Path(data_path)
        self.top_k = top_k
        self.beta = beta
        self.beta_schedule = beta_schedule.lower().replace("-", "_")
        self.beta_delta = beta_delta
        self.acq_strategy = acq_strategy.lower()
        self.qnei_mc_samples = int(qnei_mc_samples)
        self.kernel_type = kernel_type.lower()
        self.h2_penalty_weight = float(h2_penalty_weight)
        self.candidates_path = (
            Path(candidates_path) if candidates_path else None
        )
        self.n_restarts = n_restarts
        self.raw_samples = raw_samples
        self.rf_n_estimators = rf_n_estimators
        self.skip_feature_selection = skip_feature_selection
        self.strict_training_schema = strict_training_schema
        self.pre_fill_before_choice = pre_fill_before_choice
        self.seed = seed
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.kabo_mode = kabo_mode
        self.lambda_p = lambda_p
        self.lambda_k = lambda_k
        self.expert_prior_file = expert_prior_file
        self.diversity_weight = diversity_weight
        
        if self.kabo_mode:
            logger.info(
                f"KABO Mode ENABLED: lambda_p={self.lambda_p}, lambda_k={self.lambda_k}"
            )

        if self.beta_schedule not in {"fixed", "theory", "theory_strict"}:
            raise ValueError(
                f"Unsupported beta_schedule='{beta_schedule}'. "
                "Use 'fixed', 'theory', or 'theory_strict'."
            )
        if not (0.0 < self.beta_delta < 1.0):
            raise ValueError("beta_delta must be in (0, 1).")
        if self.acq_strategy not in {"ucb", "qnei"}:
            raise ValueError(
                f"Unsupported acq_strategy='{acq_strategy}'. "
                "Use 'ucb' or 'qnei'."
            )
        if self.qnei_mc_samples <= 0:
            raise ValueError("qnei_mc_samples must be a positive integer.")
        if self.kernel_type not in {"matern", "spectral_mixture"}:
            raise ValueError(
                f"Unsupported kernel_type='{kernel_type}'. "
                "Use 'matern' or 'spectral_mixture'."
            )
        if self.h2_penalty_weight < 0:
            raise ValueError("h2_penalty_weight must be >= 0.")
        if not (np.isfinite(self.diversity_weight) and self.diversity_weight >= 0):
            raise ValueError(
                f"diversity_weight must be a finite non-negative number "
                f"(got {self.diversity_weight})."
            )

        if self.seed is not None:
            set_global_seed(self.seed)
            logger.info("Global random seed set to: %d", self.seed)

        self.device = select_device(device)
        logger.info("Using device: %s", self.device)

        # Resolve target product to column name
        self.target_product = target_product.upper()
        if self.target_product in PRODUCT_COLUMNS:
            self.target_column = PRODUCT_COLUMNS[self.target_product]
        elif target_product in ALL_PRODUCT_COLUMNS:
            self.target_column = target_product
        else:
            self.target_column = TARGET_COLUMN
            logger.warning(
                "Product '%s' not recognized. "
                "Falling back to legacy target column '%s'. "
                "Valid products: %s",
                target_product, TARGET_COLUMN,
                ", ".join(PRODUCT_COLUMNS.keys()),
            )

        target_name = PRODUCT_NAMES.get(self.target_column, self.target_column)
        logger.info("Target product: %s (%s)", target_name, self.target_column)

        if self.skip_feature_selection:
            logger.info(
                "Feature selection: SKIPPED (paper-minimal BO mode, using all features)"
            )
        else:
            logger.info(
                "Feature selection: RF top-%d (engineering-enhanced default mode)",
                self.top_k,
            )
        logger.info(
            "Training schema: %s",
            "STRICT (require all 19 descriptors)"
            if self.strict_training_schema
            else "FLEXIBLE (allow descriptor subset)",
        )
        logger.info(
            "Continuous recipe pre-fill before choice: %s",
            "ENABLED" if self.pre_fill_before_choice else "DISABLED",
        )
        logger.info(
            "Beta schedule: %s (beta=%.4f, delta=%.4f)",
            self.beta_schedule,
            self.beta,
            self.beta_delta,
        )
        logger.info(
            "Acquisition strategy: %s%s",
            self.acq_strategy,
            (
                f" (mc_samples={self.qnei_mc_samples})"
                if self.acq_strategy == "qnei"
                else ""
            ),
        )
        logger.info("Surrogate kernel: %s", self.kernel_type)
        if self.h2_penalty_weight > 0:
            logger.info(
                "Composite objective enabled: %s - %.4f * Y_H2",
                self.target_column,
                self.h2_penalty_weight,
            )

        # State (populated by phases)
        self.df: pd.DataFrame = pd.DataFrame()
        self.selected_features: list[str] = []
        self.feature_importances: pd.Series = pd.Series(dtype=float)
        self.surrogate = SurrogateModel(self.device)
        self.preference_model = PreferenceModel(self.device)
        self.expert_prior: Optional[ExpertPrior] = None

        # Design-space bounds (can be customized)
        self.design_bounds: dict[str, tuple[float, float]] = dict(DESIGN_SPACE_BOUNDS)

        # Discrete candidates df (cached after first load for row lookups)
        self._discrete_candidates_df: Optional[pd.DataFrame] = None
        self._beta_trace: list[float] = []

    def _compute_beta_t(self, iteration: int, dim: int) -> float:
        """Compute effective UCB beta for a BO iteration.

        Parameters
        ----------
        iteration : int
            1-based BO iteration index.
        dim : int
            Optimization dimensionality (selected feature count).

        Returns
        -------
        float
            Effective beta value used by UCB in this iteration.
        """
        if self.beta_schedule == "fixed":
            return float(self.beta)

        t = max(1, int(iteration))
        d = max(1, int(dim))
        delta = min(max(float(self.beta_delta), 1e-12), 1.0 - 1e-12)

        # A common GP-UCB theoretical schedule variant.
        theory_beta = 2.0 * np.log(
            (t ** (d / 2.0 + 2.0)) * (np.pi ** 2) / (3.0 * delta)
        )
        theory_beta = max(theory_beta, 1e-12)
        if self.beta_schedule == "theory_strict":
            return float(theory_beta)
        return float(self.beta * theory_beta)

    # ===================================================================
    #  PHASE 1
    # ===================================================================
    def phase1_feature_selection(self) -> list[str]:
        """Load data and optionally run Random Forest feature selection.

        If ``skip_feature_selection`` is True, all available features
        are used directly without RF training.

        Returns
        -------
        list[str]
            Names of the selected features.
        """
        logger.info("=" * 60)
        logger.info("PHASE 1: Feature Weight Evaluation & Selection")
        logger.info("=" * 60)

        # Auto-detect legacy single-target datasets: if requested target
        # (e.g. Y_CO) is absent but legacy Y exists, switch automatically.
        if self.data_path.exists():
            header_cols = set(pd.read_csv(self.data_path, nrows=0).columns)
            if self.target_column not in header_cols and TARGET_COLUMN in header_cols:
                prev_target = self.target_column
                self.target_column = TARGET_COLUMN
                logger.warning(
                    "Target column '%s' not found in '%s'. Auto-switched to "
                    "legacy target column '%s'.",
                    prev_target, self.data_path.name, TARGET_COLUMN,
                )

        self.df = load_and_validate_data(
            self.data_path,
            self.target_column,
            strict_feature_schema=self.strict_training_schema,
        )

        available_features = [c for c in ALL_FEATURE_COLUMNS if c in self.df.columns]

        if self.skip_feature_selection:
            # P1-1: Skip RF, use all available features
            self.selected_features = available_features
            logger.info(
                "Feature selection SKIPPED. Using all %d available features.",
                len(self.selected_features),
            )
            for i, feat in enumerate(self.selected_features, 1):
                logger.info("  [%2d] %s", i, feat)
        else:
            _, importances, _ = train_random_forest(
                self.df,
                target_column=self.target_column,
                n_estimators=self.rf_n_estimators,
                random_state=self.seed if self.seed is not None else 42,
            )
            self.feature_importances = importances

            plot_feature_importances(
                importances, self.top_k, self.output_dir,
                target_column=self.target_column,
            )

            self.selected_features = select_top_k_features(
                importances, self.top_k
            )

        return self.selected_features

    # ===================================================================
    #  PHASE 2
    # ===================================================================
    def phase2_fit_surrogate(self) -> None:
        """Build and fit the GP surrogate model on the target product.

        Uses explicit design-space bounds for normalization.

        Raises
        ------
        RuntimeError
            If Phase 1 has not been run.
        """
        logger.info("=" * 60)
        logger.info("PHASE 2: BoTorch Surrogate Model Setup")
        logger.info("=" * 60)

        if not self.selected_features:
            raise RuntimeError("Phase 1 must be run before Phase 2.")

        X_raw = self.df[self.selected_features].values.astype(np.float64)
        Y_raw = self._build_training_target(self.df)

        self.surrogate.fit(
            X_raw, Y_raw, self.selected_features,
            design_bounds=self.design_bounds,
            kernel_type=self.kernel_type,
        )
        logger.info("Phase 2 complete.")

    def _build_training_target(self, df: pd.DataFrame) -> np.ndarray:
        """Build the surrogate training target from raw product columns.

        Returns either the original target product or a selectivity-aware
        composite objective that penalizes hydrogen evolution.
        """
        y_target = df[self.target_column].values.astype(np.float64)
        if self.h2_penalty_weight <= 0.0:
            return y_target

        if "Y_H2" not in df.columns:
            logger.warning(
                "h2_penalty_weight > 0 but 'Y_H2' column is missing. "
                "Falling back to single-target objective."
            )
            return y_target

        y_h2 = df["Y_H2"].values.astype(np.float64)
        y_composite = y_target - self.h2_penalty_weight * y_h2
        logger.info(
            "Using composite objective: %s - %.4f * Y_H2",
            self.target_column,
            self.h2_penalty_weight,
        )
        return y_composite

    # ===================================================================
    #  PHASE 3
    # ===================================================================
    def phase3_optimize(
        self,
        n_iterations: int = 10,
        top_n_recommend: int = 3,
        interactive: bool = True,
    ) -> pd.DataFrame:
        """Run the UCB-based optimization loop with human-in-the-loop.

        In interactive mode, the user is prompted to enter yields for
        ALL CO2RR products (CO, HCOOH, CH₄, C₂H₄, CH₃OH, C₂H₅OH, H₂).
        The GP model is re-fitted on the selected target product's yield.

        Parameters
        ----------
        n_iterations : int, optional
            Maximum number of BO iterations (default 10).
        top_n_recommend : int, optional
            Number of top candidates to recommend (default 3).
        interactive : bool, optional
            If ``True``, prompt user for results via CLI.
            If ``False``, simulate with random yields.

        Returns
        -------
        pd.DataFrame
            Updated dataset with all product yields.
        """
        target_name = PRODUCT_NAMES.get(self.target_column, self.target_column)

        logger.info("=" * 60)
        logger.info("PHASE 3: Acquisition & Human-in-the-Loop Optimization")
        logger.info("=" * 60)
        logger.info(
            "Target: %s | acq=%s | beta_schedule=%s | beta=%.4f | iterations=%d",
            target_name,
            self.acq_strategy,
            self.beta_schedule,
            self.beta,
            n_iterations,
        )

        if self.surrogate.model is None:
            raise RuntimeError("Phase 2 must be run before Phase 3.")

        if self.kabo_mode:
            self.expert_prior = ExpertPrior(
                config_path=self.expert_prior_file,
                selected_features=self.selected_features,
                bounds_raw=self.surrogate.bounds_raw,
                device=self.device,
            )

        self._discrete_candidates_df = load_discrete_candidates(
            self.candidates_path, self.selected_features
        )

        K = len(self.selected_features)
        self._beta_trace = []

        for iteration in range(1, n_iterations + 1):
            logger.info("-" * 50)
            logger.info("BO Iteration %d / %d", iteration, n_iterations)
            logger.info("-" * 50)

            # 1. Build acquisition function
            if self.acq_strategy == "ucb":
                beta_t = self._compute_beta_t(iteration, K)
                self._beta_trace.append(beta_t)
                logger.info("Using UCB with β_t = %.6f", beta_t)
                acq_func = build_ucb(self.surrogate.model, beta_t)
            else:
                logger.info(
                    "Using qNEI (Monte Carlo, mc_samples=%d)",
                    self.qnei_mc_samples,
                )
                acq_func = build_qnei(
                    self.surrogate.model,
                    num_mc_samples=self.qnei_mc_samples,
                )

            # 1.5 Wrap with KABO if enabled
            if self.kabo_mode:
                self.preference_model.fit()
                acq_func = build_kabo(
                    base_acq_func=acq_func,
                    preference_model=self.preference_model,
                    expert_prior=self.expert_prior,
                    lambda_p=self.lambda_p,
                    lambda_k=self.lambda_k,
                )

            # 2. Optimize over continuous [0,1]^K
            cont_cand, cont_val = optimize_continuous(
                acq_func, K, self.device,
                self.n_restarts, self.raw_samples,
            )

            # 3. Collect all candidates
            all_candidates: list[torch.Tensor] = [cont_cand]
            all_acq_values: list[float] = [cont_val]
            all_sources: list[str] = ["continuous"]
            all_orig_rows: list[int] = [-1]  # -1 = no original row

            if self._discrete_candidates_df is not None:
                disc_results = evaluate_discrete_candidates(
                    acq_func, self._discrete_candidates_df,
                    self.selected_features,
                    self.surrogate.bounds_raw,
                    self.device,
                )
                for i, (cand, val, orig_idx) in enumerate(disc_results):
                    all_candidates.append(cand)
                    all_acq_values.append(val)
                    all_sources.append(f"discrete_{i}")
                    all_orig_rows.append(orig_idx)

            pre_choice_prefills: dict[str, float] = {}
            pre_choice_oob_count = 0
            pre_choice_overrides: list[str] = []
            if interactive and self.pre_fill_before_choice:
                nonselected_pre = [
                    f for f in ALL_FEATURE_COLUMNS
                    if f not in self.selected_features
                    and f in self.df.columns
                ]
                if nonselected_pre:
                    logger.info(
                        "Pre-fill mode enabled: collecting continuous "
                        "non-selected feature values before candidate choice."
                    )
                    (
                        pre_choice_prefills,
                        pre_choice_oob_count,
                        pre_choice_overrides,
                    ) = prompt_user_nonselected_features(
                        nonselected_pre,
                        self.design_bounds,
                    )

            # 4. Print recommendations
            top_indices = print_recommendations(
                all_candidates, all_acq_values, all_sources,
                all_orig_rows, self._discrete_candidates_df,
                self.selected_features, ALL_FEATURE_COLUMNS,
                self.surrogate.bounds_raw,
                iteration, self.target_column, top_n_recommend,
                continuous_nonselected_values=(
                    pre_choice_prefills
                    if interactive and self.pre_fill_before_choice
                    else None
                ),
                diversity_weight=self.diversity_weight,
            )

            # 5. Human-in-the-Loop: expert selects candidate
            continuous_prefills: dict[str, float] = {}
            manual_raw_vals: Optional[dict[str, float]] = None
            overrides: list[str] = []
            oob_count = 0
            if interactive:
                # Expert chooses which candidate to execute
                chosen_idx = prompt_user_candidate_choice(
                    top_indices, len(all_candidates)
                )
                if chosen_idx is None:
                    logger.info("User requested exit.")
                    break

                if chosen_idx == -1:
                    manual_payload = prompt_user_manual_candidate(
                        ALL_FEATURE_COLUMNS,
                        self.design_bounds,
                    )
                    if manual_payload is None:
                        logger.info("User requested exit during manual override.")
                        break
                    manual_raw_vals, oob_count, overrides = manual_payload

                # Only prompt non-selected features when the chosen candidate
                # is continuous. This avoids unnecessary interaction overhead.
                if chosen_idx >= 0 and all_sources[chosen_idx] == "continuous":
                    if self.pre_fill_before_choice:
                        continuous_prefills = dict(pre_choice_prefills)
                        oob_count = pre_choice_oob_count
                        overrides = list(pre_choice_overrides)
                    else:
                        nonselected = [
                            f for f in ALL_FEATURE_COLUMNS
                            if f not in self.selected_features
                            and f in self.df.columns
                        ]
                        if nonselected:
                            (
                                continuous_prefills,
                                oob_count,
                                overrides,
                            ) = prompt_user_nonselected_features(
                                nonselected, self.design_bounds
                            )

                # Collect product yields
                product_yields = prompt_user_input_multiproduct(
                    self.target_column
                )
                if product_yields is None:
                    logger.info("User requested exit.")
                    break
            else:
                # Demo mode: auto-select rank #1
                chosen_idx = top_indices[0]
                product_yields = simulate_multiproduct_yields(
                    self.target_column,
                    self.surrogate.y_mean,
                    self.surrogate.y_std,
                )
                logger.info("[Demo mode] Simulated yields:")
                for col, val in product_yields.items():
                    name = PRODUCT_NAMES.get(col, col)
                    logger.info("  %s = %.2f", name, val)

            # 6. Build CandidateRecord and append observation
            raw_vals: dict[str, float] = {}
            norm_vals: dict[str, float] = {}

            if manual_raw_vals is not None:
                chosen_source = "manual_override"
                chosen_orig_row = -1
                chosen_acq_value = float("nan")
                raw_vals = dict(manual_raw_vals)
                for f in self.selected_features:
                    lo, hi = self.design_bounds.get(f, (0.0, 1.0))
                    denom = (hi - lo) if hi != lo else 1.0
                    norm_vals[f] = float((raw_vals[f] - lo) / denom)
            else:
                chosen_cand_norm = all_candidates[chosen_idx]
                chosen_orig_row = all_orig_rows[chosen_idx]
                chosen_source = all_sources[chosen_idx]
                chosen_acq_value = all_acq_values[chosen_idx]

                cand_raw_np = unnormalize_x(chosen_cand_norm, self.surrogate.bounds_raw)
                if chosen_orig_row >= 0 and self._discrete_candidates_df is not None:
                    orig_row_s = self._discrete_candidates_df.iloc[chosen_orig_row]
                    for f in ALL_FEATURE_COLUMNS:
                        if f in orig_row_s.index and pd.notna(orig_row_s[f]):
                            raw_vals[f] = float(orig_row_s[f])
                        elif f in self.selected_features:
                            f_idx = self.selected_features.index(f)
                            raw_vals[f] = float(cand_raw_np[f_idx])
                        else:
                            lo, hi = self.design_bounds.get(f, (0.0, 1.0))
                            raw_vals[f] = (lo + hi) / 2.0
                else:
                    for f in ALL_FEATURE_COLUMNS:
                        if f in self.selected_features:
                            f_idx = self.selected_features.index(f)
                            raw_vals[f] = float(cand_raw_np[f_idx])
                        elif f in continuous_prefills:
                            raw_vals[f] = continuous_prefills[f]
                        else:
                            lo, hi = self.design_bounds.get(f, (0.0, 1.0))
                            raw_vals[f] = (lo + hi) / 2.0

                norm_vals = {
                    f: float(chosen_cand_norm[i])
                    for i, f in enumerate(self.selected_features)
                }
            
            invalid_full_fields: list[str] = []
            for f in ALL_FEATURE_COLUMNS:
                val = raw_vals.get(f, np.nan)
                lo, hi = self.design_bounds.get(f, (-np.inf, np.inf))
                if pd.isna(val) or val < lo or val > hi:
                    invalid_full_fields.append(f)

            is_valid_full = len(invalid_full_fields) == 0
            if not is_valid_full:
                logger.warning(
                    "Selected candidate has out-of-bounds/missing full-feature "
                    "values in fields: %s",
                    ", ".join(invalid_full_fields),
                )
            
            expert_rank_val = -1
            if interactive and chosen_source == "manual_override":
                expert_rank_val = 0
            elif interactive and chosen_idx in top_indices:
                expert_rank_val = top_indices.index(chosen_idx) + 1
            elif not interactive:
                 expert_rank_val = top_indices.index(chosen_idx) + 1

            record = CandidateRecord(
                raw_values=raw_vals,
                normalized_values=norm_vals,
                source=chosen_source,
                is_valid_full_feature=is_valid_full,
                acq_value=chosen_acq_value,
                orig_row_idx=chosen_orig_row,
                expert_rank=expert_rank_val,
                overridden_fields=(
                    overrides if chosen_source in ("continuous", "manual_override") else []
                ),
                oob_confirmation_count=oob_count
            )

            if chosen_source == "manual_override":
                logger.info(
                    "Selected candidate: MANUAL OVERRIDE (outside top-%d list).",
                    len(top_indices),
                )
            else:
                logger.info(
                    "Selected candidate: Rank #%d (source: %s, UCB: %.4f)",
                    expert_rank_val,
                    chosen_source,
                    chosen_acq_value,
                )

            # KABO: Record preference comparisons for online learning.
            # Both normal selection AND manual override generate pairs.
            if self.kabo_mode:
                if chosen_source == "manual_override":
                    # Manual override is the strongest preference signal:
                    # the expert rejected ALL recommended candidates.
                    # Construct the normalised vector from norm_vals.
                    manual_norm_tensor = torch.tensor(
                        [norm_vals[f] for f in self.selected_features],
                        dtype=torch.double,
                        device=self.device,
                    )
                    losers_norm = [all_candidates[idx] for idx in top_indices]
                    if losers_norm:
                        self.preference_model.add_comparisons(
                            winner_norm=manual_norm_tensor,
                            losers_norm=losers_norm,
                        )
                elif chosen_idx in top_indices:
                    losers_norm = [
                        all_candidates[idx]
                        for idx in top_indices
                        if idx != chosen_idx
                    ]
                    self.preference_model.add_comparisons(
                        winner_norm=all_candidates[chosen_idx],
                        losers_norm=losers_norm,
                    )

            self._append_observation(
                record, product_yields, iteration
            )
            logger.info("Dataset now has %d rows.", len(self.df))

            self.phase2_fit_surrogate()

        print_best_found(self.df, self.selected_features, self.target_column)
        return self.df

    # ===================================================================
    #  Helpers
    # ===================================================================
    def _append_observation(
        self,
        candidate_record: CandidateRecord,
        product_yields: dict[str, float],
        iteration: int = -1,
    ) -> None:
        """Append a new observation using a CandidateRecord.

        Parameters
        ----------
        candidate_record : CandidateRecord
            The chosen candidate containing all raw values and audit fields.
        product_yields : dict[str, float]
            Dictionary mapping product column names to yield values.
        """
        new_row: dict[str, float] = dict(candidate_record.raw_values)
        
        # P1-2: Log audit information
        logger.info(
            "Observation source: %s — Audit: rank=%d, full_valid=%s",
            candidate_record.source, candidate_record.expert_rank,
            candidate_record.is_valid_full_feature
        )
        if candidate_record.overridden_fields:
            logger.info(
                "  Overridden fields: %s", 
                ", ".join(candidate_record.overridden_fields)
            )

        # Fill all product yield columns
        for col_name in ALL_PRODUCT_COLUMNS:
            new_row[col_name] = product_yields.get(col_name, np.nan)
            
        new_row["expert_rank"] = candidate_record.expert_rank
        if iteration >= 0:
            new_row["bo_iteration"] = iteration

        self.df = pd.concat(
            [self.df, pd.DataFrame([new_row])],
            ignore_index=True,
        )

    # ===================================================================
    #  Run Full Pipeline
    # ===================================================================
    def run(
        self,
        n_iterations: int = 10,
        interactive: bool = True,
    ) -> pd.DataFrame:
        """Execute the full three-phase optimization pipeline.

        Parameters
        ----------
        n_iterations : int, optional
            Number of BO iterations (default 10).
        interactive : bool, optional
            Whether to prompt for user input (default True).

        Returns
        -------
        pd.DataFrame
            The final updated dataset.
        """
        target_name = PRODUCT_NAMES.get(self.target_column, self.target_column)

        logger.info("Starting CO2RR Bayesian Optimization Pipeline")
        logger.info("Data: %s | Target: %s | K=%d | β=%.2f",
                     self.data_path.name, target_name,
                     self.top_k, self.beta)

        self.phase1_feature_selection()
        self.phase2_fit_surrogate()

        result_df = self.phase3_optimize(
            n_iterations=n_iterations,
            interactive=interactive,
        )

        output_path = self.output_dir / "data_updated.csv"
        result_df.to_csv(output_path, index=False)
        logger.info("Updated dataset saved to: %s", output_path)

        metadata = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "data_path": str(self.data_path),
            "target_column": self.target_column,
            "top_k": self.top_k,
            "acq_strategy": self.acq_strategy,
            "qnei_mc_samples": self.qnei_mc_samples,
            "kernel_type": self.kernel_type,
            "h2_penalty_weight": self.h2_penalty_weight,
            "beta": self.beta,
            "beta_schedule": self.beta_schedule,
            "beta_delta": self.beta_delta,
            "beta_trace": self._beta_trace,
            "n_iterations": n_iterations,
            "interactive": interactive,
            "skip_feature_selection": self.skip_feature_selection,
            "strict_training_schema": self.strict_training_schema,
            "pre_fill_before_choice": self.pre_fill_before_choice,
            "seed": self.seed,
            "selected_features": self.selected_features,
            "diversity_weight": self.diversity_weight,
            "kabo_mode": self.kabo_mode,
            "lambda_p": self.lambda_p if self.kabo_mode else None,
            "lambda_k": self.lambda_k if self.kabo_mode else None,
            "expert_prior_file": str(self.expert_prior_file) if self.expert_prior_file else None,
            "n_rows_final": int(len(result_df)),
            "output_data": str(output_path),
        }
        metadata_path = self.output_dir / "run_metadata.json"
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        logger.info("Run metadata saved to: %s", metadata_path)

        return result_df
