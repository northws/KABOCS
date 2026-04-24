"""
Phase 3: Acquisition Function & Human-in-the-Loop Optimization.

Implements the UCB (Upper Confidence Bound) acquisition function
optimization and the interactive human-in-the-loop CLI loop.

The human-in-the-loop input collects yields for ALL CO2RR products:
CO, HCOOH, CH₄, C₂H₄, CH₃OH, C₂H₅OH, and H₂.

Follows the paper's:
- GP-UCB algorithm (Algorithm 2): α_UCB(x) = μ(x) + β·σ(x)
- Human-in-the-Loop BO (Algorithm 3): expert reviews candidates
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from botorch.acquisition import AcquisitionFunction, UpperConfidenceBound
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.generation import MaxPosteriorSampling
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler

from kabo.utils import get_logger, round_integer_dims_to_grid, unnormalize_x
from kabo.preference import PreferenceModel
from kabo.knowledge import ExpertPrior

logger = get_logger(__name__)

class KABOAcquisition(AcquisitionFunction):
    """Knowledge-Augmented Bayesian Optimization Acquisition Function.

    Combines a base acquisition function (e.g. UCB or qNEI) with:
      - A preference score:  λ_p · Pref(x)
      - An expert prior score:  λ_k · Prior(x)
      - An approximate VOI score:  λ_v · VOI(x)  (posterior variance proxy)

    Each component is z-score normalised online (over the current
    evaluation batch) before weighted combination so that no single
    term dominates due to scale differences.

    The VOI term is a lightweight approximation of the Knowledge Gradient
    (Wu & Frazier 2016): it uses the surrogate posterior variance as an
    information-value proxy rather than a full one-step lookahead.
    """

    def __init__(
        self,
        base_acq_func: AcquisitionFunction,
        preference_model: PreferenceModel,
        expert_prior: ExpertPrior,
        lambda_p: float = 1.0,
        lambda_k: float = 1.0,
        lambda_v: float = 0.0,
    ):
        super().__init__(model=base_acq_func.model)
        self.base_acq_func = base_acq_func
        self.preference_model = preference_model
        self.expert_prior = expert_prior
        self.lambda_p = lambda_p
        self.lambda_k = lambda_k
        self.lambda_v = lambda_v

    # ------------------------------------------------------------------
    @staticmethod
    def _zscore(t: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Z-score normalise a 1-D tensor (mean=0, std=1).

        Handles edge cases:
        - Single-element tensor: returns zeros (no relative ranking).
        - All-identical values (std ≈ 0): returns zeros.
        - NaN from degenerate inputs: returns zeros.
        """
        if t.numel() <= 1:
            return torch.zeros_like(t)
        std, mean = torch.std_mean(t, unbiased=False)
        if not torch.isfinite(std) or std < eps:
            return torch.zeros_like(t)
        return (t - mean) / (std + eps)

    # ------------------------------------------------------------------
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Evaluate α_KABO(x) = z(α_base) + λ_p·z(Pref) + λ_k·z(Prior).

        All three components are z-score normalised over the current
        evaluation batch before combination.

        Parameters
        ----------
        X : torch.Tensor
            Candidates, shape ``(... x q x d)`` as per BoTorch convention.
        """
        base_val = self.base_acq_func(X)  # shape: (batch,) or ()

        # Preference / prior models work on 2-D (N, d) inputs.
        d = X.shape[-1]
        X_2d = X.reshape(-1, d)  # (N, d)

        pref_score = self.preference_model.evaluate(X_2d).squeeze(-1)  # (N,)
        prior_score = self.expert_prior.evaluate(X_2d).squeeze(-1)     # (N,)

        # Reshape auxiliary scores to match base_val
        pref_score = pref_score.reshape(base_val.shape)
        prior_score = prior_score.reshape(base_val.shape)

        # Online z-score normalisation — prevents any single term from
        # dominating solely due to numeric scale differences.
        base_z = self._zscore(base_val)
        pref_z = self._zscore(pref_score)
        prior_z = self._zscore(prior_score)

        combined = base_z + self.lambda_p * pref_z + self.lambda_k * prior_z

        # Approximate VOI: use surrogate posterior variance as information
        # value proxy (cf. Wu & Frazier 2016 KG motivation).
        if self.lambda_v > 0:
            with torch.no_grad():
                posterior = self.model.posterior(X_2d)
                voi_score = posterior.variance.squeeze(-1)  # (N,)
                voi_score = voi_score.reshape(base_val.shape)
            voi_z = self._zscore(voi_score)
            combined = combined + self.lambda_v * voi_z

        return combined


def build_kabo(
    base_acq_func: AcquisitionFunction,
    preference_model: PreferenceModel,
    expert_prior: ExpertPrior,
    lambda_p: float = 1.0,
    lambda_k: float = 1.0,
    lambda_v: float = 0.0,
) -> KABOAcquisition:
    """Construct KABO Acquisition function."""
    return KABOAcquisition(
        base_acq_func=base_acq_func,
        preference_model=preference_model,
        expert_prior=expert_prior,
        lambda_p=lambda_p,
        lambda_k=lambda_k,
        lambda_v=lambda_v,
    )


def build_ucb(model: SingleTaskGP, beta: float) -> UpperConfidenceBound:
    """Construct the UCB acquisition function.

    Parameters
    ----------
    model : SingleTaskGP
        Fitted GP surrogate model.
    beta : float
        Exploration parameter β.

    Returns
    -------
    UpperConfidenceBound
        The acquisition function.
    """
    return UpperConfidenceBound(model=model, beta=beta)


def build_qnei(
    model: SingleTaskGP,
    num_mc_samples: int = 128,
    prune_baseline: bool = True,
) -> qNoisyExpectedImprovement:
    """Construct a qNoisyExpectedImprovement acquisition function.

    Parameters
    ----------
    model : SingleTaskGP
        Fitted GP surrogate model.
    num_mc_samples : int, optional
        Number of QMC samples for MC estimation (default 128).
    prune_baseline : bool, optional
        Whether to prune dominated baseline points (default True).

    Returns
    -------
    qNoisyExpectedImprovement
        The qNEI acquisition function.
    """
    if model.train_inputs is None or len(model.train_inputs) == 0:
        raise ValueError("Model has no training inputs; cannot build qNEI.")

    x_baseline = model.train_inputs[0]
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([num_mc_samples]))
    return qNoisyExpectedImprovement(
        model=model,
        X_baseline=x_baseline,
        sampler=sampler,
        prune_baseline=prune_baseline,
    )


def optimize_continuous(
    acq_func: AcquisitionFunction,
    K: int,
    device: torch.device,
    n_restarts: int = 10,
    raw_samples: int = 256,
    integer_indices: Optional[list[int]] = None,
    bounds_raw: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, float]:
    """Optimize the acquisition function over continuous [0,1]^K bounds.

    When ``integer_indices`` is provided, the continuous candidate is
    snapped to the nearest integer-grid point on those dims after
    optimization and its acquisition value is recomputed.  This is the
    "round-trick" of Garrido-Merchán & Hernández-Lobato (JMLR 2020) and
    is applied as belt-and-suspenders on top of any GP-level integer
    handling.

    Parameters
    ----------
    acq_func : AcquisitionFunction
        The acquisition function.
    K : int
        Number of features (dimensionality).
    device : torch.device
        Torch device.
    n_restarts : int, optional
        Number of random restarts (default 10).
    raw_samples : int, optional
        Number of raw initialization samples (default 256).
    integer_indices : list[int] or None, optional
        Dimension indices that must lie on an integer grid in raw space.
        Requires ``bounds_raw`` to be supplied as well.  Default None
        (no integer handling).
    bounds_raw : torch.Tensor or None, optional
        Raw design-space bounds, shape ``(2, K)``.  Required when
        ``integer_indices`` is non-empty so that the grid step can be
        recovered.

    Returns
    -------
    tuple[torch.Tensor, float]
        ``(best_candidate, acquisition_value)`` where
        ``best_candidate`` has shape ``(K,)``.
    """
    bounds_unit = torch.stack([
        torch.zeros(K, dtype=torch.double, device=device),
        torch.ones(K, dtype=torch.double, device=device),
    ])  # shape (2, K)

    try:
        candidates, values = optimize_acqf(
            acq_function=acq_func,
            bounds=bounds_unit,
            q=1,
            num_restarts=n_restarts,
            raw_samples=raw_samples,
        )
        best = candidates.squeeze(0)
        best_val = values.item()
        # ---- P1: snap integer dims to grid (round-trick) ----
        if integer_indices:
            if bounds_raw is None:
                logger.warning(
                    "integer_indices=%s was supplied but bounds_raw is None; "
                    "skipping integer grid snap.",
                    integer_indices,
                )
            else:
                snapped = round_integer_dims_to_grid(
                    best.unsqueeze(0), integer_indices, bounds_raw,
                ).squeeze(0)
                # Recompute acquisition value on the snapped point so the
                # orchestrator ranks apples-to-apples with discrete candidates.
                try:
                    with torch.no_grad():
                        new_val = float(acq_func(snapped.unsqueeze(0).unsqueeze(0)).item())
                    logger.debug(
                        "Integer round-trick: continuous acq %.4f -> snapped acq %.4f",
                        best_val, new_val,
                    )
                    best = snapped
                    best_val = new_val
                except Exception as err:
                    logger.warning(
                        "Failed to re-evaluate acquisition on snapped "
                        "candidate (%s); keeping unsnapped.", err,
                    )
        return best, best_val
    except Exception as e:
        logger.warning("optimize_acqf failed: %s. Using random point.", e)
        rand_cand = torch.rand(K, dtype=torch.double, device=device)
        if integer_indices and bounds_raw is not None:
            rand_cand = round_integer_dims_to_grid(
                rand_cand.unsqueeze(0), integer_indices, bounds_raw,
            ).squeeze(0)
        return rand_cand, 0.0


def optimize_continuous_batch(
    acq_func: AcquisitionFunction,
    K: int,
    q: int,
    device: torch.device,
    n_restarts: int = 10,
    raw_samples: int = 256,
    integer_indices: Optional[list[int]] = None,
    bounds_raw: Optional[torch.Tensor] = None,
) -> list[tuple[torch.Tensor, float]]:
    """Propose a *batch* of ``q`` diverse continuous candidates.

    Strategy
    --------
    * For ``q == 1`` this falls through to :func:`optimize_continuous`.
    * For ``q > 1`` we first try ``optimize_acqf`` with the requested
      ``q`` — appropriate for joint-MC acquisitions such as qNEI.  If
      that raises (typical for analytic acquisitions like UCB, which
      do **not** support q > 1), we fall back to a **sequential greedy**
      strategy: solve ``q`` independent ``q=1`` sub-problems, seeded
      with different random starts so the restarts diversify.

    Each returned candidate is independently snapped to the integer
    grid when ``integer_indices`` is provided, and its acquisition value
    is recomputed post-snap so downstream ranking stays consistent.

    Parameters
    ----------
    q : int
        Number of candidates to return.  Must be >= 1.
    Others : same as :func:`optimize_continuous`.

    Returns
    -------
    list[tuple[torch.Tensor, float]]
        List of length ``q`` where each element is
        ``(candidate[K,], acq_value)``.
    """
    if q < 1:
        raise ValueError(f"q must be >= 1, got {q}")
    if q == 1:
        c, v = optimize_continuous(
            acq_func, K, device, n_restarts, raw_samples,
            integer_indices=integer_indices, bounds_raw=bounds_raw,
        )
        return [(c, v)]

    bounds_unit = torch.stack([
        torch.zeros(K, dtype=torch.double, device=device),
        torch.ones(K, dtype=torch.double, device=device),
    ])

    # ---- Path A: joint batch optimization (only works for MC acq) ----
    try:
        candidates, values = optimize_acqf(
            acq_function=acq_func,
            bounds=bounds_unit,
            q=q,
            num_restarts=n_restarts,
            raw_samples=raw_samples,
        )
        # values is a scalar (joint utility); distribute equally across candidates.
        joint_val = float(values.item())
        results: list[tuple[torch.Tensor, float]] = []
        for i in range(q):
            c = candidates[i]
            if integer_indices and bounds_raw is not None:
                c = round_integer_dims_to_grid(
                    c.unsqueeze(0), integer_indices, bounds_raw,
                ).squeeze(0)
                try:
                    with torch.no_grad():
                        v = float(acq_func(c.unsqueeze(0).unsqueeze(0)).item())
                except Exception:
                    v = joint_val
            else:
                try:
                    with torch.no_grad():
                        v = float(acq_func(c.unsqueeze(0).unsqueeze(0)).item())
                except Exception:
                    v = joint_val
            results.append((c, v))
        logger.info("optimize_acqf(q=%d) succeeded (joint batch).", q)
        return results
    except Exception as joint_err:
        logger.info(
            "Joint q=%d optimization unavailable (%s); falling back to "
            "sequential greedy q=1 restarts.",
            q, type(joint_err).__name__,
        )

    # ---- Path B: sequential q=1 restarts (works for any acq) ----
    greedy: list[tuple[torch.Tensor, float]] = []
    for i in range(q):
        c, v = optimize_continuous(
            acq_func, K, device, n_restarts, raw_samples,
            integer_indices=integer_indices, bounds_raw=bounds_raw,
        )
        greedy.append((c, v))
    return greedy


def load_discrete_candidates(
    candidates_path: Optional[Path],
    selected_features: list[str],
    all_feature_columns: list[str],
    design_bounds: dict[str, tuple[float, float]],
    strict_all_features: bool = True,
) -> Optional[pd.DataFrame]:
    """Load discrete candidate vectors from CSV and validate completeness.

    When ``strict_all_features`` is True (default), **every feature**
    declared by the active Task must be present and non-NaN --- not just
    the selected subset.  This prevents "seemingly discrete" candidates
    from containing system-inferred fields, which would undermine
    reproducibility (REVIEW_REPORT §P0-1).

    Parameters
    ----------
    candidates_path : Path or None
        Path to the candidates CSV file.
    selected_features : list[str]
        Currently selected feature names.
    all_feature_columns : list[str]
        Full ordered descriptor list (supplied by the active Task).
    design_bounds : dict[str, tuple[float, float]]
        Design-space bounds per feature (supplied by the active Task).
    strict_all_features : bool
        If True, require *all* feature columns to be present and
        non-NaN.  If False, only ``selected_features`` are validated
        (legacy behaviour).

    Returns
    -------
    pd.DataFrame or None
        Loaded and validated candidates, or ``None`` if unavailable.

    Raises
    ------
    ValueError
        If the candidates CSV is missing required features or contains
        NaN values in validated columns.
    """
    if candidates_path is None or not candidates_path.exists():
        if candidates_path is not None:
            logger.warning(
                "Candidates file not found: %s. "
                "Using continuous optimization only.",
                candidates_path,
            )
        return None

    candidates_df = pd.read_csv(candidates_path)

    # --- P0-1: Determine which columns to validate ---
    validate_cols = (
        all_feature_columns if strict_all_features else selected_features
    )

    available = [c for c in validate_cols if c in candidates_df.columns]
    missing = set(validate_cols) - set(available)

    if missing:
        scope = (
            f"all {len(all_feature_columns)} features"
            if strict_all_features else "selected features"
        )
        raise ValueError(
            f"Candidates CSV '{candidates_path.name}' is missing required "
            f"{scope}: {sorted(missing)}. "
            f"All validated features must be present in the candidates file. "
            f"Please add the missing columns, or pass "
            f"strict_all_features=False to relax to selected-only validation, "
            f"or remove --candidates to use continuous optimization only."
        )

    # --- P0-1 continued: Check for NaN values across validated scope ---
    nan_count = candidates_df[validate_cols].isna().sum().sum()
    if nan_count > 0:
        scope = (
            f"all {len(all_feature_columns)} feature"
            if strict_all_features else "selected feature"
        )
        raise ValueError(
            f"Candidates CSV contains {nan_count} NaN values in "
            f"{scope} columns. All candidate values must be complete."
        )

    # --- Validate candidate values are within design bounds ---
    n_violations = 0
    for feat in validate_cols:
        if feat in design_bounds:
            lo, hi = design_bounds[feat]
            col_vals = candidates_df[feat]
            below = (col_vals < lo).sum()
            above = (col_vals > hi).sum()
            if below + above > 0:
                logger.warning(
                    "⚠ Candidate feature '%s': %d values outside design "
                    "bounds [%.4f, %.4f] (below: %d, above: %d)",
                    feat, below + above, lo, hi, below, above,
                )
                n_violations += below + above

    if n_violations > 0:
        logger.warning(
            "Total %d candidate values outside design bounds (scope: %s). "
            "These candidates may represent infeasible experiments.",
            n_violations,
            "all features" if strict_all_features else "selected features",
        )

    logger.info(
        "Loaded %d discrete candidates from %s "
        "(validated %d features, strict=%s)",
        len(candidates_df), candidates_path.name,
        len(validate_cols), strict_all_features,
    )
    return candidates_df


def evaluate_discrete_candidates(
    acq_func: AcquisitionFunction,
    candidates_df: pd.DataFrame,
    selected_features: list[str],
    bounds_raw: torch.Tensor,
    device: torch.device,
    all_feature_columns: list[str],
    design_bounds: dict[str, tuple[float, float]],
    top_n: int = 5,
    strict_full_bounds: bool = True,
) -> list[tuple[torch.Tensor, float, int]]:
    """Evaluate the acquisition function over a discrete candidate set.

    Candidates whose feature values fall outside the design-space bounds
    are **excluded** before scoring.  When ``strict_full_bounds`` is True
    (REVIEW_REPORT §P0-2), boundary checks cover **all 19 features**,
    not just the selected subset, preventing "locally legal but globally
    illegal" candidates from being recommended.

    Parameters
    ----------
    acq_func : AcquisitionFunction
        The acquisition function.
    candidates_df : pd.DataFrame
        DataFrame of candidate vectors (all selected features present).
    selected_features : list[str]
        Currently selected feature names.
    bounds_raw : torch.Tensor
        Design-space bounds tensor, shape ``(2, K)``.
    device : torch.device
        Torch device.
    all_feature_columns : list[str]
        Full ordered descriptor list (supplied by the active Task).
    design_bounds : dict[str, tuple[float, float]]
        Design-space bounds per feature (supplied by the active Task).
    top_n : int, optional
        Number of top candidates to return (default 5).
    strict_full_bounds : bool
        If True, exclude candidates where *any* of the feature columns
        violates its design-space bound (not just selected features).

    Returns
    -------
    list[tuple[torch.Tensor, float, int]]
        List of ``(candidate_normalized, acquisition_value, original_row_idx)``
        tuples, sorted by acquisition value descending.
    """
    # ---- P0-2: Full-feature boundary pre-filter ----------------------------
    if strict_full_bounds:
        full_bounds_mask = np.ones(len(candidates_df), dtype=bool)
        for feat in all_feature_columns:
            if feat in candidates_df.columns and feat in design_bounds:
                lo, hi = design_bounds[feat]
                vals = candidates_df[feat].values.astype(np.float64)
                full_bounds_mask &= (vals >= lo) & (vals <= hi)
        n_full_excluded = (~full_bounds_mask).sum()
        if n_full_excluded > 0:
            logger.warning(
                "[strict_full_bounds] Excluded %d / %d discrete candidates "
                "with at least one feature (out of %d) outside design-space "
                "bounds.",
                n_full_excluded, len(candidates_df),
                len(all_feature_columns),
            )
        # Restrict to full-feature-valid rows
        valid_full_indices = np.where(full_bounds_mask)[0]
        working_df = candidates_df.iloc[valid_full_indices].reset_index(drop=True)
    else:
        valid_full_indices = np.arange(len(candidates_df))
        working_df = candidates_df

    if len(working_df) == 0:
        logger.warning("No discrete candidates within full design bounds.")
        return []

    # ---- Selected-feature extraction & bounds for GP normalisation ---------
    X_cand_raw = working_df[selected_features].values.astype(np.float64)

    bounds_np = bounds_raw.cpu().numpy()
    x_min, x_max = bounds_np[0], bounds_np[1]
    x_range = x_max - x_min
    x_range[x_range == 0] = 1.0

    # Selected-feature bounds filter (always applied)
    in_bounds_mask = np.all(
        (X_cand_raw >= x_min) & (X_cand_raw <= x_max), axis=1
    )
    n_excluded = (~in_bounds_mask).sum()
    if n_excluded > 0:
        logger.warning(
            "Excluded %d / %d discrete candidates with selected-feature "
            "values outside GP bounds.",
            n_excluded, len(X_cand_raw),
        )

    local_valid = np.where(in_bounds_mask)[0]
    X_valid = X_cand_raw[in_bounds_mask]

    if len(X_valid) == 0:
        logger.warning("No discrete candidates within design bounds.")
        return []

    # Normalize in-bounds candidates to [0, 1] — no clipping needed
    X_norm = (X_valid - x_min) / x_range

    X_tensor = torch.tensor(
        X_norm, dtype=torch.double, device=device
    ).unsqueeze(1)  # shape (N, 1, K) for q-batch

    with torch.no_grad():
        acq_values = acq_func(X_tensor).cpu().numpy().ravel()

    results = []
    sort_idx = np.argsort(acq_values)[::-1]
    for idx in sort_idx[:top_n]:
        cand_tensor = torch.tensor(
            X_norm[idx], dtype=torch.double, device=device
        )
        # Map back to the original DataFrame row index
        orig_row_idx = int(valid_full_indices[int(local_valid[idx])])
        results.append((cand_tensor, float(acq_values[idx]), orig_row_idx))

    return results


def evaluate_discrete_thompson(
    model: SingleTaskGP,
    candidates_df: pd.DataFrame,
    selected_features: list[str],
    bounds_raw: torch.Tensor,
    device: torch.device,
    all_feature_columns: list[str],
    design_bounds: dict[str, tuple[float, float]],
    top_n: int = 5,
    strict_full_bounds: bool = True,
) -> list[tuple[torch.Tensor, float, int]]:
    """Select top-N discrete candidates via **Thompson sampling** from the
    GP posterior (P3 of discrete variables proposal).

    Draws ``top_n`` independent posterior samples over the full candidate
    pool and returns the argmax of each sample.  Duplicate argmaxes are
    filtered out, and the pool is oversampled up to ``4 * top_n`` times
    to reach the desired unique count.  This yields naturally diverse
    recommendations without an explicit diversity regulariser and scales
    to large pools (dynamic generators with thousands of rows) in O(N·q)
    GP forward passes.

    Parameters
    ----------
    model : SingleTaskGP
        The fitted GP surrogate (its posterior is sampled directly; no
        explicit acquisition function is needed).
    candidates_df, selected_features, bounds_raw, device,
    all_feature_columns, design_bounds, top_n, strict_full_bounds
        Same meanings as :func:`evaluate_discrete_candidates`.

    Returns
    -------
    list[tuple[torch.Tensor, float, int]]
        ``(candidate_normalized, posterior_sample_value, original_row_idx)``
        triples.  The "acquisition value" field is the sample value the
        sampler assigned to the winning candidate in its draw; this is
        monotonic with the predicted target so higher is better.  The
        list is returned in draw order (not sorted) — duplicates removed.
    """
    # ---- Re-use evaluate_discrete_candidates' pre-filter path by
    #      duplicating the strict_full_bounds / GP-bounds checks here.
    if strict_full_bounds:
        full_bounds_mask = np.ones(len(candidates_df), dtype=bool)
        for feat in all_feature_columns:
            if feat in candidates_df.columns and feat in design_bounds:
                lo, hi = design_bounds[feat]
                vals = candidates_df[feat].values.astype(np.float64)
                full_bounds_mask &= (vals >= lo) & (vals <= hi)
        n_full_excluded = (~full_bounds_mask).sum()
        if n_full_excluded > 0:
            logger.warning(
                "[strict_full_bounds] Excluded %d / %d candidates with "
                "at least one feature outside design-space bounds.",
                n_full_excluded, len(candidates_df),
            )
        valid_full_indices = np.where(full_bounds_mask)[0]
        working_df = candidates_df.iloc[valid_full_indices].reset_index(drop=True)
    else:
        valid_full_indices = np.arange(len(candidates_df))
        working_df = candidates_df

    if len(working_df) == 0:
        logger.warning("No discrete candidates within full design bounds.")
        return []

    X_cand_raw = working_df[selected_features].values.astype(np.float64)
    bounds_np = bounds_raw.cpu().numpy()
    x_min, x_max = bounds_np[0], bounds_np[1]
    x_range = x_max - x_min
    x_range[x_range == 0] = 1.0

    in_bounds_mask = np.all(
        (X_cand_raw >= x_min) & (X_cand_raw <= x_max), axis=1
    )
    n_excluded = (~in_bounds_mask).sum()
    if n_excluded > 0:
        logger.warning(
            "Excluded %d / %d candidates outside selected-feature bounds.",
            n_excluded, len(X_cand_raw),
        )

    local_valid = np.where(in_bounds_mask)[0]
    X_valid = X_cand_raw[in_bounds_mask]
    if len(X_valid) == 0:
        logger.warning("No discrete candidates within design bounds.")
        return []

    X_norm = (X_valid - x_min) / x_range
    X_tensor = torch.tensor(X_norm, dtype=torch.double, device=device)
    # MaxPosteriorSampling expects shape (N, d) — no q-batch dim.

    # Draw oversampled posterior samples to get `top_n` unique argmaxes.
    sampler = MaxPosteriorSampling(model=model, replacement=False)
    max_draws = max(top_n * 4, top_n + 2)
    picked_local_indices: list[int] = []
    seen: set[int] = set()

    try:
        with torch.no_grad():
            samples = sampler(X_tensor, num_samples=max_draws)
        # `samples` has shape (num_samples, d); find each in X_tensor.
        # Use exact equality on normalized float coords (safe since the
        # sampler picks rows from X_tensor itself, so values are bit-exact).
        for i in range(samples.shape[0]):
            row = samples[i]
            matches = torch.all(X_tensor == row, dim=1).nonzero(as_tuple=False)
            if matches.numel() == 0:
                # Fallback: closest match
                dist = torch.norm(X_tensor - row, dim=1)
                local_idx = int(dist.argmin().item())
            else:
                local_idx = int(matches[0].item())
            if local_idx in seen:
                continue
            seen.add(local_idx)
            picked_local_indices.append(local_idx)
            if len(picked_local_indices) >= top_n:
                break
    except Exception as err:
        logger.warning(
            "Thompson sampling failed (%s); falling back to posterior-mean "
            "ranking on the candidate pool.",
            err,
        )
        with torch.no_grad():
            post = model.posterior(X_tensor)
            mean = post.mean.squeeze(-1).cpu().numpy()
        # Also fall back to top-N by predicted mean.
        order = np.argsort(mean)[::-1]
        picked_local_indices = [int(i) for i in order[:top_n]]

    # Pad if we still have fewer than top_n uniques (rare; only when pool
    # is smaller than top_n).
    if len(picked_local_indices) < top_n:
        remaining = [
            i for i in range(len(X_norm)) if i not in seen
        ][: top_n - len(picked_local_indices)]
        picked_local_indices.extend(remaining)

    # Score each picked candidate by its posterior mean (monotonic proxy
    # for display; actual TS ranking is implicit in the draw order).
    if picked_local_indices:
        with torch.no_grad():
            picked_tensor = X_tensor[picked_local_indices]
            post = model.posterior(picked_tensor)
            means = post.mean.squeeze(-1).cpu().numpy().ravel()
    else:
        means = np.array([])

    results: list[tuple[torch.Tensor, float, int]] = []
    for rank, (local_idx, mu) in enumerate(zip(picked_local_indices, means)):
        cand_tensor = torch.tensor(
            X_norm[local_idx], dtype=torch.double, device=device,
        )
        orig_row_idx = int(valid_full_indices[int(local_valid[local_idx])])
        # Store mu as the "acq value" so downstream ranking/printing works
        # uniformly.  Higher is better.
        results.append((cand_tensor, float(mu), orig_row_idx))
    logger.info(
        "Thompson sampling: drew %d samples, returned %d unique picks.",
        max_draws, len(results),
    )
    return results


# =============================================================================
#  Interaction helpers moved to ``kabo.interaction`` (v1.2).
#  Re-exported here for backward compatibility so existing
#  ``from kabo.acquisition import prompt_user_*`` imports keep working.
# =============================================================================
from kabo.interaction import (  # noqa: E402  (intentional late import)
    print_best_found,
    print_recommendations,
    prompt_user_candidate_choice,
    prompt_user_manual_candidate,
    prompt_user_nonselected_features,
)

__all__ = [
    # Acquisition builders / optimisers
    "KABOAcquisition",
    "build_kabo",
    "build_ucb",
    "build_qnei",
    "optimize_continuous",
    "load_discrete_candidates",
    "evaluate_discrete_candidates",
    "evaluate_discrete_thompson",
    # Re-exports from kabo.interaction
    "prompt_user_candidate_choice",
    "prompt_user_manual_candidate",
    "prompt_user_nonselected_features",
    "print_recommendations",
    "print_best_found",
]
