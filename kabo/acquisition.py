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
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler

from kabo.utils import get_logger, unnormalize_x
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
) -> tuple[torch.Tensor, float]:
    """Optimize the acquisition function over continuous [0,1]^K bounds.

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
        return candidates.squeeze(0), values.item()
    except Exception as e:
        logger.warning("optimize_acqf failed: %s. Using random point.", e)
        rand_cand = torch.rand(K, dtype=torch.double, device=device)
        return rand_cand, 0.0


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


def prompt_user_candidate_choice(
    top_indices: list[int],
    n_total: int,
) -> Optional[int]:
    """Prompt the user to select which recommended candidate to execute.

    This implements the true human-in-the-loop semantics from
    Algorithm 3 (arXiv:2604.01328v3): the expert reviews the
    recommended candidates and decides which one to actually run.

    Parameters
    ----------
    top_indices : list[int]
        Indices of the top-ranked candidates (in the all_candidates list).
    n_total : int
        Total number of candidates available.

    Returns
    -------
    int or None
        The chosen candidate index (in all_candidates list),
        ``-1`` for manual override, ``-2`` for tie (no preference),
        or ``None`` if the user wants to exit.
    """
    _ = n_total  # reserved for future UI expansion

    print("\n  ┌─────────────────────────────────────────────────────┐")
    print("  │  🔬 SELECT CANDIDATE TO EXECUTE                    │")
    print("  │     Enter rank number (1, 2, 3, ...)               │")
    print("  │     Press Enter to accept Rank #1 (default)        │")
    print("  │     Enter 'tie' if candidates are equally good     │")
    print("  │     Enter 'manual' to override all candidates      │")
    print("  │     Enter 'exit' to stop optimization              │")
    print("  └─────────────────────────────────────────────────────┘")

    while True:
        try:
            user_input = input("    Your choice: ").strip()

            if user_input.lower() in ("exit", "quit", "q"):
                return None

            if user_input.lower() in ("manual", "override", "m"):
                logger.info("Expert chose manual override mode.")
                return -1

            if user_input.lower() in ("tie", "t", "equal"):
                logger.info("Expert declared tie — no preference recorded.")
                return -2

            if user_input == "":
                # Default: accept rank #1
                chosen_idx = top_indices[0]
                logger.info("Expert accepted Rank #1 (default).")
                return chosen_idx

            rank = int(user_input)
            if 1 <= rank <= len(top_indices):
                chosen_idx = top_indices[rank - 1]
                logger.info("Expert selected Rank #%d.", rank)
                return chosen_idx
            else:
                print(f"      ⚠ Invalid rank. Enter 1–{len(top_indices)} "
                      f"or 'exit'.")
        except ValueError:
            print("      ⚠ Invalid input. Enter a rank number or 'exit'.")
        except (EOFError, KeyboardInterrupt):
            print("\n  Received interrupt. Exiting.")
            return None


def prompt_user_manual_candidate(
    all_feature_columns: list[str],
    design_bounds: dict[str, tuple[float, float]],
) -> Optional[tuple[dict[str, float], int, list[str]]]:
    """Prompt the expert to provide a fully manual candidate point.

    This implements the Algorithm 3 branch where the expert can reject
    all model-recommended candidates and submit an alternative point.

    Parameters
    ----------
    all_feature_columns : list[str]
        Full ordered feature list for the experiment recipe.
    design_bounds : dict[str, tuple[float, float]]
        Design-space bounds used for defaults and out-of-bounds warnings.

    Returns
    -------
    tuple[dict[str, float], int, list[str]] or None
        ``(values, oob_confirmation_count, overridden_fields)`` for the
        manual candidate, or ``None`` if the expert exits.
    """
    if not all_feature_columns:
        return {}, 0, []

    print("\n  ┌─────────────────────────────────────────────────────┐")
    print("  │  ✍️  MANUAL OVERRIDE CANDIDATE                      │")
    print("  │     All recommended candidates are bypassed.        │")
    print("  │     Enter full feature values for your own point.   │")
    print("  │     Press Enter to use design-space midpoint.       │")
    print("  │     Enter 'exit' to stop optimization.              │")
    print("  └─────────────────────────────────────────────────────┘")

    results: dict[str, float] = {}
    oob_confirmation_count = 0
    overridden_fields: list[str] = []

    for feat in all_feature_columns:
        lo, hi = design_bounds.get(feat, (0.0, 1.0))
        midpoint = (lo + hi) / 2.0

        while True:
            try:
                prompt = (
                    f"    {feat:35s} [{lo:.2f}, {hi:.2f}] "
                    f"(default={midpoint:.2f}): "
                )
                user_input = input(prompt).strip()

                if user_input.lower() in ("exit", "quit", "q"):
                    return None

                if user_input == "":
                    results[feat] = midpoint
                    break

                value = float(user_input)
                if value < lo or value > hi:
                    logger.warning(
                        "Manual value %.4f is outside physical bounds "
                        "[%.4f, %.4f] for '%s'.", value, lo, hi, feat,
                    )
                    confirm = input("      Confirm out-of-bounds value? (y/N): ").strip()
                    if confirm.lower() != "y":
                        continue
                    oob_confirmation_count += 1

                results[feat] = value
                if not np.isclose(value, midpoint, rtol=0.0, atol=1e-12):
                    overridden_fields.append(feat)
                break

            except ValueError:
                print("      ⚠ Invalid input. Enter a number, Enter, or 'exit'.")
            except (EOFError, KeyboardInterrupt):
                print("\n  Received interrupt. Exiting.")
                return None

    return results, oob_confirmation_count, overridden_fields


def prompt_user_nonselected_features(
    nonselected_features: list[str],
    design_bounds: dict[str, tuple[float, float]],
) -> tuple[dict[str, float], int, list[str]]:
    """Prompt the user for actual values of non-selected features.

    For continuous candidates, the GP only optimizes over selected
    features. The remaining features correspond to real experimental
    conditions that the user should specify (e.g. which solvent,
    which photosensitizer wavelength, etc.).

    Parameters
    ----------
    nonselected_features : list[str]
        Feature names not in the GP's selected set.
    design_bounds : dict[str, tuple[float, float]]
        Design-space bounds for reference display.

    Returns
    -------
    tuple[dict[str, float], int, list[str]]
        ``(values, oob_confirmation_count, overridden_fields)`` where:
        - ``values`` maps feature name → chosen value.
        - ``oob_confirmation_count`` counts confirmed out-of-bounds inputs.
        - ``overridden_fields`` includes only explicitly entered fields whose
          values differ from the default midpoint strategy.
    """
    if not nonselected_features:
        return {}, 0, []

    print("\n  ┌─────────────────────────────────────────────────────┐")
    print("  │  📝 ENTER NON-OPTIMIZED FEATURE VALUES             │")
    print("  │     These features were not in the GP model.        │")
    print("  │     Enter the actual experimental conditions.       │")
    print("  │     Press Enter to use design-space midpoint.       │")
    print("  └─────────────────────────────────────────────────────┘")

    results: dict[str, float] = {}
    oob_confirmation_count = 0
    overridden_fields: list[str] = []

    for feat in nonselected_features:
        lo, hi = design_bounds.get(feat, (0.0, 1.0))
        midpoint = (lo + hi) / 2.0

        while True:
            try:
                prompt = (
                    f"    {feat:35s} [{lo:.2f}, {hi:.2f}] "
                    f"(default={midpoint:.2f}): "
                )
                user_input = input(prompt).strip()

                if user_input == "":
                    results[feat] = midpoint
                    break

                value = float(user_input)
                if value < lo or value > hi:
                    logger.warning(
                        "Value %.4f is outside physical bounds [%.4f, %.4f] "
                        "for '%s'.", value, lo, hi, feat,
                    )
                    confirm = input("      Are you sure you want to use this "
                                    "out-of-bounds value? (y/N): ").strip()
                    if confirm.lower() != 'y':
                        continue  # Ask again
                    oob_confirmation_count += 1
                results[feat] = value
                if not np.isclose(value, midpoint, rtol=0.0, atol=1e-12):
                    overridden_fields.append(feat)
                break

            except ValueError:
                print("      ⚠ Invalid input. Enter a number or press Enter.")
            except (EOFError, KeyboardInterrupt):
                # Fill remaining with midpoints
                results[feat] = midpoint
                for remaining_feat in nonselected_features[
                    nonselected_features.index(feat) + 1:
                ]:
                    lo_r, hi_r = design_bounds.get(remaining_feat, (0.0, 1.0))
                    results[remaining_feat] = (lo_r + hi_r) / 2.0
                return results, oob_confirmation_count, overridden_fields

    return results, oob_confirmation_count, overridden_fields


def print_recommendations(
    candidates_norm: list[torch.Tensor],
    acq_values: list[float],
    source_labels: list[str],
    all_orig_rows: list[int],
    discrete_df: Optional[pd.DataFrame],
    selected_features: list[str],
    all_feature_columns: list[str],
    bounds_raw: torch.Tensor,
    iteration: int,
    target_column: str,
    product_names: dict[str, str],
    top_n: int = 3,
    continuous_nonselected_values: dict[str, float] | None = None,
    diversity_weight: float = 0.5,
) -> list[int]:
    """Print top-N recommended experiments with full feature breakdown.

    For discrete candidates, this displays all features (selected and
    non-selected) from the original DataFrame row.

    For continuous candidates:
      - If ``continuous_nonselected_values`` is provided (P1-1 "complete
        recipe" mode), the full 19-dim recipe is shown so the expert
        can compare candidates on a level playing field.
      - Otherwise, placeholders are shown (legacy behaviour).

    Parameters
    ----------
    candidates_norm : list[torch.Tensor]
        Normalized candidate vectors.
    acq_values : list[float]
        Acquisition values for each candidate.
    source_labels : list[str]
        Source label for each candidate.
    all_orig_rows : list[int]
        Original row indices in the discrete dataframe.
    discrete_df : pd.DataFrame or None
        The discrete candidates dataframe (for fetching non-selected features).
    selected_features : list[str]
        Feature names used in GP optimization.
    all_feature_columns : list[str]
        All 19 descriptor names.
    bounds_raw : torch.Tensor
        Design-space bounds for un-normalization.
    iteration : int
        Current iteration number.
    target_column : str
        The optimization target column name.
    product_names : dict[str, str]
        Mapping from product column name to display name (from Task).
    top_n : int, optional
        Number of recommendations to print (default 3).
    continuous_nonselected_values : dict[str, float] or None
        Pre-filled non-selected feature values for continuous candidates.
        When provided, continuous candidates are displayed as complete
        recipes rather than partial + placeholder.

    Returns
    -------
    list[int]
        Indices of the top-N candidates.
    """
    target_name = product_names.get(target_column, target_column)

    # ------------------------------------------------------------------
    # Diversity-aware Top-N selection (greedy submodular)
    #
    # Following the "menu" concept from Astudillo & Frazier 2019, we
    # present diverse candidates rather than the N highest-scoring
    # (which may cluster in one region).
    #
    # Strategy:
    #   Slot 1 = best acquisition value (unchanged).
    #   Slot k = argmax_{i not selected} [  acq_norm(i)
    #              + diversity_weight * min-L2-dist(i, selected) ]
    # ------------------------------------------------------------------
    # diversity_weight passed as parameter (0 = pure score, 1 = strong diversity)

    n_cands = len(acq_values)
    acq_arr = np.array(acq_values, dtype=float)

    # Normalise acquisition values to [0, 1] for fair weighting
    acq_min, acq_max = acq_arr.min(), acq_arr.max()
    acq_range = acq_max - acq_min if acq_max > acq_min else 1.0
    acq_norm = (acq_arr - acq_min) / acq_range

    # Precompute candidate feature vectors as numpy for distance calc
    cand_vecs = np.array([c.cpu().numpy() for c in candidates_norm])

    selected: list[int] = []
    remaining = set(range(n_cands))

    for slot in range(min(top_n, n_cands)):
        if slot == 0:
            # First slot: pure best acquisition
            best = int(np.argmax(acq_arr))
        else:
            best_score = -np.inf
            best = -1
            sel_vecs = cand_vecs[selected]  # (k, d)
            for i in remaining:
                # Minimum L2 distance to any already-selected candidate
                dists = np.linalg.norm(sel_vecs - cand_vecs[i], axis=1)
                min_dist = dists.min()
                score = acq_norm[i] + diversity_weight * min_dist
                if score > best_score:
                    best_score = score
                    best = i
        selected.append(best)
        remaining.discard(best)

    top_indices = selected

    print("\n" + "=" * 65)
    print(f"  🔬 TOP {min(top_n, len(top_indices))} "
          f"RECOMMENDED EXPERIMENTS (Iteration {iteration})")
    print(f"     Optimizing: {target_name} ({target_column})")
    print("=" * 65)

    for rank, idx in enumerate(top_indices, 1):
        cand_norm = candidates_norm[idx]
        acq_val = acq_values[idx]
        source = source_labels[idx]
        orig_row = all_orig_rows[idx]

        cand_raw = unnormalize_x(cand_norm, bounds_raw)

        print(f"\n  Rank #{rank}  (acquisition value: {acq_val:.4f}, "
              f"source: {source})")

        if orig_row >= 0 and discrete_df is not None:
            # Discrete: show complete recipe from the original row
            row_data = discrete_df.iloc[orig_row]
            for feat in all_feature_columns:
                val = row_data.get(feat, np.nan)
                marker = " [selected]" if feat in selected_features else " [fixed]"
                print(f"    {feat:35s} = {val:.4f}{marker}")
        else:
            # Continuous candidate
            for feat in all_feature_columns:
                if feat in selected_features:
                    f_idx = selected_features.index(feat)
                    val = cand_raw[f_idx]
                    print(f"    {feat:35s} = {val:.4f} [selected]")
                elif (continuous_nonselected_values is not None
                      and feat in continuous_nonselected_values):
                    val = continuous_nonselected_values[feat]
                    print(f"    {feat:35s} = {val:.4f} [expert]")
                else:
                    print(f"    {feat:35s} = <to be provided by expert>")

    print("\n" + "-" * 65)
    return top_indices


def print_best_found(
    df: pd.DataFrame,
    selected_features: list[str],
    target_column: str,
    all_product_columns: list[str],
    product_names: dict[str, str],
) -> None:
    """Print the best observed experiment with all product yields.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset.
    selected_features : list[str]
        Feature names to display.
    target_column : str
        The optimization target column name.
    all_product_columns : list[str]
        Ordered list of every product yield column (from Task).
    product_names : dict[str, str]
        Mapping from product column name to display name (from Task).
    """
    if df.empty or target_column not in df.columns:
        return

    target_name = product_names.get(target_column, target_column)

    best_idx = df[target_column].idxmax()
    best_target = df.loc[best_idx, target_column]
    best_row = df.loc[best_idx]

    print("\n" + "=" * 65)
    print(f"  🏆 BEST EXPERIMENT FOUND (target: {target_name})")
    print("=" * 65)
    print(f"  {target_name} = {best_target:.4f}  ★")

    # Print all product yields if available
    available_products = [c for c in all_product_columns
                          if c in df.columns and c != target_column]
    if available_products:
        print("\n  All product yields:")
        print(f"    {'Product':10s} {'Yield':>10s}")
        print(f"    {'─' * 10}  {'─' * 10}")

        # Target first
        print(f"    {target_name + ' ★':10s} {best_target:10.4f}")
        for col in available_products:
            name = product_names.get(col, col)
            val = best_row.get(col, np.nan)
            if pd.notna(val):
                print(f"    {name:10s} {val:10.4f}")

    print("\n  Selected features:")
    for feat in selected_features:
        val = best_row.get(feat, np.nan)
        if pd.notna(val):
            print(f"    {feat:35s} = {val:.4f}")
    print("=" * 65 + "\n")
