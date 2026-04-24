"""
Human-in-the-Loop interaction helpers.

This module hosts every CLI I/O helper invoked by the
``KABOOptimizer.phase3_optimize`` loop.  It was previously embedded in
``kabo.acquisition`` — split out here so that:

* ``kabo.acquisition`` becomes a pure "math layer" (acquisition-function
  construction, discrete-pool evaluation, candidate loading).
* WebUI monkey-patches and unit tests can target a focused surface.

Backward compatibility
----------------------
``kabo.acquisition`` re-exports every public symbol of this module, so
downstream code using ``from kabo.acquisition import prompt_user_...``
continues to work unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd

from kabo.utils import get_logger, unnormalize_x

logger = get_logger(__name__)


if TYPE_CHECKING:  # pragma: no cover
    import torch


# =============================================================================
#  Interactive prompts
# =============================================================================
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


# =============================================================================
#  Pretty-printers
# =============================================================================
def print_recommendations(
    candidates_norm: list["torch.Tensor"],
    acq_values: list[float],
    source_labels: list[str],
    all_orig_rows: list[int],
    discrete_df: Optional[pd.DataFrame],
    selected_features: list[str],
    all_feature_columns: list[str],
    bounds_raw: "torch.Tensor",
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


__all__ = [
    "prompt_user_candidate_choice",
    "prompt_user_manual_candidate",
    "prompt_user_nonselected_features",
    "print_recommendations",
    "print_best_found",
]
