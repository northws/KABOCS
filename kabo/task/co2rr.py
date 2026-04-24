"""
CO2RRTask — photocatalytic CO2 reduction reaction.

Uses the 19-descriptor schema and seven-product yield columns defined
in ``kabo/constants.py``.  Supports a composite objective that penalizes
competing hydrogen evolution (HER).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from kabo.constants import (
    ALL_FEATURE_COLUMNS,
    ALL_PRODUCT_COLUMNS,
    DEFAULT_TARGET_PRODUCT,
    DESIGN_SPACE_BOUNDS,
    PRODUCT_COLUMNS,
    PRODUCT_NAMES,
)
from kabo.multi_objective import ObjectiveSpec
from kabo.utils import get_logger

from kabo.task.base import TaskBase, register_task

logger = get_logger(__name__)


@register_task
class CO2RRTask(TaskBase):
    """Photocatalytic CO2 reduction reaction task.

    Product schema includes seven outputs (CO, HCOOH, CH₄, C₂H₄, CH₃OH,
    C₂H₅OH, H₂) with H₂ treated as competing HER side reaction.  The
    training target can optionally subtract a scaled H₂ penalty to
    encourage selectivity toward the chosen primary product.
    """

    def task_name(self) -> str:
        return "CO2RR"

    def feature_columns(self) -> list[str]:
        return list(ALL_FEATURE_COLUMNS)

    def design_space_bounds(self) -> dict[str, tuple[float, float]]:
        return dict(DESIGN_SPACE_BOUNDS)

    def target_columns(self) -> dict[str, str]:
        return dict(PRODUCT_COLUMNS)

    def default_target(self) -> str:
        return DEFAULT_TARGET_PRODUCT

    def all_product_columns(self) -> list[str]:
        # Preserve the exact ordering used by the legacy constants module.
        return list(ALL_PRODUCT_COLUMNS)

    def product_names(self) -> dict[str, str]:
        return dict(PRODUCT_NAMES)

    # -------------------------------------------------------------------------
    #  Feature-type declarations (P0 of discrete variables proposal)
    #
    #  CO2RR has 6 integer-valued descriptors (hydrogen-bond counts) that
    #  must not be interpolated by the continuous GP / acquisition:
    #
    #      * A_hbond_acceptors, A_hbond_donors   (amino acid A, counts)
    #      * B_hbond_acceptors, B_hbond_donors   (amino acid B, counts)
    #      * Solvent_hbond_acceptors, Solvent_hbond_donors
    #
    #  The remaining 13 descriptors (pI, distances, binding energies,
    #  wavelengths, potentials, dielectrics, concentrations) are
    #  continuous.  Amino acid / solvent identity is currently encoded
    #  indirectly via these continuous descriptors; lifting them to
    #  categorical variables is left for P4 (chemical embeddings).
    # -------------------------------------------------------------------------
    _INTEGER_FEATURES: frozenset[str] = frozenset({
        "A_hbond_acceptors",
        "A_hbond_donors",
        "B_hbond_acceptors",
        "B_hbond_donors",
        "Solvent_hbond_acceptors",
        "Solvent_hbond_donors",
    })

    def feature_types(self) -> dict[str, str]:
        return {
            f: ("integer" if f in self._INTEGER_FEATURES else "continuous")
            for f in self.feature_columns()
        }

    # -------------------------------------------------------------------------
    #  Dynamic candidate pool (P2 of discrete variables proposal)
    #
    #  Generates ``n`` synthetic candidate recipes inside the declared
    #  design-space bounds.  Continuous dims use a Sobol sequence (better
    #  space-filling than uniform i.i.d.); integer dims use uniform random
    #  integers between their ``(lo, hi)`` bounds.  Values are laid out in
    #  the canonical ``feature_columns()`` order so downstream validation
    #  and scoring can treat this frame like ``candidates.csv``.
    # -------------------------------------------------------------------------
    def generate_candidates(
        self,
        n: int = 1000,
        seed: int = 0,
    ) -> pd.DataFrame:
        from torch.quasirandom import SobolEngine

        feature_columns = self.feature_columns()
        bounds = self.design_space_bounds()
        types = self.feature_types()

        cont_features = [f for f in feature_columns if types[f] == "continuous"]
        int_features = [f for f in feature_columns if types[f] == "integer"]

        rng = np.random.default_rng(seed)

        # ---- Continuous dims via Sobol (quasi-random, better coverage) ----
        K_cont = len(cont_features)
        cont_vals: dict[str, np.ndarray] = {}
        if K_cont > 0:
            sobol = SobolEngine(dimension=K_cont, scramble=True, seed=seed)
            U = sobol.draw(n).numpy()
            for j, f in enumerate(cont_features):
                lo, hi = bounds[f]
                cont_vals[f] = U[:, j] * (hi - lo) + lo

        # ---- Integer dims via uniform random ints ----
        int_vals: dict[str, np.ndarray] = {}
        for f in int_features:
            lo, hi = bounds[f]
            int_vals[f] = rng.integers(
                low=int(round(lo)), high=int(round(hi)) + 1, size=n,
            ).astype(np.float64)

        # ---- Assemble DataFrame in canonical column order ----
        data: dict[str, np.ndarray] = {}
        for f in feature_columns:
            if types[f] == "integer":
                data[f] = int_vals[f]
            else:
                data[f] = cont_vals[f]

        df = pd.DataFrame(data, columns=feature_columns)
        logger.info(
            "CO2RRTask.generate_candidates: produced %d rows "
            "(continuous=%d Sobol dims, integer=%d uniform dims, seed=%d).",
            len(df), K_cont, len(int_features), seed,
        )
        return df

    def build_training_target(
        self,
        df: pd.DataFrame,
        target_column: str,
        **kwargs,
    ) -> np.ndarray:
        """Construct the surrogate training target.

        If ``h2_penalty_weight`` (kwarg) is > 0 and a ``Y_H2`` column
        exists, returns ``Y_target - h2_penalty_weight * Y_H2`` to
        discourage HER-dominant conditions.
        """
        y_target = df[target_column].values.astype(np.float64)

        h2_penalty_weight = float(kwargs.get("h2_penalty_weight", 0.0))
        if h2_penalty_weight <= 0.0:
            return y_target

        if "Y_H2" not in df.columns:
            logger.warning(
                "h2_penalty_weight > 0 but 'Y_H2' column is missing. "
                "Falling back to single-target objective."
            )
            return y_target

        y_h2 = df["Y_H2"].values.astype(np.float64)
        y_composite = y_target - h2_penalty_weight * y_h2
        logger.info(
            "Using composite objective: %s - %.4f * Y_H2",
            target_column,
            h2_penalty_weight,
        )
        return y_composite

    def multi_objectives(self) -> list[ObjectiveSpec]:
        """Default MO preset: maximise CO selectivity vs. minimise HER.

        Rationale: CO2RR catalysis trades off the target carbon-product
        yield against competing hydrogen evolution (HER).  Instead of
        the scalar ``Y_CO − w·Y_H2`` composite used in single-objective
        mode, the MO preset treats both as first-class objectives so the
        optimiser can expose the full Pareto front.

        Users can always override this at run-time via
        ``--objectives CO HCOOH`` or similar.
        """
        return [
            ObjectiveSpec(
                column="Y_CO", direction="max",
                display_name="CO yield",
            ),
            ObjectiveSpec(
                column="Y_H2", direction="min",
                display_name="H2 (HER, to minimise)",
            ),
        ]

    # -------------------------------------------------------------------------
    #  Interactive / simulated observation collection
    # -------------------------------------------------------------------------
    def prompt_observation(
        self,
        target_column: str,
    ) -> Optional[dict[str, float]]:
        """Collect CO2RR product yields interactively.

        Iterates over ``self.target_columns()`` in declared order,
        flags the target product, and allows the user to enter ``"exit"``
        to abort the run.
        """
        product_cols = self.target_columns()
        product_names = self.product_names()

        print("\n  ┌─────────────────────────────────────────────────────┐")
        print("  │  📋 ENTER EXPERIMENTAL RESULTS (all product yields) │")
        print("  │     Enter '0' for undetected products               │")
        print("  │     Enter 'exit' to stop optimization               │")
        print("  └─────────────────────────────────────────────────────┘")

        results: dict[str, float] = {}
        for product_name, col_name in product_cols.items():
            is_target = (col_name == target_column)
            marker = " ★ TARGET" if is_target else ""

            while True:
                try:
                    prompt = (
                        f"    {product_name:8s} ({col_name}){marker}: "
                    )
                    user_input = input(prompt).strip()

                    if user_input.lower() in ("exit", "quit", "q"):
                        return None

                    value = float(user_input)
                    results[col_name] = value
                    break
                except ValueError:
                    print("      ⚠ Invalid input. Enter a number or 'exit'.")
                except (EOFError, KeyboardInterrupt):
                    print("\n  Received interrupt. Exiting.")
                    return None

        print("\n  Recorded yields:")
        for col_name, value in results.items():
            name = product_names.get(col_name, col_name)
            marker = " ★" if col_name == target_column else ""
            print(f"    {name:8s} = {value:8.2f}{marker}")

        return results

    def simulate_observation(
        self,
        target_column: str,
        y_mean: float,
        y_std: float,
    ) -> dict[str, float]:
        """Simulate CO2RR product yields for demo mode.

        Preserves the legacy distribution assumptions:

        * The target product is sampled around the training-data mean
          (±30% of training std).
        * ``Y_H2`` (competing HER) is uniform in ``[2, 15]``.
        * Other CO2RR products follow an exponential distribution
          (scale 2.0) to emulate minor-product noise.
        """
        results: dict[str, float] = {}
        for col_name in self.all_product_columns():
            if col_name == target_column:
                value = y_mean + np.random.normal(0, y_std * 0.3)
                results[col_name] = round(max(0.0, value), 2)
            elif col_name == "Y_H2":
                results[col_name] = round(np.random.uniform(2.0, 15.0), 2)
            else:
                results[col_name] = round(np.random.exponential(2.0), 2)
        return results
