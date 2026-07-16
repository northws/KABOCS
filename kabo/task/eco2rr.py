"""
ECO2RRTask — electrocatalytic CO2 reduction reaction.

Uses the 19-descriptor schema and eight Faradaic-efficiency columns
defined in ``kabo/constants.py``.  Supports a composite objective that
penalizes competing hydrogen evolution (HER).

Differences from the photocatalytic ``CO2RRTask``:

* The driving force is an applied electrode potential (V vs. RHE)
  rather than photosensitizer excitation, so the descriptor set covers
  electrode, electrolyte and cell engineering instead of MOF /
  photosensitizer / sacrificial-agent chemistry.
* Products are reported as Faradaic efficiencies (``FE_*``, percent)
  rather than yields.  FE values of one measurement are non-negative and
  sum to at most 100%; this task enforces that in both the interactive
  prompt and demo-mode simulation.
* Three descriptors are categorical (metal, cation, cell type), which
  routes the surrogate to ``MixedSingleTaskGP``.
"""

from __future__ import annotations

import itertools
import math
from typing import Optional

import numpy as np
import pandas as pd

from kabo.constants import (
    ECO2RR_ALL_PRODUCT_COLUMNS,
    ECO2RR_CATEGORICAL_VALUES,
    ECO2RR_DEFAULT_TARGET_PRODUCT,
    ECO2RR_DESIGN_SPACE_BOUNDS,
    ECO2RR_FE_TOTAL_MAX,
    ECO2RR_FEATURE_COLUMNS,
    ECO2RR_INTEGER_FEATURES,
    ECO2RR_PRODUCT_COLUMNS,
    ECO2RR_PRODUCT_NAMES,
)
from kabo.multi_objective import ObjectiveSpec
from kabo.utils import get_logger

from kabo.task.base import TaskBase, register_task

logger = get_logger(__name__)


@register_task
class ECO2RRTask(TaskBase):
    """Electrocatalytic CO2 reduction reaction task.

    Product schema includes eight outputs (CO, HCOOH, CH₄, C₂H₄, CH₃OH,
    C₂H₅OH, n-C₃H₇OH, H₂) reported as Faradaic efficiencies, with H₂
    treated as the competing HER side reaction.  The training target can
    optionally subtract a scaled H₂ penalty to encourage selectivity
    toward the chosen primary product.
    """

    def task_name(self) -> str:
        return "ECO2RR"

    def feature_columns(self) -> list[str]:
        return list(ECO2RR_FEATURE_COLUMNS)

    def design_space_bounds(self) -> dict[str, tuple[float, float]]:
        return dict(ECO2RR_DESIGN_SPACE_BOUNDS)

    def target_columns(self) -> dict[str, str]:
        return dict(ECO2RR_PRODUCT_COLUMNS)

    def default_target(self) -> str:
        return ECO2RR_DEFAULT_TARGET_PRODUCT

    def all_product_columns(self) -> list[str]:
        # Preserve the exact ordering used by the constants module.
        return list(ECO2RR_ALL_PRODUCT_COLUMNS)

    def product_names(self) -> dict[str, str]:
        return dict(ECO2RR_PRODUCT_NAMES)

    # -------------------------------------------------------------------------
    #  Feature-type declarations (P0 of discrete variables proposal)
    #
    #  Electrocatalytic CO2RR has three genuinely categorical descriptors
    #  that must not be interpolated by the continuous GP / acquisition:
    #
    #      * Metal_identity  (Cu, Ag, Au, Zn, Sn, Bi)
    #      * Cation          (Li, Na, K, Cs)
    #      * Cell_type       (H-cell, flow-cell, MEA)
    #
    #  Declaring these as "categorical" routes the surrogate to
    #  MixedSingleTaskGP (CategoricalKernel on those dims) instead of
    #  pretending that "half-way between Ag and Au" is a real electrode.
    #  Unlike CO2RRTask — which encodes identity indirectly via continuous
    #  descriptors only — this task keeps BOTH: the categorical identity
    #  AND its continuous binding-energy descriptors, since the two carry
    #  complementary information (scaling relations vs. metal-specific
    #  effects such as Cu's unique C–C coupling ability).
    #
    #  Two further descriptors are integer counts (d-electron count,
    #  cation charge); the remaining 14 are continuous.
    # -------------------------------------------------------------------------
    def feature_types(self) -> dict[str, str]:
        types: dict[str, str] = {}
        for f in self.feature_columns():
            if f in ECO2RR_CATEGORICAL_VALUES:
                types[f] = "categorical"
            elif f in ECO2RR_INTEGER_FEATURES:
                types[f] = "integer"
            else:
                types[f] = "continuous"
        return types

    def categorical_values(self) -> dict[str, list]:
        return {k: list(v) for k, v in ECO2RR_CATEGORICAL_VALUES.items()}

    # -------------------------------------------------------------------------
    #  Ordinal-encoding helpers
    #
    #  Categorical descriptors travel through the pipeline as their index
    #  in ``categorical_values()`` (Cu → 0, Ag → 1, …), because the GP,
    #  the acquisition optimizer and candidates.csv all speak floats.
    #  These two helpers are the only place that mapping is defined, and
    #  they are what the interactive prompt uses to show "Cu" instead of
    #  "0.0" to an experimentalist.
    # -------------------------------------------------------------------------
    def encode_categorical(self, feature: str, value: str) -> float:
        """Map a category label to its ordinal code.

        Raises ``ValueError`` when ``value`` is not a declared category,
        which keeps a typo in a hand-edited CSV from silently becoming a
        different electrode material.
        """
        values = ECO2RR_CATEGORICAL_VALUES.get(feature)
        if values is None:
            raise ValueError(f"'{feature}' is not a categorical feature.")
        try:
            return float(values.index(value))
        except ValueError:
            raise ValueError(
                f"'{value}' is not a valid {feature}. Choose one of: {values}"
            ) from None

    def decode_categorical(self, feature: str, code: float) -> str:
        """Map an ordinal code back to its category label.

        Codes are rounded before lookup: the acquisition optimizer snaps
        categorical dims to the integer grid, but a float that survived a
        normalize / unnormalize round-trip can still land on 1.9999997.
        """
        values = ECO2RR_CATEGORICAL_VALUES.get(feature)
        if values is None:
            raise ValueError(f"'{feature}' is not a categorical feature.")
        idx = int(round(float(code)))
        if not 0 <= idx < len(values):
            raise ValueError(
                f"Code {code!r} is out of range for {feature} "
                f"(expected 0..{len(values) - 1})."
            )
        return values[idx]

    # -------------------------------------------------------------------------
    #  Candidate validation
    # -------------------------------------------------------------------------
    def validate_candidates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Schema check plus a categorical-code range check.

        Extends the base implementation: an out-of-range ordinal code
        (e.g. ``Metal_identity = 9`` when only six metals are declared)
        would otherwise be silently clipped to the design-space bound by
        the surrogate's normalization step, quietly turning one electrode
        into another.
        """
        df = super().validate_candidates(df)

        for feature, values in ECO2RR_CATEGORICAL_VALUES.items():
            if feature not in df.columns:
                continue
            codes = pd.to_numeric(df[feature], errors="coerce")
            bad = codes.isna() | (codes < 0) | (codes > len(values) - 1)
            if bad.any():
                logger.warning(
                    "Task '%s': %d candidate row(s) have an out-of-range or "
                    "non-numeric '%s' code (expected 0..%d for %s).",
                    self.task_name(), int(bad.sum()), feature,
                    len(values) - 1, values,
                )
        return df

    # -------------------------------------------------------------------------
    #  Dynamic candidate pool (P2 of discrete variables proposal)
    #
    #  Categorical dims are ENUMERATED rather than sampled: the Cartesian
    #  product of all category combinations (6 metals × 4 cations × 3
    #  cell types = 72) is tiled across the requested rows, so every
    #  combination is represented roughly equally regardless of ``n``.
    #  Random sampling would instead leave some electrode/cation pairs
    #  unrepresented at small ``n``.  Continuous dims use a Sobol
    #  sequence (better space-filling than uniform i.i.d.); integer dims
    #  use uniform random integers between their ``(lo, hi)`` bounds.
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

        cat_features = [f for f in feature_columns if types[f] == "categorical"]
        int_features = [f for f in feature_columns if types[f] == "integer"]
        cont_features = [f for f in feature_columns if types[f] == "continuous"]

        rng = np.random.default_rng(seed)

        # ---- Categorical dims via tiled Cartesian product ----
        cat_vals: dict[str, np.ndarray] = {}
        if cat_features:
            grids = [
                range(len(ECO2RR_CATEGORICAL_VALUES[f])) for f in cat_features
            ]
            combos = np.array(list(itertools.product(*grids)), dtype=np.float64)
            # Tile to cover n rows, then shuffle so the categorical block
            # is not correlated with the Sobol row index.
            reps = int(np.ceil(n / len(combos)))
            tiled = np.tile(combos, (reps, 1))[:n]
            rng.shuffle(tiled, axis=0)
            for j, f in enumerate(cat_features):
                cat_vals[f] = tiled[:, j]

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
            if types[f] == "categorical":
                data[f] = cat_vals[f]
            elif types[f] == "integer":
                data[f] = int_vals[f]
            else:
                data[f] = cont_vals[f]

        df = pd.DataFrame(data, columns=feature_columns)
        logger.info(
            "ECO2RRTask.generate_candidates: produced %d rows "
            "(categorical=%d enumerated dims, continuous=%d Sobol dims, "
            "integer=%d uniform dims, seed=%d).",
            len(df), len(cat_features), K_cont, len(int_features), seed,
        )
        return df

    def build_training_target(
        self,
        df: pd.DataFrame,
        target_column: str,
        **kwargs,
    ) -> np.ndarray:
        """Construct the surrogate training target.

        If ``h2_penalty_weight`` (kwarg) is > 0 and an ``FE_H2`` column
        exists, returns ``FE_target - h2_penalty_weight * FE_H2`` to
        discourage HER-dominant conditions.  HER is a stiffer competitor
        in electrocatalysis than in photocatalysis — it shares the same
        applied potential and proton source — so this penalty matters
        more here than in the photocatalytic task.
        """
        y_target = df[target_column].values.astype(np.float64)

        h2_penalty_weight = float(kwargs.get("h2_penalty_weight", 0.0))
        if h2_penalty_weight <= 0.0:
            return y_target

        if "FE_H2" not in df.columns:
            logger.warning(
                "h2_penalty_weight > 0 but 'FE_H2' column is missing. "
                "Falling back to single-target objective."
            )
            return y_target

        y_h2 = df["FE_H2"].values.astype(np.float64)
        y_composite = y_target - h2_penalty_weight * y_h2
        logger.info(
            "Using composite objective: %s - %.4f * FE_H2",
            target_column,
            h2_penalty_weight,
        )
        return y_composite

    def multi_objectives(self) -> list[ObjectiveSpec]:
        """Default MO preset: maximise CO Faradaic efficiency vs. minimise HER.

        Rationale: identical in spirit to the photocatalytic preset — the
        target carbon product trades off against competing hydrogen
        evolution — but the trade-off is sharper here because FE values
        are constrained to sum to <= 100%, so every percent of HER is a
        percent taken directly from the carbon products.  Exposing both
        as first-class objectives lets the optimiser map the full Pareto
        front instead of committing to one scalar penalty weight.

        Users can always override this at run-time via
        ``--objectives CO C2H4`` or similar.
        """
        return [
            ObjectiveSpec(
                column="FE_CO", direction="max",
                display_name="CO Faradaic efficiency",
            ),
            ObjectiveSpec(
                column="FE_H2", direction="min",
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
        """Collect CO2RR Faradaic efficiencies interactively.

        Iterates over ``self.target_columns()`` in declared order, flags
        the target product, and allows the user to enter ``"exit"`` to
        abort the run.  Rejects negative values and warns (without
        rejecting) when the total exceeds 100% — a real measurement can
        overshoot slightly through accumulated calibration error, but a
        large overshoot usually means a mis-entered value.
        """
        product_cols = self.target_columns()
        product_names = self.product_names()

        print("\n  ┌─────────────────────────────────────────────────────┐")
        print("  │  ⚡ ENTER EXPERIMENTAL RESULTS (Faradaic eff., %)    │")
        print("  │     Enter '0' for undetected products               │")
        print("  │     Total FE across all products should be ≤ 100%   │")
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
                    if value < 0:
                        print("      ⚠ Faradaic efficiency cannot be negative.")
                        continue
                    results[col_name] = value
                    break
                except ValueError:
                    print("      ⚠ Invalid input. Enter a number or 'exit'.")
                except (EOFError, KeyboardInterrupt):
                    print("\n  Received interrupt. Exiting.")
                    return None

        total = sum(results.values())
        if total > ECO2RR_FE_TOTAL_MAX:
            print(
                f"\n  ⚠ Total FE = {total:.2f}% exceeds "
                f"{ECO2RR_FE_TOTAL_MAX:.0f}%. Recorded as entered — please "
                f"double-check the values."
            )

        print("\n  Recorded Faradaic efficiencies:")
        for col_name, value in results.items():
            name = product_names.get(col_name, col_name)
            marker = " ★" if col_name == target_column else ""
            print(f"    {name:8s} = {value:8.2f} %{marker}")
        print(f"    {'TOTAL':8s} = {total:8.2f} %")

        return results

    def simulate_observation(
        self,
        target_column: str,
        y_mean: float,
        y_std: float,
    ) -> dict[str, float]:
        """Simulate CO2RR Faradaic efficiencies for demo mode.

        Mirrors the photocatalytic distribution assumptions, adapted to
        the FE domain:

        * The target product is sampled around the training-data mean
          (±30% of training std).
        * ``FE_H2`` (competing HER) is uniform in ``[5, 40]`` — a wider
          and higher band than the photocatalytic ``[2, 15]``, because
          HER competes directly for the same applied potential.
        * Other products follow an exponential distribution (scale 2.0)
          to emulate minor-product noise.
        * The whole vector is rescaled if it would exceed 100% total, so
          that simulated data respects the same constraint the prompt
          asks experimentalists to respect.
        """
        results: dict[str, float] = {}
        for col_name in self.all_product_columns():
            if col_name == target_column:
                value = y_mean + np.random.normal(0, y_std * 0.3)
                results[col_name] = max(0.0, value)
            elif col_name == "FE_H2":
                results[col_name] = np.random.uniform(5.0, 40.0)
            else:
                results[col_name] = np.random.exponential(2.0)

        # Enforce the FE budget: scale the whole vector down proportionally
        # rather than clipping one product, which would distort selectivity.
        total = sum(results.values())
        if total <= ECO2RR_FE_TOTAL_MAX:
            return {k: round(v, 2) for k, v in results.items()}

        scale = ECO2RR_FE_TOTAL_MAX / total
        # Truncate rather than round on this path: rounding eight products
        # half-up after scaling to exactly 100% can carry the total back
        # over the cap (100.01), which is the one thing this branch exists
        # to prevent.  Truncation can only ever undershoot.
        return {
            k: math.floor(v * scale * 100.0) / 100.0
            for k, v in results.items()
        }
