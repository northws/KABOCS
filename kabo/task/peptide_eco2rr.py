"""
PeptideECO2RRTask — electrocatalytic CO2RR on a peptide-ligated metal centre.

A metal centre (Fe) carrying an amino-acid or short-peptide ligand, swept
over applied potential, with GC-quantified gas products reported as
Faradaic efficiencies.

Why this Task exists separately from ``ECO2RRTask``
---------------------------------------------------
The obvious way to model "which catalyst" is a categorical feature — and
that is exactly what makes the model useless for this system's actual
question, "what should we try next?".

A categorical dimension carries no metric between its levels: a
``CategoricalKernel`` only asks "same or different", ``design_space_bounds``
is ``(0, n-1)`` over the levels already declared, and
``generate_candidates()`` can only enumerate those same levels.  A residue
that has never been tested therefore has no coordinate, and BO can never
propose it.  That is a property of the encoding, not of the data volume —
no amount of extra measurement fixes it.

So the ligand enters the design space **only** through the averaged
physicochemical descriptors of its residues (see
``kabo.constants.AMINO_ACID_DESCRIPTORS``).  Every residue — tested or not
— has a coordinate, the bounds span all 20 canonical residues, and
:meth:`generate_candidates` proposes residues the model has never seen.
``CO2RRTask`` uses the same idea for its amino acids (``A_pI`` /
``A_hbond_*``); this Task simply makes it the whole ligand representation.

The cost of the descriptor encoding
-----------------------------------
Continuous relaxation inside ``optimize_acqf`` will happily return a
descriptor vector that corresponds to no real residue (a "chimera").  The
discrete candidate pool from :meth:`generate_candidates` is therefore the
load-bearing path here, and :meth:`nearest_residue` exists to interpret a
continuous proposal by snapping it back to the closest real residue.

Averaging over the sequence assumes the residues do not individually
coordinate the metal — i.e. there is no single "dominant" binding residue
whose descriptors should dominate.  If that stops being true for a given
ligand family, this representation must be revisited.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from kabo.constants import (
    AA_ALIASES,
    AA_DESCRIPTOR_NAMES,
    AMINO_ACID_DESCRIPTORS,
    ECO2RR_FE_TOTAL_MAX,
    PEPTIDE_ALL_PRODUCT_COLUMNS,
    PEPTIDE_DEFAULT_TARGET_PRODUCT,
    PEPTIDE_FEATURE_COLUMNS,
    PEPTIDE_LIGAND_DESCRIPTOR_MAP,
    PEPTIDE_POTENTIAL_BOUNDS,
    PEPTIDE_PRODUCT_COLUMNS,
    PEPTIDE_PRODUCT_NAMES,
    aa_descriptor_bounds,
)
from kabo.multi_objective import ObjectiveSpec
from kabo.utils import get_logger

from kabo.task.base import TaskBase, register_task

logger = get_logger(__name__)


def parse_sequence(ligand: str) -> list[str]:
    """Parse a ligand label into canonical residue names.

    Accepts the shapes that actually turn up in lab spreadsheets:
    ``"His-Arg-His"``, ``"his arg his"``, ``"Gln-Met"``, ``"met"``.

    Raises ``ValueError`` on an unknown residue rather than silently
    dropping it — a typo that quietly became a shorter peptide would
    change the averaged descriptors without any signal that it had.
    """
    raw = str(ligand).replace("_", "-").replace(" ", "-")
    parts = [p for p in raw.split("-") if p]
    if not parts:
        raise ValueError(f"Empty ligand sequence: {ligand!r}")

    out: list[str] = []
    for p in parts:
        key = AA_ALIASES.get(p.lower(), p.capitalize())
        if key not in AMINO_ACID_DESCRIPTORS:
            raise ValueError(
                f"Unknown residue {p!r} in ligand {ligand!r}. "
                f"Known: {sorted(AMINO_ACID_DESCRIPTORS)}"
            )
        out.append(key)
    return out


def featurize_sequence(residues: list[str]) -> dict[str, float]:
    """Average residue descriptors over a sequence.

    Returns a mapping keyed by :data:`AA_DESCRIPTOR_NAMES`.
    """
    M = np.array(
        [AMINO_ACID_DESCRIPTORS[r] for r in residues], dtype=float,
    )
    # strict=: guards against the descriptor table and the name list drifting
    # apart — a truncated zip would silently mislabel every residue.
    return dict(zip(AA_DESCRIPTOR_NAMES, M.mean(axis=0), strict=True))


@register_task
class PeptideECO2RRTask(TaskBase):
    """Peptide-ligated electrocatalytic CO2RR: ligand descriptors x potential.

    Every feature is continuous — deliberately.  See the module docstring
    for why a categorical catalyst identity is not used.
    """

    def task_name(self) -> str:
        return "peptide"

    def feature_columns(self) -> list[str]:
        return list(PEPTIDE_FEATURE_COLUMNS)

    def design_space_bounds(self) -> dict[str, tuple[float, float]]:
        aa = aa_descriptor_bounds()
        bounds = {
            col: aa[desc]
            for col, desc in PEPTIDE_LIGAND_DESCRIPTOR_MAP.items()
        }
        bounds["Applied_potential"] = PEPTIDE_POTENTIAL_BOUNDS
        return bounds

    # Every dim is continuous: no feature_types() / categorical_values()
    # override, so the base class reports all-continuous and the surrogate
    # stays on a plain SingleTaskGP (no MixedSingleTaskGP, and --gp-model
    # auto can still upgrade to SVGP once the dataset grows).

    def target_columns(self) -> dict[str, str]:
        return dict(PEPTIDE_PRODUCT_COLUMNS)

    def default_target(self) -> str:
        return PEPTIDE_DEFAULT_TARGET_PRODUCT

    def all_product_columns(self) -> list[str]:
        return list(PEPTIDE_ALL_PRODUCT_COLUMNS)

    def product_names(self) -> dict[str, str]:
        return dict(PEPTIDE_PRODUCT_NAMES)

    # -------------------------------------------------------------------------
    #  Ligand <-> descriptor plumbing
    # -------------------------------------------------------------------------
    def ligand_features(self, ligand: str) -> dict[str, float]:
        """Map a ligand label (e.g. ``"His-Arg-His"``) to feature columns."""
        desc = featurize_sequence(parse_sequence(ligand))
        return {
            col: desc[key]
            for col, key in PEPTIDE_LIGAND_DESCRIPTOR_MAP.items()
        }

    def add_ligand_descriptors(
        self,
        df: pd.DataFrame,
        ligand_column: str = "ligand",
    ) -> pd.DataFrame:
        """Return ``df`` with the six ``Ligand_*`` feature columns added.

        This is the bridge from an experimental CSV — which stores the
        human-readable ligand (``"His-Arg-His"``) — to the numeric design
        space the GP consumes.  Call it once when preparing data.
        """
        if ligand_column not in df.columns:
            raise KeyError(
                f"Column {ligand_column!r} not in DataFrame. "
                f"Available: {list(df.columns)}"
            )
        out = df.copy()
        feats = out[ligand_column].apply(
            lambda s: pd.Series(self.ligand_features(s))
        )
        for col in PEPTIDE_LIGAND_DESCRIPTOR_MAP:
            out[col] = feats[col]
        logger.info(
            "PeptideECO2RRTask: featurized %d row(s) / %d distinct ligand(s) "
            "into %d descriptor columns.",
            len(out), out[ligand_column].nunique(),
            len(PEPTIDE_LIGAND_DESCRIPTOR_MAP),
        )
        return out

    def nearest_residue(self, features: dict[str, float]) -> tuple[str, float]:
        """Snap a descriptor vector back onto the closest real residue.

        ``optimize_acqf`` relaxes the ligand dims to a continuous box, so a
        continuous recommendation is generally a chimera that corresponds to
        no synthesisable residue.  This maps it to the nearest single
        residue in bounds-normalised descriptor space.

        Returns
        -------
        (residue_name, normalised_distance)
            The distance is worth reading: a large value means the proposal
            does not resemble any real residue, and the "nearest" label is
            not a faithful realisation of what the acquisition asked for.
        """
        bounds = self.design_space_bounds()
        cols = list(PEPTIDE_LIGAND_DESCRIPTOR_MAP)

        def norm(vec: dict[str, float]) -> np.ndarray:
            out = []
            for col in cols:
                lo, hi = bounds[col]
                out.append((float(vec[col]) - lo) / (hi - lo if hi > lo else 1.0))
            return np.array(out)

        target = norm(features)
        best, best_d = "", float("inf")
        for name in AMINO_ACID_DESCRIPTORS:
            d = float(np.linalg.norm(norm(self.ligand_features(name)) - target))
            if d < best_d:
                best, best_d = name, d
        return best, best_d

    # -------------------------------------------------------------------------
    #  Dynamic candidate pool
    #
    #  This is what actually lets BO propose an UNTESTED residue: the pool
    #  enumerates all 20 canonical residues (as single-residue ligands),
    #  not just the ones present in the training data.  Peptide candidates
    #  can be supplied through a candidates.csv instead.
    # -------------------------------------------------------------------------
    def generate_candidates(
        self,
        n: int = 1000,
        seed: int = 0,
    ) -> pd.DataFrame:
        from torch.quasirandom import SobolEngine

        residues = list(AMINO_ACID_DESCRIPTORS)
        per = max(1, n // len(residues))
        lo, hi = PEPTIDE_POTENTIAL_BOUNDS
        sobol = SobolEngine(dimension=1, scramble=True, seed=seed)
        pots = sobol.draw(per).numpy().ravel() * (hi - lo) + lo

        rows = []
        for r in residues:
            feats = self.ligand_features(r)
            for p in pots:
                row = dict(feats)
                row["Applied_potential"] = float(p)
                rows.append(row)

        df = pd.DataFrame(rows, columns=self.feature_columns())
        logger.info(
            "PeptideECO2RRTask.generate_candidates: %d rows "
            "(%d canonical residues x %d Sobol potentials, seed=%d).",
            len(df), len(residues), per, seed,
        )
        return df

    # -------------------------------------------------------------------------
    #  Objectives
    # -------------------------------------------------------------------------
    def build_training_target(
        self,
        df: pd.DataFrame,
        target_column: str,
        **kwargs,
    ) -> np.ndarray:
        """Construct the surrogate training target.

        If ``h2_penalty_weight`` > 0 and ``FE_H2`` exists, returns
        ``FE_target - h2_penalty_weight * FE_H2``.
        """
        y = df[target_column].values.astype(np.float64)

        w = float(kwargs.get("h2_penalty_weight", 0.0))
        if w <= 0.0:
            return y
        if "FE_H2" not in df.columns:
            logger.warning(
                "h2_penalty_weight > 0 but 'FE_H2' column is missing. "
                "Falling back to single-target objective."
            )
            return y

        logger.info(
            "Using composite objective: %s - %.4f * FE_H2", target_column, w,
        )
        return y - w * df["FE_H2"].values.astype(np.float64)

    def multi_objectives(self) -> list[ObjectiveSpec]:
        """Default MO preset: maximise CO Faradaic efficiency vs. minimise HER."""
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
    #  Observation collection
    # -------------------------------------------------------------------------
    def prompt_observation(
        self,
        target_column: str,
    ) -> Optional[dict[str, float]]:
        """Collect gas-product Faradaic efficiencies interactively."""
        product_cols = self.target_columns()
        product_names = self.product_names()

        print("\n  ┌─────────────────────────────────────────────────────┐")
        print("  │  ⚡ ENTER GAS-PRODUCT RESULTS (Faradaic eff., %)     │")
        print("  │     Enter '0' for undetected products               │")
        print("  │     Total FE should be ≤ 100%                       │")
        print("  │     Enter 'exit' to stop optimization               │")
        print("  └─────────────────────────────────────────────────────┘")

        results: dict[str, float] = {}
        for name, col in product_cols.items():
            marker = " ★ TARGET" if col == target_column else ""
            while True:
                try:
                    raw = input(f"    {name:6s} ({col}){marker}: ").strip()
                    if raw.lower() in ("exit", "quit", "q"):
                        return None
                    v = float(raw)
                    if v < 0:
                        print("      ⚠ Faradaic efficiency cannot be negative.")
                        continue
                    results[col] = v
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
        for col, v in results.items():
            marker = " ★" if col == target_column else ""
            print(f"    {product_names.get(col, col):6s} = {v:8.2f} %{marker}")
        print(f"    {'TOTAL':6s} = {total:8.2f} %")
        return results

    def simulate_observation(
        self,
        target_column: str,
        y_mean: float,
        y_std: float,
    ) -> dict[str, float]:
        """Simulate gas-product Faradaic efficiencies for demo mode."""
        import math

        results: dict[str, float] = {}
        for col in self.all_product_columns():
            if col == target_column:
                results[col] = max(0.0, y_mean + np.random.normal(0, y_std * 0.3))
            elif col == "FE_H2":
                results[col] = np.random.uniform(5.0, 40.0)
            else:
                results[col] = np.random.exponential(1.0)

        total = sum(results.values())
        if total <= ECO2RR_FE_TOTAL_MAX:
            return {k: round(v, 2) for k, v in results.items()}
        # Truncate rather than round on the rescale path: rounding after
        # scaling to exactly 100% can carry the total back over the cap.
        scale = ECO2RR_FE_TOTAL_MAX / total
        return {
            k: math.floor(v * scale * 100.0) / 100.0
            for k, v in results.items()
        }
