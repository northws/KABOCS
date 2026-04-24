"""
TestTask — minimal synthetic task for end-to-end smoke testing.

Declares three generic continuous features (``x1``, ``x2``, ``x3``)
with unit bounds and a single product column (``y``).  Designed to
verify that the KABO engine and orchestrator stay system-agnostic
and can drive a run without any CO2RR-specific knowledge.

The filename is ``test_task.py`` (rather than ``test.py``) to avoid
collision with pytest's default test-discovery pattern.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from kabo.multi_objective import ObjectiveSpec
from kabo.task.base import TaskBase, register_task


@register_task
class TestTask(TaskBase):
    """Three-feature synthetic smoke task.

    Exposes two product columns (``y`` as the primary scalar target and
    ``y2`` as a secondary, anti-correlated objective) so that the same
    Task can drive both the single-objective legacy smoke path and the
    v1.2 multi-objective (qNEHVI) path.
    """

    def task_name(self) -> str:
        return "test"

    def feature_columns(self) -> list[str]:
        return ["x1", "x2", "x3"]

    def design_space_bounds(self) -> dict[str, tuple[float, float]]:
        return {
            "x1": (0.0, 1.0),
            "x2": (0.0, 1.0),
            "x3": (0.0, 1.0),
        }

    def target_columns(self) -> dict[str, str]:
        # Y remains the default scalar target so legacy tests using
        # --target-product Y keep their observed behaviour.  Y2 is a
        # secondary product that only matters in multi-objective mode.
        return {"Y": "y", "Y2": "y2"}

    def default_target(self) -> str:
        return "Y"

    def multi_objectives(self) -> list[ObjectiveSpec]:
        """Default dual-objective preset: maximise both ``y`` and ``y2``.

        Activated by ``--multi-objective`` on the CLI.  The two columns
        are synthesised to be mildly anti-correlated so the Pareto front
        is non-trivial for demos.
        """
        return [
            ObjectiveSpec(column="y", direction="max", display_name="y"),
            ObjectiveSpec(column="y2", direction="max", display_name="y2"),
        ]

    def build_training_target(
        self,
        df: pd.DataFrame,
        target_column: str,
        **kwargs,
    ) -> np.ndarray:
        return df[target_column].values.astype(np.float64)

    def prompt_observation(
        self,
        target_column: str,
    ) -> Optional[dict[str, float]]:
        """Interactive prompt: ask for every known product column.

        ``target_column`` is kept in the signature for API compatibility
        with the single-objective orchestrator, but in multi-objective
        mode the caller typically ignores it and reads every entry.
        """
        print(f"\n  [TestTask] Enter observations (primary target: "
              f"'{target_column}') or 'exit':")
        obs: dict[str, float] = {}
        for col in self.all_product_columns():
            while True:
                try:
                    raw = input(f"    {col}: ").strip()
                    if raw.lower() in ("exit", "quit", "q"):
                        return None
                    if raw == "" and col != target_column:
                        # Secondary product is optional — leave as NaN.
                        obs[col] = float("nan")
                        break
                    obs[col] = float(raw)
                    break
                except ValueError:
                    print("      ⚠ Invalid input. Enter a number, Enter to "
                          "skip (secondary only), or 'exit'.")
                except (EOFError, KeyboardInterrupt):
                    return None
        return obs

    def simulate_observation(
        self,
        target_column: str,
        y_mean: float,
        y_std: float,
    ) -> dict[str, float]:
        """Simulate BOTH product yields.

        ``y`` is drawn around the GP mean (single-objective behaviour).
        ``y2`` is derived from the same latent so that the two
        objectives are mildly anti-correlated — useful for producing
        non-degenerate Pareto fronts in the MO smoke tests.
        """
        y_val = y_mean + np.random.normal(0, max(y_std, 1e-3) * 0.5)
        # Anti-correlated secondary: peaks when y is low, with jitter.
        y2_val = (
            10.0 - 0.3 * float(y_val)
            + np.random.normal(0, max(y_std, 1e-3) * 0.3)
        )
        return {
            "y": round(float(y_val), 4),
            "y2": round(float(y2_val), 4),
        }
