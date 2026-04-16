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

from kabo.task.base import TaskBase, register_task


@register_task
class TestTask(TaskBase):
    """Three-feature, single-product synthetic smoke task."""

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
        return {"Y": "y"}

    def default_target(self) -> str:
        return "Y"

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
        """Minimal interactive prompt: ask only for the single target."""
        print(f"\n  [TestTask] Enter observation for '{target_column}' "
              "(or 'exit'):")
        while True:
            try:
                user_input = input("    value: ").strip()
                if user_input.lower() in ("exit", "quit", "q"):
                    return None
                return {target_column: float(user_input)}
            except ValueError:
                print("      ⚠ Invalid input. Enter a number or 'exit'.")
            except (EOFError, KeyboardInterrupt):
                return None

    def simulate_observation(
        self,
        target_column: str,
        y_mean: float,
        y_std: float,
    ) -> dict[str, float]:
        """Simulate a single-product observation centred on the GP mean."""
        value = y_mean + np.random.normal(0, max(y_std, 1e-3) * 0.5)
        return {target_column: round(float(value), 4)}
