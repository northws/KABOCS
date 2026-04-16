"""
TaskBase abstract contract and task registry.

A ``TaskBase`` encapsulates all domain-specific knowledge of a catalytic
reaction system (feature schema, design-space bounds, product columns,
training target construction, interactive prompts, etc.) so that the
underlying Bayesian Optimization engine (``KABOEngine``) remains
system-agnostic.

Concrete implementations live in sibling modules (``co2rr.py``,
``test_task.py``, …) and register themselves via the ``@register_task``
decorator.  Importing the ``kabo.task`` package triggers registration
for all built-in tasks, after which ``get_task(name)`` can construct
any task by lowercased name.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd

from kabo.constants import TARGET_COLUMN
from kabo.utils import get_logger

logger = get_logger(__name__)


# =============================================================================
#  Task registry (phase 4 uses this to resolve --task CLI arg)
# =============================================================================
TASK_REGISTRY: dict[str, type["TaskBase"]] = {}


def register_task(cls: type["TaskBase"]) -> type["TaskBase"]:
    """Class decorator: register a TaskBase subclass by its lowercased name."""
    key = cls().task_name().lower()
    if key in TASK_REGISTRY:
        logger.warning("Task '%s' already registered; overwriting.", key)
    TASK_REGISTRY[key] = cls
    return cls


def get_task(name: str) -> "TaskBase":
    """Instantiate a registered task by name (case-insensitive)."""
    key = name.lower()
    if key not in TASK_REGISTRY:
        raise KeyError(
            f"Unknown task '{name}'. Registered tasks: "
            f"{sorted(TASK_REGISTRY.keys())}"
        )
    return TASK_REGISTRY[key]()


# =============================================================================
#  Abstract base class
# =============================================================================
class TaskBase(ABC):
    """Abstract base class defining the contract a catalytic-system task
    must fulfill to drive the KABO engine.

    Concrete subclasses (e.g. ``CO2RRTask``, ``TestTask``) provide the
    domain layer: feature schema, design-space bounds, product columns,
    training-target construction, and interactive prompts.

    See ``docs/KABO_Engine_Task_Feasibility_Report.md`` §4.2 for the
    full rationale behind this interface.
    """

    # -------------------------------------------------------------------------
    #  Identity / schema
    # -------------------------------------------------------------------------
    @abstractmethod
    def task_name(self) -> str:
        """Short system identifier, e.g. ``"CO2RR"``, ``"OER"``."""

    @abstractmethod
    def feature_columns(self) -> list[str]:
        """Ordered list of ALL descriptor column names for this system."""

    @abstractmethod
    def design_space_bounds(self) -> dict[str, tuple[float, float]]:
        """Physical design-space bounds for every feature column."""

    # -------------------------------------------------------------------------
    #  Product / target schema
    # -------------------------------------------------------------------------
    @abstractmethod
    def target_columns(self) -> dict[str, str]:
        """Mapping from product short-name to DataFrame column name.

        For CO2RR: ``{"CO": "Y_CO", "HCOOH": "Y_HCOOH", ...}``.
        For systems with a single target, return e.g. ``{"O2": "Y_O2"}``.
        """

    @abstractmethod
    def default_target(self) -> str:
        """Default product short-name used when none is specified."""

    def all_product_columns(self) -> list[str]:
        """Ordered list of every product DataFrame column (derived)."""
        return list(self.target_columns().values())

    def product_names(self) -> dict[str, str]:
        """Inverse of ``target_columns``: column-name → short-name."""
        return {v: k for k, v in self.target_columns().items()}

    def resolve_target_column(self, target: str) -> str:
        """Resolve a user-supplied target (short-name or column name)
        into a DataFrame column name. Falls back to legacy ``Y``.
        """
        cols = self.target_columns()
        t_upper = target.upper() if isinstance(target, str) else target
        if t_upper in cols:
            return cols[t_upper]
        if target in cols.values():
            return target
        logger.warning(
            "Task '%s': target '%s' not recognized; falling back to "
            "legacy target column '%s'.",
            self.task_name(), target, TARGET_COLUMN,
        )
        return TARGET_COLUMN

    # -------------------------------------------------------------------------
    #  Surrogate training target
    # -------------------------------------------------------------------------
    @abstractmethod
    def build_training_target(
        self,
        df: pd.DataFrame,
        target_column: str,
        **kwargs,
    ) -> np.ndarray:
        """Construct the scalar surrogate training vector from raw columns.

        Implementations may define composite objectives (e.g. CO2RR's
        ``Y_target - weight * Y_H2`` selectivity form).
        """

    # -------------------------------------------------------------------------
    #  Optional hooks (default implementations provided)
    # -------------------------------------------------------------------------
    def validate_candidates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return ``df`` unchanged if it passes basic feature-schema checks."""
        missing = [c for c in self.feature_columns() if c not in df.columns]
        if missing:
            logger.warning(
                "Task '%s' candidate frame is missing %d feature columns: %s",
                self.task_name(), len(missing), missing,
            )
        return df

    def parse_expert_prior(self, json_obj: dict) -> dict:
        """Hook for validating/normalizing a domain-specific expert prior.

        Default implementation is a pass-through; override to add
        system-specific schema checks.
        """
        return json_obj

    @abstractmethod
    def prompt_observation(
        self,
        target_column: str,
    ) -> Optional[dict[str, float]]:
        """Interactive hook: collect all product observations from the user.

        Concrete subclasses own the CLI wording, product list, and any
        validation rules (e.g. "enter 0 for undetected", negative-yield
        rejection).  Return ``None`` if the user requests exit.
        """

    @abstractmethod
    def simulate_observation(
        self,
        target_column: str,
        y_mean: float,
        y_std: float,
    ) -> dict[str, float]:
        """Demo-mode hook: simulate product observations.

        Concrete subclasses define the system-specific product distribution
        (e.g. CO2RR has H₂ as HER side reaction in a 2–15% range).
        """
