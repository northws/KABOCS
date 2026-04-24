"""
Project registry — declarative optimization projects defined as JSON.

A *project* is a JSON file under ``projects/<name>.json`` that describes
everything :class:`kabo.task.TaskBase` needs to drive a new catalytic
system: feature schema, design-space bounds, product columns, default
target, and optional competing side-reactions.  On startup we synthesise
a ``TaskBase`` subclass for every project and register it via the
existing ``TASK_REGISTRY``.

This module intentionally does **not** touch any file inside ``kabo/``.
The plumbing is entirely additive: CLI users that don't import this
module keep seeing exactly the Python-defined built-in tasks
(``co2rr``, ``test``).  The web UI backend imports and calls
:func:`register_all` at application startup.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator

from kabo.task.base import TASK_REGISTRY, TaskBase, register_task

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECTS_DIR: Path = Path(__file__).resolve().parent.parent.parent / "projects"
PROJECTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Pydantic schema
# ---------------------------------------------------------------------------
FeatureType = str  # "continuous" | "integer"


class FeatureSpec(BaseModel):
    """One design-space feature (descriptor column)."""

    name: str = Field(..., description="Exact column name in the CSV.")
    type: FeatureType = Field(
        "continuous", description="'continuous' or 'integer'."
    )
    lo: float
    hi: float
    unit: Optional[str] = None
    display_name: Optional[str] = None

    @field_validator("name")
    @classmethod
    def _no_blank_name(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("feature name cannot be blank")
        return v

    @field_validator("type")
    @classmethod
    def _valid_type(cls, v: str) -> str:
        v = v.lower()
        if v not in ("continuous", "integer"):
            raise ValueError("feature type must be 'continuous' or 'integer'")
        return v

    @model_validator(mode="after")
    def _bounds_ok(self) -> "FeatureSpec":
        if self.hi <= self.lo:
            raise ValueError(
                f"feature '{self.name}': hi ({self.hi}) must be > lo ({self.lo})"
            )
        return self


class TargetSpec(BaseModel):
    """One product / target column."""

    short_name: str = Field(..., description="Short product name e.g. 'CO'.")
    column: str = Field(..., description="CSV column name e.g. 'Y_CO'.")
    display_name: Optional[str] = None
    unit: Optional[str] = "%"
    is_competing: bool = Field(
        False,
        description=(
            "If True, this product is treated as a side-reaction whose "
            "yield is subtracted from the target during training."
        ),
    )

    @field_validator("short_name", "column")
    @classmethod
    def _strip_required(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("short_name and column cannot be blank")
        return v


class ProjectSpec(BaseModel):
    """Full declarative project definition."""

    name: str = Field(..., description="Unique lowercase task name.")
    display_name: str = ""
    description: str = ""
    features: list[FeatureSpec] = Field(default_factory=list)
    targets: list[TargetSpec] = Field(default_factory=list)
    default_target: str = ""
    notes: str = ""

    @field_validator("name")
    @classmethod
    def _valid_name(cls, v: str) -> str:
        v = v.strip().lower()
        if not v:
            raise ValueError("name cannot be blank")
        if not all(c.isalnum() or c in "_-" for c in v):
            raise ValueError(
                "name may only contain alphanumerics, '_' and '-'"
            )
        return v

    @model_validator(mode="after")
    def _check_structure(self) -> "ProjectSpec":
        if not self.features:
            raise ValueError("project must declare at least one feature")
        if not self.targets:
            raise ValueError("project must declare at least one target")
        # Unique feature names
        names = [f.name for f in self.features]
        if len(set(names)) != len(names):
            raise ValueError("duplicate feature names are not allowed")
        # Unique target columns & short names
        cols = [t.column for t in self.targets]
        shorts = [t.short_name for t in self.targets]
        if len(set(cols)) != len(cols):
            raise ValueError("duplicate target columns are not allowed")
        if len(set(shorts)) != len(shorts):
            raise ValueError("duplicate target short_names are not allowed")
        # Default target
        if not self.default_target:
            self.default_target = self.targets[0].short_name
        elif self.default_target not in shorts:
            raise ValueError(
                f"default_target '{self.default_target}' must be one of "
                f"{shorts}"
            )
        if not self.display_name:
            self.display_name = self.name.upper()
        return self


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------
def _project_path(name: str) -> Path:
    return PROJECTS_DIR / f"{name.lower()}.json"


def list_specs() -> list[ProjectSpec]:
    """Return every project spec currently on disk (validated)."""
    specs: list[ProjectSpec] = []
    for p in sorted(PROJECTS_DIR.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            specs.append(ProjectSpec.model_validate(data))
        except Exception as exc:
            logger.warning("Skipping invalid project file %s: %s", p, exc)
    return specs


def load_spec(name: str) -> ProjectSpec:
    path = _project_path(name)
    if not path.exists():
        raise FileNotFoundError(f"Project '{name}' not found")
    data = json.loads(path.read_text(encoding="utf-8"))
    return ProjectSpec.model_validate(data)


def save_spec(spec: ProjectSpec) -> ProjectSpec:
    path = _project_path(spec.name)
    path.write_text(
        json.dumps(spec.model_dump(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return spec


def delete_spec(name: str) -> None:
    path = _project_path(name)
    if path.exists():
        path.unlink()


# ---------------------------------------------------------------------------
# Dynamic TaskBase factory
# ---------------------------------------------------------------------------
_BUILTIN_TASKS: set[str] = set()
_REGISTERED_DYNAMIC: set[str] = set()


def _snapshot_builtins() -> None:
    """Record the built-in task names so we never overwrite them silently."""
    global _BUILTIN_TASKS
    if not _BUILTIN_TASKS:
        _BUILTIN_TASKS = set(TASK_REGISTRY.keys()) - _REGISTERED_DYNAMIC


def is_builtin(name: str) -> bool:
    _snapshot_builtins()
    return name.lower() in _BUILTIN_TASKS


def _make_task_class(spec: ProjectSpec) -> type[TaskBase]:
    """Synthesise a concrete ``TaskBase`` subclass from a ``ProjectSpec``.

    All abstract methods are implemented; the composite objective is
    delegated to the kwarg ``h2_penalty_weight`` (reused for historical
    symmetry with ``CO2RRTask``) and applied to every target flagged
    ``is_competing``.
    """

    # Pre-compute schemas once so the closures don't walk the list every call.
    feature_columns = [f.name for f in spec.features]
    design_space = {
        f.name: (float(f.lo), float(f.hi)) for f in spec.features
    }
    feature_type_map = {f.name: f.type for f in spec.features}
    target_map = {t.short_name: t.column for t in spec.targets}
    all_cols = [t.column for t in spec.targets]
    display_names = {
        t.column: (t.display_name or t.short_name) for t in spec.targets
    }
    competing_cols = [t.column for t in spec.targets if t.is_competing]

    # Simulation ranges per product (used in demo mode).
    def _default_sim_range(col: str) -> tuple[float, float]:
        return (2.0, 15.0) if col in competing_cols else (0.0, 5.0)

    spec_name = spec.name

    class ProjectTask(TaskBase):
        """Dynamically generated task for project ``{spec.name}``."""

        _SPEC = spec  # exposed for introspection

        def task_name(self) -> str:
            return spec_name

        def feature_columns(self) -> list[str]:
            return list(feature_columns)

        def design_space_bounds(self) -> dict[str, tuple[float, float]]:
            return dict(design_space)

        def target_columns(self) -> dict[str, str]:
            return dict(target_map)

        def default_target(self) -> str:
            return spec.default_target

        def all_product_columns(self) -> list[str]:
            return list(all_cols)

        def product_names(self) -> dict[str, str]:
            # column → display / short name (prefer display for humans)
            return dict(display_names)

        def feature_types(self) -> dict[str, str]:
            return dict(feature_type_map)

        # -- training target ------------------------------------------------
        def build_training_target(
            self,
            df: pd.DataFrame,
            target_column: str,
            **kwargs: Any,
        ) -> np.ndarray:
            y = df[target_column].values.astype(np.float64)
            weight = float(kwargs.get("h2_penalty_weight", 0.0))
            if weight <= 0.0 or not competing_cols:
                return y
            penalty = np.zeros_like(y)
            for col in competing_cols:
                if col in df.columns:
                    penalty = penalty + df[col].values.astype(np.float64)
            logger.info(
                "Task '%s': composite objective = %s - %.4f * sum(%s)",
                task_name, target_column, weight, competing_cols,
            )
            return y - weight * penalty

        # -- interactive prompt --------------------------------------------
        def prompt_observation(
            self,
            target_column: str,
        ) -> Optional[dict[str, float]]:
            """Default CLI prompt. Web UI replaces this with a structured
            form via the bridge; this implementation is only used from
            the CLI (ui_bridge swaps the bound method out at runtime)."""
            print(
                f"\n  ┌─ [{task_name.upper()}] enter yields (or 'exit') ─┐"
            )
            results: dict[str, float] = {}
            for short, col in target_map.items():
                marker = " ★ TARGET" if col == target_column else ""
                while True:
                    try:
                        raw = input(
                            f"    {short:10s} ({col}){marker}: "
                        ).strip()
                        if raw.lower() in ("exit", "quit", "q"):
                            return None
                        results[col] = float(raw)
                        break
                    except ValueError:
                        print("      ⚠ Invalid number; try again.")
                    except (EOFError, KeyboardInterrupt):
                        return None
            return results

        # -- simulator ------------------------------------------------------
        def simulate_observation(
            self,
            target_column: str,
            y_mean: float,
            y_std: float,
        ) -> dict[str, float]:
            results: dict[str, float] = {}
            for col in all_cols:
                if col == target_column:
                    value = y_mean + np.random.normal(0, max(y_std, 1e-3) * 0.3)
                    results[col] = round(max(0.0, value), 3)
                elif col in competing_cols:
                    lo, hi = _default_sim_range(col)
                    results[col] = round(np.random.uniform(lo, hi), 3)
                else:
                    results[col] = round(np.random.exponential(2.0), 3)
            return results

        # -- optional dynamic candidates (Sobol + uniform int) -------------
        def generate_candidates(
            self,
            n: int = 1000,
            seed: int = 0,
        ) -> pd.DataFrame:
            try:
                from torch.quasirandom import SobolEngine
            except ImportError:
                return None  # type: ignore[return-value]

            rng = np.random.default_rng(seed)
            cont = [f for f in feature_columns if feature_type_map[f] == "continuous"]
            ints = [f for f in feature_columns if feature_type_map[f] == "integer"]

            data: dict[str, np.ndarray] = {}
            if cont:
                sobol = SobolEngine(dimension=len(cont), scramble=True, seed=seed)
                u = sobol.draw(n).numpy()
                for j, f in enumerate(cont):
                    lo, hi = design_space[f]
                    data[f] = u[:, j] * (hi - lo) + lo
            for f in ints:
                lo, hi = design_space[f]
                data[f] = rng.integers(
                    low=int(round(lo)),
                    high=int(round(hi)) + 1,
                    size=n,
                ).astype(np.float64)
            return pd.DataFrame(
                {f: data[f] for f in feature_columns},
                columns=feature_columns,
            )

    ProjectTask.__name__ = f"Project_{spec.name}_Task"
    ProjectTask.__qualname__ = ProjectTask.__name__
    ProjectTask.__doc__ = (
        f"Dynamic project task '{spec.name}' — {spec.display_name}"
    )
    return ProjectTask


# ---------------------------------------------------------------------------
# Registry sync
# ---------------------------------------------------------------------------
def register_spec(spec: ProjectSpec) -> None:
    """Add or replace a dynamic task in ``TASK_REGISTRY``."""
    _snapshot_builtins()
    key = spec.name.lower()
    if key in _BUILTIN_TASKS:
        raise ValueError(
            f"name '{key}' collides with a built-in task and cannot be "
            f"used for a dynamic project"
        )
    cls = _make_task_class(spec)
    # Bypass the @register_task decorator to avoid double-instantiation:
    # we already validated the spec and constructed a clean class.
    TASK_REGISTRY[key] = cls
    _REGISTERED_DYNAMIC.add(key)
    logger.info("Registered dynamic project task '%s'.", key)


def unregister_spec(name: str) -> None:
    key = name.lower()
    if key in _REGISTERED_DYNAMIC and key in TASK_REGISTRY:
        del TASK_REGISTRY[key]
        _REGISTERED_DYNAMIC.discard(key)
        logger.info("Unregistered dynamic project task '%s'.", key)


def register_all() -> list[ProjectSpec]:
    """Load every project JSON from disk and register them. Call on startup."""
    _snapshot_builtins()
    specs = list_specs()
    for spec in specs:
        try:
            register_spec(spec)
        except Exception as exc:  # pragma: no cover
            logger.error(
                "Failed to register project '%s': %s", spec.name, exc
            )
    return specs


def list_dynamic_names() -> list[str]:
    return sorted(_REGISTERED_DYNAMIC)


__all__ = [
    "FeatureSpec",
    "TargetSpec",
    "ProjectSpec",
    "PROJECTS_DIR",
    "list_specs",
    "load_spec",
    "save_spec",
    "delete_spec",
    "register_spec",
    "unregister_spec",
    "register_all",
    "list_dynamic_names",
    "is_builtin",
]
