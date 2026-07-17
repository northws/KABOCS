"""
KABO — Task-driven Bayesian Optimization Package
================================================

A modular Python pipeline for Bayesian Optimization of catalytic
reaction systems.  Domain specifics (feature schema, product columns,
prompts, simulation) are supplied by a ``TaskBase`` implementation,
while the algorithmic core (``KABOEngine``) remains system-agnostic.

Built-in tasks: ``CO2RRTask`` (photocatalytic CO2 reduction),
``ECO2RRTask`` (electrocatalytic CO2 reduction), ``PeptideECO2RRTask``
(peptide-ligated electrocatalytic CO2RR, descriptor-encoded ligand),
``TestTask`` (minimal synthetic task for smoke tests).

The methodology follows arXiv:2604.01328v3:
    "Efficient and Principled Scientific Discovery through
     Bayesian Optimization: A Tutorial"

Typical usage::

    from kabo import KABOOptimizer, CO2RRTask

    optimizer = KABOOptimizer(
        data_path="data/data.csv",
        task=CO2RRTask(),
    )
    optimizer.run(n_iterations=10, interactive=True)
"""

#
# Lazy top-level re-exports
# ---------------------------------------------------------------------------
# Importing ``kabo`` itself should not force the full torch / botorch /
# gpytorch stack to load — that prevents lightweight test suites (and
# docs tooling) from touching pure-Python helpers such as
# ``kabo.candidate.CandidateRecord`` without installing the optional
# optimizer stack.  We therefore defer every public symbol via the
# PEP 562 ``__getattr__`` hook, so ``kabo.KABOOptimizer`` / ``kabo.KABOEngine``
# still resolve on first access exactly like before.
#
from __future__ import annotations

from typing import TYPE_CHECKING

__version__ = "1.2.0"

__all__ = [
    "KABOOptimizer",
    "KABOEngine",
    "TaskBase",
    "CO2RRTask",
    "ECO2RRTask",
    "PeptideECO2RRTask",
    "TestTask",
    "TASK_REGISTRY",
    "get_task",
    "register_task",
    # Backward-compatibility alias
    "CO2RROptimizer",
]

# Map public name -> (submodule, attribute) for the lazy loader.
_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "KABOEngine": ("kabo.engine", "KABOEngine"),
    "KABOOptimizer": ("kabo.optimizer", "KABOOptimizer"),
    "CO2RROptimizer": ("kabo.optimizer", "CO2RROptimizer"),
    "TASK_REGISTRY": ("kabo.task", "TASK_REGISTRY"),
    "CO2RRTask": ("kabo.task", "CO2RRTask"),
    "ECO2RRTask": ("kabo.task", "ECO2RRTask"),
    "PeptideECO2RRTask": ("kabo.task", "PeptideECO2RRTask"),
    "TaskBase": ("kabo.task", "TaskBase"),
    "TestTask": ("kabo.task", "TestTask"),
    "get_task": ("kabo.task", "get_task"),
    "register_task": ("kabo.task", "register_task"),
}


def __getattr__(name: str):  # PEP 562
    try:
        modname, attr = _LAZY_ATTRS[name]
    except KeyError as exc:
        raise AttributeError(f"module 'kabo' has no attribute {name!r}") from exc
    import importlib

    mod = importlib.import_module(modname)
    value = getattr(mod, attr)
    globals()[name] = value  # cache to avoid repeat lookups
    return value


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)


if TYPE_CHECKING:  # static type checkers still see concrete imports
    from kabo.engine import KABOEngine
    from kabo.optimizer import CO2RROptimizer, KABOOptimizer
    from kabo.task import (
        TASK_REGISTRY,
        CO2RRTask,
        ECO2RRTask,
        PeptideECO2RRTask,
        TaskBase,
        TestTask,
        get_task,
        register_task,
    )
