"""
Backward-compatibility shim package for ``kabo``.

Before stage 4 of the KABO_Engine + Task generalization
(see ``docs/KABO_Engine_Task_Feasibility_Report.md``), the package was
named ``co2rr_bo``.  It has been renamed to ``kabo`` to reflect the
multi-task architecture (CO2RR, OER, …).  This shim re-exports every
public symbol from ``kabo`` so that legacy import paths continue to
work:

    from co2rr_bo import CO2RROptimizer          # still works
    from co2rr_bo.task import CO2RRTask          # still works
    from co2rr_bo.acquisition import build_ucb   # still works

New code should use the ``kabo`` package directly.
"""

from __future__ import annotations

import sys
import warnings
from importlib import import_module

import kabo as _kabo

# -- Submodule aliasing ------------------------------------------------------
# Expose every loaded kabo submodule under the legacy ``co2rr_bo`` prefix so
# ``import co2rr_bo.xxx`` keeps working.  Future imports are handled lazily
# below via ``__getattr__``.
_SUBMODULES = [
    "acquisition",
    "candidate",
    "cli",
    "constants",
    "engine",
    "feature_selection",
    "knowledge",
    "optimizer",
    "preference",
    "surrogate",
    "task",
    "utils",
]

for _name in _SUBMODULES:
    try:
        sys.modules[f"co2rr_bo.{_name}"] = import_module(f"kabo.{_name}")
    except ImportError:
        # Submodule may be optional/absent; skip silently.
        pass

# -- Re-export the kabo public surface ---------------------------------------
from kabo import (  # noqa: E402  (after submodule aliasing)
    CO2RROptimizer,
    CO2RRTask,
    KABOEngine,
    KABOOptimizer,
    TASK_REGISTRY,
    TaskBase,
    TestTask,
    get_task,
    register_task,
)

__all__ = [
    "CO2RROptimizer",
    "KABOOptimizer",
    "KABOEngine",
    "TaskBase",
    "CO2RRTask",
    "TestTask",
    "TASK_REGISTRY",
    "get_task",
    "register_task",
]
__version__ = _kabo.__version__


def __getattr__(name: str):
    """Lazy fall-through: resolve any unknown attribute against ``kabo``."""
    if name.startswith("_"):
        raise AttributeError(name)
    try:
        return getattr(_kabo, name)
    except AttributeError as exc:
        raise AttributeError(
            f"module 'co2rr_bo' has no attribute '{name}'"
        ) from exc


warnings.warn(
    "The 'co2rr_bo' package has been renamed to 'kabo'. "
    "This shim is retained for backward compatibility and will be "
    "removed in a future release. Please update your imports.",
    DeprecationWarning,
    stacklevel=2,
)
