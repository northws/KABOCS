"""
KABO — Task-driven Bayesian Optimization Package
================================================

A modular Python pipeline for Bayesian Optimization of catalytic
reaction systems.  Domain specifics (feature schema, product columns,
prompts, simulation) are supplied by a ``TaskBase`` implementation,
while the algorithmic core (``KABOEngine``) remains system-agnostic.

Built-in tasks: ``CO2RRTask`` (photocatalytic CO2 reduction),
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

from kabo.engine import KABOEngine
from kabo.optimizer import CO2RROptimizer, KABOOptimizer
from kabo.task import (
    TASK_REGISTRY,
    CO2RRTask,
    TaskBase,
    TestTask,
    get_task,
    register_task,
)

__all__ = [
    "KABOOptimizer",
    "KABOEngine",
    "TaskBase",
    "CO2RRTask",
    "TestTask",
    "TASK_REGISTRY",
    "get_task",
    "register_task",
    # Backward-compatibility alias
    "CO2RROptimizer",
]
__version__ = "1.1.0"
