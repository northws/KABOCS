"""
``kabo.task`` — Task package.

The task layer is organised as a package so each catalytic system lives
in its own module.  Importing the package triggers ``@register_task``
decorators for every built-in task, populating ``TASK_REGISTRY``:

    from kabo.task import CO2RRTask, TestTask, TaskBase, get_task, TASK_REGISTRY

To add a new system (e.g. OER):

    1. Create ``kabo/task/oer.py`` with an ``OERTask`` class decorated
       by ``@register_task``.
    2. Add an ``from kabo.task.oer import OERTask`` line below so the
       task is imported (and therefore registered) on package load.
    3. Optionally export the class in ``__all__``.

No changes to the engine, orchestrator, or CLI are required.
"""

from kabo.task.base import (
    TASK_REGISTRY,
    TaskBase,
    get_task,
    register_task,
)

# Importing each concrete task triggers its @register_task decorator.
from kabo.task.co2rr import CO2RRTask
from kabo.task.eco2rr import ECO2RRTask
from kabo.task.test_task import TestTask

__all__ = [
    "TaskBase",
    "TASK_REGISTRY",
    "register_task",
    "get_task",
    "CO2RRTask",
    "ECO2RRTask",
    "TestTask",
]
