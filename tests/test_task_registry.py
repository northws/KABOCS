"""Unit tests for ``kabo.task`` registry semantics.

These only touch ``TaskBase`` + ``register_task`` + ``get_task`` — they
skip gracefully when ``kabo.task`` (re-exported through ``kabo``) cannot
be imported because of missing optional deps like torch.
"""

from __future__ import annotations

import pytest

# TaskBase itself does not need torch — but the ``kabo.task`` __init__ imports
# CO2RR / TestTask, whose definitions are torch-free.  Import at module load.
from kabo.task import TASK_REGISTRY, TaskBase, TestTask, get_task


class TestRegistry:
    def test_builtin_tasks_registered(self):
        keys = set(TASK_REGISTRY)
        assert {"co2rr", "eco2rr", "test"}.issubset(keys)

    def test_get_task_is_case_insensitive(self):
        a = get_task("test")
        b = get_task("TEST")
        c = get_task("Test")
        assert type(a) is type(b) is type(c) is TestTask

    def test_get_task_raises_on_unknown(self):
        with pytest.raises(KeyError, match="Unknown task"):
            get_task("does_not_exist_42")


class TestTestTaskContract:
    """The TestTask is the canonical smoke fixture — its contract matters."""

    @pytest.fixture
    def task(self) -> TestTask:
        return TestTask()

    def test_task_name(self, task):
        assert task.task_name() == "test"

    def test_feature_schema_is_consistent(self, task):
        feats = task.feature_columns()
        bounds = task.design_space_bounds()
        assert set(feats) == set(bounds)
        for name, (lo, hi) in bounds.items():
            assert lo < hi, f"bounds for {name!r} must be strictly increasing"

    def test_default_target_resolves_to_column(self, task):
        target = task.default_target()
        col = task.resolve_target_column(target)
        assert col in task.target_columns().values()

    def test_simulate_observation_returns_mapping(self, task):
        out = task.simulate_observation(target_column="y", y_mean=5.0, y_std=1.0)
        assert isinstance(out, dict)
        assert "y" in out
        assert isinstance(out["y"], float)

    def test_is_subclass_of_taskbase(self, task):
        assert isinstance(task, TaskBase)
