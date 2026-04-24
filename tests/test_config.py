"""Unit tests for ``kabo.config``: YAML/TOML/JSON loading and merge precedence.

The merge test relies on building the CLI parser itself, which in turn
triggers ``from kabo.task import TASK_REGISTRY``.  That import path is
torch-free thanks to the lazy ``utils`` module, so these tests run even
in a minimal environment.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kabo.config import load_config_file, merge_config_into_args


# ---------------------------------------------------------------------------
# load_config_file
# ---------------------------------------------------------------------------
class TestLoadConfigFile:
    def test_loads_yaml(self, tmp_path: Path):
        p = tmp_path / "cfg.yaml"
        p.write_text("task: co2rr\niterations: 7\n")
        cfg = load_config_file(p)
        assert cfg == {"task": "co2rr", "iterations": 7}

    def test_loads_json(self, tmp_path: Path):
        p = tmp_path / "cfg.json"
        p.write_text(json.dumps({"task": "test", "seed": 5}))
        cfg = load_config_file(p)
        assert cfg == {"task": "test", "seed": 5}

    def test_loads_toml(self, tmp_path: Path):
        tomllib = pytest.importorskip("tomllib")  # Python 3.11+
        del tomllib
        p = tmp_path / "cfg.toml"
        p.write_text('task = "co2rr"\niterations = 3\n')
        cfg = load_config_file(p)
        assert cfg == {"task": "co2rr", "iterations": 3}

    def test_raises_on_missing_file(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_config_file(tmp_path / "absent.yaml")

    def test_unknown_extension_falls_back(self, tmp_path: Path):
        p = tmp_path / "cfg.txt"
        p.write_text("task: test\nseed: 1\n")  # valid YAML
        cfg = load_config_file(p)
        assert cfg["task"] == "test"


# ---------------------------------------------------------------------------
# merge_config_into_args
# ---------------------------------------------------------------------------
class TestMergeConfig:
    @pytest.fixture
    def parser(self):
        # Avoid importing the full cli (which imports optimizer → torch).
        import argparse

        p = argparse.ArgumentParser()
        p.add_argument("--task", default="co2rr")
        p.add_argument("--iterations", type=int, default=10)
        p.add_argument("--top-k", type=int, default=10)
        p.add_argument("--seed", type=int, default=None)
        p.add_argument("--non-interactive", action="store_true")
        return p

    def test_config_fills_in_when_cli_leaves_defaults(self, parser):
        args = parser.parse_args([])
        cfg = {"task": "test", "iterations": 3, "seed": 42}
        merged = merge_config_into_args(args, parser, cfg)
        assert merged.task == "test"
        assert merged.iterations == 3
        assert merged.seed == 42

    def test_cli_explicit_flags_win(self, parser):
        args = parser.parse_args(["--iterations", "99"])
        cfg = {"task": "test", "iterations": 3}
        merged = merge_config_into_args(args, parser, cfg)
        assert merged.iterations == 99, "CLI must beat the config value"
        assert merged.task == "test"

    def test_hyphen_underscore_key_aliasing(self, parser):
        args = parser.parse_args([])
        # User wrote top-k in YAML; should map to args.top_k
        cfg = {"top-k": 5}
        merged = merge_config_into_args(args, parser, cfg)
        assert merged.top_k == 5

    def test_unknown_keys_are_ignored(self, parser, caplog):
        args = parser.parse_args([])
        cfg = {"task": "test", "mystery_key": "x"}
        with caplog.at_level("WARNING"):
            merged = merge_config_into_args(args, parser, cfg)
        assert merged.task == "test"
        assert any("mystery_key" in msg for msg in caplog.messages)

    def test_boolean_flag_merge(self, parser):
        args = parser.parse_args([])
        cfg = {"non_interactive": True}
        merged = merge_config_into_args(args, parser, cfg)
        assert merged.non_interactive is True
