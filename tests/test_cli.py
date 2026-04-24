"""CLI parser / config-merge regression tests.

These exercise the argparse layer without constructing the actual
optimizer, so they run in torch-free environments.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kabo.cli import build_parser, parse_args


class TestBuildParser:
    def test_contains_core_flags(self):
        parser = build_parser()
        dests = {a.dest for a in parser._actions}
        assert {
            "config",
            "task",
            "data",
            "iterations",
            "top_k",
            "q_batch",
            "max_stagnation",
            "stagnation_tol",
            "acq_strategy",
        }.issubset(dests)

    def test_defaults_are_stable(self):
        args = build_parser().parse_args([])
        assert args.q_batch == 1
        assert args.max_stagnation == 0
        assert args.stagnation_tol == pytest.approx(1e-4)
        assert args.acq_strategy == "ucb"

    def test_q_batch_parses_to_int(self):
        args = build_parser().parse_args(["--q-batch", "4"])
        assert args.q_batch == 4

    def test_max_stagnation_parses_to_int(self):
        args = build_parser().parse_args(
            ["--max-stagnation", "3", "--stagnation-tol", "1e-3"],
        )
        assert args.max_stagnation == 3
        assert args.stagnation_tol == pytest.approx(1e-3)


class TestParseArgsConfigMerge:
    def test_config_file_fills_defaults(self, tmp_path: Path):
        cfg = tmp_path / "run.yaml"
        cfg.write_text(
            "task: test\niterations: 5\ntop_k: 3\nq_batch: 2\n"
        )
        args = parse_args(["--config", str(cfg)])
        assert args.task == "test"
        assert args.iterations == 5
        assert args.top_k == 3
        assert args.q_batch == 2

    def test_cli_overrides_config(self, tmp_path: Path):
        cfg = tmp_path / "run.yaml"
        cfg.write_text("iterations: 5\nq_batch: 2\n")
        args = parse_args(["--config", str(cfg), "--iterations", "77"])
        assert args.iterations == 77
        assert args.q_batch == 2
