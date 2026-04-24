"""Declarative YAML / TOML configuration support for the KABO CLI.

Motivation
----------
CLI invocations of KABO accumulate 20+ flags.  For reproducibility,
team-sharing and batch experimentation, it is more ergonomic to keep
the parameter set in a versioned file and override only a handful of
flags from the command line.

Workflow
--------
1. User prepares ``experiments/my_run.yaml`` (example below).
2. Runs ``python -m kabo --config experiments/my_run.yaml --seed 42``.
3. This module merges the three sources, with precedence::

       defaults  <  config file  <  explicit CLI flags

Only ``argparse`` defaults that the user did **not** override on the
command line are replaced by config-file values — this makes CLI the
ultimate authority and keeps the existing flow backward-compatible.

Example YAML
------------
::

    # experiments/co2rr_base.yaml
    task: co2rr
    target_product: CO
    data: data/data.csv
    candidates: data/candidates.csv
    iterations: 10
    top_k: 10
    beta: 2.5
    beta_schedule: theory
    acq_strategy: ucb
    seed: 42
    kabo_mode: true
    lambda_p: 1.0
    lambda_k: 0.5

Supported file formats
----------------------
* ``.yaml`` / ``.yml`` — parsed with ``PyYAML`` (optional dependency).
* ``.toml`` — parsed with the stdlib ``tomllib`` (Python ≥ 3.11).
* ``.json`` — parsed with ``json``.

If the extension is unrecognized, this module first tries YAML, then
TOML, then JSON, and reports the last parser error.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def load_config_file(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load a YAML / TOML / JSON configuration file.

    Parameters
    ----------
    path : str or PathLike
        Path to the config file.  Extension determines the parser; if
        it is unrecognized, several parsers are tried in turn.

    Returns
    -------
    dict
        Top-level mapping from config keys to values.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If parsing fails for every supported format.
    """
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")

    ext = p.suffix.lower()
    text = p.read_text(encoding="utf-8")

    # Dispatch on extension with automatic fall-through.
    parsers: list[tuple[str, Any]] = []
    if ext in {".yaml", ".yml"}:
        parsers = [("yaml", _try_yaml), ("toml", _try_toml), ("json", _try_json)]
    elif ext == ".toml":
        parsers = [("toml", _try_toml), ("yaml", _try_yaml), ("json", _try_json)]
    elif ext == ".json":
        parsers = [("json", _try_json), ("yaml", _try_yaml), ("toml", _try_toml)]
    else:
        parsers = [("yaml", _try_yaml), ("toml", _try_toml), ("json", _try_json)]

    last_err: Exception | None = None
    for label, fn in parsers:
        try:
            data = fn(text)
        except Exception as exc:
            last_err = exc
            continue
        if not isinstance(data, dict):
            last_err = ValueError(
                f"{label} parser produced a {type(data).__name__}, expected a mapping."
            )
            continue
        return data

    raise ValueError(
        f"Failed to parse {p} with any supported format "
        f"(yaml / toml / json). Last error: {last_err!r}"
    )


def merge_config_into_args(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    config: dict[str, Any],
) -> argparse.Namespace:
    """Return a new ``Namespace`` with config-file values folded in.

    Precedence (low → high):
        1. argparse defaults
        2. config file
        3. explicit CLI flags (anything the user typed)

    Detection rule: an argument is considered *explicit* if its current
    value differs from the parser-declared default.  This works because
    the KABO CLI does not use any ``None`` defaults ambiguously.

    Parameters
    ----------
    args : argparse.Namespace
        Result of ``parser.parse_args()``.
    parser : argparse.ArgumentParser
        The parser used to create ``args`` (needed to recover defaults).
    config : dict
        Parsed config-file mapping.

    Returns
    -------
    argparse.Namespace
        A namespace in which non-explicit attributes have been replaced
        by config-file values where available, and CLI-explicit values
        are preserved.
    """
    defaults: dict[str, Any] = {
        a.dest: a.default for a in parser._actions if a.dest != "help"
    }
    merged = argparse.Namespace(**vars(args))

    unknown_keys: list[str] = []
    applied: list[str] = []
    for key, value in config.items():
        # Normalise hyphen/underscore variants ("top-k" <-> "top_k").
        dest = key.replace("-", "_")
        if dest not in defaults:
            unknown_keys.append(key)
            continue
        current = getattr(merged, dest, defaults[dest])
        if current == defaults[dest]:
            # Not explicitly set on CLI → accept config value.
            setattr(merged, dest, value)
            applied.append(dest)
        else:
            # CLI already overrode it; keep CLI.
            pass

    if applied:
        logger.info("Config file applied %d parameters: %s",
                    len(applied), ", ".join(sorted(applied)))
    if unknown_keys:
        logger.warning(
            "Config file contained %d unknown keys (ignored): %s",
            len(unknown_keys), ", ".join(sorted(unknown_keys)),
        )
    return merged


# ---------------------------------------------------------------------------
# Parser shims (kept private so callers import only load_config_file)
# ---------------------------------------------------------------------------
def _try_yaml(text: str) -> Any:
    try:
        import yaml  # noqa: WPS433 — optional import
    except ImportError as exc:  # pragma: no cover - env-specific
        raise ImportError(
            "PyYAML is required to parse .yaml / .yml config files. "
            "Install with: pip install pyyaml"
        ) from exc
    return yaml.safe_load(text)


def _try_toml(text: str) -> Any:
    # tomllib is stdlib from Python 3.11+.
    try:
        import tomllib  # type: ignore[import-not-found]
    except ModuleNotFoundError:  # pragma: no cover - python 3.10 fallback
        import tomli as tomllib  # type: ignore[import-not-found]
    return tomllib.loads(text)


def _try_json(text: str) -> Any:
    return json.loads(text)
