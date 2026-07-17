# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

KABOCS — Knowledge-Augmented Bayesian Optimization for Catalytic Systems. A BoTorch/GPyTorch BO pipeline for catalysis experiments (CO₂RR and friends), driven either from a CLI or a React + FastAPI web console.

`README.md` (Chinese) is the authoritative reference for the science, CLI flags, and v1.2 feature mechanics (multi-objective qNEHVI, SVGP, expert priors, feature-selection methods, PE queries). Read it before changing algorithm behavior rather than re-deriving intent from code.

## Commands

```bash
# Tests — the suite auto-skips the torch tier when torch/botorch/gpytorch are absent
pytest -q                                    # whole suite
pytest tests/test_config.py -q               # one file
pytest tests/test_config.py::test_name -q    # one test
pytest -m "not requires_torch" -q            # pure tier only
pytest --cov=kabo --cov-report=term-missing  # coverage (CI's full lane)

ruff check kabo tests                        # lint (non-gating in CI)

pip install -e ".[dev]"                      # dev deps
pip install -r requirements.txt              # full torch stack

# Run the optimizer (run.py, python -m kabo, and the `kabo` script are equivalent)
python -m kabo --config configs/testtask_smoke.yaml --non-interactive
python -m kabo --task co2rr --iterations 20 --seed 42

# WebUI
pip install -r webui/requirements.txt
(cd webui/frontend && npm install && npm run build)   # → webui/frontend/dist
python webui/run_webui.py                             # http://127.0.0.1:8000
(cd webui/frontend && npm run dev)                    # hot-reload on :5173, proxies /api → :8000
```

## Environment gotchas

- **`data/` is gitignored and not present in a fresh clone.** Every README example referencing `data/data.csv` or `data/test_data.csv` will fail until you supply the CSVs. Tests that need them (`test_data_csv` / `test_candidates_csv` fixtures in `tests/conftest.py`) skip rather than fail — a green run does not mean the data-dependent paths were exercised.
- **torch is often not installed locally.** ~38 of 109 tests skip in that state. Don't read those skips as breakage, and don't assume an integration path works because `pytest -q` was green.

## Architecture

Three layers, deliberately decoupled — the split is the main thing to preserve:

- **`kabo/engine.py` · `KABOEngine`** — system-agnostic algorithm core, owning `SurrogateModel` (GP), `PreferenceModel` (PairwiseGP), and `ExpertPrior`. **No domain keywords (product names, feature schemas, catalysis constants) may appear at this layer.**
- **`kabo/task/base.py` · `TaskBase`** — the domain layer. One file per catalytic system, self-registering via `@register_task` into `TASK_REGISTRY`; resolved by lowercased name through `get_task(name)`. Registration happens as an import side effect of `kabo.task`, so a new task must be imported in `kabo/task/__init__.py` to exist.
- **`kabo/optimizer.py` · `KABOOptimizer`** — the orchestrator that composes the two into `phase1_feature_selection` → `phase2_fit_surrogate` → `phase3_optimize`. The BO loop policy (interactivity, what to observe) lives here, not in the engine.

Adding a catalytic system means adding one `TaskBase` subclass plus one import line — `engine.py`, `optimizer.py`, `cli.py`, `acquisition.py`, and `feature_selection.py` should not need edits. `kabo/task/test_task.py` is the minimal template; `kabo/task/co2rr.py` (photocatalytic), `kabo/task/eco2rr.py` (electrocatalytic), and `kabo/task/peptide_eco2rr.py` (peptide-ligated electrocatalytic) are the full ones.

### Choosing a feature encoding — the decision that constrains everything

This is the first design question for a new Task, and it is very expensive to get wrong, because it decides what BO is *capable of proposing* — not merely how well it fits.

**Categorical identity cannot extrapolate, ever.** `CategoricalKernel` is Hamming-style: it only asks "same or different", so there is no metric between levels. `design_space_bounds` is `(0, n-1)` over the levels already declared, and `generate_candidates()` can only enumerate those same levels. A member of the family that was never measured has no coordinate, so BO can never suggest it. **No amount of extra data fixes this** — it is a property of the encoding.

**Continuous descriptors can extrapolate, at a cost.** Give each member physicochemical coordinates and the GP can speak about unseen ones. `CO2RRTask` does this for amino acids (`A_pI`, `A_hbond_*`); `PeptideECO2RRTask` makes it the entire ligand representation, with bounds derived from all 20 canonical residues (`kabo.constants.aa_descriptor_bounds()`) so an untested residue provably stays in-bounds rather than being silently clipped onto a neighbour. The cost: `optimize_acqf` relaxes those dims to a continuous box and returns *chimeras* — descriptor vectors matching no real member (e.g. `Ligand_hbond_donors = 1.1424`). So the discrete pool from `generate_candidates()` is the load-bearing path, and `PeptideECO2RRTask.nearest_residue()` exists to decode a continuous proposal (and to report the distance, i.e. how badly it lied).

Rule of thumb: **categorical when the levels are the whole world** (three cell types), **descriptors when you want to generalize beyond the tested set** (a new residue, a new metal). Descriptors also need enough distinct members to be identifiable — with 4 ligands, 6 descriptors span at most rank 3 after centering and carry no signal past the trend of the other features; verify with leave-one-member-out against a control that omits the descriptors entirely, not against a mean-predictor baseline.

If you do declare `"categorical"` (only `ECO2RRTask` today), three engine behaviours follow: the surrogate switches to `MixedSingleTaskGP`, `--gp-model auto` can no longer upgrade to SVGP (no Mixed/variational equivalent), and the categories must be ordinal-encoded (`0…n-1`) with `design_space_bounds` of `(0, n-1)` — nothing in `kabo/` consumes `categorical_values()`, it is metadata for the Task and the WebUI only.

`kabo/interaction.py` holds CLI prompt/print helpers; `kabo/acquisition.py` is kept to pure acquisition math. Keep that boundary — it was an explicit refactor (see `improve.md`).

### Lazy-import invariant (CI-enforced)

`kabo/__init__.py` resolves every public symbol through a PEP 562 `__getattr__`, and `kabo.utils` imports `torch` inside function bodies. CI's fast lane asserts `'torch' not in sys.modules` after importing `kabo`, `kabo.utils`, `kabo.candidate`, `kabo.feature_selection`, `kabo.task`, `kabo.config`, and `kabo.cli`. **Adding a module-level torch import to any of those breaks CI**, even though the tests still pass.

### Config precedence

`kabo/config.py` merges `argparse defaults < config file < explicit CLI flags`. "Explicit" is detected by comparing the parsed value against the parser default — so a CLI flag passed with a value equal to its default is indistinguishable from omission and will lose to the config file. Samples live in `configs/`.

### WebUI bridge

`webui/backend/ui_bridge.py` drives the optimizer **without modifying `kabo/`**, by monkey-patching at run time inside a worker thread: the `prompt_user_*` / `print_*` names on `kabo.optimizer`, `task.prompt_observation` (bound method), `builtins.input`, `sys.stdout`, and the engine's `build_acquisition` / `build_mo_acquisition` (to cache the acq func for visualization). Install/uninstall are paired in `SessionRunner._worker()`'s `try/finally`. Preserve that pairing and the "no edits to `kabo/`" property when touching this file.

`webui/backend/projects.py` synthesizes `TaskBase` subclasses at startup from declarative `projects/*.json` (`register_all()`), so users can define a system without writing Python. These are visible to the CLI only if you call `register_all()` yourself. Built-in task names (`co2rr`, `eco2rr`, `peptide`, `test`) are protected against collision — the guard reads `TASK_REGISTRY` at call time, so a newly added built-in is covered automatically. Note `projects/*.json` only supports `continuous` / `integer` feature types, so a system needing categorical or descriptor-derived features has to be a real Task in `kabo/task/`.

Note `webui/README.md` still documents single-session as a limitation; v1.2 replaced the `SessionManager` singleton with a `run_id`-keyed registry and `allow_concurrent` on `POST /api/runs`. Trust `webui/backend/runner.py` over that section.

## Conventions

- ruff, line-length 100, target py310, `select = E,W,F,I,B,UP`. Lint does not gate merges.
- `co2rr_bo/` is a deprecation shim over `kabo`; don't build on it.
- Prefer `kabo/constants.py` for CO2RR-domain constants, read by `CO2RRTask` — not by the engine.
