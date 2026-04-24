# KABO Run Configurations

This folder collects **declarative configuration files** for reproducible
KABO runs.  They are consumed via:

```bash
python -m kabo --config configs/co2rr_base.yaml
```

Precedence (low → high):

1. Built-in argparse defaults
2. Config-file keys (this folder)
3. Any flag passed explicitly on the command line

Supported formats: **YAML** (`.yaml` / `.yml`), **TOML** (`.toml`),
**JSON** (`.json`).  YAML is recommended for readability; TOML/JSON
fall back automatically.

## Key naming

Use either the CLI flag form (`top-k`) or its Python-identifier form
(`top_k`).  Both resolve to the same argparse destination.

## Starter templates

| File | Purpose |
|---|---|
| `co2rr_base.yaml` | Canonical CO2RR UCB run with default priors |
| `testtask_smoke.yaml` | Minimal TestTask smoke configuration (CI / demo) |

To add a new template, copy an existing one and tweak keys — unknown
keys are ignored with a warning (they do not abort the run).
