# Optimization projects

Every `.json` file in this directory declares a **project** — a
declarative `TaskBase` definition that the web UI loads at startup and
registers as a dynamic task (alongside built-ins like `co2rr` and
`test`). CLI users see exactly the built-in tasks; only the web UI
backend discovers these JSON files.

Create, edit, and delete projects from the **Projects** tab in the
web UI. The backend persists each project as a JSON file here; you
can also edit them by hand if you prefer — the backend re-validates
on the next startup.

## JSON schema (minimal example)

```json
{
  "name": "orr",
  "display_name": "Oxygen Reduction Reaction",
  "description": "ORR catalyst screening",
  "features": [
    { "name": "temperature", "type": "continuous", "lo": 25.0, "hi": 80.0, "unit": "°C" },
    { "name": "n_layers",   "type": "integer",    "lo": 1,    "hi": 10 }
  ],
  "targets": [
    { "short_name": "OH",  "column": "Y_OH",  "display_name": "Hydroxide" },
    { "short_name": "H2O2","column": "Y_H2O2","display_name": "Hydrogen peroxide", "is_competing": true }
  ],
  "default_target": "OH",
  "notes": ""
}
```

### Fields

- `name` — lowercase task identifier (alphanumerics, `_`, `-`). Must
  not collide with a built-in task.
- `features[]` — list of design-space features; each with `name`,
  `type` (`continuous` or `integer`), and bounds `lo < hi`.
- `targets[]` — list of product / yield columns. Set `is_competing:
  true` on side-reactions whose yield should be subtracted from the
  training target when `--h2-penalty-weight > 0` is supplied at run
  time.
- `default_target` — one of the `short_name` values above.
