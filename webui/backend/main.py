"""
KABOCS WebUI — FastAPI application.

Run with::

    python -m webui.backend.main          # default :8000
    uvicorn webui.backend.main:app --reload

Serves:
  * REST endpoints for task discovery, file management, and run control.
  * Server-Sent Events (SSE) streaming for live logs, recommendations,
    and interactive prompts.
  * A static build of the React frontend (``webui/frontend/dist``)
    when present.
"""

from __future__ import annotations

import asyncio
import json
import mimetypes
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles

# Important: import the kabo package so task registry is populated.
import kabo.task as _kabo_task  # noqa: F401

from webui.backend import projects as _projects
from webui.backend.event_hub import EventHub
from webui.backend.runner import (
    RunConfig,
    SessionManager,
    delete_archived_run,
    list_archived_runs,
    PROJECT_ROOT,
    RUNS_ROOT,
)
from webui.backend.schemas import (
    AnswerRequest,
    SaveFileRequest,
    SaveJsonFileRequest,
    StartRunRequest,
    StatusResponse,
)

# Snapshot built-in task names before we register any dynamic projects so
# CRUD endpoints can reliably detect collisions.
_projects._snapshot_builtins()
_projects.register_all()


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
PRIORS_DIR = PROJECT_ROOT / "priors"
FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"
FRONTEND_PUBLIC = Path(__file__).resolve().parent.parent / "frontend" / "public"


# ---------------------------------------------------------------------------
# App & global singletons
# ---------------------------------------------------------------------------
app = FastAPI(
    title="KABOCS WebUI",
    description=(
        "Browser-based control panel for the Knowledge-Augmented Bayesian "
        "Optimization pipeline."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

hub = EventHub()
sessions = SessionManager(hub)


@app.on_event("startup")
async def _startup() -> None:
    await hub.start()
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)


@app.on_event("shutdown")
async def _shutdown() -> None:
    await hub.stop()


# ---------------------------------------------------------------------------
# Task discovery
# ---------------------------------------------------------------------------
@app.get("/api/tasks")
async def list_tasks() -> dict:
    """Return every registered task with its schema (features/bounds/products)."""
    from kabo.task import TASK_REGISTRY

    dynamic = set(_projects.list_dynamic_names())
    tasks_out: list[dict] = []
    for name, cls in TASK_REGISTRY.items():
        inst = cls()
        tasks_out.append({
            "name": name,
            "display_name": inst.task_name(),
            "features": inst.feature_columns(),
            "feature_types": inst.feature_types(),
            "categorical_values": inst.categorical_values(),
            "design_bounds": {
                k: list(v) for k, v in inst.design_space_bounds().items()
            },
            "target_columns": inst.target_columns(),
            "all_product_columns": inst.all_product_columns(),
            "product_names": inst.product_names(),
            "default_target": inst.default_target(),
            "source": "project" if name in dynamic else "builtin",
        })
    return {"tasks": tasks_out}


@app.get("/api/tasks/{name}")
async def get_task_schema(name: str) -> dict:
    from kabo.task import TASK_REGISTRY
    if name.lower() not in TASK_REGISTRY:
        raise HTTPException(404, f"Unknown task '{name}'")
    inst = TASK_REGISTRY[name.lower()]()
    return {
        "name": name.lower(),
        "display_name": inst.task_name(),
        "features": inst.feature_columns(),
        "feature_types": inst.feature_types(),
        "categorical_values": inst.categorical_values(),
        "design_bounds": {k: list(v) for k, v in inst.design_space_bounds().items()},
        "target_columns": inst.target_columns(),
        "all_product_columns": inst.all_product_columns(),
        "product_names": inst.product_names(),
        "default_target": inst.default_target(),
    }


# ---------------------------------------------------------------------------
# Project (dynamic task) management
# ---------------------------------------------------------------------------
@app.get("/api/projects")
async def list_projects() -> dict:
    """List every declarative project (JSON-defined dynamic task)."""
    return {
        "projects": [s.model_dump() for s in _projects.list_specs()],
        "builtins": sorted(_projects._BUILTIN_TASKS),
    }


@app.get("/api/projects/{name}")
async def get_project(name: str) -> dict:
    try:
        spec = _projects.load_spec(name)
    except FileNotFoundError:
        raise HTTPException(404, f"Project '{name}' not found")
    return spec.model_dump()


@app.post("/api/projects")
async def create_project(spec: _projects.ProjectSpec) -> dict:
    """Create a new project. Fails if the name collides with a built-in
    task or an existing project."""
    if _projects.is_builtin(spec.name):
        raise HTTPException(
            409,
            f"Name '{spec.name}' is reserved by a built-in task.",
        )
    if _projects._project_path(spec.name).exists():
        raise HTTPException(
            409, f"Project '{spec.name}' already exists; use PUT to update."
        )
    _projects.save_spec(spec)
    try:
        _projects.register_spec(spec)
    except Exception as exc:
        raise HTTPException(400, f"Registration failed: {exc}") from exc
    return {"project": spec.model_dump(), "created": True}


@app.put("/api/projects/{name}")
async def update_project(name: str, spec: _projects.ProjectSpec) -> dict:
    """Update an existing project. The path-segment name must match the
    spec's ``name`` field (projects cannot be renamed in-place — delete
    and recreate to rename)."""
    if name.lower() != spec.name:
        raise HTTPException(
            400,
            f"URL name '{name}' does not match body name '{spec.name}'.",
        )
    if _projects.is_builtin(spec.name):
        raise HTTPException(
            409, f"Name '{spec.name}' is reserved by a built-in task.",
        )
    if not _projects._project_path(spec.name).exists():
        raise HTTPException(404, f"Project '{spec.name}' not found.")
    # Refuse to rewrite the spec of a task that is currently running.
    runner = sessions.current
    if (
        runner is not None
        and runner.status in ("running", "pending")
        and runner.config.task.lower() == spec.name
    ):
        raise HTTPException(
            409,
            "Cannot edit a project while one of its runs is active.",
        )
    _projects.save_spec(spec)
    try:
        _projects.register_spec(spec)  # idempotent overwrite
    except Exception as exc:
        raise HTTPException(400, f"Registration failed: {exc}") from exc
    return {"project": spec.model_dump(), "updated": True}


@app.delete("/api/projects/{name}")
async def delete_project(name: str) -> dict:
    key = name.lower()
    if _projects.is_builtin(key):
        raise HTTPException(409, f"Cannot delete built-in task '{key}'.")
    runner = sessions.current
    if (
        runner is not None
        and runner.status in ("running", "pending")
        and runner.config.task.lower() == key
    ):
        raise HTTPException(
            409, "Cannot delete a project while one of its runs is active.",
        )
    _projects.delete_spec(key)
    _projects.unregister_spec(key)
    return {"ok": True, "deleted": key}


# ---------------------------------------------------------------------------
# Data management
# ---------------------------------------------------------------------------
def _ensure_within(base: Path, target: Path) -> Path:
    """Reject path traversal attempts (``..`` or absolute paths)."""
    base_resolved = base.resolve()
    target_resolved = (base / target).resolve()
    if base_resolved not in target_resolved.parents and target_resolved != base_resolved:
        raise HTTPException(400, "Path escapes allowed directory.")
    return target_resolved


def _list_dir(directory: Path, pattern: str = "*") -> list[dict]:
    if not directory.exists():
        return []
    items = []
    for p in sorted(directory.glob(pattern)):
        if p.is_file():
            stat = p.stat()
            items.append({
                "name": p.name,
                "path": str(p.relative_to(PROJECT_ROOT)),
                "size": stat.st_size,
                "mtime": stat.st_mtime,
            })
    return items


@app.get("/api/files/data")
async def list_data_files() -> dict:
    return {"files": _list_dir(DATA_DIR, "*.csv")}


@app.get("/api/files/data/{name}")
async def read_data_file(name: str, raw: bool = False) -> Any:
    target = _ensure_within(DATA_DIR, Path(name))
    if not target.exists():
        raise HTTPException(404, "File not found")
    if raw:
        return FileResponse(target, media_type="text/csv", filename=name)
    content = target.read_text(encoding="utf-8")
    return {"name": name, "content": content}


@app.put("/api/files/data/{name}")
async def write_data_file(name: str, payload: SaveFileRequest) -> dict:
    target = _ensure_within(DATA_DIR, Path(name))
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    target.write_text(payload.content, encoding="utf-8")
    return {"name": name, "size": target.stat().st_size}


@app.delete("/api/files/data/{name}")
async def delete_data_file(name: str) -> dict:
    target = _ensure_within(DATA_DIR, Path(name))
    if target.exists():
        target.unlink()
    return {"ok": True}


@app.get("/api/files/priors")
async def list_prior_files() -> dict:
    flat: list[dict] = []
    if PRIORS_DIR.exists():
        for p in sorted(PRIORS_DIR.rglob("*.json")):
            stat = p.stat()
            flat.append({
                "name": p.name,
                "path": str(p.relative_to(PROJECT_ROOT)),
                "size": stat.st_size,
                "mtime": stat.st_mtime,
            })
    return {"files": flat}


@app.get("/api/files/priors/{name:path}")
async def read_prior_file(name: str) -> dict:
    target = _ensure_within(PRIORS_DIR, Path(name))
    if not target.exists():
        raise HTTPException(404, "File not found")
    try:
        return {"name": name, "content": json.loads(target.read_text())}
    except Exception:
        return {"name": name, "content_text": target.read_text()}


@app.put("/api/files/priors/{name:path}")
async def write_prior_file(name: str, payload: SaveJsonFileRequest) -> dict:
    target = _ensure_within(PRIORS_DIR, Path(name))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload.content, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return {"name": name, "size": target.stat().st_size}


@app.delete("/api/files/priors/{name:path}")
async def delete_prior_file(name: str) -> dict:
    target = _ensure_within(PRIORS_DIR, Path(name))
    if target.exists():
        target.unlink()
    return {"ok": True}


# ---------------------------------------------------------------------------
# Run control
# ---------------------------------------------------------------------------
@app.post("/api/runs", response_model=StatusResponse)
async def start_run(
    req: StartRunRequest,
    allow_concurrent: bool = False,
) -> StatusResponse:
    """Start a new BO run.

    Pass ``?allow_concurrent=true`` (v1.2) to allow starting a new run
    while an older one is still running.  The default is single-active
    to preserve v1.1 semantics — concurrent runs share the global
    interaction bridge, so callers are expected to keep them in
    ``--non-interactive`` mode.
    """
    cfg = RunConfig(**req.model_dump())
    try:
        runner = sessions.start(cfg, allow_concurrent=allow_concurrent)
    except RuntimeError as exc:
        raise HTTPException(409, str(exc))
    return StatusResponse(
        status=runner.status,
        run_id=runner.run_id,
    )


@app.get("/api/runs/current", response_model=StatusResponse)
async def current_run_status() -> StatusResponse:
    runner = sessions.current
    if runner is None:
        return StatusResponse(status="idle")
    snap = runner.snapshot()
    return StatusResponse(
        status=snap["status"],
        run_id=snap["run_id"],
        pending_prompt=snap.get("pending_prompt"),
        error=snap.get("error"),
    )


@app.post("/api/runs/current/answer")
async def submit_answer(req: AnswerRequest) -> dict:
    runner = sessions.current
    if runner is None:
        raise HTTPException(404, "No active run")
    pending = runner.bridge.get_pending_prompt()
    if pending is None:
        raise HTTPException(409, "No prompt awaiting answer")
    accepted = runner.bridge.submit_answer(
        req.model_dump(exclude_none=True),
        prompt_id=req.prompt_id,
    )
    if not accepted:
        raise HTTPException(409, "Answer rejected (stale prompt_id)")
    return {"ok": True}


@app.post("/api/runs/current/abort")
async def abort_run() -> dict:
    ok = sessions.abort()
    if not ok:
        raise HTTPException(404, "No active run")
    return {"ok": True}


# ---------------------------------------------------------------------------
# Multi-session registry (v1.2) — id-addressed run management
# ---------------------------------------------------------------------------
# These endpoints expose the same sessions that ``/api/runs/current`` sees,
# but addressed by ``run_id`` so the UI can track multiple concurrent
# (non-interactive) runs.  Historical, on-disk archives remain under the
# legacy ``/api/runs/{run_id}`` read-only endpoints.
@app.get("/api/sessions")
async def list_sessions() -> dict:
    """Live session registry (not archived runs on disk)."""
    return {"sessions": sessions.list_snapshots()}


@app.get("/api/sessions/{run_id}", response_model=StatusResponse)
async def get_session(run_id: str) -> StatusResponse:
    runner = sessions.get(run_id)
    if runner is None:
        raise HTTPException(404, "Session not found")
    snap = runner.snapshot()
    return StatusResponse(
        status=snap["status"],
        run_id=snap["run_id"],
        pending_prompt=snap.get("pending_prompt"),
        error=snap.get("error"),
    )


@app.post("/api/sessions/{run_id}/answer")
async def submit_session_answer(run_id: str, req: AnswerRequest) -> dict:
    runner = sessions.get(run_id)
    if runner is None:
        raise HTTPException(404, "Session not found")
    pending = runner.bridge.get_pending_prompt()
    if pending is None:
        raise HTTPException(409, "No prompt awaiting answer")
    accepted = runner.bridge.submit_answer(
        req.model_dump(exclude_none=True),
        prompt_id=req.prompt_id,
    )
    if not accepted:
        raise HTTPException(409, "Answer rejected (stale prompt_id)")
    return {"ok": True}


@app.post("/api/sessions/{run_id}/abort")
async def abort_session(run_id: str) -> dict:
    ok = sessions.abort(run_id)
    if not ok:
        raise HTTPException(404, "Session not found")
    return {"ok": True}


@app.delete("/api/sessions/{run_id}")
async def remove_session(run_id: str) -> dict:
    """Drop a terminal session from the live registry (does NOT delete
    the archived run directory — use ``DELETE /api/runs/{run_id}`` for that)."""
    removed = sessions.remove(run_id)
    if not removed:
        raise HTTPException(
            409,
            "Session cannot be removed (not found or still active)",
        )
    return {"ok": True}


@app.get("/api/runs/current/events")
async def stream_events(request: Request) -> StreamingResponse:
    """SSE stream: live logs, recommendations and prompts."""
    queue_ = await hub.subscribe()

    async def generator():
        try:
            # Send an initial heartbeat so browsers detect the connection fast
            yield "event: ready\ndata: {}\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(queue_.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    # keep-alive comment (prevents proxies from closing)
                    yield ": ping\n\n"
                    continue
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        finally:
            hub.unsubscribe(queue_)

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Historical runs
# ---------------------------------------------------------------------------
@app.get("/api/runs")
async def list_runs() -> dict:
    return {"runs": list_archived_runs()}


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str) -> dict:
    runs = {r["run_id"]: r for r in list_archived_runs()}
    if run_id not in runs:
        raise HTTPException(404, "Run not found")
    return runs[run_id]


@app.delete("/api/runs/{run_id}")
async def delete_run(run_id: str) -> dict:
    if not delete_archived_run(run_id):
        raise HTTPException(404, "Run not found")
    return {"ok": True}


@app.get("/api/runs/{run_id}/data")
async def run_data(run_id: str) -> dict:
    target = (RUNS_ROOT / run_id / "data_updated.csv").resolve()
    if not target.exists():
        raise HTTPException(404, "data_updated.csv not found")
    return {"name": "data_updated.csv", "content": target.read_text(encoding="utf-8")}


@app.get("/api/runs/{run_id}/metadata")
async def run_metadata(run_id: str) -> dict:
    target = (RUNS_ROOT / run_id / "run_metadata.json").resolve()
    if not target.exists():
        raise HTTPException(404, "run_metadata.json not found")
    return json.loads(target.read_text())


@app.get("/api/runs/{run_id}/file/{name:path}")
async def run_file(run_id: str, name: str) -> FileResponse:
    base = RUNS_ROOT / run_id
    target = (base / name).resolve()
    if base.resolve() not in target.parents and target != base.resolve():
        raise HTTPException(400, "Path escapes allowed directory")
    if not target.exists():
        raise HTTPException(404, "File not found")
    mimetype, _ = mimetypes.guess_type(target.name)
    return FileResponse(target, media_type=mimetype or "application/octet-stream")


# ---------------------------------------------------------------------------
# Static frontend (built React bundle)
# ---------------------------------------------------------------------------
if FRONTEND_DIST.exists():
    # Mount assets under /assets to match Vite's default build output layout.
    assets_dir = FRONTEND_DIST / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/{full_path:path}")
    async def spa_fallback(full_path: str) -> Any:
        # Serve static file if it exists, otherwise the SPA index.
        candidate = FRONTEND_DIST / full_path
        if full_path and candidate.exists() and candidate.is_file():
            return FileResponse(candidate)
        index_file = FRONTEND_DIST / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return HTMLResponse(_fallback_html(), status_code=200)
else:
    @app.get("/")
    async def _index_fallback() -> HTMLResponse:
        return HTMLResponse(_fallback_html(), status_code=200)


def _fallback_html() -> str:
    return """<!doctype html>
<html><head><meta charset="utf-8"><title>KABOCS WebUI</title>
<style>
  body { font-family: -apple-system, system-ui, sans-serif; max-width: 680px;
         margin: 4rem auto; padding: 0 1rem; color: #1f2937; line-height: 1.6; }
  code { background: #f3f4f6; padding: 2px 6px; border-radius: 4px; }
  pre  { background: #f3f4f6; padding: 1rem; border-radius: 6px; overflow: auto; }
  h1   { margin-bottom: 0.2rem; }
  .badge { display: inline-block; padding: 2px 10px; border-radius: 999px;
           background: #dbeafe; color: #1d4ed8; font-size: 0.85rem; }
</style></head><body>
<h1>KABOCS WebUI</h1>
<p class="badge">backend is running</p>
<p>The React frontend has not been built yet. From the project root run:</p>
<pre>cd webui/frontend
npm install
npm run build</pre>
<p>Or, during development, run <code>npm run dev</code> in <code>webui/frontend</code>
and visit the Vite dev server (default <code>http://localhost:5173</code>).
The Vite dev server is pre-configured to proxy <code>/api</code> to this FastAPI
server.</p>
<p>API docs: <a href="/docs">/docs</a>.</p>
</body></html>"""


# ---------------------------------------------------------------------------
# Dev entry
# ---------------------------------------------------------------------------
def main() -> None:
    import uvicorn
    uvicorn.run(
        "webui.backend.main:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
