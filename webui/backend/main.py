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
async def start_run(req: StartRunRequest) -> StatusResponse:
    cfg = RunConfig(**req.model_dump())
    try:
        runner = sessions.start(cfg)
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
