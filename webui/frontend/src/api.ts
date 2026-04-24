/**
 * Thin REST client around the FastAPI backend.
 *
 * All endpoints are relative paths so the Vite dev server proxy
 * (`/api` → `http://127.0.0.1:8000`) and production single-origin
 * deployment both work without configuration.
 */

import type {
  ArchivedRun,
  FileEntry,
  ProjectSpec,
  ProjectsListResponse,
  RunConfig,
  StatusResponse,
  TaskSchema,
} from "./types";

async function jsonFetch<T>(
  url: string,
  init?: RequestInit,
): Promise<T> {
  const resp = await fetch(url, {
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    ...init,
  });
  if (!resp.ok) {
    let message = resp.statusText;
    try {
      const body = await resp.json();
      if (body?.detail) message = body.detail;
    } catch {
      // ignore JSON parse errors on non-JSON error bodies
    }
    throw new Error(`${resp.status} ${message}`);
  }
  if (resp.status === 204) return undefined as unknown as T;
  return (await resp.json()) as T;
}

// ---------------------------------------------------------------------------
// Tasks
// ---------------------------------------------------------------------------
export async function listTasks(): Promise<{ tasks: TaskSchema[] }> {
  return jsonFetch("/api/tasks");
}

export async function getTaskSchema(name: string): Promise<TaskSchema> {
  return jsonFetch(`/api/tasks/${encodeURIComponent(name)}`);
}

// ---------------------------------------------------------------------------
// Runs
// ---------------------------------------------------------------------------
export async function startRun(
  config: Partial<RunConfig>,
): Promise<StatusResponse> {
  return jsonFetch("/api/runs", {
    method: "POST",
    body: JSON.stringify(config),
  });
}

export async function currentRunStatus(): Promise<StatusResponse> {
  return jsonFetch("/api/runs/current");
}

export async function submitAnswer(answer: Record<string, unknown>): Promise<void> {
  await jsonFetch("/api/runs/current/answer", {
    method: "POST",
    body: JSON.stringify(answer),
  });
}

export async function abortRun(): Promise<void> {
  await jsonFetch("/api/runs/current/abort", { method: "POST" });
}

export async function listRuns(): Promise<{ runs: ArchivedRun[] }> {
  return jsonFetch("/api/runs");
}

export async function getRun(runId: string): Promise<ArchivedRun> {
  return jsonFetch(`/api/runs/${encodeURIComponent(runId)}`);
}

export async function deleteRun(runId: string): Promise<void> {
  await jsonFetch(`/api/runs/${encodeURIComponent(runId)}`, { method: "DELETE" });
}

export async function getRunData(runId: string): Promise<{ content: string }> {
  return jsonFetch(`/api/runs/${encodeURIComponent(runId)}/data`);
}

export async function getRunMetadata(runId: string): Promise<Record<string, unknown>> {
  return jsonFetch(`/api/runs/${encodeURIComponent(runId)}/metadata`);
}

export function runFileUrl(runId: string, name: string): string {
  return `/api/runs/${encodeURIComponent(runId)}/file/${encodeURIComponent(name)}`;
}

// ---------------------------------------------------------------------------
// Files
// ---------------------------------------------------------------------------
export async function listDataFiles(): Promise<{ files: FileEntry[] }> {
  return jsonFetch("/api/files/data");
}

export async function readDataFile(name: string): Promise<{ content: string }> {
  return jsonFetch(`/api/files/data/${encodeURIComponent(name)}`);
}

export async function writeDataFile(
  name: string, content: string,
): Promise<{ size: number }> {
  return jsonFetch(`/api/files/data/${encodeURIComponent(name)}`, {
    method: "PUT",
    body: JSON.stringify({ content }),
  });
}

export async function deleteDataFile(name: string): Promise<void> {
  await jsonFetch(`/api/files/data/${encodeURIComponent(name)}`, {
    method: "DELETE",
  });
}

export async function listPriorFiles(): Promise<{ files: FileEntry[] }> {
  return jsonFetch("/api/files/priors");
}

export async function readPriorFile(name: string): Promise<{
  name: string;
  content?: Record<string, unknown>;
  content_text?: string;
}> {
  return jsonFetch(`/api/files/priors/${encodeURIComponent(name)}`);
}

export async function writePriorFile(
  name: string, content: Record<string, unknown> | unknown[],
): Promise<{ size: number }> {
  return jsonFetch(`/api/files/priors/${encodeURIComponent(name)}`, {
    method: "PUT",
    body: JSON.stringify({ content }),
  });
}

export async function deletePriorFile(name: string): Promise<void> {
  await jsonFetch(`/api/files/priors/${encodeURIComponent(name)}`, {
    method: "DELETE",
  });
}

// ---------------------------------------------------------------------------
// Projects (declarative dynamic tasks)
// ---------------------------------------------------------------------------
export async function listProjects(): Promise<ProjectsListResponse> {
  return jsonFetch("/api/projects");
}

export async function getProject(name: string): Promise<ProjectSpec> {
  return jsonFetch(`/api/projects/${encodeURIComponent(name)}`);
}

export async function createProject(spec: ProjectSpec): Promise<{
  project: ProjectSpec;
  created: boolean;
}> {
  return jsonFetch("/api/projects", {
    method: "POST",
    body: JSON.stringify(spec),
  });
}

export async function updateProject(
  name: string,
  spec: ProjectSpec,
): Promise<{ project: ProjectSpec; updated: boolean }> {
  return jsonFetch(`/api/projects/${encodeURIComponent(name)}`, {
    method: "PUT",
    body: JSON.stringify(spec),
  });
}

export async function deleteProject(name: string): Promise<void> {
  await jsonFetch(`/api/projects/${encodeURIComponent(name)}`, {
    method: "DELETE",
  });
}
