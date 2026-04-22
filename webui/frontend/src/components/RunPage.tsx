import { useEffect, useMemo, useState } from "react";
import { CheckCircle2, CircleSlash, Play, StopCircle, Trophy } from "lucide-react";

import ConfigPanel from "./ConfigPanel";
import LogStream from "./LogStream";
import PromptPanel from "./PromptPanel";
import RecommendationList from "./RecommendationList";
import { useEventStream } from "../hooks/useEventStream";
import { abortRun, currentRunStatus, listTasks, startRun } from "../api";
import type {
  BestFoundEvent,
  KaboEvent,
  LogEvent,
  PromptEvent,
  RecommendationsEvent,
  RunConfig,
  StatusResponse,
  TaskSchema,
} from "../types";

const INITIAL_CONFIG: RunConfig = {
  task: "co2rr",
  data_path: "data/data.csv",
  candidates_path: "data/candidates.csv",
  target_product: null,
  top_k: 10,
  beta: 2.0,
  beta_schedule: "fixed",
  beta_delta: 0.1,
  acq_strategy: "ucb",
  qnei_mc_samples: 128,
  kernel_type: "matern",
  h2_penalty_weight: 0.0,
  skip_feature_selection: false,
  strict_training_schema: false,
  pre_fill_before_choice: false,
  seed: null,
  device: "auto",
  iterations: 10,
  interactive: true,
  kabo_mode: false,
  lambda_p: 1.0,
  lambda_k: 1.0,
  lambda_v: 0.0,
  expert_prior_file: null,
  diversity_weight: 0.5,
  pe_budget: 0,
  generate_candidates_n: 1000,
  prefer_file_candidates: false,
  discrete_strategy: "acq",
};

interface Props {
  onStatusChange?: (s: StatusResponse) => void;
}

export default function RunPage({ onStatusChange }: Props) {
  const [tasks, setTasks] = useState<TaskSchema[]>([]);
  const [config, setConfig] = useState<RunConfig>(INITIAL_CONFIG);
  const [status, setStatus] = useState<StatusResponse>({ status: "idle" });
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const { events, connected, clear: clearEvents } = useEventStream();

  // Load registered tasks once.
  useEffect(() => {
    listTasks()
      .then((r) => {
        setTasks(r.tasks);
        // Auto-select default task if the initial one isn't registered.
        if (!r.tasks.find((t) => t.name === config.task) && r.tasks.length > 0) {
          setConfig((c) => ({ ...c, task: r.tasks[0].name }));
        }
      })
      .catch((e) => setError(`Failed to load tasks: ${e}`));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Poll status (primary source of truth for the prompt modal).
  useEffect(() => {
    let alive = true;
    const tick = async () => {
      try {
        const s = await currentRunStatus();
        if (alive) {
          setStatus(s);
          onStatusChange?.(s);
        }
      } catch {
        /* ignore */
      }
    };
    tick();
    const id = window.setInterval(tick, 1500);
    return () => {
      alive = false;
      window.clearInterval(id);
    };
  }, [onStatusChange]);

  const runActive = status.status === "running" || status.status === "pending";
  const { logs, recommendations, best, lastLifecycle } = useDerived(events);

  async function handleStart() {
    setStarting(true);
    setError(null);
    try {
      clearEvents();
      const resp = await startRun(config);
      setStatus({ status: resp.status, run_id: resp.run_id });
    } catch (e) {
      setError(String(e));
    } finally {
      setStarting(false);
    }
  }

  async function handleAbort() {
    try {
      await abortRun();
    } catch (e) {
      setError(String(e));
    }
  }

  return (
    <div className="grid grid-cols-1 xl:grid-cols-[420px_1fr] gap-6">
      <section className="space-y-4">
        <Card title="Configuration">
          <ConfigPanel
            tasks={tasks}
            config={config}
            onChange={(patch) => setConfig((c) => ({ ...c, ...patch }))}
            disabled={runActive}
          />
        </Card>

        <Card>
          <div className="flex items-center gap-3">
            {runActive ? (
              <button
                onClick={handleAbort}
                className="inline-flex items-center gap-2 px-4 py-2 rounded-md border border-rose-300 bg-white hover:bg-rose-50 text-rose-700 text-sm"
              >
                <StopCircle className="w-4 h-4" /> Stop
              </button>
            ) : (
              <button
                onClick={handleStart}
                disabled={starting}
                className="inline-flex items-center gap-2 px-4 py-2 rounded-md bg-brand-600 text-white hover:bg-brand-700 text-sm disabled:opacity-60"
              >
                <Play className="w-4 h-4" />
                {starting ? "Starting…" : "Start run"}
              </button>
            )}
            <StatusLabel status={status} connected={connected} />
          </div>
          {error ? (
            <div className="mt-3 text-xs text-rose-700 bg-rose-50 border border-rose-200 p-2 rounded">
              {error}
            </div>
          ) : null}
          {status.error ? (
            <div className="mt-3 text-xs text-rose-700 bg-rose-50 border border-rose-200 p-2 rounded">
              Run error: <span className="font-mono">{status.error}</span>
            </div>
          ) : null}
        </Card>
      </section>

      <section className="space-y-4 min-w-0">
        <PromptPanel
          prompt={(status.pending_prompt as PromptEvent | null) ?? null}
          latestRecommendations={recommendations}
        />

        {best ? <BestCard best={best} /> : null}
        <RecommendationList event={recommendations} />
        <LogStream logs={logs} />

        {lastLifecycle ? (
          <LifecycleCard ev={lastLifecycle} />
        ) : null}
      </section>
    </div>
  );
}

// ---------------------------------------------------------------------------
function Card({
  title,
  children,
}: {
  title?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white">
      {title ? (
        <div className="px-4 py-3 border-b bg-slate-50 text-sm font-semibold text-slate-700">
          {title}
        </div>
      ) : null}
      <div className="p-4">{children}</div>
    </div>
  );
}

function StatusLabel({
  status,
  connected,
}: {
  status: StatusResponse;
  connected: boolean;
}) {
  const txt = status.status.toUpperCase();
  return (
    <div className="text-xs flex items-center gap-2">
      <span
        className={`inline-block w-2 h-2 rounded-full ${
          connected ? "bg-emerald-500" : "bg-slate-400"
        }`}
      />
      <span className="text-slate-600">
        stream {connected ? "connected" : "offline"}
      </span>
      <span className="text-slate-400">·</span>
      <span className="font-medium text-slate-700">status {txt}</span>
      {status.run_id ? (
        <span className="text-slate-400 font-mono">· {status.run_id}</span>
      ) : null}
    </div>
  );
}

function BestCard({ best }: { best: BestFoundEvent }) {
  const entries = Object.entries(best.products).filter(
    ([, v]) => v !== null,
  );
  return (
    <div className="rounded-lg border border-emerald-200 bg-emerald-50 p-4 text-sm">
      <div className="flex items-center gap-2 text-emerald-800 font-semibold mb-2">
        <Trophy className="w-4 h-4" /> Best experiment so far
      </div>
      <div className="text-slate-700">
        <span className="font-semibold">{best.target_name}</span> ={" "}
        <span className="num">{best.best_value?.toFixed(4) ?? "—"}</span>{" "}
        <span className="text-slate-500 text-xs">({best.target_column})</span>
      </div>
      {entries.length > 0 ? (
        <div className="mt-2 grid grid-cols-2 md:grid-cols-3 xl:grid-cols-4 gap-x-3 gap-y-1 text-xs">
          {entries.map(([col, val]) => (
            <div key={col} className="flex justify-between">
              <span className="font-mono text-slate-500">{col}</span>
              <span className="num">{val?.toFixed(2)}</span>
            </div>
          ))}
        </div>
      ) : null}
    </div>
  );
}

function LifecycleCard({ ev }: { ev: KaboEvent }) {
  if (ev.type === "run_completed") {
    return (
      <div className="rounded-md border border-sky-200 bg-sky-50 p-3 text-sm text-sky-800 flex items-center gap-2">
        <CheckCircle2 className="w-4 h-4" /> Run completed.
      </div>
    );
  }
  if (ev.type === "run_failed") {
    return (
      <div className="rounded-md border border-rose-200 bg-rose-50 p-3 text-sm text-rose-800">
        <div className="flex items-center gap-2 mb-1 font-semibold">
          <CircleSlash className="w-4 h-4" /> Run failed
        </div>
        <div className="font-mono text-xs break-all">{ev.error}</div>
      </div>
    );
  }
  return null;
}

// ---------------------------------------------------------------------------
// Helper: derive the latest-of-kind views from the event stream.
// ---------------------------------------------------------------------------
function useDerived(events: KaboEvent[]) {
  return useMemo(() => {
    const logs: LogEvent[] = [];
    let recommendations: RecommendationsEvent | null = null;
    let best: BestFoundEvent | null = null;
    let lastLifecycle: KaboEvent | null = null;

    for (const ev of events) {
      if (ev.type === "log") logs.push(ev);
      else if (ev.type === "recommendations") recommendations = ev;
      else if (ev.type === "best_found") best = ev;
      else if (
        ev.type === "run_started"
        || ev.type === "run_completed"
        || ev.type === "run_failed"
      ) {
        lastLifecycle = ev;
      }
    }

    return { logs, recommendations, best, lastLifecycle };
  }, [events]);
}
