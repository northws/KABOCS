import { useEffect, useState } from "react";
import {
  Beaker,
  Database,
  FlaskConical,
  FolderKanban,
  History,
  Sparkles,
  RefreshCw,
} from "lucide-react";

import RunPage from "./components/RunPage";
import DataManager from "./components/DataManager";
import PriorsManager from "./components/PriorsManager";
import ProjectsManager from "./components/ProjectsManager";
import Dashboard from "./components/Dashboard";
import type { StatusResponse } from "./types";
import { currentRunStatus } from "./api";

type Tab = "run" | "projects" | "data" | "priors" | "history";

const TABS: Array<{ key: Tab; label: string; icon: React.ReactNode }> = [
  { key: "run", label: "Run", icon: <Sparkles className="w-4 h-4" /> },
  { key: "projects", label: "Projects", icon: <FlaskConical className="w-4 h-4" /> },
  { key: "data", label: "Data", icon: <Database className="w-4 h-4" /> },
  { key: "priors", label: "Priors", icon: <FolderKanban className="w-4 h-4" /> },
  { key: "history", label: "History", icon: <History className="w-4 h-4" /> },
];

export default function App() {
  const [tab, setTab] = useState<Tab>("run");
  const [status, setStatus] = useState<StatusResponse>({ status: "idle" });
  const [pulse, setPulse] = useState(0);

  // Lightweight polling to keep the top-bar badge current.
  useEffect(() => {
    let alive = true;
    const tick = async () => {
      try {
        const s = await currentRunStatus();
        if (alive) setStatus(s);
      } catch {
        // ignore transient errors
      }
    };
    tick();
    const id = window.setInterval(tick, 2000);
    return () => {
      alive = false;
      window.clearInterval(id);
    };
  }, [pulse]);

  return (
    <div className="min-h-screen flex flex-col">
      <TopBar status={status} onRefresh={() => setPulse((p) => p + 1)} />

      <nav className="border-b bg-white">
        <div className="max-w-7xl mx-auto px-6 flex gap-1">
          {TABS.map((t) => (
            <button
              key={t.key}
              onClick={() => setTab(t.key)}
              className={`flex items-center gap-2 px-4 py-3 text-sm font-medium border-b-2 transition-colors ${
                tab === t.key
                  ? "border-brand-600 text-brand-700"
                  : "border-transparent text-slate-600 hover:text-slate-900"
              }`}
            >
              {t.icon}
              {t.label}
            </button>
          ))}
        </div>
      </nav>

      <main className="flex-1 max-w-7xl w-full mx-auto px-6 py-6">
        {tab === "run" && <RunPage onStatusChange={setStatus} />}
        {tab === "projects" && <ProjectsManager />}
        {tab === "data" && <DataManager />}
        {tab === "priors" && <PriorsManager />}
        {tab === "history" && <Dashboard />}
      </main>

      <footer className="border-t bg-white py-3 text-center text-xs text-slate-500">
        KABOCS WebUI · Knowledge-Augmented Bayesian Optimization for Catalytic
        Systems
      </footer>
    </div>
  );
}

function TopBar({
  status,
  onRefresh,
}: {
  status: StatusResponse;
  onRefresh: () => void;
}) {
  const color =
    status.status === "running"
      ? "bg-emerald-100 text-emerald-700 border-emerald-200"
      : status.status === "error"
      ? "bg-rose-100 text-rose-700 border-rose-200"
      : status.status === "done"
      ? "bg-sky-100 text-sky-700 border-sky-200"
      : status.status === "aborted"
      ? "bg-amber-100 text-amber-700 border-amber-200"
      : "bg-slate-100 text-slate-600 border-slate-200";
  return (
    <header className="bg-white border-b">
      <div className="max-w-7xl mx-auto px-6 h-14 flex items-center gap-3">
        <Beaker className="w-6 h-6 text-brand-600" />
        <div>
          <div className="font-semibold leading-tight">KABOCS</div>
          <div className="text-[11px] text-slate-500 leading-tight">
            Knowledge-Augmented Bayesian Optimization
          </div>
        </div>
        <div className="flex-1" />
        <span
          className={`px-2.5 py-1 rounded-full text-xs border font-medium ${color}`}
        >
          {status.status.toUpperCase()}
          {status.run_id ? (
            <span className="ml-1 text-slate-500 font-normal">
              · {status.run_id.slice(0, 15)}
            </span>
          ) : null}
        </span>
        <button
          onClick={onRefresh}
          className="p-1.5 rounded-md hover:bg-slate-100 text-slate-600"
          title="Refresh status"
        >
          <RefreshCw className="w-4 h-4" />
        </button>
      </div>
    </header>
  );
}
