import { useEffect, useMemo, useState } from "react";
import {
  Download,
  Image as ImageIcon,
  RefreshCw,
  Trash2,
} from "lucide-react";
import {
  deleteRun,
  getRunData,
  getRunMetadata,
  listRuns,
  runFileUrl,
} from "../api";
import type { ArchivedRun } from "../types";

/** Historical run browser + inline viewer for metadata, data, and plots. */
export default function Dashboard() {
  const [runs, setRuns] = useState<ArchivedRun[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [metadata, setMetadata] = useState<Record<string, unknown> | null>(null);
  const [dataCsv, setDataCsv] = useState<string>("");
  const [msg, setMsg] = useState<string | null>(null);

  async function refresh() {
    try {
      const r = await listRuns();
      setRuns(r.runs);
      if (!selected && r.runs.length > 0) setSelected(r.runs[0].run_id);
    } catch (e) {
      setMsg(String(e));
    }
  }

  useEffect(() => {
    refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!selected) {
      setMetadata(null);
      setDataCsv("");
      return;
    }
    let alive = true;
    (async () => {
      setMetadata(null);
      setDataCsv("");
      try {
        const m = await getRunMetadata(selected);
        if (alive) setMetadata(m);
      } catch {
        /* metadata file may be missing */
      }
      try {
        const d = await getRunData(selected);
        if (alive) setDataCsv(d.content);
      } catch {
        /* data file may be missing */
      }
    })();
    return () => {
      alive = false;
    };
  }, [selected]);

  async function onDelete(run_id: string) {
    if (!confirm(`Delete run ${run_id}?`)) return;
    try {
      await deleteRun(run_id);
      if (selected === run_id) setSelected(null);
      await refresh();
    } catch (e) {
      setMsg(String(e));
    }
  }

  const betaTrace = useMemo(() => {
    const arr = (metadata?.beta_trace as number[]) ?? [];
    return Array.isArray(arr) ? arr : [];
  }, [metadata]);
  const selectedFeatures = useMemo(() => {
    const f = metadata?.selected_features;
    return Array.isArray(f) ? (f as string[]) : [];
  }, [metadata]);

  return (
    <div className="grid grid-cols-1 xl:grid-cols-[320px_1fr] gap-6">
      <aside className="rounded-lg border border-slate-200 bg-white">
        <div className="px-3 py-2 border-b bg-slate-50 text-xs font-semibold uppercase text-slate-600 flex items-center justify-between">
          <span>Archived runs ({runs.length})</span>
          <button
            onClick={refresh}
            className="p-1 rounded hover:bg-slate-200 text-slate-500"
          >
            <RefreshCw className="w-3.5 h-3.5" />
          </button>
        </div>
        <ul className="max-h-[520px] overflow-auto log-scroll">
          {runs.length === 0 ? (
            <li className="p-3 text-xs text-slate-400">
              No archived runs yet. Start one from the Run tab.
            </li>
          ) : (
            runs.map((r) => (
              <li
                key={r.run_id}
                className={`px-3 py-2 border-b flex items-start gap-2 cursor-pointer text-xs ${
                  selected === r.run_id
                    ? "bg-brand-50 text-brand-800"
                    : "hover:bg-slate-50"
                }`}
                onClick={() => setSelected(r.run_id)}
              >
                <div className="flex-1 min-w-0">
                  <div className="font-mono truncate">{r.run_id}</div>
                  <div className="flex items-center gap-1 text-[10px] text-slate-500 mt-0.5">
                    <StatusPill status={r.status ?? "?"} />
                    {r.config?.task ? (
                      <span className="uppercase">{r.config.task}</span>
                    ) : null}
                    {typeof r.finished_at === "number" ? (
                      <span className="num">
                        {new Date(r.finished_at * 1000).toLocaleString()}
                      </span>
                    ) : null}
                  </div>
                </div>
                <button
                  className="text-slate-400 hover:text-rose-500"
                  onClick={(e) => {
                    e.stopPropagation();
                    onDelete(r.run_id);
                  }}
                >
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
              </li>
            ))
          )}
        </ul>
      </aside>

      <section className="space-y-4 min-w-0">
        {!selected ? (
          <div className="rounded-lg border border-slate-200 bg-white p-8 text-sm text-slate-500 text-center">
            Pick an archived run from the sidebar to inspect it.
          </div>
        ) : (
          <>
            <SummaryCard runId={selected} metadata={metadata} />
            <FeatureImportanceCard runId={selected} />
            {betaTrace.length > 0 ? <BetaTraceCard values={betaTrace} /> : null}
            <DataPreviewCard
              runId={selected}
              csv={dataCsv}
              selectedFeatures={selectedFeatures}
            />
            {msg ? (
              <div className="text-xs text-rose-700 bg-rose-50 border border-rose-200 p-2 rounded">
                {msg}
              </div>
            ) : null}
          </>
        )}
      </section>
    </div>
  );
}

// ---------------------------------------------------------------------------
function StatusPill({ status }: { status: string }) {
  const c =
    status === "done"
      ? "bg-emerald-100 text-emerald-700"
      : status === "error"
      ? "bg-rose-100 text-rose-700"
      : status === "aborted"
      ? "bg-amber-100 text-amber-700"
      : "bg-slate-100 text-slate-600";
  return (
    <span className={`px-1.5 py-0.5 rounded font-medium uppercase ${c}`}>
      {status}
    </span>
  );
}

function SummaryCard({
  runId,
  metadata,
}: {
  runId: string;
  metadata: Record<string, unknown> | null;
}) {
  const rows: [string, unknown][] = [
    ["task", metadata?.task],
    ["target_column", metadata?.target_column],
    ["seed", metadata?.seed],
    ["iterations", metadata?.n_iterations],
    ["top_k", metadata?.top_k],
    ["acq_strategy", metadata?.acq_strategy],
    ["beta", metadata?.beta],
    ["beta_schedule", metadata?.beta_schedule],
    ["kabo_mode", metadata?.kabo_mode],
    ["n_rows_final", metadata?.n_rows_final],
  ];
  return (
    <div className="rounded-lg border border-slate-200 bg-white">
      <header className="px-4 py-3 border-b bg-slate-50 flex items-center gap-2">
        <span className="text-sm font-semibold text-slate-700">Metadata</span>
        <span className="text-xs text-slate-500 font-mono">({runId})</span>
        <div className="flex-1" />
        <a
          href={runFileUrl(runId, "run_metadata.json")}
          target="_blank"
          rel="noreferrer"
          className="text-xs text-slate-600 inline-flex items-center gap-1 hover:text-brand-700"
        >
          <Download className="w-3.5 h-3.5" /> run_metadata.json
        </a>
      </header>
      <div className="p-4 grid grid-cols-2 md:grid-cols-3 xl:grid-cols-4 gap-3 text-sm">
        {rows.map(([k, v]) => (
          <div key={k}>
            <div className="text-[10px] uppercase text-slate-400">{k}</div>
            <div className="font-mono text-slate-800 truncate">
              {formatValue(v)}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function FeatureImportanceCard({ runId }: { runId: string }) {
  const url = runFileUrl(runId, "feature_importances.png");
  return (
    <div className="rounded-lg border border-slate-200 bg-white">
      <header className="px-4 py-3 border-b bg-slate-50 flex items-center gap-2">
        <ImageIcon className="w-4 h-4 text-slate-500" />
        <span className="text-sm font-semibold text-slate-700">
          Feature importances
        </span>
      </header>
      <div className="p-4">
        <img
          src={url}
          alt="feature importances"
          className="max-w-full border border-slate-100 rounded"
          onError={(e) => {
            (e.target as HTMLImageElement).replaceWith(
              Object.assign(document.createElement("div"), {
                innerText: "(image not available — feature selection was skipped?)",
                className: "text-xs text-slate-500",
              }),
            );
          }}
        />
      </div>
    </div>
  );
}

function BetaTraceCard({ values }: { values: number[] }) {
  if (values.length === 0) return null;
  const max = Math.max(...values);
  return (
    <div className="rounded-lg border border-slate-200 bg-white">
      <header className="px-4 py-3 border-b bg-slate-50">
        <span className="text-sm font-semibold text-slate-700">
          β schedule trace
        </span>
      </header>
      <div className="p-4">
        <svg viewBox={`0 0 ${values.length * 40} 120`} className="w-full h-32">
          {values.map((v, i) => {
            const h = (v / max) * 100;
            return (
              <g key={i}>
                <rect
                  x={i * 40 + 8}
                  y={120 - h - 10}
                  width={24}
                  height={h}
                  className="fill-brand-400"
                />
                <text
                  x={i * 40 + 20}
                  y={115}
                  textAnchor="middle"
                  className="text-[10px] fill-slate-500"
                >
                  {i + 1}
                </text>
                <text
                  x={i * 40 + 20}
                  y={120 - h - 14}
                  textAnchor="middle"
                  className="text-[10px] fill-slate-700"
                >
                  {v.toFixed(2)}
                </text>
              </g>
            );
          })}
        </svg>
      </div>
    </div>
  );
}

function DataPreviewCard({
  runId,
  csv,
  selectedFeatures,
}: {
  runId: string;
  csv: string;
  selectedFeatures: string[];
}) {
  const parsed = useMemo(() => parseCsv(csv), [csv]);
  if (!csv) {
    return (
      <div className="rounded-lg border border-slate-200 bg-white p-4 text-sm text-slate-500">
        No updated dataset for this run.
      </div>
    );
  }
  const targetCol =
    typeof parsed.headers === "object" && parsed.rows.length > 0
      ? findTargetHeader(parsed.headers)
      : null;
  return (
    <div className="rounded-lg border border-slate-200 bg-white">
      <header className="px-4 py-3 border-b bg-slate-50 flex items-center gap-2">
        <span className="text-sm font-semibold text-slate-700">
          data_updated.csv
        </span>
        <span className="text-xs text-slate-500">
          {parsed.rows.length} rows · {parsed.headers.length} columns
        </span>
        <div className="flex-1" />
        <a
          href={runFileUrl(runId, "data_updated.csv")}
          target="_blank"
          rel="noreferrer"
          className="text-xs text-slate-600 inline-flex items-center gap-1 hover:text-brand-700"
        >
          <Download className="w-3.5 h-3.5" /> Download
        </a>
      </header>
      <div className="overflow-auto max-h-[400px] log-scroll">
        <table className="text-[11px] w-max">
          <thead className="bg-slate-50 sticky top-0">
            <tr>
              {parsed.headers.map((h) => (
                <th
                  key={h}
                  className={`px-2 py-1 text-left font-semibold border-b ${
                    h === targetCol
                      ? "text-amber-700 bg-amber-50"
                      : selectedFeatures.includes(h)
                      ? "text-brand-700"
                      : "text-slate-700"
                  }`}
                >
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {parsed.rows.slice(-50).map((row, i) => (
              <tr key={i} className="odd:bg-slate-50/50">
                {parsed.headers.map((h) => (
                  <td
                    key={h}
                    className={`px-2 py-1 num border-b border-slate-100 ${
                      h === targetCol ? "bg-amber-50/50 font-semibold" : ""
                    }`}
                  >
                    {row[h] ?? ""}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
function formatValue(v: unknown): string {
  if (v === null || v === undefined) return "—";
  if (typeof v === "boolean") return v ? "true" : "false";
  if (typeof v === "number") return String(v);
  if (typeof v === "string") return v;
  try {
    return JSON.stringify(v);
  } catch {
    return String(v);
  }
}

function parseCsv(csv: string): {
  headers: string[];
  rows: Array<Record<string, string>>;
} {
  if (!csv) return { headers: [], rows: [] };
  const lines = csv.split(/\r?\n/).filter((l) => l.length > 0);
  if (lines.length === 0) return { headers: [], rows: [] };
  const headers = splitCsvLine(lines[0]);
  const rows = lines.slice(1).map((ln) => {
    const cells = splitCsvLine(ln);
    const rec: Record<string, string> = {};
    headers.forEach((h, i) => {
      rec[h] = cells[i] ?? "";
    });
    return rec;
  });
  return { headers, rows };
}

function splitCsvLine(line: string): string[] {
  // Simplistic split; KABO writes straightforward CSVs without quoting.
  return line.split(",");
}

function findTargetHeader(headers: string[]): string | null {
  const target = headers.find((h) => h.startsWith("Y_") && !h.includes("_H2"))
    ?? headers.find((h) => h === "Y");
  return target ?? null;
}
