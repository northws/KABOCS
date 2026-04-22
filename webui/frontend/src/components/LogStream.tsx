import { useEffect, useRef } from "react";
import type { LogEvent } from "../types";

/** Scrolling log pane with colour-coded severity levels. */
export default function LogStream({ logs }: { logs: LogEvent[] }) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    // Only auto-scroll if already near the bottom.
    const nearBottom =
      el.scrollHeight - el.clientHeight - el.scrollTop < 80;
    if (nearBottom) el.scrollTop = el.scrollHeight;
  }, [logs]);

  return (
    <div className="rounded-lg border border-slate-200 bg-slate-950 text-slate-100 overflow-hidden">
      <div className="px-3 py-2 border-b border-slate-800 text-xs text-slate-400 flex items-center justify-between">
        <span>Log stream ({logs.length} lines)</span>
        <span className="font-mono">
          {logs.length > 0
            ? new Date(
                logs[logs.length - 1].ts * 1000,
              ).toLocaleTimeString()
            : "—"}
        </span>
      </div>
      <div
        ref={ref}
        className="log-scroll h-[360px] overflow-auto px-3 py-2 text-[12px] leading-relaxed font-mono"
      >
        {logs.length === 0 ? (
          <div className="text-slate-500">
            (no logs yet — start a run to see the optimizer's output)
          </div>
        ) : (
          logs.map((ev, i) => (
            <div key={i} className="whitespace-pre-wrap">
              <span className="text-slate-500 mr-2">
                {new Date(ev.ts * 1000).toLocaleTimeString()}
              </span>
              <LevelBadge level={ev.level} />
              <span className="ml-2">{ev.message}</span>
            </div>
          ))
        )}
      </div>
    </div>
  );
}

function LevelBadge({ level }: { level: string }) {
  const colour =
    level === "error" || level === "critical"
      ? "text-rose-400"
      : level === "warning" || level === "warn"
      ? "text-amber-300"
      : level === "info"
      ? "text-sky-300"
      : level === "debug"
      ? "text-slate-500"
      : level === "stdout"
      ? "text-emerald-300"
      : "text-slate-400";
  return (
    <span className={`${colour} uppercase text-[10px] font-semibold`}>
      {level}
    </span>
  );
}
