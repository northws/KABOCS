import { useState } from "react";
import { ChevronDown, ChevronRight, Target } from "lucide-react";
import { t } from "../i18n";
import type { RecommendationsEvent } from "../types";

/**
 * Renders the latest batch of candidate recommendations emitted by the
 * optimizer. Every candidate can be expanded to see its full physical
 * recipe (both selected and non-selected features).
 */
export default function RecommendationList({
  event,
}: {
  event: RecommendationsEvent | null;
}) {
  if (!event) {
    return (
      <div className="rounded-lg border border-slate-200 bg-white p-4 text-sm text-slate-500">
        <Target className="w-4 h-4 inline mr-2 text-slate-400" />
        {t("rec.none")}
      </div>
    );
  }

  return (
    <div className="rounded-lg border border-slate-200 bg-white">
      <div className="px-4 py-3 border-b bg-slate-50">
        <div className="flex items-center gap-2">
          <Target className="w-4 h-4 text-brand-600" />
          <span className="font-semibold text-slate-800">
            {t("rec.title", { iteration: event.iteration, n: event.top_n })}
          </span>
          <span className="text-xs text-slate-500 ml-2">
            {t("rec.optimizing", { target: event.target_name })}
            <span className="font-mono"> ({event.target_column})</span>
          </span>
        </div>
      </div>
      <ul className="divide-y divide-slate-100">
        {event.recommendations.map((rec) => (
          <RecRow
            key={rec.rank}
            rec={rec}
            allFeatures={event.all_features}
            selectedFeatures={event.selected_features}
          />
        ))}
      </ul>
    </div>
  );
}

function RecRow({
  rec,
  allFeatures,
  selectedFeatures,
}: {
  rec: RecommendationsEvent["recommendations"][number];
  allFeatures: string[];
  selectedFeatures: string[];
}) {
  const [open, setOpen] = useState(rec.rank === 1);

  return (
    <li className="px-4 py-3">
      <button
        onClick={() => setOpen((o) => !o)}
        className="w-full flex items-center gap-3 text-left"
      >
        {open ? (
          <ChevronDown className="w-4 h-4 text-slate-400" />
        ) : (
          <ChevronRight className="w-4 h-4 text-slate-400" />
        )}
        <span
          className={`w-8 text-center rounded px-1 py-0.5 text-xs font-semibold ${
            rec.rank === 1
              ? "bg-brand-600 text-white"
              : "bg-slate-100 text-slate-700"
          }`}
        >
          #{rec.rank}
        </span>
        <span className="text-sm text-slate-700 flex-1">
          Acquisition {rec.acq_value.toFixed(4)}{" "}
          <span className="text-slate-400">·</span>{" "}
          <span className="text-xs text-slate-500">{rec.source}</span>
        </span>
      </button>

      {open && (
        <div className="mt-3 ml-10 grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-x-4 gap-y-1">
          {allFeatures.map((feat) => {
            const cell = rec.features[feat];
            const origin = cell?.origin ?? "pending";
            const val = cell?.value;
            const isSelected = selectedFeatures.includes(feat);
            return (
              <div
                key={feat}
                className="flex items-baseline gap-2 text-xs border-b border-dashed border-slate-100 py-1"
              >
                <span className="font-mono text-slate-600 flex-1 truncate" title={feat}>
                  {feat}
                </span>
                <span
                  className={`num tabular-nums ${
                    val === null ? "text-slate-400" : "text-slate-900"
                  }`}
                >
                  {val === null ? "—" : val.toFixed(4)}
                </span>
                <OriginBadge origin={origin} isSelected={isSelected} />
              </div>
            );
          })}
        </div>
      )}
    </li>
  );
}

function OriginBadge({
  origin,
  isSelected,
}: {
  origin: string;
  isSelected: boolean;
}) {
  const base =
    "px-1.5 py-0.5 rounded text-[10px] font-medium uppercase tracking-wider";
  if (origin === "selected") {
    return (
      <span className={`${base} bg-brand-100 text-brand-700`}>{t("rec.selected")}</span>
    );
  }
  if (origin === "expert") {
    return (
      <span className={`${base} bg-amber-100 text-amber-700`}>{t("rec.expert")}</span>
    );
  }
  if (origin === "fixed") {
    return (
      <span className={`${base} bg-slate-100 text-slate-500`}>
        {isSelected ? t("rec.selected") : t("rec.fixed")}
      </span>
    );
  }
  return <span className={`${base} bg-rose-50 text-rose-500`}>{t("rec.pending")}</span>;
}
