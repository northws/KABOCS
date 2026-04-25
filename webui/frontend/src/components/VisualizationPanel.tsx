import { useState } from "react";
import { Activity, ChevronDown, ChevronRight } from "lucide-react";

import { t } from "../i18n";
import type { VisualizationEvent } from "../types";

interface Props {
  event: VisualizationEvent | null;
}

export default function VisualizationPanel({ event }: Props) {
  const [open, setOpen] = useState(true);

  if (event === null) {
    return null;
  }

  const hasGp = event.gp_landscape !== null;
  const hasPca = event.pca_projection !== null;
  if (!hasGp && !hasPca) return null;

  return (
    <div className="rounded-lg border border-slate-200 bg-white">
      <button
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center justify-between px-4 py-3 border-b bg-slate-50 text-sm font-semibold text-slate-700 hover:bg-slate-100"
      >
        <span className="flex items-center gap-2">
          <Activity className="w-4 h-4 text-brand-600" />
          {t("viz.title", { iteration: event.iteration })}
          <span className="text-xs font-normal text-slate-500">
            · {event.target_name}
          </span>
        </span>
        {open ? (
          <ChevronDown className="w-4 h-4 text-slate-400" />
        ) : (
          <ChevronRight className="w-4 h-4 text-slate-400" />
        )}
      </button>

      {open ? (
        <div className="p-4 space-y-4">
          {hasGp && event.gp_landscape ? (
            <div>
              <div className="text-xs font-semibold uppercase tracking-wider text-slate-500 mb-1">
                {t("viz.gp")}
                {event.gp_landscape.dims ? (
                  <span className="ml-2 font-normal text-slate-400 normal-case tracking-normal">
                    {t("viz.gp.dims", {
                      a: event.gp_landscape.dims[0],
                      b: event.gp_landscape.dims[1],
                    })}
                  </span>
                ) : null}
              </div>
              <img
                src={event.gp_landscape.image}
                alt="GP posterior and acquisition"
                className="max-w-full rounded border border-slate-200"
              />
            </div>
          ) : null}

          {hasPca && event.pca_projection ? (
            <div>
              <div className="text-xs font-semibold uppercase tracking-wider text-slate-500 mb-1">
                {t("viz.pca")}
                <span className="ml-2 font-normal text-slate-400 normal-case tracking-normal">
                  {t("viz.pca.legend", {
                    train: event.pca_projection.n_train ?? 0,
                    cand: event.pca_projection.n_candidates ?? 0,
                  })}
                </span>
              </div>
              <img
                src={event.pca_projection.image}
                alt="PCA projection of design space"
                className="max-w-full rounded border border-slate-200"
              />
            </div>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}
