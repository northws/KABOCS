import { useEffect, useMemo, useState } from "react";
import { AlertTriangle, MessageSquare, Send } from "lucide-react";
import { submitAnswer } from "../api";
import type {
  PromptEvent,
  RecommendationsEvent,
} from "../types";

interface Props {
  prompt: PromptEvent | null;
  latestRecommendations: RecommendationsEvent | null;
}

/**
 * Container for whichever prompt is currently awaiting an answer from
 * the backend optimizer. Delegates to a kind-specific sub-component.
 */
export default function PromptPanel({ prompt, latestRecommendations }: Props) {
  if (!prompt) {
    return (
      <div className="rounded-lg border border-slate-200 bg-slate-50 p-4 text-sm text-slate-500">
        <div className="flex items-center gap-2">
          <MessageSquare className="w-4 h-4" />
          Waiting for the optimizer to emit the next prompt…
        </div>
      </div>
    );
  }

  return (
    <div className="rounded-lg border-2 border-brand-400 bg-brand-50/50 p-5 shadow-sm">
      <div className="flex items-center gap-2 text-brand-800 mb-4">
        <MessageSquare className="w-5 h-5" />
        <div className="font-semibold">Awaiting your input</div>
        <span className="text-xs text-slate-500 ml-2">
          prompt #{prompt.prompt_id} · {prompt.kind}
        </span>
      </div>

      {prompt.kind === "candidate_choice" && (
        <CandidateChoiceForm prompt={prompt} recs={latestRecommendations} />
      )}
      {prompt.kind === "manual_candidate" && (
        <ManualCandidateForm prompt={prompt} />
      )}
      {prompt.kind === "nonselected_features" && (
        <NonselectedFeaturesForm prompt={prompt} />
      )}
      {prompt.kind === "product_yields" && (
        <ProductYieldsForm prompt={prompt} />
      )}
      {prompt.kind === "raw_input" && <RawInputForm prompt={prompt} />}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Candidate choice
// ---------------------------------------------------------------------------
function CandidateChoiceForm({
  prompt,
  recs,
}: {
  prompt: Extract<PromptEvent, { kind: "candidate_choice" }>;
  recs: RecommendationsEvent | null;
}) {
  const [selected, setSelected] = useState<number>(1);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const ranks = prompt.top_indices.map((_, i) => i + 1);

  async function go(action: string, rank?: number) {
    setSubmitting(true);
    setError(null);
    try {
      const payload: Record<string, unknown> = { prompt_id: prompt.prompt_id };
      if (action === "submit" && rank !== undefined) {
        payload.rank = rank;
      }
      if (action !== "submit") payload.action = action;
      await submitAnswer(payload);
    } catch (e) {
      setError(String(e));
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="space-y-3">
      <p className="text-sm text-slate-600">
        Choose which candidate to execute. The{" "}
        <span className="font-semibold">bold rank</span> is the model's top
        recommendation; picking another rank will also be recorded as a
        preference signal when KABO mode is active.
      </p>

      <div className="flex flex-wrap gap-2">
        {ranks.map((r) => {
          const rec = recs?.recommendations.find((x) => x.rank === r);
          const isFirst = r === 1;
          return (
            <button
              key={r}
              disabled={submitting}
              onClick={() => {
                setSelected(r);
                go("submit", r);
              }}
              className={`px-4 py-2 rounded-md border text-sm transition-colors ${
                selected === r
                  ? "bg-brand-600 text-white border-brand-600"
                  : "bg-white border-slate-300 hover:border-brand-400"
              } ${isFirst ? "font-semibold" : ""}`}
              title={rec ? `source: ${rec.source} · acq ${rec.acq_value.toFixed(4)}` : ""}
            >
              Rank #{r}
              {rec ? (
                <span className="text-[10px] ml-1 opacity-70">
                  ({rec.acq_value.toFixed(3)})
                </span>
              ) : null}
            </button>
          );
        })}
      </div>

      <div className="flex flex-wrap gap-2 text-sm">
        <button
          disabled={submitting}
          onClick={() => go("tie")}
          className="px-3 py-1.5 rounded-md border border-slate-300 bg-white hover:border-amber-400 text-slate-700"
        >
          Declare tie
        </button>
        <button
          disabled={submitting}
          onClick={() => go("manual")}
          className="px-3 py-1.5 rounded-md border border-slate-300 bg-white hover:border-amber-400 text-slate-700"
        >
          Manual override
        </button>
        <button
          disabled={submitting}
          onClick={() => go("exit")}
          className="px-3 py-1.5 rounded-md border border-rose-300 bg-white hover:bg-rose-50 text-rose-700"
        >
          Stop optimization
        </button>
      </div>

      {error ? <ErrorBanner text={error} /> : null}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Manual candidate / non-selected features (shared form scaffolding)
// ---------------------------------------------------------------------------
function featureTable(
  features: string[],
  bounds: Record<string, [number, number]>,
  values: Record<string, string>,
  setValues: (v: Record<string, string>) => void,
  disabled: boolean,
) {
  return (
    <div className="max-h-[400px] overflow-auto log-scroll rounded-md border border-slate-200 bg-white">
      <table className="w-full text-sm">
        <thead className="bg-slate-50 sticky top-0">
          <tr className="text-left text-xs uppercase text-slate-500">
            <th className="px-3 py-2 font-medium">Feature</th>
            <th className="px-3 py-2 font-medium">Bounds</th>
            <th className="px-3 py-2 font-medium">Value</th>
          </tr>
        </thead>
        <tbody>
          {features.map((f) => {
            const b = bounds[f] ?? [0, 1];
            const mid = ((b[0] + b[1]) / 2).toFixed(4);
            return (
              <tr key={f} className="border-t border-slate-100">
                <td className="px-3 py-1.5 font-mono text-xs">{f}</td>
                <td className="px-3 py-1.5 num text-xs text-slate-500">
                  [{b[0]}, {b[1]}]
                </td>
                <td className="px-3 py-1.5">
                  <input
                    disabled={disabled}
                    className="w-full px-2 py-1 rounded border border-slate-300 bg-white text-sm num focus:outline-none focus:ring-2 focus:ring-brand-200"
                    placeholder={`mid=${mid}`}
                    value={values[f] ?? ""}
                    onChange={(e) =>
                      setValues({ ...values, [f]: e.target.value })
                    }
                  />
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function useFeatureFormState(features: string[]) {
  const [values, setValues] = useState<Record<string, string>>({});
  useEffect(() => {
    setValues({});
  }, [features.join("|")]);
  return [values, setValues] as const;
}

// ---------------------------------------------------------------------------
function ManualCandidateForm({
  prompt,
}: {
  prompt: Extract<PromptEvent, { kind: "manual_candidate" }>;
}) {
  const [values, setValues] = useFeatureFormState(prompt.features);
  const [oob, setOob] = useState<number>(0);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const hasOOB = useMemo(() => {
    const list: string[] = [];
    for (const f of prompt.features) {
      const raw = values[f];
      if (raw === undefined || raw === "") continue;
      const v = parseFloat(raw);
      if (!Number.isFinite(v)) continue;
      const [lo, hi] = prompt.bounds[f] ?? [-Infinity, Infinity];
      if (v < lo || v > hi) list.push(f);
    }
    return list;
  }, [values, prompt]);

  async function submit(action: "submit" | "exit") {
    setSubmitting(true);
    setError(null);
    try {
      if (action === "exit") {
        await submitAnswer({ prompt_id: prompt.prompt_id, action: "exit" });
      } else {
        await submitAnswer({
          prompt_id: prompt.prompt_id,
          values,
          oob_confirmations: oob,
        });
      }
    } catch (e) {
      setError(String(e));
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="space-y-3">
      <p className="text-sm text-slate-600">
        Supply full feature values for a custom candidate. Empty cells will
        default to the design-space midpoint.
      </p>

      {featureTable(prompt.features, prompt.bounds, values, setValues, submitting)}

      {hasOOB.length > 0 ? (
        <div className="flex items-start gap-2 text-xs text-amber-700 bg-amber-50 border border-amber-200 p-2 rounded">
          <AlertTriangle className="w-3.5 h-3.5 mt-0.5" />
          <div>
            Out-of-bounds values for{" "}
            <code className="font-mono">{hasOOB.join(", ")}</code>. Confirm
            intentional override{" "}
            <label className="ml-2 inline-flex items-center gap-1">
              <input
                type="checkbox"
                checked={oob > 0}
                onChange={(e) => setOob(e.target.checked ? hasOOB.length : 0)}
              />
              I accept
            </label>
          </div>
        </div>
      ) : null}

      <SubmitRow
        onSubmit={() => submit("submit")}
        onExit={() => submit("exit")}
        submitting={submitting}
      />
      {error ? <ErrorBanner text={error} /> : null}
    </div>
  );
}

// ---------------------------------------------------------------------------
function NonselectedFeaturesForm({
  prompt,
}: {
  prompt: Extract<PromptEvent, { kind: "nonselected_features" }>;
}) {
  const [values, setValues] = useFeatureFormState(prompt.features);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit() {
    setSubmitting(true);
    setError(null);
    try {
      await submitAnswer({
        prompt_id: prompt.prompt_id,
        values,
      });
    } catch (e) {
      setError(String(e));
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="space-y-3">
      <p className="text-sm text-slate-600">
        These features are outside the GP model; fill in the actual
        experimental conditions you plan to use.
      </p>

      {featureTable(prompt.features, prompt.bounds, values, setValues, submitting)}

      <div className="flex gap-2">
        <button
          disabled={submitting}
          onClick={submit}
          className="inline-flex items-center gap-1.5 px-4 py-2 rounded-md bg-brand-600 text-white hover:bg-brand-700 text-sm disabled:opacity-60"
        >
          <Send className="w-4 h-4" />
          Submit
        </button>
      </div>
      {error ? <ErrorBanner text={error} /> : null}
    </div>
  );
}

// ---------------------------------------------------------------------------
function ProductYieldsForm({
  prompt,
}: {
  prompt: Extract<PromptEvent, { kind: "product_yields" }>;
}) {
  const [values, setValues] = useState<Record<string, string>>({});
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setValues({});
  }, [prompt.prompt_id]);

  async function submit(action: "submit" | "exit") {
    setSubmitting(true);
    setError(null);
    try {
      if (action === "exit") {
        await submitAnswer({ prompt_id: prompt.prompt_id, action: "exit" });
      } else {
        const yields: Record<string, number> = {};
        for (const p of prompt.products) {
          const v = parseFloat(values[p.column] ?? "");
          yields[p.column] = Number.isFinite(v) ? v : 0;
        }
        await submitAnswer({
          prompt_id: prompt.prompt_id,
          yields,
        });
      }
    } catch (e) {
      setError(String(e));
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="space-y-3">
      <p className="text-sm text-slate-600">
        Record the experimental yield for every product. Target is highlighted.
        Enter <code>0</code> for undetected products.
      </p>

      <div className="max-h-[400px] overflow-auto log-scroll rounded-md border border-slate-200 bg-white">
        <table className="w-full text-sm">
          <thead className="bg-slate-50 sticky top-0">
            <tr className="text-left text-xs uppercase text-slate-500">
              <th className="px-3 py-2 font-medium">Product</th>
              <th className="px-3 py-2 font-medium">Column</th>
              <th className="px-3 py-2 font-medium">Yield</th>
            </tr>
          </thead>
          <tbody>
            {prompt.products.map((p) => (
              <tr
                key={p.column}
                className={`border-t border-slate-100 ${
                  p.is_target ? "bg-amber-50" : ""
                }`}
              >
                <td className="px-3 py-1.5">
                  <span className="font-medium">{p.display}</span>
                  {p.is_target ? (
                    <span className="ml-2 text-[10px] text-amber-700 font-semibold">
                      ★ TARGET
                    </span>
                  ) : null}
                </td>
                <td className="px-3 py-1.5 font-mono text-xs text-slate-500">
                  {p.column}
                </td>
                <td className="px-3 py-1.5">
                  <input
                    disabled={submitting}
                    className="w-full px-2 py-1 rounded border border-slate-300 bg-white text-sm num focus:outline-none focus:ring-2 focus:ring-brand-200"
                    placeholder="e.g. 42.5"
                    value={values[p.column] ?? ""}
                    onChange={(e) =>
                      setValues({ ...values, [p.column]: e.target.value })
                    }
                  />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <SubmitRow
        onSubmit={() => submit("submit")}
        onExit={() => submit("exit")}
        submitting={submitting}
      />
      {error ? <ErrorBanner text={error} /> : null}
    </div>
  );
}

// ---------------------------------------------------------------------------
function RawInputForm({
  prompt,
}: {
  prompt: Extract<PromptEvent, { kind: "raw_input" }>;
}) {
  const [value, setValue] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setValue("");
  }, [prompt.prompt_id]);

  async function go() {
    setSubmitting(true);
    setError(null);
    try {
      await submitAnswer({ prompt_id: prompt.prompt_id, value });
    } catch (e) {
      setError(String(e));
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="space-y-3">
      <pre className="bg-slate-900 text-slate-100 p-3 rounded text-xs whitespace-pre-wrap">
        {prompt.prompt_text || "(no prompt text)"}
      </pre>
      <div className="flex gap-2">
        <input
          value={value}
          disabled={submitting}
          onChange={(e) => setValue(e.target.value)}
          className="flex-1 px-3 py-2 rounded border border-slate-300 bg-white text-sm focus:outline-none focus:ring-2 focus:ring-brand-200"
          placeholder="Your response (e.g. a / b / tie)"
        />
        <button
          disabled={submitting}
          onClick={go}
          className="px-4 py-2 rounded-md bg-brand-600 text-white hover:bg-brand-700 text-sm disabled:opacity-60"
        >
          Send
        </button>
      </div>
      {error ? <ErrorBanner text={error} /> : null}
    </div>
  );
}

// ---------------------------------------------------------------------------
function SubmitRow({
  onSubmit,
  onExit,
  submitting,
}: {
  onSubmit: () => void;
  onExit: () => void;
  submitting: boolean;
}) {
  return (
    <div className="flex gap-2">
      <button
        disabled={submitting}
        onClick={onSubmit}
        className="inline-flex items-center gap-1.5 px-4 py-2 rounded-md bg-brand-600 text-white hover:bg-brand-700 text-sm disabled:opacity-60"
      >
        <Send className="w-4 h-4" />
        Submit
      </button>
      <button
        disabled={submitting}
        onClick={onExit}
        className="px-4 py-2 rounded-md border border-rose-300 bg-white hover:bg-rose-50 text-rose-700 text-sm"
      >
        Stop optimization
      </button>
    </div>
  );
}

function ErrorBanner({ text }: { text: string }) {
  return (
    <div className="text-xs text-rose-700 bg-rose-50 border border-rose-200 p-2 rounded">
      {text}
    </div>
  );
}
