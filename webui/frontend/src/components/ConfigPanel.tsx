import { useMemo } from "react";
import { Info } from "lucide-react";
import type { RunConfig, TaskSchema } from "../types";

interface Props {
  tasks: TaskSchema[];
  config: RunConfig;
  onChange: (patch: Partial<RunConfig>) => void;
  disabled?: boolean;
}

const LABEL_CLS = "text-sm text-slate-700 font-medium";
const INPUT_CLS =
  "w-full px-2.5 py-1.5 rounded-md border border-slate-300 bg-white text-sm " +
  "focus:outline-none focus:ring-2 focus:ring-brand-200 focus:border-brand-500 " +
  "disabled:bg-slate-50 disabled:text-slate-400";
const SELECT_CLS = INPUT_CLS + " pr-8";

export default function ConfigPanel({ tasks, config, onChange, disabled }: Props) {
  const currentTask = useMemo(
    () => tasks.find((t) => t.name === config.task),
    [tasks, config.task],
  );
  const targetOptions = useMemo(() => {
    if (!currentTask) return [];
    return Object.entries(currentTask.target_columns).map(([short, col]) => ({
      short,
      col,
    }));
  }, [currentTask]);

  const [builtinTasks, projectTasks] = useMemo(() => {
    const b: TaskSchema[] = [];
    const p: TaskSchema[] = [];
    for (const t of tasks) (t.source === "project" ? p : b).push(t);
    return [b, p];
  }, [tasks]);

  const isProjectTask = currentTask?.source === "project";

  return (
    <div className="space-y-6">
      {/* Group: Core ------------------------------------------------ */}
      <fieldset className="space-y-3">
        <Legend>Core</Legend>

        <Grid2>
          <Field
            label="Task"
            hint={
              isProjectTask
                ? "project definition — edit in the Projects tab"
                : "built-in (Python-defined)"
            }
          >
            <select
              disabled={disabled}
              className={SELECT_CLS}
              value={config.task}
              onChange={(e) => onChange({ task: e.target.value })}
            >
              {builtinTasks.length > 0 && (
                <optgroup label="Built-in">
                  {builtinTasks.map((t) => (
                    <option key={t.name} value={t.name}>
                      {t.display_name} ({t.name})
                    </option>
                  ))}
                </optgroup>
              )}
              {projectTasks.length > 0 && (
                <optgroup label="Projects">
                  {projectTasks.map((t) => (
                    <option key={t.name} value={t.name}>
                      {t.display_name} ({t.name})
                    </option>
                  ))}
                </optgroup>
              )}
            </select>
          </Field>

          <Field label="Target product">
            <select
              disabled={disabled || !currentTask}
              className={SELECT_CLS}
              value={config.target_product ?? ""}
              onChange={(e) =>
                onChange({ target_product: e.target.value || null })
              }
            >
              <option value="">
                (default: {currentTask?.default_target ?? "—"})
              </option>
              {targetOptions.map((opt) => (
                <option key={opt.col} value={opt.short}>
                  {opt.short} → {opt.col}
                </option>
              ))}
            </select>
          </Field>

          <Field label="Data CSV">
            <input
              disabled={disabled}
              className={INPUT_CLS}
              value={config.data_path}
              onChange={(e) => onChange({ data_path: e.target.value })}
              placeholder="data/data.csv"
            />
          </Field>

          <Field
            label="Candidates CSV"
            hint="'none' to skip discrete pool"
          >
            <input
              disabled={disabled}
              className={INPUT_CLS}
              value={config.candidates_path ?? ""}
              onChange={(e) =>
                onChange({
                  candidates_path:
                    e.target.value.trim() === "" ? null : e.target.value,
                })
              }
              placeholder="data/candidates.csv"
            />
          </Field>

          <Field label="Iterations">
            <input
              disabled={disabled}
              type="number"
              min={1}
              className={INPUT_CLS}
              value={config.iterations}
              onChange={(e) =>
                onChange({ iterations: parseInt(e.target.value, 10) || 1 })
              }
            />
          </Field>

          <Field label="Top K features">
            <input
              disabled={disabled}
              type="number"
              min={1}
              className={INPUT_CLS}
              value={config.top_k}
              onChange={(e) =>
                onChange({ top_k: parseInt(e.target.value, 10) || 1 })
              }
            />
          </Field>

          <Field label="Seed">
            <input
              disabled={disabled}
              type="number"
              className={INPUT_CLS}
              value={config.seed ?? ""}
              onChange={(e) =>
                onChange({
                  seed: e.target.value === "" ? null : parseInt(e.target.value, 10),
                })
              }
              placeholder="(none)"
            />
          </Field>

          <Field label="Device">
            <select
              disabled={disabled}
              className={SELECT_CLS}
              value={config.device}
              onChange={(e) => onChange({ device: e.target.value })}
            >
              <option value="auto">auto</option>
              <option value="cpu">cpu</option>
              <option value="cuda">cuda</option>
            </select>
          </Field>
        </Grid2>

        {currentTask && (
          <div
            className={
              "flex flex-wrap items-center gap-2 text-[11px] rounded border px-2.5 py-1.5 " +
              (isProjectTask
                ? "bg-indigo-50 border-indigo-200 text-indigo-900"
                : "bg-slate-50 border-slate-200 text-slate-600")
            }
          >
            <span className="font-semibold uppercase tracking-wide">
              {isProjectTask ? "Project" : "Built-in"}
            </span>
            <span className="font-mono text-slate-500">
              {currentTask.features.length} features
            </span>
            <span className="font-mono text-slate-500">
              {currentTask.all_product_columns.length} products
            </span>
            <span className="text-slate-500">
              default target →{" "}
              <span className="font-mono">{currentTask.default_target}</span>
            </span>
          </div>
        )}

        <div className="flex flex-wrap gap-3 text-sm pt-1">
          <Checkbox
            disabled={disabled}
            checked={config.interactive}
            onChange={(v) => onChange({ interactive: v })}
            label="Interactive (expert-in-the-loop)"
          />
          <Checkbox
            disabled={disabled}
            checked={config.skip_feature_selection}
            onChange={(v) => onChange({ skip_feature_selection: v })}
            label="Skip feature selection"
          />
          <Checkbox
            disabled={disabled}
            checked={config.strict_training_schema}
            onChange={(v) => onChange({ strict_training_schema: v })}
            label="Strict training schema"
          />
          <Checkbox
            disabled={disabled}
            checked={config.pre_fill_before_choice}
            onChange={(v) => onChange({ pre_fill_before_choice: v })}
            label="Pre-fill recipes before choice"
          />
        </div>
      </fieldset>

      {/* Group: Acquisition ---------------------------------------- */}
      <fieldset className="space-y-3">
        <Legend>Acquisition</Legend>
        <Grid2>
          <Field label="Strategy">
            <select
              disabled={disabled}
              className={SELECT_CLS}
              value={config.acq_strategy}
              onChange={(e) => onChange({ acq_strategy: e.target.value })}
            >
              <option value="ucb">UCB</option>
              <option value="qnei">qNEI</option>
            </select>
          </Field>

          <Field label="Kernel">
            <select
              disabled={disabled}
              className={SELECT_CLS}
              value={config.kernel_type}
              onChange={(e) => onChange({ kernel_type: e.target.value })}
            >
              <option value="matern">Matérn 2.5 (ARD)</option>
              <option value="spectral_mixture">Spectral mixture</option>
            </select>
          </Field>

          <Field label="β">
            <input
              disabled={disabled}
              type="number"
              step="0.1"
              className={INPUT_CLS}
              value={config.beta}
              onChange={(e) =>
                onChange({ beta: parseFloat(e.target.value) || 0 })
              }
            />
          </Field>

          <Field label="β schedule">
            <select
              disabled={disabled}
              className={SELECT_CLS}
              value={config.beta_schedule}
              onChange={(e) => onChange({ beta_schedule: e.target.value })}
            >
              <option value="fixed">fixed</option>
              <option value="theory">theory</option>
              <option value="theory-strict">theory-strict</option>
            </select>
          </Field>

          <Field label="β δ">
            <input
              disabled={disabled}
              type="number"
              step="0.01"
              min="0.001"
              max="0.999"
              className={INPUT_CLS}
              value={config.beta_delta}
              onChange={(e) =>
                onChange({ beta_delta: parseFloat(e.target.value) || 0.1 })
              }
            />
          </Field>

          <Field label="qNEI MC samples">
            <input
              disabled={disabled}
              type="number"
              min={1}
              className={INPUT_CLS}
              value={config.qnei_mc_samples}
              onChange={(e) =>
                onChange({ qnei_mc_samples: parseInt(e.target.value, 10) || 1 })
              }
            />
          </Field>

          <Field label="H₂ penalty weight">
            <input
              disabled={disabled}
              type="number"
              step="0.05"
              min={0}
              className={INPUT_CLS}
              value={config.h2_penalty_weight}
              onChange={(e) =>
                onChange({ h2_penalty_weight: parseFloat(e.target.value) || 0 })
              }
            />
          </Field>

          <Field label="Diversity weight">
            <input
              disabled={disabled}
              type="number"
              step="0.05"
              min={0}
              className={INPUT_CLS}
              value={config.diversity_weight}
              onChange={(e) =>
                onChange({ diversity_weight: parseFloat(e.target.value) || 0 })
              }
            />
          </Field>

          <Field label="Discrete strategy">
            <select
              disabled={disabled}
              className={SELECT_CLS}
              value={config.discrete_strategy}
              onChange={(e) => onChange({ discrete_strategy: e.target.value })}
            >
              <option value="acq">acq</option>
              <option value="thompson">thompson</option>
            </select>
          </Field>

          <Field label="Generated candidates N">
            <input
              disabled={disabled}
              type="number"
              min={0}
              className={INPUT_CLS}
              value={config.generate_candidates_n}
              onChange={(e) =>
                onChange({
                  generate_candidates_n: parseInt(e.target.value, 10) || 0,
                })
              }
            />
          </Field>
        </Grid2>

        <div className="flex flex-wrap gap-3 text-sm pt-1">
          <Checkbox
            disabled={disabled}
            checked={config.prefer_file_candidates}
            onChange={(v) => onChange({ prefer_file_candidates: v })}
            label="Prefer candidates CSV over generator"
          />
        </div>
      </fieldset>

      {/* Group: KABO ----------------------------------------------- */}
      <fieldset className="space-y-3">
        <Legend>KABO mode</Legend>
        <div className="flex flex-wrap gap-3 text-sm">
          <Checkbox
            disabled={disabled}
            checked={config.kabo_mode}
            onChange={(v) => onChange({ kabo_mode: v })}
            label="Enable knowledge-augmented mode"
          />
        </div>

        <Grid2>
          <Field label="λ_p (preference)">
            <input
              disabled={disabled || !config.kabo_mode}
              type="number"
              step="0.5"
              className={INPUT_CLS}
              value={config.lambda_p}
              onChange={(e) =>
                onChange({ lambda_p: parseFloat(e.target.value) || 0 })
              }
            />
          </Field>

          <Field label="λ_k (expert prior)">
            <input
              disabled={disabled || !config.kabo_mode}
              type="number"
              step="0.5"
              className={INPUT_CLS}
              value={config.lambda_k}
              onChange={(e) =>
                onChange({ lambda_k: parseFloat(e.target.value) || 0 })
              }
            />
          </Field>

          <Field label="λ_v (VOI)">
            <input
              disabled={disabled || !config.kabo_mode}
              type="number"
              step="0.1"
              className={INPUT_CLS}
              value={config.lambda_v}
              onChange={(e) =>
                onChange({ lambda_v: parseFloat(e.target.value) || 0 })
              }
            />
          </Field>

          <Field label="PE budget / iter">
            <input
              disabled={disabled || !config.kabo_mode}
              type="number"
              min={0}
              className={INPUT_CLS}
              value={config.pe_budget}
              onChange={(e) =>
                onChange({ pe_budget: parseInt(e.target.value, 10) || 0 })
              }
            />
          </Field>

          <Field
            label="Expert prior JSON"
            hint="relative to project root"
          >
            <input
              disabled={disabled || !config.kabo_mode}
              className={INPUT_CLS}
              value={config.expert_prior_file ?? ""}
              onChange={(e) =>
                onChange({
                  expert_prior_file:
                    e.target.value.trim() === "" ? null : e.target.value,
                })
              }
              placeholder="priors/my_prior.json"
            />
          </Field>
        </Grid2>

        <div className="flex items-start gap-2 text-xs text-slate-500">
          <Info className="w-3.5 h-3.5 flex-shrink-0 mt-0.5" />
          <span>
            PE queries (<code className="text-xs">pe_budget &gt; 0</code>) use
            generic text prompts in the web UI. Set to 0 unless you need
            preference exploration.
          </span>
        </div>
      </fieldset>
    </div>
  );
}

// ---------------------------------------------------------------------------
function Legend({ children }: { children: React.ReactNode }) {
  return (
    <legend className="text-xs font-semibold uppercase tracking-wider text-slate-500">
      {children}
    </legend>
  );
}

function Grid2({ children }: { children: React.ReactNode }) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-x-4 gap-y-3">
      {children}
    </div>
  );
}

function Field({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: React.ReactNode;
}) {
  return (
    <label className="block">
      <div className="flex items-baseline justify-between">
        <span className={LABEL_CLS}>{label}</span>
        {hint ? (
          <span className="text-[10px] text-slate-400 ml-2">{hint}</span>
        ) : null}
      </div>
      <div className="mt-1">{children}</div>
    </label>
  );
}

function Checkbox({
  label,
  checked,
  onChange,
  disabled,
}: {
  label: string;
  checked: boolean;
  onChange: (v: boolean) => void;
  disabled?: boolean;
}) {
  return (
    <label className="inline-flex items-center gap-2 cursor-pointer text-slate-700 select-none">
      <input
        type="checkbox"
        disabled={disabled}
        className="accent-brand-600 w-4 h-4"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
      />
      {label}
    </label>
  );
}
