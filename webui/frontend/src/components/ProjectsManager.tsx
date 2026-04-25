import { useCallback, useEffect, useMemo, useState } from "react";
import {
  FolderPlus,
  Lock,
  Plus,
  RefreshCw,
  Save,
  Trash2,
  X,
  FlaskConical,
  Target,
  Settings2,
} from "lucide-react";

import {
  createProject,
  deleteProject,
  getProject,
  listProjects,
  updateProject,
} from "../api";
import { t } from "../i18n";
import type { FeatureSpec, ProjectSpec, TargetSpec } from "../types";

// ---------------------------------------------------------------------------
// Factory helpers
// ---------------------------------------------------------------------------
function blankFeature(): FeatureSpec {
  return {
    name: "",
    type: "continuous",
    lo: 0,
    hi: 1,
    unit: null,
    display_name: null,
  };
}

function blankTarget(): TargetSpec {
  return {
    short_name: "",
    column: "",
    display_name: null,
    unit: "%",
    is_competing: false,
  };
}

function blankProject(): ProjectSpec {
  return {
    name: "",
    display_name: "",
    description: "",
    features: [blankFeature()],
    targets: [blankTarget()],
    default_target: "",
    notes: "",
  };
}

const NAME_RE = /^[a-z0-9_-]+$/;

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------
function validate(spec: ProjectSpec, builtins: string[]): string[] {
  const errors: string[] = [];
  const key = spec.name.trim().toLowerCase();
  if (!key) errors.push(t("proj.err.name_required"));
  else if (!NAME_RE.test(key))
    errors.push(t("proj.err.name_chars"));
  else if (builtins.includes(key))
    errors.push(t("proj.err.name_collision", { name: key }));

  if (spec.features.length === 0)
    errors.push(t("proj.err.need_feature"));
  const featNames = new Set<string>();
  spec.features.forEach((f, i) => {
    const n = f.name.trim();
    if (!n) errors.push(t("proj.err.feature_name", { i: i + 1 }));
    else if (featNames.has(n))
      errors.push(t("proj.err.feature_dup", { name: n }));
    featNames.add(n);
    if (!Number.isFinite(f.lo) || !Number.isFinite(f.hi))
      errors.push(t("proj.err.feature_bounds", { name: n }));
    else if (f.hi <= f.lo)
      errors.push(t("proj.err.feature_hi_lo", { name: n, lo: f.lo, hi: f.hi }));
  });

  if (spec.targets.length === 0)
    errors.push(t("proj.err.need_target"));
  const shortNames = new Set<string>();
  const cols = new Set<string>();
  spec.targets.forEach((tg, i) => {
    const s = tg.short_name.trim();
    const c = tg.column.trim();
    if (!s) errors.push(t("proj.err.target_name", { i: i + 1 }));
    else if (shortNames.has(s))
      errors.push(t("proj.err.target_dup_name", { name: s }));
    shortNames.add(s);
    if (!c) errors.push(t("proj.err.target_column", { i: i + 1 }));
    else if (cols.has(c)) errors.push(t("proj.err.target_dup_col", { col: c }));
    cols.add(c);
  });

  if (spec.default_target && !shortNames.has(spec.default_target))
    errors.push(
      t("proj.err.target_short_mismatch", { target: spec.default_target }),
    );
  return errors;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------
export default function ProjectsManager() {
  const [projects, setProjects] = useState<ProjectSpec[]>([]);
  const [builtins, setBuiltins] = useState<string[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [draft, setDraft] = useState<ProjectSpec | null>(null);
  const [isNew, setIsNew] = useState(false);
  const [dirty, setDirty] = useState(false);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState<
    { text: string; tone: "ok" | "error" } | null
  >(null);

  const refresh = useCallback(async () => {
    try {
      const resp = await listProjects();
      setProjects(resp.projects);
      setBuiltins(resp.builtins ?? []);
    } catch (e) {
      setMessage({
        text: t("proj.list.failed", { msg: (e as Error).message }),
        tone: "error",
      });
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const errors = useMemo(
    () => (draft ? validate(draft, builtins) : []),
    [draft, builtins],
  );

  const canSave = draft != null && errors.length === 0 && !saving && dirty;

  const selectProject = useCallback(
    async (name: string) => {
      if (dirty && !window.confirm(t("proj.dirty.confirm"))) return;
      try {
        const spec = await getProject(name);
        setDraft(spec);
        setSelected(name);
        setIsNew(false);
        setDirty(false);
        setMessage(null);
      } catch (e) {
        setMessage({
          text: t("proj.load.failed", { msg: (e as Error).message }),
          tone: "error",
        });
      }
    },
    [dirty],
  );

  const startNew = useCallback(() => {
    if (dirty && !window.confirm(t("proj.dirty.confirm"))) return;
    setDraft(blankProject());
    setSelected(null);
    setIsNew(true);
    setDirty(true);
    setMessage(null);
  }, [dirty]);

  const update = useCallback((patch: Partial<ProjectSpec>) => {
    setDraft((d) => (d ? { ...d, ...patch } : d));
    setDirty(true);
  }, []);

  const updateFeature = useCallback(
    (idx: number, patch: Partial<FeatureSpec>) => {
      setDraft((d) =>
        d
          ? {
              ...d,
              features: d.features.map((f, i) =>
                i === idx ? { ...f, ...patch } : f,
              ),
            }
          : d,
      );
      setDirty(true);
    },
    [],
  );

  const updateTarget = useCallback(
    (idx: number, patch: Partial<TargetSpec>) => {
      setDraft((d) =>
        d
          ? {
              ...d,
              targets: d.targets.map((t, i) =>
                i === idx ? { ...t, ...patch } : t,
              ),
            }
          : d,
      );
      setDirty(true);
    },
    [],
  );

  const addFeature = () =>
    setDraft((d) =>
      d ? { ...d, features: [...d.features, blankFeature()] } : d,
    );

  const removeFeature = (idx: number) => {
    setDraft((d) =>
      d
        ? {
            ...d,
            features: d.features.filter((_, i) => i !== idx),
          }
        : d,
    );
    setDirty(true);
  };

  const addTarget = () =>
    setDraft((d) =>
      d ? { ...d, targets: [...d.targets, blankTarget()] } : d,
    );

  const removeTarget = (idx: number) => {
    setDraft((d) => {
      if (!d) return d;
      const next = d.targets.filter((_, i) => i !== idx);
      const removed = d.targets[idx]?.short_name;
      return {
        ...d,
        targets: next,
        default_target:
          d.default_target === removed
            ? next[0]?.short_name ?? ""
            : d.default_target,
      };
    });
    setDirty(true);
  };

  const onSave = async () => {
    if (!draft) return;
    setSaving(true);
    setMessage(null);
    const payload: ProjectSpec = {
      ...draft,
      name: draft.name.trim().toLowerCase(),
      default_target:
        draft.default_target || draft.targets[0]?.short_name || "",
    };
    try {
      if (isNew) await createProject(payload);
      else await updateProject(payload.name, payload);
      await refresh();
      setSelected(payload.name);
      setIsNew(false);
      setDirty(false);
      setMessage({
        text: isNew
          ? t("proj.save.ok.created", { name: payload.name })
          : t("proj.save.ok.updated", { name: payload.name }),
        tone: "ok",
      });
    } catch (e) {
      setMessage({
        text: t("proj.save.failed", { msg: (e as Error).message }),
        tone: "error",
      });
    } finally {
      setSaving(false);
    }
  };

  const onDelete = async () => {
    if (!selected) return;
    if (!window.confirm(t("proj.delete.confirm", { name: selected }))) return;
    try {
      await deleteProject(selected);
      await refresh();
      setSelected(null);
      setDraft(null);
      setIsNew(false);
      setDirty(false);
      setMessage({ text: t("proj.delete.ok", { name: selected }), tone: "ok" });
    } catch (e) {
      setMessage({
        text: t("proj.delete.failed", { msg: (e as Error).message }),
        tone: "error",
      });
    }
  };

  return (
    <div className="grid grid-cols-12 gap-4">
      <aside className="col-span-3 bg-white rounded-lg shadow-sm border p-3 flex flex-col gap-3 max-h-[75vh]">
        <div className="flex items-center justify-between">
          <h3 className="font-semibold text-sm text-slate-700 flex items-center gap-1.5">
            <FlaskConical size={16} className="text-indigo-500" />
            {t("proj.title")}
          </h3>
          <div className="flex items-center gap-1">
            <button
              onClick={refresh}
              title={t("proj.refresh")}
              className="p-1 rounded hover:bg-slate-100"
            >
              <RefreshCw size={14} />
            </button>
            <button
              onClick={startNew}
              title={t("proj.new")}
              className="p-1 rounded hover:bg-slate-100 text-indigo-600"
            >
              <FolderPlus size={14} />
            </button>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto flex flex-col gap-1">
          {projects.length === 0 && (
            <p className="text-xs text-slate-400 italic p-2">
              {t("proj.none")}
            </p>
          )}
          {projects.map((p) => {
            const isActive = selected === p.name && !isNew;
            return (
              <button
                key={p.name}
                onClick={() => selectProject(p.name)}
                className={
                  "text-left px-2 py-1.5 rounded text-sm border transition " +
                  (isActive
                    ? "bg-indigo-50 border-indigo-300 text-indigo-800"
                    : "bg-white border-transparent hover:bg-slate-50")
                }
              >
                <div className="font-mono text-xs text-slate-500">{p.name}</div>
                <div className="text-slate-800 truncate">
                  {p.display_name || p.name.toUpperCase()}
                </div>
                <div className="text-[10px] text-slate-400">
                  {p.features.length} {t("proj.form.features.col_name")} · {p.targets.length} {t("proj.form.targets.col_short")}
                </div>
              </button>
            );
          })}

          {builtins.length > 0 && (
            <div className="mt-3 border-t pt-2">
              <div className="flex items-center gap-1 text-[10px] uppercase tracking-wide text-slate-400 font-semibold px-2">
                <Lock size={10} /> {t("proj.builtin")}
              </div>
              {builtins.map((n) => (
                <div
                  key={n}
                  className="px-2 py-1 text-xs text-slate-400 font-mono"
                  title={t("proj.builtin.tooltip")}
                >
                  {n}
                </div>
              ))}
            </div>
          )}
        </div>
      </aside>

      <section className="col-span-9 bg-white rounded-lg shadow-sm border p-4">
        {!draft ? (
          <EmptyPane />
        ) : (
          <Editor
            draft={draft}
            update={update}
            updateFeature={updateFeature}
            updateTarget={updateTarget}
            addFeature={addFeature}
            removeFeature={removeFeature}
            addTarget={addTarget}
            removeTarget={removeTarget}
            isNew={isNew}
            dirty={dirty}
            errors={errors}
            canSave={canSave}
            saving={saving}
            message={message}
            onSave={onSave}
            onDelete={onDelete}
            builtins={builtins}
          />
        )}
      </section>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------
function EmptyPane() {
  return (
    <div className="h-full min-h-[60vh] flex flex-col items-center justify-center text-slate-400 gap-2">
      <FolderPlus size={36} className="text-slate-300" />
      <p className="text-sm">{t("proj.empty.title")}</p>
      <p className="text-xs max-w-md text-center mt-2 leading-relaxed">
        {t("proj.empty.desc")}
      </p>
    </div>
  );
}

interface EditorProps {
  draft: ProjectSpec;
  update: (patch: Partial<ProjectSpec>) => void;
  updateFeature: (idx: number, patch: Partial<FeatureSpec>) => void;
  updateTarget: (idx: number, patch: Partial<TargetSpec>) => void;
  addFeature: () => void;
  removeFeature: (idx: number) => void;
  addTarget: () => void;
  removeTarget: (idx: number) => void;
  isNew: boolean;
  dirty: boolean;
  errors: string[];
  canSave: boolean;
  saving: boolean;
  message: { text: string; tone: "ok" | "error" } | null;
  onSave: () => void;
  onDelete: () => void;
  builtins: string[];
}

function Editor(props: EditorProps) {
  const {
    draft,
    update,
    updateFeature,
    updateTarget,
    addFeature,
    removeFeature,
    addTarget,
    removeTarget,
    isNew,
    dirty,
    errors,
    canSave,
    saving,
    message,
    onSave,
    onDelete,
  } = props;

  return (
    <div className="flex flex-col gap-5">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold text-slate-800">
            {isNew
              ? t("proj.editing.new")
              : t("proj.editing.existing", { name: draft.display_name || draft.name })}
          </h2>
          <p className="text-xs text-slate-500">
            {isNew
              ? t("proj.editing.new.desc")
              : t("proj.editing.existing.desc", { name: draft.name })}
          </p>
        </div>
        <div className="flex gap-2">
          {!isNew && (
            <button
              onClick={onDelete}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs text-red-600 border border-red-200 rounded hover:bg-red-50"
            >
              <Trash2 size={14} /> {t("proj.delete")}
            </button>
          )}
          <button
            disabled={!canSave}
            onClick={onSave}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs text-white bg-indigo-600 rounded shadow-sm hover:bg-indigo-700 disabled:bg-slate-300 disabled:cursor-not-allowed"
          >
            <Save size={14} /> {saving ? t("proj.saving") : isNew ? t("proj.create") : t("proj.save")}
          </button>
        </div>
      </div>

      {message && (
        <div
          className={
            "text-xs rounded px-3 py-2 " +
            (message.tone === "ok"
              ? "bg-emerald-50 text-emerald-800 border border-emerald-200"
              : "bg-red-50 text-red-800 border border-red-200")
          }
        >
          {message.text}
        </div>
      )}
      {dirty && errors.length > 0 && (
        <div className="bg-amber-50 border border-amber-200 rounded px-3 py-2 text-xs text-amber-800">
          <strong>{t("proj.err.fix_before_save")}</strong>
          <ul className="mt-1 list-disc pl-5 space-y-0.5">
            {errors.map((e, i) => (
              <li key={i}>{e}</li>
            ))}
          </ul>
        </div>
      )}

      <Section icon={<Settings2 size={14} />} title={t("proj.form.identity")}>
        <div className="grid grid-cols-2 gap-3">
          <Field label={t("proj.form.name")} hint={t("proj.form.name.hint")}>
            <input
              value={draft.name}
              disabled={!isNew}
              onChange={(e) => update({ name: e.target.value.toLowerCase() })}
              placeholder={t("proj.form.name.placeholder")}
              className="w-full rounded border border-slate-300 px-2 py-1 text-sm disabled:bg-slate-100 font-mono"
            />
          </Field>
          <Field label={t("proj.form.display")}>
            <input
              value={draft.display_name}
              onChange={(e) => update({ display_name: e.target.value })}
              placeholder={t("proj.form.display.placeholder")}
              className="w-full rounded border border-slate-300 px-2 py-1 text-sm"
            />
          </Field>
        </div>
        <Field label={t("proj.form.desc")}>
          <textarea
            value={draft.description}
            onChange={(e) => update({ description: e.target.value })}
            rows={2}
            placeholder={t("proj.form.desc.placeholder")}
            className="w-full rounded border border-slate-300 px-2 py-1 text-sm"
          />
        </Field>
      </Section>

      <Section
        icon={<FlaskConical size={14} />}
        title={t("proj.form.features")}
        action={
          <button
            onClick={addFeature}
            className="inline-flex items-center gap-1 text-xs text-indigo-600 hover:text-indigo-800"
          >
            <Plus size={12} /> {t("proj.form.features.add")}
          </button>
        }
      >
        <div className="grid grid-cols-[1fr_100px_80px_80px_90px_1fr_30px] gap-2 text-[11px] uppercase tracking-wide text-slate-500 font-semibold pb-1">
          <div>{t("proj.form.features.col_name")}</div>
          <div>{t("proj.form.features.col_type")}</div>
          <div>{t("proj.form.features.col_lo")}</div>
          <div>{t("proj.form.features.col_hi")}</div>
          <div>{t("proj.form.features.col_unit")}</div>
          <div>{t("proj.form.features.col_display")}</div>
          <div />
        </div>
        <div className="flex flex-col gap-2">
          {draft.features.map((f, i) => (
            <div
              key={i}
              className="grid grid-cols-[1fr_100px_80px_80px_90px_1fr_30px] gap-2 items-center"
            >
              <input
                value={f.name}
                onChange={(e) => updateFeature(i, { name: e.target.value })}
                placeholder="temperature"
                className="rounded border border-slate-300 px-2 py-1 text-sm font-mono"
              />
              <select
                value={f.type}
                onChange={(e) =>
                  updateFeature(i, {
                    type: e.target.value as "continuous" | "integer",
                  })
                }
                className="rounded border border-slate-300 px-1.5 py-1 text-sm bg-white"
              >
                <option value="continuous">{t("proj.form.features.cont")}</option>
                <option value="integer">{t("proj.form.features.int")}</option>
              </select>
              <input
                type="number"
                value={f.lo}
                step={f.type === "integer" ? 1 : "any"}
                onChange={(e) =>
                  updateFeature(i, { lo: parseFloat(e.target.value) })
                }
                className="rounded border border-slate-300 px-2 py-1 text-sm"
              />
              <input
                type="number"
                value={f.hi}
                step={f.type === "integer" ? 1 : "any"}
                onChange={(e) =>
                  updateFeature(i, { hi: parseFloat(e.target.value) })
                }
                className="rounded border border-slate-300 px-2 py-1 text-sm"
              />
              <input
                value={f.unit ?? ""}
                onChange={(e) =>
                  updateFeature(i, { unit: e.target.value || null })
                }
                placeholder="°C"
                className="rounded border border-slate-300 px-2 py-1 text-sm"
              />
              <input
                value={f.display_name ?? ""}
                onChange={(e) =>
                  updateFeature(i, { display_name: e.target.value || null })
                }
                placeholder="(optional)"
                className="rounded border border-slate-300 px-2 py-1 text-sm"
              />
              <button
                onClick={() => removeFeature(i)}
                disabled={draft.features.length <= 1}
                title="Remove feature"
                className="p-1 rounded hover:bg-red-50 text-red-500 disabled:text-slate-300 disabled:hover:bg-transparent"
              >
                <X size={14} />
              </button>
            </div>
          ))}
        </div>
      </Section>

      <Section
        icon={<Target size={14} />}
        title={t("proj.form.targets")}
        action={
          <button
            onClick={addTarget}
            className="inline-flex items-center gap-1 text-xs text-indigo-600 hover:text-indigo-800"
          >
            <Plus size={12} /> {t("proj.form.targets.add")}
          </button>
        }
      >
        <div className="grid grid-cols-[100px_1fr_1fr_60px_90px_30px] gap-2 text-[11px] uppercase tracking-wide text-slate-500 font-semibold pb-1">
          <div>{t("proj.form.targets.col_short")}</div>
          <div>{t("proj.form.targets.col_csv")}</div>
          <div>{t("proj.form.targets.col_display")}</div>
          <div>{t("proj.form.targets.col_unit")}</div>
          <div>{t("proj.form.targets.col_competing")}</div>
          <div />
        </div>
        <div className="flex flex-col gap-2">
          {draft.targets.map((tg, i) => (
            <div
              key={i}
              className="grid grid-cols-[100px_1fr_1fr_60px_90px_30px] gap-2 items-center"
            >
              <input
                value={tg.short_name}
                onChange={(e) =>
                  updateTarget(i, { short_name: e.target.value })
                }
                placeholder="CO"
                className="rounded border border-slate-300 px-2 py-1 text-sm font-mono"
              />
              <input
                value={tg.column}
                onChange={(e) => updateTarget(i, { column: e.target.value })}
                placeholder="Y_CO"
                className="rounded border border-slate-300 px-2 py-1 text-sm font-mono"
              />
              <input
                value={tg.display_name ?? ""}
                onChange={(e) =>
                  updateTarget(i, { display_name: e.target.value || null })
                }
                placeholder="Carbon monoxide"
                className="rounded border border-slate-300 px-2 py-1 text-sm"
              />
              <input
                value={tg.unit ?? ""}
                onChange={(e) =>
                  updateTarget(i, { unit: e.target.value || null })
                }
                placeholder="%"
                className="rounded border border-slate-300 px-2 py-1 text-sm"
              />
              <label className="inline-flex items-center gap-1.5 text-xs text-slate-600 pl-1">
                <input
                  type="checkbox"
                  checked={tg.is_competing}
                  onChange={(e) =>
                    updateTarget(i, { is_competing: e.target.checked })
                  }
                />
                {t("proj.form.targets.side_rxn")}
              </label>
              <button
                onClick={() => removeTarget(i)}
                disabled={draft.targets.length <= 1}
                title="Remove target"
                className="p-1 rounded hover:bg-red-50 text-red-500 disabled:text-slate-300 disabled:hover:bg-transparent"
              >
                <X size={14} />
              </button>
            </div>
          ))}
        </div>

        <div className="mt-3 flex items-center gap-3">
          <Field label={t("proj.form.targets.default")} inline>
            <select
              value={draft.default_target}
              onChange={(e) => update({ default_target: e.target.value })}
              className="rounded border border-slate-300 px-2 py-1 text-sm bg-white"
            >
              {draft.targets.map((t, i) => (
                <option key={i} value={t.short_name || `#${i}`}>
                  {t.short_name || `(target #${i + 1})`}
                </option>
              ))}
            </select>
          </Field>
          <p className="text-[11px] text-slate-400 max-w-md">
            {t("proj.form.targets.hint")}
          </p>
        </div>
      </Section>

      <Section title={t("proj.form.notes")} icon={<Settings2 size={14} />}>
        <textarea
          value={draft.notes}
          onChange={(e) => update({ notes: e.target.value })}
          rows={3}
          placeholder={t("proj.form.notes.placeholder")}
          className="w-full rounded border border-slate-300 px-2 py-1 text-sm font-mono"
        />
      </Section>
    </div>
  );
}

function Section(props: {
  title: string;
  icon?: React.ReactNode;
  action?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <div className="border rounded-lg p-3 bg-slate-50/60">
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-sm font-semibold text-slate-700 flex items-center gap-1.5">
          {props.icon}
          {props.title}
        </h3>
        {props.action}
      </div>
      {props.children}
    </div>
  );
}

function Field(props: {
  label: string;
  hint?: string;
  inline?: boolean;
  children: React.ReactNode;
}) {
  return (
    <label
      className={
        props.inline
          ? "inline-flex items-center gap-2 text-xs text-slate-600"
          : "block text-xs text-slate-600 mb-2"
      }
    >
      <span className="font-semibold tracking-wide uppercase text-[10px] text-slate-500">
        {props.label}
      </span>
      <div className="mt-0.5">{props.children}</div>
      {props.hint && (
        <span className="text-[10px] text-slate-400 italic">{props.hint}</span>
      )}
    </label>
  );
}
