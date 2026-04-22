import { useEffect, useState } from "react";
import {
  FilePlus2,
  RefreshCw,
  Save,
  Trash2,
} from "lucide-react";
import {
  deletePriorFile,
  listPriorFiles,
  readPriorFile,
  writePriorFile,
} from "../api";
import type { FileEntry } from "../types";

/** JSON editor for expert prior files under `priors/`. */
export default function PriorsManager() {
  const [files, setFiles] = useState<FileEntry[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [text, setText] = useState("");
  const [dirty, setDirty] = useState(false);
  const [saving, setSaving] = useState(false);
  const [parseOk, setParseOk] = useState(true);
  const [msg, setMsg] = useState<string | null>(null);
  const [newFile, setNewFile] = useState("");

  async function refresh() {
    try {
      const r = await listPriorFiles();
      setFiles(r.files);
    } catch (e) {
      setMsg(String(e));
    }
  }
  useEffect(() => {
    refresh();
  }, []);

  async function loadFile(path: string) {
    try {
      setSelected(path);
      const r = await readPriorFile(path);
      const body = r.content ?? r.content_text ?? {};
      const pretty =
        typeof body === "string"
          ? body
          : JSON.stringify(body, null, 2);
      setText(pretty);
      setDirty(false);
      setParseOk(true);
      setMsg(null);
    } catch (e) {
      setMsg(String(e));
    }
  }

  function updateText(t: string) {
    setText(t);
    setDirty(true);
    try {
      JSON.parse(t);
      setParseOk(true);
    } catch {
      setParseOk(false);
    }
  }

  async function save() {
    if (!selected) return;
    try {
      const obj = JSON.parse(text);
      setSaving(true);
      await writePriorFile(selected, obj);
      setDirty(false);
      setMsg(`Saved ${selected}`);
      await refresh();
    } catch (e) {
      setMsg(`Invalid JSON or write failed: ${e}`);
    } finally {
      setSaving(false);
    }
  }

  async function remove(path: string) {
    if (!confirm(`Delete ${path}?`)) return;
    try {
      await deletePriorFile(path);
      if (selected === path) {
        setSelected(null);
        setText("");
      }
      await refresh();
    } catch (e) {
      setMsg(String(e));
    }
  }

  async function createNew() {
    if (!newFile.trim()) return;
    const name = newFile.trim().endsWith(".json")
      ? newFile.trim()
      : `${newFile.trim()}.json`;
    const path = name.startsWith("priors/") ? name.slice(7) : name;
    try {
      await writePriorFile(path, { _description: "New prior file" });
      setNewFile("");
      await refresh();
      loadFile(path);
    } catch (e) {
      setMsg(String(e));
    }
  }

  return (
    <div className="grid grid-cols-1 xl:grid-cols-[320px_1fr] gap-6">
      <aside className="rounded-lg border border-slate-200 bg-white">
        <div className="px-3 py-2 border-b bg-slate-50 text-xs font-semibold uppercase text-slate-600 flex items-center justify-between">
          <span>JSON priors (priors/)</span>
          <button
            onClick={refresh}
            className="p-1 rounded hover:bg-slate-200 text-slate-500"
            title="Refresh"
          >
            <RefreshCw className="w-3.5 h-3.5" />
          </button>
        </div>

        <ul className="max-h-[420px] overflow-auto log-scroll">
          {files.length === 0 ? (
            <li className="p-3 text-xs text-slate-400">No JSON files yet.</li>
          ) : (
            files.map((f) => {
              const rel = f.path.replace(/^priors\//, "");
              return (
                <li
                  key={f.path}
                  className={`px-3 py-2 border-b flex items-center gap-2 cursor-pointer text-xs ${
                    selected === rel
                      ? "bg-brand-50 text-brand-800"
                      : "hover:bg-slate-50"
                  }`}
                  onClick={() => loadFile(rel)}
                >
                  <span className="truncate flex-1 font-mono">{rel}</span>
                  <button
                    className="text-slate-400 hover:text-rose-500"
                    onClick={(e) => {
                      e.stopPropagation();
                      remove(rel);
                    }}
                    title="Delete"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </li>
              );
            })
          )}
        </ul>

        <div className="p-3 border-t">
          <div className="flex gap-2">
            <input
              value={newFile}
              onChange={(e) => setNewFile(e.target.value)}
              placeholder="new_prior.json"
              className="flex-1 px-2 py-1 rounded border border-slate-300 text-sm"
            />
            <button
              onClick={createNew}
              className="px-2 py-1 rounded bg-slate-100 hover:bg-slate-200 text-xs inline-flex items-center gap-1"
            >
              <FilePlus2 className="w-3.5 h-3.5" /> New
            </button>
          </div>
        </div>
      </aside>

      <section className="rounded-lg border border-slate-200 bg-white flex flex-col min-w-0">
        <header className="px-4 py-3 border-b bg-slate-50 flex items-center gap-3">
          <span className="text-sm font-semibold text-slate-700 font-mono">
            {selected ?? "(no file selected)"}
          </span>
          {dirty ? (
            <span className="text-[10px] uppercase text-amber-700 bg-amber-100 rounded px-1.5 py-0.5">
              unsaved
            </span>
          ) : null}
          {!parseOk ? (
            <span className="text-[10px] uppercase text-rose-700 bg-rose-100 rounded px-1.5 py-0.5">
              invalid JSON
            </span>
          ) : null}
          <div className="flex-1" />
          <button
            onClick={save}
            disabled={!selected || !dirty || saving || !parseOk}
            className="inline-flex items-center gap-1 px-3 py-1.5 rounded-md bg-brand-600 text-white hover:bg-brand-700 text-sm disabled:opacity-50"
          >
            <Save className="w-3.5 h-3.5" />
            {saving ? "Saving…" : "Save"}
          </button>
        </header>
        <textarea
          value={text}
          onChange={(e) => updateText(e.target.value)}
          spellCheck={false}
          className="flex-1 w-full font-mono text-xs p-3 outline-none resize-none min-h-[420px] log-scroll"
          placeholder='{ "feature_name": { "type": "gaussian", "mean": 0, "std": 1 } }'
          disabled={!selected}
        />
        {msg ? (
          <div className="px-4 py-2 border-t text-xs text-slate-600">{msg}</div>
        ) : null}
      </section>
    </div>
  );
}
