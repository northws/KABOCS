import { useEffect, useState } from "react";
import {
  Download,
  FilePlus2,
  RefreshCw,
  Save,
  Trash2,
  Upload,
} from "lucide-react";
import {
  deleteDataFile,
  listDataFiles,
  readDataFile,
  writeDataFile,
} from "../api";
import { t } from "../i18n";
import type { FileEntry } from "../types";

/** CSV file browser + in-browser editor for the `data/` directory. */
export default function DataManager() {
  const [files, setFiles] = useState<FileEntry[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [content, setContent] = useState<string>("");
  const [dirty, setDirty] = useState(false);
  const [saving, setSaving] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);
  const [newFile, setNewFile] = useState("");

  async function refresh() {
    try {
      const r = await listDataFiles();
      setFiles(r.files);
      if (!r.files.find((f) => f.name === selected) && r.files.length > 0) {
        loadFile(r.files[0].name);
      }
    } catch (e) {
      setMsg(String(e));
    }
  }

  useEffect(() => {
    refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function loadFile(name: string) {
    try {
      setSelected(name);
      const r = await readDataFile(name);
      setContent(r.content);
      setDirty(false);
      setMsg(null);
    } catch (e) {
      setMsg(String(e));
    }
  }

  async function save() {
    if (!selected) return;
    setSaving(true);
    try {
      await writeDataFile(selected, content);
      setDirty(false);
      setMsg(`${t("data.save")} ${selected}`);
      await refresh();
    } catch (e) {
      setMsg(String(e));
    } finally {
      setSaving(false);
    }
  }

  async function remove(name: string) {
    if (!confirm(`Delete ${name}?`)) return;
    try {
      await deleteDataFile(name);
      if (selected === name) {
        setSelected(null);
        setContent("");
      }
      await refresh();
    } catch (e) {
      setMsg(String(e));
    }
  }

  async function onUpload(file: File) {
    const text = await file.text();
    try {
      await writeDataFile(file.name, text);
      await refresh();
      loadFile(file.name);
      setMsg(`Uploaded ${file.name}`);
    } catch (e) {
      setMsg(String(e));
    }
  }

  async function createNew() {
    if (!newFile.trim()) return;
    const name = newFile.trim().endsWith(".csv") ? newFile.trim() : `${newFile.trim()}.csv`;
    try {
      await writeDataFile(name, "");
      setNewFile("");
      await refresh();
      loadFile(name);
    } catch (e) {
      setMsg(String(e));
    }
  }

  return (
    <div className="grid grid-cols-1 xl:grid-cols-[280px_1fr] gap-6">
      <aside className="rounded-lg border border-slate-200 bg-white">
        <div className="px-3 py-2 border-b bg-slate-50 text-xs font-semibold uppercase text-slate-600 flex items-center justify-between">
          <span>{t("data.title")}</span>
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
            <li className="p-3 text-xs text-slate-400">{t("data.none")}</li>
          ) : (
            files.map((f) => (
              <li
                key={f.name}
                className={`px-3 py-2 border-b flex items-center gap-2 cursor-pointer text-sm ${
                  selected === f.name
                    ? "bg-brand-50 text-brand-800"
                    : "hover:bg-slate-50"
                }`}
                onClick={() => loadFile(f.name)}
              >
                <span className="truncate flex-1">{f.name}</span>
                <span className="text-[10px] text-slate-500 num">
                  {fmtSize(f.size)}
                </span>
                <button
                  className="text-slate-400 hover:text-rose-500"
                  onClick={(e) => {
                    e.stopPropagation();
                    remove(f.name);
                  }}
                    title="Delete"
                  >
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
              </li>
            ))
          )}
        </ul>

        <div className="p-3 border-t space-y-2">
          <div className="flex gap-2">
            <input
              value={newFile}
              onChange={(e) => setNewFile(e.target.value)}
              placeholder="new_file.csv"
              className="flex-1 px-2 py-1 rounded border border-slate-300 text-sm"
            />
            <button
              onClick={createNew}
              className="px-2 py-1 rounded bg-slate-100 hover:bg-slate-200 text-xs inline-flex items-center gap-1"
            >
              <FilePlus2 className="w-3.5 h-3.5" /> {t("data.new")}
            </button>
          </div>
          <label className="flex items-center gap-2 text-xs text-slate-600 cursor-pointer">
            <Upload className="w-3.5 h-3.5" /> {t("data.upload")}
            <input
              type="file"
              accept=".csv,text/csv"
              className="hidden"
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) onUpload(f);
              }}
            />
          </label>
        </div>
      </aside>

      <section className="rounded-lg border border-slate-200 bg-white flex flex-col min-w-0">
        <header className="px-4 py-3 border-b bg-slate-50 flex items-center gap-3">
          <span className="text-sm font-semibold text-slate-700">
            {selected ?? t("data.nofile")}
          </span>
          {dirty ? (
            <span className="text-[10px] uppercase text-amber-700 bg-amber-100 rounded px-1.5 py-0.5">
              {t("data.unsaved")}
            </span>
          ) : null}
          <div className="flex-1" />
          {selected ? (
            <a
              href={`/api/files/data/${encodeURIComponent(selected)}?raw=true`}
              target="_blank"
              rel="noreferrer"
              className="text-xs text-slate-600 inline-flex items-center gap-1 hover:text-brand-700"
            >
              <Download className="w-3.5 h-3.5" /> {t("data.download")}
            </a>
          ) : null}
          <button
            onClick={save}
            disabled={!selected || !dirty || saving}
            className="inline-flex items-center gap-1 px-3 py-1.5 rounded-md bg-brand-600 text-white hover:bg-brand-700 text-sm disabled:opacity-50"
          >
            <Save className="w-3.5 h-3.5" />
            {saving ? t("data.saving") : t("data.save")}
          </button>
        </header>
        <textarea
          value={content}
          onChange={(e) => {
            setContent(e.target.value);
            setDirty(true);
          }}
          spellCheck={false}
          className="flex-1 w-full font-mono text-xs p-3 outline-none resize-none min-h-[420px] log-scroll"
          placeholder={t("data.placeholder")}
          disabled={!selected}
        />
        {msg ? (
          <div className="px-4 py-2 border-t text-xs text-slate-600">{msg}</div>
        ) : null}
      </section>
    </div>
  );
}

function fmtSize(n: number): string {
  if (n < 1024) return `${n}B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)}KB`;
  return `${(n / 1024 / 1024).toFixed(1)}MB`;
}
