import { useSyncExternalStore } from "react";

export type Lang = "zh" | "en";

export const zh: Record<string, string> = {
  // ---------- App ----------
  "tab.run": "运行",
  "tab.projects": "项目",
  "tab.data": "数据",
  "tab.priors": "先验",
  "tab.history": "历史",
  "app.footer": "KABOCS 网页界面 · 面向催化系统的知识增强贝叶斯优化",

  // ---------- TopBar ----------
  "topbar.title": "KABOCS",
  "topbar.subtitle": "知识增强贝叶斯优化",
  "topbar.refresh": "刷新状态",

  // ---------- RunPage ----------
  "run.config": "配置",
  "run.stop": "停止",
  "run.start": "开始运行",
  "run.starting": "启动中…",
  "run.stream": "流",
  "run.stream.connected": "已连接",
  "run.stream.offline": "离线",
  "run.status": "状态",
  "run.error": "运行出错：",
  "run.loading.tasks": "加载任务失败：",
  "run.best": "当前最佳实验",
  "run.completed": "运行已完成。",
  "run.failed": "运行失败",
  "run.norecs": "尚未收到推荐。",

  // ---------- ConfigPanel ----------
  "config.core": "核心",
  "config.task": "任务",
  "config.task.hint.builtin": "内置（Python 定义）",
  "config.task.hint.project": "项目定义 — 可在「项目」标签编辑",
  "config.target": "目标产物",
  "config.target.default": "（默认：",
  "config.data": "数据 CSV",
  "config.candidates": "候选 CSV",
  "config.candidates.hint": "输入 none 跳过候选池",
  "config.iterations": "迭代次数",
  "config.topk": "Top K 特征",
  "config.seed": "随机种子",
  "config.seed.ph": "（不指定）",
  "config.device": "设备",
  "config.features": "个特征",
  "config.products": "个产物",
  "config.default_target": "默认目标 →",
  "config.interactive": "交互模式（专家介入）",
  "config.skip_fs": "跳过特征选择",
  "config.strict_schema": "严格训练模式",
  "config.prefill": "选择前预先填充配方",

  "config.acq": "采集函数",
  "config.strategy": "策略",
  "config.kernel": "核函数",
  "config.beta": "β",
  "config.beta_schedule": "β 调度",
  "config.beta_delta": "β δ",
  "config.qnei_mc": "qNEI MC 采样数",
  "config.h2_penalty": "H₂ 惩罚权重",
  "config.diversity": "多样性权重",
  "config.discrete_strategy": "离散策略",
  "config.gen_candidates_n": "生成候选数 N",
  "config.prefer_csv": "优先使用 CSV 候选文件而非生成器",

  "config.kabo": "KABO 模式",
  "config.kabo.enable": "启 用知识增强模式",
  "config.kabo.lambda_p": "λ_p（偏好）",
  "config.kabo.lambda_k": "λ_k（专家先验）",
  "config.kabo.lambda_v": "λ_v（VOI）",
  "config.kabo.pe_budget": "PE 预算 / 轮",
  "config.kabo.prior": "专家先验 JSON",
  "config.kabo.prior.hint": "相对于项目根目录",
  "config.kabo.pe_hint":
    "PE 查询（pe_budget > 0）在 WebUI 中使用通用文字提示。如无需求可设为 0。",

  "config.builtin": "内置",
  "config.project": "项目",

  // ---------- PromptPanel ----------
  "prompt.waiting": "正在等待优化器发出下一次提示…",
  "prompt.awaiting": "等待您的输入",
  "prompt.candidate.title": "请选择要执行的候选方案",
  "prompt.candidate.desc":
    "粗体编号为模型首选推荐；选择其他方案在启用 KABO 模式时将\n记录为偏好信号。",
  "prompt.candidate.rank": "Rank #",
  "prompt.candidate.tie": "声明平局",
  "prompt.candidate.manual": "手动覆盖",
  "prompt.candidate.stop": "停止优化",
  "prompt.manual.title": "手动输入候选方案",
  "prompt.manual.desc":
    "填写自定义方案的完整特征值。空单元格将使用设计空间的中点。",
  "prompt.manual.oob": "以下特征值超出设计空间边界：",
  "prompt.manual.accept": "我确认",
  "prompt.nonselected.title": "填写非选择特征值",
  "prompt.nonselected.desc": "这些特征不在 GP 模型中，请填入您计划使用的实际实验条件。",
  "prompt.yields.title": "输入实验产率",
  "prompt.yields.desc":
    "记录每种产物的实验产率。目标产物已高亮。未检测到的产物输入 0。",
  "prompt.yields.target": "★ 目标",
  "prompt.yields.placeholder": "例如 42.5",
  "prompt.raw.title": "原始输入",
  "prompt.raw.placeholder": "您的回答（如 a / b / tie）",
  "prompt.submit": "提交",
  "prompt.submit.exit": "停止优化",

  "prompt.column.feature": "特征",
  "prompt.column.bounds": "范围",
  "prompt.column.value": "值",
  "prompt.column.product": "产物",
  "prompt.column.col": "列",
  "prompt.column.yield": "产率",

  // ---------- RecommendationList ----------
  "rec.title": "迭代 {iteration} · 前 {n} 项推荐",
  "rec.optimizing": "优化 {target}",
  "rec.none": "此运行尚未产生推荐。",
  "rec.selected": "已选",
  "rec.expert": "专家",
  "rec.fixed": "固定",
  "rec.pending": "待定",

  // ---------- LogStream ----------
  "log.lines": "日志流（{count} 行）",
  "log.none": "（尚无日志 — 启动运行后可查看优化器输出）",

  // ---------- Dashboard ----------
  "dash.title": "历史运行（{count}）",
  "dash.none": "暂无归档运行。请从「运行」标签页启动一个优化。",
  "dash.hint": "从侧边栏选择一个归档运行查看详情。",
  "dash.metadata": "元数据",
  "dash.feature_imp": "特征重要性",
  "dash.feature_imp.na": "（图像不可用 — 可能跳过了特征选择？）",
  "dash.beta_trace": "β 调度轨迹",
  "dash.data": "data_updated.csv",
  "dash.data.rows": "行 · ",
  "dash.data.cols": "列",
  "dash.data.none": "此运行没有更新后的数据集。",
  "dash.delete.confirm": "确定删除运行 {id}？",

  // ---------- DataManager ----------
  "data.title": "CSV 文件（data/）",
  "data.none": "暂无 CSV 文件。",
  "data.unsaved": "未保存",
  "data.download": "下载",
  "data.saving": "保存中…",
  "data.save": "保存",
  "data.new": "新建",
  "data.upload": "上传 CSV",
  "data.delete.confirm": "确定删除 {name}？",
  "data.nofile": "（未选择文件）",
  "data.placeholder": "CSV 内容…",

  // ---------- PriorsManager ----------
  "priors.title": "JSON 先验（priors/）",
  "priors.none": "暂无 JSON 文件。",
  "priors.unsaved": "未保存",
  "priors.invalid": "无效 JSON",
  "priors.saving": "保存中…",
  "priors.save": "保存",
  "priors.new": "新建",
  "priors.delete.confirm": "确定删除 {path}？",
  "priors.nofile": "（未选择文件）",
  "priors.placeholder":
    '{ "特征名": { "type": "gaussian", "mean": 0, "std": 1 } }',

  // ---------- ProjectsManager ----------
  "proj.title": "项目",
  "proj.refresh": "刷新",
  "proj.new": "新建项目",
  "proj.none": "暂无项目。点击文件夹图标创建一个。",
  "proj.builtin": "内置（只读）",
  "proj.builtin.tooltip": "在 Python 中定义（kabo/task/*.py）",
  "proj.empty.title": "选择一个项目，或点击新建",
  "proj.empty.desc":
    "项目声明催化优化目标的特征模式、设计空间边界和产物列（如 CO2RR、ORR、OER）。保存后将注册为动态 TaskBase，可在「运行」标签页中选择。",
  "proj.editing.new": "新建项目",
  "proj.editing.existing": "编辑：{name}",
  "proj.editing.new.desc": "定义一个新的催化优化目标。",
  "proj.editing.existing.desc": "已保存到 projects/{name}.json",
  "proj.delete": "删除",
  "proj.creating": "创建中…",
  "proj.create": "创建",
  "proj.saving": "保存中…",
  "proj.save": "保存",
  "proj.save.ok.created": "项目「{name}」已创建。",
  "proj.save.ok.updated": "项目「{name}」已更新。",
  "proj.save.failed": "保存失败：{msg}",
  "proj.delete.confirm": "确定删除项目「{name}」？此操作不可撤销。",
  "proj.delete.ok": "项目「{name}」已删除。",
  "proj.delete.failed": "删除失败：{msg}",
  "proj.dirty.confirm": "放弃未保存的更改？",
  "proj.list.failed": "加载项目列表失败：{msg}",
  "proj.load.failed": "加载项目失败：{msg}",

  // Validation
  "proj.err.name_required": "名称不能为空。",
  "proj.err.name_chars": "名称只能包含小写字母、数字、下划线和连字符。",
  "proj.err.name_collision": "名称「{name}」与内置任务冲突。",
  "proj.err.need_feature": "至少需要一个特征。",
  "proj.err.feature_name": "特征 #{i}：名称不能为空。",
  "proj.err.feature_dup": "特征名称重复：「{name}」。",
  "proj.err.feature_bounds": "特征「{name}」：边界必须为数值。",
  "proj.err.feature_hi_lo": "特征「{name}」：最大值（{hi}）必须大于最小值（{lo}）。",
  "proj.err.need_target": "至少需要一个目标。",
  "proj.err.target_name": "目标 #{i}：短名称不能为空。",
  "proj.err.target_dup_name": "目标短名称重复：「{name}」。",
  "proj.err.target_column": "目标 #{i}：列名不能为空。",
  "proj.err.target_dup_col": "目标列重复：「{col}」。",
  "proj.err.target_short_mismatch":
    "默认目标「{target}」必须匹配一个目标的短名称。",
  "proj.err.fix_before_save": "保存前请修正以下错误：",

  // Project form
  "proj.form.identity": "标识",
  "proj.form.name": "名称（小写标识符）",
  "proj.form.name.hint": "用作 --task 参数",
  "proj.form.name.placeholder": "例如 orr",
  "proj.form.display": "显示名称",
  "proj.form.display.placeholder": "Oxygen Reduction Reaction",
  "proj.form.desc": "描述",
  "proj.form.desc.placeholder": "系统的简短摘要",
  "proj.form.features": "特征（描述符）",
  "proj.form.features.add": "添加",
  "proj.form.features.col_name": "名称",
  "proj.form.features.col_type": "类型",
  "proj.form.features.col_lo": "最小值",
  "proj.form.features.col_hi": "最大值",
  "proj.form.features.col_unit": "单位",
  "proj.form.features.col_display": "显示名称",
  "proj.form.features.cont": "连续",
  "proj.form.features.int": "整数",
  "proj.form.targets": "目标（产物产率）",
  "proj.form.targets.add": "添加",
  "proj.form.targets.col_short": "短名称",
  "proj.form.targets.col_csv": "CSV 列",
  "proj.form.targets.col_display": "显示名称",
  "proj.form.targets.col_unit": "单位",
  "proj.form.targets.col_competing": "副反应",
  "proj.form.targets.default": "默认目标",
  "proj.form.targets.side_rxn": "副反应",
  "proj.form.targets.hint":
    "当 CLI/WebUI 使用 h2_penalty_weight > 0 运行时，副反应目标将从训练目标中减去。",
  "proj.form.notes": "备注",
  "proj.form.notes.placeholder": "关于此项目的自由文本（优化器不使用）。",
};

export const en: Record<string, string> = {
  // ---------- App ----------
  "tab.run": "Run",
  "tab.projects": "Projects",
  "tab.data": "Data",
  "tab.priors": "Priors",
  "tab.history": "History",
  "app.footer":
    "KABOCS WebUI · Knowledge-Augmented Bayesian Optimization for Catalytic Systems",

  // ---------- TopBar ----------
  "topbar.title": "KABOCS",
  "topbar.subtitle": "Knowledge-Augmented Bayesian Optimization",
  "topbar.refresh": "Refresh status",

  // ---------- RunPage ----------
  "run.config": "Configuration",
  "run.stop": "Stop",
  "run.start": "Start run",
  "run.starting": "Starting…",
  "run.stream": "stream",
  "run.stream.connected": "connected",
  "run.stream.offline": "offline",
  "run.status": "status",
  "run.error": "Run error:",
  "run.loading.tasks": "Failed to load tasks:",
  "run.best": "Best experiment so far",
  "run.completed": "Run completed.",
  "run.failed": "Run failed",
  "run.norecs": "No recommendations yet.",

  // ---------- ConfigPanel ----------
  "config.core": "Core",
  "config.task": "Task",
  "config.task.hint.builtin": "Built-in (Python-defined)",
  "config.task.hint.project": "Project-defined — edit in Projects tab",
  "config.target": "Target product",
  "config.target.default": "(default: ",
  "config.data": "Data CSV",
  "config.candidates": "Candidates CSV",
  "config.candidates.hint": "'none' to skip discrete pool",
  "config.iterations": "Iterations",
  "config.topk": "Top K features",
  "config.seed": "Seed",
  "config.seed.ph": "(none)",
  "config.device": "Device",
  "config.features": "features",
  "config.products": "products",
  "config.default_target": "default target →",
  "config.interactive": "Interactive (expert-in-the-loop)",
  "config.skip_fs": "Skip feature selection",
  "config.strict_schema": "Strict training schema",
  "config.prefill": "Pre-fill recipes before choice",

  "config.acq": "Acquisition",
  "config.strategy": "Strategy",
  "config.kernel": "Kernel",
  "config.beta": "β",
  "config.beta_schedule": "β schedule",
  "config.beta_delta": "β δ",
  "config.qnei_mc": "qNEI MC samples",
  "config.h2_penalty": "H₂ penalty weight",
  "config.diversity": "Diversity weight",
  "config.discrete_strategy": "Discrete strategy",
  "config.gen_candidates_n": "Generated candidates N",
  "config.prefer_csv": "Prefer candidates CSV over generator",

  "config.kabo": "KABO mode",
  "config.kabo.enable": "Enable knowledge-augmented mode",
  "config.kabo.lambda_p": "λ_p (preference)",
  "config.kabo.lambda_k": "λ_k (expert prior)",
  "config.kabo.lambda_v": "λ_v (VOI)",
  "config.kabo.pe_budget": "PE budget / iter",
  "config.kabo.prior": "Expert prior JSON",
  "config.kabo.prior.hint": "relative to project root",
  "config.kabo.pe_hint":
    "PE queries (pe_budget > 0) use generic text prompts in the web UI. Set to 0 unless you need preference exploration.",

  "config.builtin": "Built-in",
  "config.project": "Projects",

  // ---------- PromptPanel ----------
  "prompt.waiting": "Waiting for the optimizer to issue the next prompt…",
  "prompt.awaiting": "Awaiting your input",
  "prompt.candidate.title": "Choose the candidate to run",
  "prompt.candidate.desc":
    "The bold rank is the model's top recommendation; choosing a different one is recorded as a preference signal when KABO mode is enabled.",
  "prompt.candidate.rank": "Rank #",
  "prompt.candidate.tie": "Declare tie",
  "prompt.candidate.manual": "Manual override",
  "prompt.candidate.stop": "Stop optimization",
  "prompt.manual.title": "Manual candidate entry",
  "prompt.manual.desc":
    "Supply full feature values for a custom candidate. Empty cells will default to the design-space midpoint.",
  "prompt.manual.oob": "Out-of-bounds values for",
  "prompt.manual.accept": "I accept",
  "prompt.nonselected.title": "Fill non-selected features",
  "prompt.nonselected.desc":
    "These features are outside the GP model; fill in the actual experimental conditions you plan to use.",
  "prompt.yields.title": "Enter experimental yields",
  "prompt.yields.desc":
    "Record the experimental yield for every product. Target is highlighted. Enter 0 for undetected products.",
  "prompt.yields.target": "★ TARGET",
  "prompt.yields.placeholder": "e.g. 42.5",
  "prompt.raw.title": "Raw input",
  "prompt.raw.placeholder": "Your response (e.g. a / b / tie)",
  "prompt.submit": "Submit",
  "prompt.submit.exit": "Stop optimization",

  "prompt.column.feature": "Feature",
  "prompt.column.bounds": "Bounds",
  "prompt.column.value": "Value",
  "prompt.column.product": "Product",
  "prompt.column.col": "Column",
  "prompt.column.yield": "Yield",

  // ---------- RecommendationList ----------
  "rec.title": "Iter {iteration} · top {n} recommendations",
  "rec.optimizing": "optimizing {target}",
  "rec.none": "No recommendations yet for this run.",
  "rec.selected": "selected",
  "rec.expert": "expert",
  "rec.fixed": "fixed",
  "rec.pending": "pending",

  // ---------- LogStream ----------
  "log.lines": "Log stream ({count} lines)",
  "log.none": "(no logs yet — start a run to see optimizer output)",

  // ---------- Dashboard ----------
  "dash.title": "Past runs ({count})",
  "dash.none": "No archived runs yet. Launch an optimization from the Run tab.",
  "dash.hint": "Select an archived run from the sidebar to inspect details.",
  "dash.metadata": "Metadata",
  "dash.feature_imp": "Feature importance",
  "dash.feature_imp.na":
    "(plot unavailable — feature selection might have been skipped?)",
  "dash.beta_trace": "β schedule trace",
  "dash.data": "data_updated.csv",
  "dash.data.rows": "rows · ",
  "dash.data.cols": "cols",
  "dash.data.none": "This run does not have an updated dataset.",
  "dash.delete.confirm": "Delete run {id}?",

  // ---------- DataManager ----------
  "data.title": "CSV files (data/)",
  "data.none": "No CSV files yet.",
  "data.unsaved": "unsaved",
  "data.download": "Download",
  "data.saving": "Saving…",
  "data.save": "Save",
  "data.new": "New",
  "data.upload": "Upload CSV",
  "data.delete.confirm": "Delete {name}?",
  "data.nofile": "(no file selected)",
  "data.placeholder": "CSV contents…",

  // ---------- PriorsManager ----------
  "priors.title": "JSON priors (priors/)",
  "priors.none": "No JSON files yet.",
  "priors.unsaved": "unsaved",
  "priors.invalid": "invalid JSON",
  "priors.saving": "Saving…",
  "priors.save": "Save",
  "priors.new": "New",
  "priors.delete.confirm": "Delete {path}?",
  "priors.nofile": "(no file selected)",
  "priors.placeholder":
    '{ "feature": { "type": "gaussian", "mean": 0, "std": 1 } }',

  // ---------- ProjectsManager ----------
  "proj.title": "Projects",
  "proj.refresh": "Refresh",
  "proj.new": "New project",
  "proj.none": "No projects yet. Click the folder icon to create one.",
  "proj.builtin": "Built-in (read-only)",
  "proj.builtin.tooltip": "Defined in Python (kabo/task/*.py)",
  "proj.empty.title": "Select a project, or click New",
  "proj.empty.desc":
    "Projects declare feature schemas, design-space bounds, and product columns for catalytic optimization targets (e.g. CO2RR, ORR, OER). Saved projects are registered as dynamic TaskBase instances selectable from the Run tab.",
  "proj.editing.new": "New project",
  "proj.editing.existing": "Editing: {name}",
  "proj.editing.new.desc": "Define a new catalytic optimization target.",
  "proj.editing.existing.desc": "Saved to projects/{name}.json",
  "proj.delete": "Delete",
  "proj.creating": "Creating…",
  "proj.create": "Create",
  "proj.saving": "Saving…",
  "proj.save": "Save",
  "proj.save.ok.created": "Project '{name}' created.",
  "proj.save.ok.updated": "Project '{name}' updated.",
  "proj.save.failed": "Save failed: {msg}",
  "proj.delete.confirm":
    "Delete project '{name}'? This cannot be undone.",
  "proj.delete.ok": "Project '{name}' deleted.",
  "proj.delete.failed": "Delete failed: {msg}",
  "proj.dirty.confirm": "Discard unsaved changes?",
  "proj.list.failed": "Failed to list projects: {msg}",
  "proj.load.failed": "Failed to load project: {msg}",

  // Validation
  "proj.err.name_required": "Name is required.",
  "proj.err.name_chars":
    "Name may contain only lowercase letters, digits, underscores and hyphens.",
  "proj.err.name_collision": "Name '{name}' collides with a built-in task.",
  "proj.err.need_feature": "At least one feature is required.",
  "proj.err.feature_name": "Feature #{i}: name is required.",
  "proj.err.feature_dup": "Duplicate feature name: '{name}'.",
  "proj.err.feature_bounds": "Feature '{name}': bounds must be numeric.",
  "proj.err.feature_hi_lo":
    "Feature '{name}': hi ({hi}) must be greater than lo ({lo}).",
  "proj.err.need_target": "At least one target is required.",
  "proj.err.target_name": "Target #{i}: short name is required.",
  "proj.err.target_dup_name": "Duplicate target short name: '{name}'.",
  "proj.err.target_column": "Target #{i}: column is required.",
  "proj.err.target_dup_col": "Duplicate target column: '{col}'.",
  "proj.err.target_short_mismatch":
    "Default target '{target}' must match one of the target short names.",
  "proj.err.fix_before_save": "Please fix the following before saving:",

  // Project form
  "proj.form.identity": "Identity",
  "proj.form.name": "Name (lowercase id)",
  "proj.form.name.hint": "Used as --task argument",
  "proj.form.name.placeholder": "e.g. orr",
  "proj.form.display": "Display name",
  "proj.form.display.placeholder": "Oxygen Reduction Reaction",
  "proj.form.desc": "Description",
  "proj.form.desc.placeholder": "Short summary of the system",
  "proj.form.features": "Features (descriptors)",
  "proj.form.features.add": "Add",
  "proj.form.features.col_name": "Name",
  "proj.form.features.col_type": "Type",
  "proj.form.features.col_lo": "Lo",
  "proj.form.features.col_hi": "Hi",
  "proj.form.features.col_unit": "Unit",
  "proj.form.features.col_display": "Display name",
  "proj.form.features.cont": "continuous",
  "proj.form.features.int": "integer",
  "proj.form.targets": "Targets (product yields)",
  "proj.form.targets.add": "Add",
  "proj.form.targets.col_short": "Short name",
  "proj.form.targets.col_csv": "CSV column",
  "proj.form.targets.col_display": "Display name",
  "proj.form.targets.col_unit": "Unit",
  "proj.form.targets.col_competing": "Competing",
  "proj.form.targets.default": "Default target",
  "proj.form.targets.side_rxn": "side-rxn",
  "proj.form.targets.hint":
    "Competing targets are subtracted from the training target when the CLI/WebUI is run with h2_penalty_weight > 0.",
  "proj.form.notes": "Notes",
  "proj.form.notes.placeholder":
    "Free-form notes about this project (not used by the optimizer).",
};

const DICTS: Record<Lang, Record<string, string>> = { zh, en };
const STORAGE_KEY = "kabocs.lang";

function readStoredLang(): Lang {
  try {
    const v = localStorage.getItem(STORAGE_KEY);
    if (v === "zh" || v === "en") return v;
  } catch {
    // ignore
  }
  return "zh";
}

let currentLang: Lang = readStoredLang();
const listeners = new Set<() => void>();

export function getLang(): Lang {
  return currentLang;
}

export function setLang(lang: Lang): void {
  if (lang === currentLang) return;
  currentLang = lang;
  try {
    localStorage.setItem(STORAGE_KEY, lang);
  } catch {
    // ignore
  }
  listeners.forEach((l) => l());
}

function subscribe(cb: () => void): () => void {
  listeners.add(cb);
  return () => {
    listeners.delete(cb);
  };
}

export function useLang(): Lang {
  return useSyncExternalStore(subscribe, getLang, getLang);
}

export function t(key: string, params?: Record<string, string | number>): string {
  const dict = DICTS[currentLang] ?? zh;
  let msg = dict[key] ?? zh[key] ?? key;
  if (params) {
    for (const [k, v] of Object.entries(params)) {
      msg = msg.replace(`{${k}}`, String(v));
    }
  }
  return msg;
}
