# KABOCS WebUI

一个面向 **KABOCS 贝叶斯优化平台**的浏览器控制台：把 CLI 的完整人机交互循环、数据管理与历史运行仪表盘搬进浏览器，**不修改 `kabo/` 任何源码**，通过运行期 monkey-patch 将 CLI 的 `input()` / `print()` 交互桥接到 REST + SSE 之上。

## 功能总览

| 区域 | 说明 |
|---|---|
| **Run** | 配置任务参数并一键启动；实时查看日志、Top-N 候选、产率输入、手动覆盖等交互 |
| **Data** | 在浏览器里浏览/编辑/上传/下载 `data/` 目录下的 CSV |
| **Priors** | 带 JSON 校验的专家先验编辑器（`priors/*.json`） |
| **History** | 历次运行的元数据、特征重要性图、β 轨迹、更新后的 CSV 表格 |

前端技术栈：**React 18 + TypeScript + Vite + TailwindCSS + lucide-react**
后端技术栈：**FastAPI + SSE（Server-Sent Events）+ 工作线程 + monkey-patch 桥接**

## 架构

```
 浏览器 (Vite/React)
    │  ▲
    │  │  SSE: /api/runs/current/events
    │  │  REST: /api/runs, /api/files/*, /api/tasks
    ▼  │
 FastAPI (webui.backend.main)
    │
    │  start → SessionRunner (一个工作线程)
    │           │
    │           └── WebUIBridge.install(task, worker_thread)
    │                    │
    │                    │ monkey-patch 提示函数 & print/input
    ▼                    ▼
 KABOOptimizer.run() → kabo.optimizer.* → kabo.acquisition.* / task.prompt_observation
```

`WebUIBridge` 运行时替换以下引用（无需改 `kabo/` 源码）：

- `kabo.optimizer.prompt_user_candidate_choice`
- `kabo.optimizer.prompt_user_manual_candidate`
- `kabo.optimizer.prompt_user_nonselected_features`
- `kabo.optimizer.print_recommendations`
- `kabo.optimizer.print_best_found`
- `task.prompt_observation`（实例方法）
- `builtins.input`（仅工作线程内生效；用于 PE 循环）
- `sys.stdout`（工作线程内捕获为 log 事件）

会话结束后自动还原。

## 目录结构

```
webui/
├─ requirements.txt          # fastapi / uvicorn / pydantic
├─ run_webui.py              # 启动脚本
├─ backend/
│  ├─ main.py                # FastAPI 应用入口（REST + SSE + 静态前端托管）
│  ├─ runner.py              # SessionRunner + SessionManager + 归档
│  ├─ ui_bridge.py           # WebUIBridge（monkey-patch & 队列桥接）
│  ├─ event_hub.py           # 工作线程 → 多路 SSE 订阅的 pub-sub
│  └─ schemas.py             # Pydantic 请求/响应
└─ frontend/
   ├─ package.json           # vite / react / tailwind
   ├─ vite.config.ts         # dev 代理 /api → 127.0.0.1:8000
   ├─ tailwind.config.js
   ├─ postcss.config.js
   ├─ tsconfig.json
   ├─ index.html
   └─ src/
      ├─ main.tsx
      ├─ App.tsx            # 顶栏 + Tabs
      ├─ api.ts             # REST 客户端
      ├─ types.ts           # 前后端共享的 TS 类型
      ├─ index.css          # @tailwind 基础样式
      ├─ hooks/useEventStream.ts
      └─ components/
         ├─ ConfigPanel.tsx
         ├─ RunPage.tsx
         ├─ PromptPanel.tsx      # 五种交互表单（候选选择 / 手动覆盖 / 非选特征 / 产率 / 原始输入）
         ├─ RecommendationList.tsx
         ├─ LogStream.tsx
         ├─ DataManager.tsx
         ├─ PriorsManager.tsx
         └─ Dashboard.tsx
```

## 首次运行步骤

### 1. 安装后端依赖（Python 环境）

在现有 `co2rr` / 虚拟环境中追加安装 webui 的额外依赖：

```bash
pip install -r webui/requirements.txt
```

你应当已经装过根目录的 `requirements.txt`（`torch` / `botorch` / `gpytorch` / `scikit-learn` / `pandas` / `numpy` / `matplotlib`）。

### 2. 构建（或以开发模式启动）前端

**方式 A：生产构建（推荐，一条命令即可用）**

```bash
cd webui/frontend
npm install
npm run build       # 产出 webui/frontend/dist
```

然后回到项目根目录启动后端：

```bash
python webui/run_webui.py
```

浏览器访问：<http://127.0.0.1:8000>

FastAPI 会自动把 `webui/frontend/dist` 作为静态资源托管。

**方式 B：开发模式（热更新）**

两个终端：

```bash
# 终端 A —— 后端
python webui/run_webui.py

# 终端 B —— 前端 dev server（代理 /api 到 :8000）
cd webui/frontend
npm install
npm run dev
```

浏览器访问：<http://127.0.0.1:5173>

### 3. 启动一次完整运行

1. 打开 **Run** tab。
2. 在 *Configuration* 选择 `task`（`co2rr` 或 `test`）、目标产物、迭代次数等。
3. 点击 **Start run**。配置面板会禁用，SSE 状态变成 `connected`。
4. 日志、Top-N 推荐会实时出现在右侧。当后端需要输入时（选候选、填产率、选特征等），顶部 **Awaiting your input** 卡片会弹出相应表单。
5. 迭代完成后，右侧出现 *Best experiment so far* 与 *Run completed* 提示；结果同步归档到 `output/runs/<run_id>/`。
6. 去 **History** tab 审阅历史运行的元数据、特征重要性图、β 轨迹、`data_updated.csv` 等。

## 已知限制 / 设计取舍

- **单会话**：同一时刻只能跑一个 BO 运行。`SessionManager` 会拒绝在已有运行时再次启动。
- **PE 原始输入**：偏好探索（`pe_budget > 0`）循环里原 CLI 用 `input("PE choice: ")` 直接收取 `a`/`b`/`tie`。web 端把它呈现成一个「原始输入」表单；要用请理解这层抽象，或把 `pe_budget=0` 禁用掉。
- **stdout 全局 patch**：进入运行期会把 `sys.stdout`/`builtins.input` 全局替换，但在非工作线程（如 uvicorn 的请求线程）会直接 pass-through，确保服务器日志不受影响；运行结束还原。
- **SSE 历史**：EventHub 仅保留最近 500 条事件给**后进订阅者**做 replay，更早的需要翻 `output/runs/<run_id>/` 的归档。
- **监听地址**：默认 `127.0.0.1`；如需在 LAN 使用，启动时加 `--host 0.0.0.0`。注意本 UI 没有鉴权机制。

## 常见问题

**Q: 浏览器显示「backend is running, but frontend is not built」**
A: 这是 FastAPI 的 fallback HTML，说明 `webui/frontend/dist/` 缺失。执行方式 A（`npm run build`）或改用方式 B（`npm run dev`）。

**Q: 启动报错 `No module named fastapi`**
A: 先 `pip install -r webui/requirements.txt`。

**Q: 「A run is already active」**
A: 之前的运行没有跑完。去 **Run** tab 按 *Stop*，或等它自己结束。

**Q: 如何在 WebUI 里添加新的 Task？**
A: 和 CLI 完全一致——按 `@register_task` 注册的 `TaskBase` 子类会自动出现在 *Configuration* → *Task* 下拉框。实现 `prompt_observation` 时继续用 `input()`，桥接层会自动把这些 CLI 语义的提示转成浏览器表单（前提：该 Task 使用的交互 API 是 `ask_product_yields` / `ask_*` 家族）。自定义 Task 如果需要特殊表单，沿用默认 `input()` 即可——会作为 `raw_input` prompt 在 web 端出现。

**Q: Monkey-patch 安全吗？**
A: 安装和卸载在同一个 `SessionRunner._worker()` 的 `try/finally` 中成对执行，即使优化器抛异常也会还原。如果担心，可以只在 dev 模式下跑 web UI，或在 production 中用 `--reload=False`。
