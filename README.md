# KABOCS — Knowledge-Augmented Bayesian Optimization for Catalytic Systems

<p align="center">
  <strong>面向多类仿酶催化体系的可扩展贝叶斯优化平台（CO₂RR / OER / NRR …）</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/BoTorch-0.9+-orange" alt="BoTorch">
  <img src="https://img.shields.io/badge/GPyTorch-1.10+-green" alt="GPyTorch">
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License">
</p>

本项目将**纯数学优化能力**（GP 拟合、采集函数、偏好学习、专家先验）与**体系相关逻辑**（特征、产物、交互文案、目标构造）彻底解耦：

- **`KABOEngine`** — 算法内核，体系无关
- **`TaskBase`** — 领域接口，一个体系一个 Task 文件
- **`KABOOptimizer`** — 编排层，把 Engine 与 Task 组合成端到端流程



---

## 文档

- [技术报告 · /docs/report.pdf](/docs/report.pdf)
- [参数说明 · /docs/Parameters.md](/docs/Parameters.md)

---

## 架构总览

```
┌──────────────────────────────────────────────────┐
│  CLI / run.py           (kabo/cli.py)            │   装配层
│  - 解析 --task / --data / --iterations           │
│  - 从 TASK_REGISTRY 构造 Task                    │
└────────────┬─────────────────────────────────────┘
             │
┌────────────┴─────────────────────────────────────┐
│  KABOOptimizer          (kabo/optimizer.py)      │   编排层
│  - Phase 1: 特征选择 (RF)                        │
│  - Phase 2: GP 拟合                              │
│  - Phase 3: 采集 + 人机交互循环                  │
└────────────┬────────────────────┬────────────────┘
             │                    │
┌────────────┴──────────┐  ┌──────┴──────────────┐
│  KABOEngine           │  │  TaskBase (子类)     │
│  (kabo/engine.py)     │  │  (kabo/task/*.py)    │
│  - SurrogateModel     │  │  - 特征 & 边界       │
│  - PreferenceModel    │  │  - 产物列 & 目标     │
│  - ExpertPrior        │  │  - 交互 & 模拟       │
│  - UCB / qNEI / KABO  │  │  - 先验 schema       │
│  ⚠ 无体系关键词       │  │                      │
└───────────────────────┘  └──────────────────────┘
         算法层                     领域层
```

内置任务：

| Task | 文件 | 用途 |
|---|---|---|
| `CO2RRTask` | `kabo/task/co2rr.py` | 光催化 CO₂ 还原（19 特征 / 7 产物） |
| `TestTask` | `kabo/task/test_task.py` | 3 特征 / 1 产物，端到端 smoke 测试 |

---

## 快速开始

### 1. 环境配置

```bash
conda create -n co2rr python=3.11 -y
conda activate co2rr
pip install -r requirements.txt
```

**依赖列表：**

| 包 | 版本 | 用途 |
|---|---|---|
| `torch` | ≥ 2.0 | 张量运算 / GPU 加速 |
| `botorch` | ≥ 0.9 | 贝叶斯优化框架 |
| `gpytorch` | ≥ 1.10 | 高斯过程建模 |
| `scikit-learn` | ≥ 1.3 | 随机森林特征选择 |
| `pandas` | ≥ 2.0 | 数据加载与处理 |
| `numpy` | ≥ 1.24 | 数值计算 |
| `matplotlib` | ≥ 3.7 | 可视化绘图 |

### 2. 运行

```bash
# 默认 CO2RR 任务，交互模式
python run.py

# 非交互演示
python run.py --non-interactive --iterations 5

# 优化特定产物
python run.py --task co2rr --target-product HCOOH
python run.py --task co2rr --target-product CH4

# 自定义参数
python run.py --top-k 8 --beta 3.0 --iterations 20

# 选点前预填连续候选完整配方
python run.py --pre-fill-before-choice

# 严格复现实验
python run.py --skip-feature-selection --strict-training-schema \
              --pre-fill-before-choice --seed 42

# 理论导向 β_t 调度
python run.py --beta-schedule theory --beta-delta 0.1 --seed 42
python run.py --beta-schedule theory-strict --beta-delta 0.1 --seed 42

# 高噪声场景：qNEI
python run.py --acq-strategy qnei --qnei-mc-samples 256 --seed 42

# 谱混合核（CatBOX 启发）
python run.py --kernel-type spectral_mixture --seed 42

# KABO 模式（偏好 + 专家先验）
python run.py --kabo-mode --lambda-p 5.0 --lambda-k 2.0 \
              --expert-prior-file priors/my_prior.json --seed 42

# 选择性优化（惩罚 HER 副反应）
python run.py --target-product CO --h2-penalty-weight 0.3 --seed 42

# 切换到最小 smoke task（TestTask）
python -m kabo --task test --non-interactive --iterations 2 --seed 42 \
    --data data/test_data.csv --candidates none --skip-feature-selection

# 模块入口（与 run.py 等价）
python -m kabo --non-interactive
```

> **兼容性**：老包名 `co2rr_bo` 仍可用（`from co2rr_bo import CO2RROptimizer`），会弹出 `DeprecationWarning`。新代码请使用 `from kabo import KABOOptimizer`。

---

## 流程（以 `CO2RRTask` 为例）

### Phase 1 · 特征权重评估与选择

```
data.csv → RandomForestRegressor → 特征重要性排序 → Top-K 筛选
```

- 用 `RandomForestRegressor` 对**用户选定的目标产物**做非线性拟合
- 排序所有 19 个描述符的重要性
- 自动选 Top-K（默认 K=10）
- 生成 `feature_importances.png`（标题前缀取自 `task.task_name()`）
- **小数据集保护**：<10 行时自动降低 RF 复杂度

### Phase 2 · BoTorch 代理模型构建

```
Top-K 特征 → 归一化 [0,1] → 标准化 Y → SingleTaskGP(ARD Matérn) → MLL 拟合
```

- **输入归一化**：按 Task 提供的 `design_space_bounds` 缩放到 `[0, 1]`
- **目标标准化**：零均值、单位方差
- **核函数**：`ScaleKernel(MaternKernel(ν=2.5, ard_num_dims=K))`
  - Matérn 2.5 不假设无限可微，适合物化数据
  - ARD 为每维学习独立长度尺度 ℓ
- **超参数拟合**：`ExactMarginalLogLikelihood` + `fit_gpytorch_mll`
- **可选谱混合核**：`--kernel-type spectral_mixture`

### Phase 3 · 采集函数与人机交互优化

```
GP → {UCB(β) 或 qNEI} → optimize_acqf → Top-N 推荐 → 用户输入产量 → 更新 → 循环
```

**UCB 采集函数：**

$$\alpha_{\text{UCB}}(x) = \mu(x) + \beta \cdot \sigma(x)$$

其中 $\mu(x)$、$\sigma(x)$ 是 GP 后验均值与标准差。本实现采用 BoTorch 常见参数化 $\mu + \beta\sigma$（部分文献写作 $\mu + \sqrt{\beta_t}\sigma$）。`--beta-schedule` 支持 `fixed` / `theory` / `theory-strict` 三种策略。

**KABO 联合采集函数（`--kabo-mode`）：**

$$\alpha_{\text{KABO}}(x) = \alpha_{\text{base}}(x) + \lambda_p \cdot \text{Pref}(x) + \lambda_k \cdot \text{KG}_{\text{expert}}(x)$$

- $\text{Pref}(x)$：根据专家历次选择记录的成对比较，拟合 PairwiseGP（Bradley-Terry）后验均值
- $\text{KG}_{\text{expert}}(x)$：读取 `--expert-prior-file` JSON 配置，按高斯 / 均匀分布计算物理空间惩罚
- 数据不足时自动回退到 $\alpha_{\text{base}}$

**qNEI 采集函数：** Monte-Carlo 噪声 EI；噪声较大或重复测量方差大时比 UCB 更稳健。

**迭代流程：**

1. 构建采集函数（UCB 或 qNEI，可选 KABO 包装）
2. 在连续 $[0,1]^K$ 上优化
3. 在离散候选集上评估（若提供 `--candidates`）
4. 合并排序，打印 Top-N 推荐方案（还原到物理尺度）
5. 交互模式：调用 `task.prompt_observation()` 收集实验结果
6. 追加数据、重拟合 GP，进入下一迭代

---

## 编程接口

```python
from kabo import KABOOptimizer, CO2RRTask, TestTask, get_task

# 方式 A：通过 Task 注册表
task = get_task("co2rr")            # 或 "test"

# 方式 B：直接实例化
task = CO2RRTask()

optimizer = KABOOptimizer(
    data_path="data/data.csv",
    task=task,
    target_product="CO",              # 不传则用 task.default_target()
    top_k=10,
    beta=2.0,
    candidates_path="data/candidates.csv",
    output_dir="output",
)

# 运行完整流水线
result_df = optimizer.run(n_iterations=10, interactive=True)

# 或按阶段调用
optimizer.phase1_feature_selection()
optimizer.phase2_fit_surrogate()
optimizer.phase3_optimize(n_iterations=5, interactive=False)
```

**兼容别名：** `from kabo import CO2RROptimizer` 等价于 `KABOOptimizer`。

---

## 添加新体系（Task）

新增 OER / NRR 等体系**只需新建一个 Task 文件**：

### 1. 创建 `kabo/task/oer.py`

```python
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from kabo.task.base import TaskBase, register_task


@register_task
class OERTask(TaskBase):

    def task_name(self) -> str:
        return "OER"

    def feature_columns(self) -> list[str]:
        return ["overpotential", "metal_loading", "pH", ...]

    def design_space_bounds(self) -> dict[str, tuple[float, float]]:
        return {
            "overpotential": (0.2, 1.2),
            "metal_loading": (0.0, 10.0),
            "pH": (0.0, 14.0),
            ...
        }

    def target_columns(self) -> dict[str, str]:
        return {"O2": "Y_O2"}

    def default_target(self) -> str:
        return "O2"

    def build_training_target(self, df, target_column, **kwargs):
        return df[target_column].values.astype(np.float64)

    def prompt_observation(self, target_column):
        # 体系专属交互逻辑
        ...

    def simulate_observation(self, target_column, y_mean, y_std):
        # 体系专属演示模式模拟
        ...
```

### 2. 在 `kabo/task/__init__.py` 追加一行

```python
from kabo.task.oer import OERTask
```

### 3. 直接使用

```bash
python run.py --task oer --data data/oer_data.csv
```

**不需要改动**：`KABOEngine`、`KABOOptimizer`、`cli.py`、`acquisition.py`、`feature_selection.py`。

参考 `kabo/task/test_task.py`（最小样板，约 60 行）和 `kabo/task/co2rr.py`（完整样板）。

---

## 输出文件

| 文件 | 说明 |
|---|---|
| `output/feature_importances.png` | 特征重要性柱状图（含 Top-K 截断线，标题由 Task 提供） |
| `output/data_updated.csv` | 包含所有新增实验的更新数据集 |
| `output/run_metadata.json` | 运行元数据（参数、seed、β 调度、每轮 β_t、特征集合、时间戳等） |

---

## 工具脚本

```bash
# 验证专家先验 JSON 是否与 task 设计空间相符
python scripts/validate_prior.py priors/my_prior.json --task co2rr

# 非默认 task 的先验校验
python scripts/validate_prior.py priors/oer_prior.json --task oer --n-samples 2000
```

---

## 项目结构

```
ml-co2rr/
├─ kabo/                       # 主包
│  ├─ engine.py                # KABOEngine（算法核心，体系无关）
│  ├─ optimizer.py             # KABOOptimizer（编排层）
│  ├─ cli.py                   # CLI（--task 路由）
│  ├─ task/                    # 体系实现
│  │  ├─ __init__.py
│  │  ├─ base.py               # TaskBase + TASK_REGISTRY
│  │  ├─ co2rr.py              # CO2RRTask
│  │  └─ test_task.py          # TestTask（smoke 基线）
│  ├─ surrogate.py             # GP 代理模型
│  ├─ acquisition.py           # UCB / qNEI / KABO / 交互工具
│  ├─ preference.py            # PairwiseGP 偏好学习
│  ├─ knowledge.py             # ExpertPrior
│  ├─ feature_selection.py     # RF 特征选择
│  ├─ candidate.py             # CandidateRecord dataclass
│  ├─ constants.py             # CO2RR 领域常量（由 CO2RRTask 读取）
│  └─ utils.py                 # 日志 / 设备 / 归一化
├─ co2rr_bo/                   # 兼容性 shim（弹 DeprecationWarning）
│  └─ __init__.py
├─ data/
├─ docs/
├─ priors/
├─ scripts/
├─ run.py                      # 便捷入口
└─ requirements.txt
```

---

## 注意事项

1. **数据量**：GP 在 <10 个点时不稳定，建议初始至少 15–20 个点
2. **特征清洁**：数值列需无 NaN / Inf
3. **β 参数调优**：
   - β < 1.0：偏利用，倾向已知高产率区域
   - β ≈ 2.0：平衡（推荐默认值）
   - β > 3.0：偏探索，倾向高不确定性区域
4. **离散候选集**：`candidates.csv` 应包含 task 声明的全部特征列；若 task 不匹配，传 `--candidates none` 禁用
5. **设备**：CPU / CUDA 自动选择，可用 `--device` 强制指定

---

## License

MIT License
