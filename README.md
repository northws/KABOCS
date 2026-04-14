# Bayesian Optimization for Photocatalytic CO₂ Reduction Reaction (CO₂RR)

<p align="center">
  <strong>基于贝叶斯优化的光催化 CO₂ 还原反应体系智能优化平台</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/BoTorch-0.9+-orange" alt="BoTorch">
  <img src="https://img.shields.io/badge/GPyTorch-1.10+-green" alt="GPyTorch">
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License">
</p>

---

## 📖 概述

本项目实现了一套完整的**贝叶斯优化 (Bayesian Optimization, BO)** Python 流水线，用于高效优化**光催化 CO₂ 还原反应 (CO₂RR)** 体系。催化剂为**二肽修饰的单层卟啉 MOF**，涉及 19 个物理化学描述符和 7 种产物的产量/法拉第效率优化。

本实现严格遵循以下方法论：

> **"Efficient and Principled Scientific Discovery through Bayesian Optimization: A Tutorial"**
> *(arXiv:2604.01328v3)*

核心技术栈：`BoTorch` (UCB/qNEI 采集函数) + `GPyTorch` (高斯过程代理模型) + `scikit-learn` (随机森林特征选择)。

### 核心特性

| 特性 | 描述 |
|------|------|
| 🧪 **多产物优化** | 支持 CO、HCOOH、CH₄、C₂H₄、CH₃OH、C₂H₅OH、H₂ 七种 CO₂RR 产物 |
| 🌲 **智能特征选择** | Random Forest 自动评估 19 个描述符的特征重要性，筛选 Top-K |
| 🔬 **ARD Matérn 核** | `SingleTaskGP` + `ScaleKernel(MaternKernel(ν=2.5, ARD))`，每个特征独立学习长度尺度 |
| 🎯 **多采集策略** | 支持 `UCB` 与 `qNEI`，覆盖低噪声与高噪声实验场景 |
| 👨‍🔬 **Human-in-the-Loop** | 交互式 CLI 推荐实验方案 → 输入全部产物产量 → 自动更新模型 |
| 📊 **离散候选评估** | 支持从候选 CSV 中评估离散设计空间（氨基酸、溶剂组合） |
| 🖥️ **GPU/CPU 自适应** | 自动检测 CUDA 设备，支持 CPU 回退 |

---

## 📁 项目结构

```
Bayesian-Optimization-in-CO2RR/
│
├── run.py                      # 入口脚本
├── requirements.txt            # Python 依赖
├── prompt.md                   # 需求文档
│
├── co2rr_bo/                   # 核心 Python 包
│   ├── __init__.py             # 包入口，导出 CO2RROptimizer
│   ├── __main__.py             # 支持 python -m co2rr_bo
│   ├── constants.py            # 19个描述符 + 7种产物列常量定义
│   ├── utils.py                # 日志、设备选择、归一化/标准化工具
│   ├── feature_selection.py    # Phase 1: 随机森林特征重要性评估与选择
│   ├── surrogate.py            # Phase 2: GP 代理模型 (SingleTaskGP + ARD Matérn)
│   ├── acquisition.py          # Phase 3: UCB/qNEI 采集函数 + 人机交互 CLI
│   ├── optimizer.py            # CO2RROptimizer 编排器
│   └── cli.py                  # 命令行参数解析
│
├── data/                       # 数据文件
│   ├── data.csv                # 训练数据集 (19 描述符 + 7 产物产量)
│   └── candidates.csv          # 离散候选实验向量
│
├── output/                     # 输出目录 (自动生成)
│   ├── feature_importances.png # 特征重要性图
│   └── data_updated.csv        # 更新后的数据集
│
├── reference/                  # 参考文献 PDF（arXiv）
└── arXiv-2604.01328v3/         # 参考论文 LaTeX 源文件
```

---

## 🚀 快速开始

### 0. 正式入口（重要）

- 正式维护入口仅有：`python run.py` 或 `python -m co2rr_bo`
- 论文复现与审计建议使用正式入口，避免双实现分叉带来的结果偏差

### 1. 环境配置

```bash
# 创建 Conda 环境
conda create -n co2rr python=3.11 -y
conda activate co2rr

# 安装依赖
pip install -r requirements.txt
```

**依赖列表：**

| 包 | 版本 | 用途 |
|---|---|---|
| `torch` | ≥ 2.0 | 张量运算、GPU 加速 |
| `botorch` | ≥ 0.9 | 贝叶斯优化框架 |
| `gpytorch` | ≥ 1.10 | 高斯过程建模 |
| `scikit-learn` | ≥ 1.3 | 随机森林特征选择 |
| `pandas` | ≥ 2.0 | 数据加载与处理 |
| `numpy` | ≥ 1.24 | 数值计算 |
| `matplotlib` | ≥ 3.7 | 可视化绘图 |

### 2. 准备数据

将实验数据整理为 CSV 格式，放入 `data/data.csv`。**必需列：**

**19 个输入描述符：**

| 类别 | 描述符列名 | 物理含义 | 单位 |
|------|-----------|---------|------|
| 氨基酸 A | `A_pI` | 等电点 | pH (无量纲) |
| | `A_distance` | 侧链到金属节点长度 | Å (埃) |
| | `A_hbond_acceptors` | 氢键受体数 | 个 (无量纲) |
| | `A_hbond_donors` | 氢键供体数 | 个 (无量纲) |
| 氨基酸 B | `B_pI` | 等电点 | pH (无量纲) |
| | `B_distance` | 侧链到金属节点长度 | Å (埃) |
| | `B_hbond_acceptors` | 氢键受体数 | 个 (无量纲) |
| | `B_hbond_donors` | 氢键供体数 | 个 (无量纲) |
| 卟啉 MOF | `MOF_potential` | 氧化还原电位 | V vs. NHE |
| | `M_CO_binding_energy` | 金属-CO 结合能 | eV |
| 光敏剂 | `PS_absorption_wavelength` | 最大吸收波长 | nm |
| | `PS_potential` | 激发态氧化还原电位 | V vs. NHE |
| 溶剂 | `Solvent_dielectric` | 相对介电常数 | 无量纲 (ε_r) |
| | `Solvent_hbond_acceptors` | 氢键受体数 | 个 (无量纲) |
| | `Solvent_hbond_donors` | 氢键供体数 | 个 (无量纲) |
| | `CO2_solubility` | CO₂ 溶解度 | mol/L |
| 反应条件 | `H2O_concentration` | 水浓度 | vol% |
| | `Sacrificial_agent_potential` | 牺牲剂氧化电位 | V vs. NHE |
| | `Sacrificial_agent_concentration` | 牺牲剂浓度 | mol/L |

**7 个可能产物列：**

> 产量单位可统一采用 **µmol·g⁻¹·h⁻¹** (产率) 或 **FE%** (法拉第效率)，
> 在同一数据集中保持一致即可。

| 产物 | 列名 | 说明 | 推荐单位 |
|------|------|------|----------|
| CO | `Y_CO` | 一氧化碳 (**默认优化目标**) | µmol·g⁻¹·h⁻¹ 或 FE% |
| HCOOH | `Y_HCOOH` | 甲酸 / 甲酸盐 | µmol·g⁻¹·h⁻¹ 或 FE% |
| CH₄ | `Y_CH4` | 甲烷 | µmol·g⁻¹·h⁻¹ 或 FE% |
| C₂H₄ | `Y_C2H4` | 乙烯 | µmol·g⁻¹·h⁻¹ 或 FE% |
| CH₃OH | `Y_CH3OH` | 甲醇 | µmol·g⁻¹·h⁻¹ 或 FE% |
| C₂H₅OH | `Y_C2H5OH` | 乙醇 | µmol·g⁻¹·h⁻¹ 或 FE% |
| H₂ | `Y_H2` | 氢气 (HER 竞争副反应) | µmol·g⁻¹·h⁻¹ 或 FE% |

> **注意：** 也可以直接写含碳产物总量 (用单列 `Y`表示)。
> 若使用正式入口且数据仅包含 legacy 单目标列 `Y`，系统会在 `Y_CO` 缺失时自动切换到 `Y` 并给出日志提示。

### 3. 运行

```bash
# 交互模式 (默认优化 CO 产率)
python run.py

# 非交互演示模式
python run.py --non-interactive --iterations 5

# 优化特定产物
python run.py --target-product HCOOH     # 优化甲酸产率
python run.py --target-product CH4       # 优化甲烷产率

# 自定义参数
python run.py --top-k 8 --beta 3.0 --iterations 20

# 选点前预填连续候选完整配方（可选）
python run.py --pre-fill-before-choice

# 严格复现实验（推荐）
python run.py --skip-feature-selection --strict-training-schema --pre-fill-before-choice --seed 42

# 理论导向 beta_t 调度（可选）
python run.py --beta-schedule theory --beta-delta 0.1 --seed 42

# 严格理论 beta_t（不使用 beta 缩放）
python run.py --beta-schedule theory-strict --beta-delta 0.1 --seed 42

# 高噪声场景推荐：qNEI
python run.py --acq-strategy qnei --qnei-mc-samples 256 --seed 42

# 也可通过模块方式运行
python -m co2rr_bo --non-interactive
```

---

## ⚙️ 运行的参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data` | `data/data.csv` | 输入数据集路径 |
| `--candidates` | `data/candidates.csv` | 离散候选向量 CSV 路径 |
| `--target-product` | `CO` | 优化目标产物：`CO`, `HCOOH`, `CH4`, `C2H4`, `CH3OH`, `C2H5OH`, `H2` |
| `--top-k` | `10` | 特征选择保留的 Top-K 特征数 |
| `--beta` | `2.0` | UCB 探索参数 β（本实现使用 $\mu + \beta\sigma$，即直接乘以 $\sigma$） |
| `--beta-schedule` | `fixed` | β 调度策略：`fixed`、`theory`（缩放理论 β_t）、`theory-strict`（纯理论 β_t） |
| `--beta-delta` | `0.1` | 理论 β_t 调度的置信参数 δ（在 `theory/theory-strict` 下生效） |
| `--acq-strategy` | `ucb` | 采集策略：`ucb` 或 `qnei`（噪声实验更推荐） |
| `--qnei-mc-samples` | `128` | qNEI 的 QMC 采样数（越大越稳健，计算更慢） |
| `--iterations` | `10` | 贝叶斯优化迭代次数 |
| `--non-interactive` | `False` | 非交互演示模式 (自动模拟实验结果) |
| `--skip-feature-selection` | `False` | 跳过 RF 特征筛选，直接使用全部 19 维特征（论文最小闭环模式） |
| `--strict-training-schema` | `False` | 严格训练模式：要求训练数据具备完整 19/19 描述符，否则报错 |
| `--pre-fill-before-choice` | `False` | 选点前补全连续候选的非选中特征，支持按完整 19 维配方比较 |
| `--seed` | `None` | 全局随机种子，统一控制 NumPy/Torch/Python 随机性以提高复现稳定性 |
| `--output-dir` | `output` | 输出文件目录 |
| `--device` | `auto` | 设备选择：`auto`, `cpu`, `cuda` |

### 何时开启 pre-fill-before-choice

建议开启 `--pre-fill-before-choice` 的场景：

- 你希望在“选点前”就比较连续候选的完整 19 维实验配方，而不是先选点再补录。
- 体系中非选中特征与选中特征存在明显耦合（例如溶剂/牺牲剂与关键电位共同决定可行性）。
- 专家评审强调实验可执行性，需要先看完整 recipe 再做决策。
- 目标是论文高保真复现，希望在选点前基于完整 19 维信息做一致性评审。

建议保持默认关闭（`False`）的场景：

- 你更关注快速迭代，希望减少每轮交互输入成本。
- 大多数迭代都会优先选择离散候选，连续候选较少被执行。
- 先选点后补录已足够满足当前实验流程。

---

## 🔬贝叶斯优化如何实现

> **[核心提示]** 本实现主要落地原论文（arXiv:2604.01328v3）的理论框架，同时也包含了一项工程增强（Engineering Enhancement）。为保证审阅与复现的清晰性，特此澄清两者边界：
> - **论文主流程**：Phase 2（BoTorch 代理模型构建）和 Phase 3（采集函数 + 人机交互优化）严格遵循论文设定（对应 Algorithm 2 / 3）。建议高保真复现时，组合使用 `--skip-feature-selection --strict-training-schema --pre-fill-before-choice --seed 42`；若偏理论配置，可进一步启用 `--beta-schedule theory-strict`。
> - **RF 工程增强流程（默认）**：Phase 1（特征选择）是针对全量 19 维特征在极小样本量下可能遭遇降维打击而附加的启发式工程包（基于 RandomForest）。此设计虽能加速模型收敛，但不属于原论文核心闭环体系。


### Phase 1: 特征权重评估与选择

```
data.csv → RandomForestRegressor → 特征重要性排序 → Top-K 筛选
```

- 使用 `RandomForestRegressor` 训练**针对选定目标产物**的非线性回归模型
- 提取并排序所有 19 个描述符的特征重要性
- 自动选择 Top-K 最重要的特征 (默认 K=10)
- 生成特征重要性横向柱状图 (`feature_importances.png`)
- **小数据集保护**：当数据 <10 行时自动降低 RF 复杂度

**示例输出：**

```
PHASE 1: Feature Weight Evaluation & Selection
Random Forest R² (train): 0.9742  (n_estimators=200)
Selected top 10 features:
  [ 1] Sacrificial_agent_potential          importance=0.2590
  [ 2] A_pI                                 importance=0.2110
  [ 3] B_pI                                 importance=0.1690
  ...
```

### Phase 2: BoTorch 代理模型构建

```
Top-K 特征 → 归一化 [0,1] → 标准化 Y → SingleTaskGP(ARD Matérn) → MLL 拟合
```

- **输入归一化**：各特征缩放到 `[0, 1]` 范围
- **目标标准化**：零均值、单位方差
- **核函数**：`ScaleKernel(MaternKernel(ν=2.5, ard_num_dims=K))`
  - **Matérn 2.5**：适合科学问题（不假设无限可微性）
  - **ARD**：每个特征独立学习长度尺度 ℓ，反映不同物理量的不同相关性
- **超参数拟合**：`ExactMarginalLogLikelihood` + `fit_gpytorch_mll`

**示例输出：**

```
GP Hyperparameters:
  Output scale:  1.4924
  Noise variance: 0.006144
  ARD length-scales (per feature):
    Sacrificial_agent_potential          ℓ=0.6752   ← 短尺度 = 高相关
    A_pI                                 ℓ=1.6906
    B_pI                                 ℓ=122.5229 ← 长尺度 = 低相关
```

### Phase 3: 采集函数与人机交互优化（UCB / qNEI）

```
GP → {UCB(β) 或 qNEI} → optimize_acqf → Top-3 推荐 → 用户输入产量 → 更新数据 → 循环
```

**UCB 采集函数：**

$$\alpha_{\text{UCB}}(x) = \mu(x) + \beta \cdot \sigma(x)$$

其中 $\mu(x)$ 是 GP 后验均值，$\sigma(x)$ 是后验标准差，$\beta$ 控制探索-利用平衡。
在 `--beta-schedule fixed` 下，使用常数 $\beta$；在 `--beta-schedule theory` 下，使用按用户 `beta` 缩放的 $\beta_t$；在 `--beta-schedule theory-strict` 下，使用纯理论 $\beta_t$。
注：部分文献写作 $\mu + \sqrt{\beta_t}\sigma$；本实现采用 BoTorch 常见参数化 $\mu + \beta\sigma$。

**qNEI 采集函数（适合高噪声实验）：**

qNEI 使用 Monte Carlo 方式估计 noisy expected improvement，在随机实验噪声较大或存在重复测量方差时，通常比纯 UCB 更稳健。

**优化流程：**

1. 按 `--acq-strategy` 构建 UCB 或 qNEI 采集函数
2. 在连续 `[0,1]^K` 空间上优化 (`optimize_acqf`)
3. 在离散候选集上评估采集值 (如果提供 `candidates.csv`)
4. 合并并排序，打印 Top-3 推荐实验方案 (还原到原始物理尺度)
5. **交互模式**：用户输入所有 7 种产物的实验产量
6. 追加数据到训练集，重新拟合 GP，进入下一迭代

**交互模式示例：**

```
=================================================================
  🔬 TOP 3 RECOMMENDED EXPERIMENTS (Iteration 1)
     Optimizing: CO (Y_CO)
=================================================================

  Rank #1  (acquisition value: 2.0876, source: continuous)
    Sacrificial_agent_potential         = -0.3200
    A_pI                                = 6.5200
    ...

  ┌─────────────────────────────────────────────────────┐
  │  📋 ENTER EXPERIMENTAL RESULTS (all product yields) │
  │     Enter '0' for undetected products               │
  │     Enter 'exit' to stop optimization               │
  └─────────────────────────────────────────────────────┘
    CO       (Y_CO) ★ TARGET: 62.5
    HCOOH    (Y_HCOOH): 3.2
    CH4      (Y_CH4): 0.5
    C2H4     (Y_C2H4): 0
    CH3OH    (Y_CH3OH): 0
    C2H5OH   (Y_C2H5OH): 0
    H2       (Y_H2): 8.1
```

**最终输出：**

```
=================================================================
  🏆 BEST EXPERIMENT FOUND (target: CO)
=================================================================
  CO = 65.1000  ★

  All product yields:
    Product         Yield
    ──────────  ──────────
    CO ★          65.1000
    HCOOH          2.2000
    CH4            0.2000
    C2H4           0.0000
    CH3OH          0.0000
    C2H5OH         0.0000
    H2             5.2000

  Selected features:
    Sacrificial_agent_potential         = -0.3200
    A_pI                                = 6.3800
    ...
=================================================================
```

---

## 🐍 编程接口 (API)

除命令行外，也可作为 Python 库直接调用：

```python
from co2rr_bo import CO2RROptimizer

# 创建优化器
optimizer = CO2RROptimizer(
    data_path="data/data.csv",
    target_product="CO",          # 优化目标产物
    top_k=10,                     # Top-K 特征数
    beta=2.0,                     # UCB 探索参数
    candidates_path="data/candidates.csv",
    output_dir="output",
)

# 运行完整流水线
result_df = optimizer.run(
    n_iterations=10,
    interactive=True,             # True=交互模式, False=演示模式
)

# 也可以分阶段调用
optimizer.phase1_feature_selection()    # 特征选择
optimizer.phase2_fit_surrogate()        # GP 拟合
optimizer.phase3_optimize(              # 优化循环
    n_iterations=5,
    interactive=False,
)
```

### 关键属性

```python
optimizer.selected_features      # Top-K 特征名列表
optimizer.feature_importances    # 特征重要性 (pd.Series)
optimizer.surrogate.model        # BoTorch SingleTaskGP 模型
optimizer.surrogate.bounds_raw   # 原始尺度下的特征边界
optimizer.df                     # 当前数据集 (pd.DataFrame)
```

---

## 📊 输出文件

| 文件 | 说明 |
|------|------|
| `output/feature_importances.png` | 特征重要性柱状图 (含 Top-K 截断线) |
| `output/data_updated.csv` | 包含所有新增实验数据的更新数据集 |
| `output/run_metadata.json` | 运行元数据（参数、seed、β 调度配置、每轮 β_t、特征集合、时间戳等） |

---

## 🧠 方法论基础

本项目基于论文 **arXiv:2604.01328v3** 的以下核心方法：

| 论文章节 | 对应实现 |
|----------|---------|
| **Part II: Surrogate Models** — GP 代理模型理论 | `surrogate.py`: `SurrogateModel` 类 |
| **§Kernel Function** — Matérn 核与 ARD | `ScaleKernel(MaternKernel(ν=2.5, ard_num_dims=K))` |
| **§Hyperparameter Adaptation** — 边际似然最大化 | `fit_gpytorch_mll(ExactMarginalLogLikelihood)` |
| **Part III: Acquisition Functions** — UCB 理论 | `acquisition.py`: `UpperConfidenceBound` |
| **Algorithm 2: GP-UCB** — 优化循环 | `optimizer.py`: `phase3_optimize()` |
| **Algorithm 3: Human-in-the-Loop** — 专家交互 | `acquisition.py`: `print_recommendations()`, `prompt_user_candidate_choice()`, `prompt_user_manual_candidate()`, `prompt_user_input_multiproduct()` |
| **§Feature Representation** — 特征工程 | `feature_selection.py`: Random Forest 重要性 |

### 为什么？

- **Matérn 2.5 (而非 RBF)**：不假设目标函数无限可微（实际化学体系通常不满足），是科学优化的推荐选择
- **ARD 核**：CO₂RR 描述符跨越多种物理量（电位、波长、浓度等），各维度需要独立的长度尺度
- **UCB 采集函数**：具有理论遗憾界保证 (Theorems 1 & 2)，$\beta$ 参数可调控探索-利用平衡
- **Random Forest 特征选择**：在小样本下稳定评估非线性特征-目标关系，降维后提升 GP 拟合效率

---

## ⚠️ 注意事项

1. **数据量要求**：GP 在 <10 个数据点时不稳定，建议至少 15-20 个初始实验点
2. **特征范围**：确保数值型特征无 NaN 或 Inf 值
3. **β 参数调优**：
   - β < 1.0：偏向利用（exploitation），倾向已知高产率区域
   - β ≈ 2.0：平衡（推荐默认值）
   - β > 3.0：偏向探索（exploration），倾向不确定性高的区域
4. **离散候选集**：`candidates.csv` 应包含与 `data.csv` 相同的特征列（不含产量列）
5. **设备兼容性**：macOS 上默认使用 CPU；NVIDIA GPU 可加速大数据集场景

---

## 📄 License

MIT License
