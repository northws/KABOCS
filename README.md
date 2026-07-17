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
- [WebUI · /webui/README.md](/webui/README.md)

---

## WebUI（浏览器控制台）

除了 CLI，本项目还提供一个 **React + FastAPI** 的浏览器控制台，支持：

- 配置任何注册的 Task、一键启动/停止贝叶斯优化运行
- 完整人机交互循环（选候选点 / 手动覆盖 / 填产率 / 非选特征 / PE 成对比较）
- **项目编辑器**：以声明式 JSON 新建 CO2RR / ORR / OER … 等「优化项目」，启动即动态注册为 `TaskBase`（无需写 Python）
- 在浏览器里直接编辑 `data/*.csv` 与 `priors/*.json`
- 历史运行仪表盘（元数据、特征重要性图、β 轨迹、更新后的数据集）

```bash
pip install -r webui/requirements.txt
(cd webui/frontend && npm install && npm run build)
python webui/run_webui.py    # 打开 http://127.0.0.1:8000
```

详细启动与架构说明见 [`webui/README.md`](/webui/README.md)。

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
| `ECO2RRTask` | `kabo/task/eco2rr.py` | **电催化 CO₂ 还原（19 特征 / 8 产物，含 3 个类别特征）** |
| `PeptideECO2RRTask` | `kabo/task/peptide_eco2rr.py` | **肽配体电催化 CO₂RR（7 连续特征 / 10 气相产物）；配体以氨基酸描述符编码，可外推到未测残基** |
| `TestTask` | `kabo/task/test_task.py` | 3 特征 / 1 产物，端到端 smoke 测试 |

---

## v1.2 新增特性（2026-04）

| 能力 | CLI 标志 | 说明 |
|---|---|---|
| **电催化 CO₂RR 体系** | `--task eco2rr` | 新增 `ECO2RRTask`（19 特征 / 8 产物 FE%），与光催化 `CO2RRTask` 并存；首个使用**类别特征**的 Task（金属 / 阳离子 / 电解池类型），自动路由 `MixedSingleTaskGP`；产物受 FE 总和 ≤ 100% 约束 |
| **肽配体电催化体系** | `--task peptide` | 新增 `PeptideECO2RRTask`（7 连续特征 / 10 气相产物）；配体以**氨基酸描述符**而非类别编码，边界覆盖全部 20 种天然残基，候选池枚举 20 残基 × 电位，**BO 因此能提议未测过的残基**；`nearest_residue()` 把连续提议解码回真实残基 |
| **多目标 BO (qNEHVI)** | `--multi-objective --objectives CO H2 --ref-point 0 5` | `ModelListGP`（每个目标一条 GP）+ `qNoisyExpectedHypervolumeImprovement` 采集，内置 Task 预设（`TestTask`: y/y2；`CO2RRTask`: CO 最大化 vs HER 最小化；`ECO2RRTask`: FE_CO 最大化 vs FE_H2 最小化）；运行结束写 `pareto_front.csv` + `pareto_front.png`，`ObjectiveSpec(direction="min")` 自动符号翻转 |
| **稀疏 / 变分 GP (SVGP)** | `--gp-model auto` (默认) / `variational` / `exact` · `--num-inducing-points 100` | 大数据集场景切 BoTorch `SingleTaskVariationalGP`，Adam + `VariationalELBO` 训练，O(N·m²) 替代 ExactGP 的 O(N³)。`auto` 在 N ≥ 200 且无类别维时自动升档；类别任务或小数据自动留在 ExactGP。采集函数（UCB/qNEI/qNEHVI）无需改造 |
| **ExpertPrior 分布扩展** | `--expert-prior-file priors/my_prior.json` | 新增 Beta（浓度/比例）、Log-Normal（正值偏斜）、Categorical（类别特征）分布，支持多先验组合与可微 log-score 评估 |
| **特征选择扩展** | `--fs-method rf` (默认) / `permutation` / `mutual_info` / `shap` · `--permutation-repeats 10` · `--mi-neighbors 3` · `--correlation-heatmap` | 新增排列重要性、互信息、SHAP 值三种特征排序方法；可选生成特征相关性热力图 `correlation_heatmap.png` |
| **PE 查询向量化** | `--pe-pool-cap 50` · `--pe-strategy variance` (默认) / `random` | 候选池上限控制避免 O(n²) 穷举，支持随机策略冷启动；全向量化实现提升大候选池性能 |
| **WebUI 多会话** | API: `allow_concurrent=true` | SessionManager 改为按 `run_id` 键控的注册表，支持多并发 BO 运行；新增 `/api/sessions/*` 端点管理会话列表、查询、中止 |
| 声明式配置文件 | `--config run.yaml` | 支持 YAML / TOML / JSON；CLI 显式标志优先级高于配置文件，见 `configs/` 样例 |
| 批量推荐 q>1 | `--q-batch 3` | 每轮给出多个连续候选，qNEI 走联合优化、UCB 走 sequential-greedy 退化，全部进入 Top-N 排序 |
| 早停/收敛检测 | `--max-stagnation 3 --stagnation-tol 1e-3` | 最佳产率连续 N 轮无 >tol 改进自动终止，`run_metadata.json` 记录 `stopped_early` / `stop_reason` |
| 自动化测试 | `pytest -q` | 80+ 测试覆盖归一化、特征选择、CandidateRecord、Task 注册、CLI、YAML 合并、**MO 帕累托/参考点**、**SVGP e2e**、**ExpertPrior 分布**、**特征选择扩展**、**PE 查询向量化**、**SessionManager 多会话**、TestTask 端到端 smoke；`.github/workflows/ci.yml` 已接入 |
| 打包与工程化 | `pyproject.toml` | 标准化元数据 + pytest / coverage / ruff 配置；`pip install -e .[dev]` 一键安装开发依赖 |
| 模块重构 | — | CLI 交互助手（`prompt_user_*` / `print_*`）拆至 `kabo/interaction.py`，`kabo/acquisition.py` 回归到采集函数纯数学层，全部向后兼容 |
| 懒加载 | — | `kabo/__init__.py` 采用 PEP 562 按需解析，`kabo.utils` 的 `torch` 变为函数体内导入；无 torch 环境也能跑轻量测试与配置校验 |

```bash
# 示例 1：单目标 YAML + 批量 q=3 + 3 轮无改进自动停止
python -m kabo --config configs/co2rr_base.yaml \
               --q-batch 3 --max-stagnation 3 --iterations 30

# 示例 2：CO2RR 多目标（CO 最大化 vs HER 最小化，使用 Task 预设）
python -m kabo --task co2rr --multi-objective \
               --iterations 20 --seed 42

# 示例 3：显式指定多目标列 + 参考点
python -m kabo --task co2rr \
               --objectives CO HCOOH --ref-point 0.0 0.0 \
               --iterations 20
```

### 多目标 BO 机制

- **独立代理**：每个目标拟合一条 `SingleTaskGP`（ARD Matérn 2.5，与单目标路径完全相同），再由 BoTorch 的 `ModelListGP` 统一出后验。
- **采集函数**：`qNoisyExpectedHypervolumeImprovement`（qNEHVI）而非 `qEHVI`，因为实验催化数据永远带噪声，qNEHVI 在后验采样上估算 Pareto 前沿，避免了 qEHVI 假设「观测无噪」带来的偏差。
- **方向处理**：`ObjectiveSpec(direction="min")` 在训练时对 `Y` 取负，对外展示仍为原始刻度；参考点同步翻转符号。
- **参考点推断**：未指定 `--ref-point` 时按 `y_min − 0.1 × |range|` 对每个目标推断，并在每轮随新观测自动更新。
- **输出**：`output/<run>/pareto_front.csv` 逐行按 Pareto rank 排序；2D/3D 目标自动生成 `pareto_front.png` 散点图（更高维跳过图输出）。
- **`run_metadata.json` 追加**：`multi_objective` / `objectives[]` / `ref_point` / `qnehvi_mc_samples` / `pareto_front_csv`。

自定义 Task 只需在子类里覆盖 `multi_objectives()`：

```python
from kabo.multi_objective import ObjectiveSpec
from kabo.task.base import TaskBase, register_task

@register_task
class OERTask(TaskBase):
    ...
    def multi_objectives(self):
        return [
            ObjectiveSpec("O2_faradaic_efficiency", direction="max"),
            ObjectiveSpec("overpotential_mV",       direction="min"),
        ]
```

### 稀疏 / 变分 GP (SVGP) 机制

**背景**：精确 GP 的 Cholesky 分解复杂度 O(N³)。~25 行数据毫无压力，200+ 行开始让单轮 BO 进入数秒-分钟级别，500+ 行就不再实用。SVGP 通过 m 个可学习的诱导点近似完整后验，把训练从 O(N³) 降到 **O(N·m² + m³)**（默认 m=100 时即使 N=5000 也只需不到一秒）。

- **后端**：BoTorch `SingleTaskVariationalGP`（底层 `ApproximateGP` + `CholeskyVariationalDistribution`），ARD Matérn 2.5 协方差
- **训练**：`VariationalELBO` + Adam 手动 loop（`--svgp-epochs 200 --svgp-lr 1e-2`，全批）；比 `fit_gpytorch_mll` 对 SVGP 的 dispatch 更稳定跨版本
- **诱导点**：默认 `min(N, 100)` 随机子采样 `train_X` 作为初始位置，`learn_inducing_points=True` 让 Adam 同时优化位置 + 变分参数
- **自动路由**（`--gp-model auto`，默认）：
  - N ≥ 200 **且** 无类别维 → `variational`
  - 否则 / 类别任务 / `SingleTaskVariationalGP` 不可用 → `exact`
- **采集函数无感**：SVGP 实现了标准 `posterior()`，UCB / qNEI / qNEHVI / KABO 组合采集全部直接复用；多目标路径（每个目标一条 `SurrogateModel`）同样自动继承
- **审计字段**：`run_metadata.json` 追加 `gp_model_type`（用户请求）+ `gp_model_type_resolved`（实际使用）+ `num_inducing_points` + `svgp_epochs` + `svgp_lr`

```bash
# 示例 4：在 500+ 行数据集上强制变分 GP
python -m kabo --task co2rr --gp-model variational \
               --num-inducing-points 150 --svgp-epochs 300 \
               --iterations 20

# 示例 5：小数据也强制走 SVGP（诊断 / 对比用）
python -m kabo --task test --gp-model variational \
               --num-inducing-points 5 --svgp-epochs 50 \
               --non-interactive --iterations 2 --seed 42 \
               --data data/test_data.csv --candidates none \
               --skip-feature-selection
```

### ExpertPrior 分布扩展机制

**背景**：原 `ExpertPrior` 仅支持 Gaussian 和 Uniform 分布，无法覆盖浓度/比例参数（Beta）、正值偏斜量（Log-Normal）、类别特征（Categorical）等常见场景。

- **新增分布**：
  - **Beta**：适用于浓度、比例等 [0,1] 有界参数，支持 `alpha`、`beta` 形状参数
  - **Log-Normal**：适用于正值偏斜分布（如催化活性、反应速率），支持 `mu`、`sigma` 参数
  - **Categorical**：适用于类别特征（如催化剂类型、溶剂种类），支持 `probs` 概率向量
- **多先验组合**：单个 JSON 文件可定义多个先验分量，自动按特征名匹配并叠加 log-score
- **可微评分**：所有分布实现可微 log-score，支持梯度优化；验证失败时抛出 `ValueError`
- **配置验证**：加载时自动检查参数合法性（如 Beta 的 alpha/beta > 0，Categorical 的 probs 归一化）

**JSON 配置示例**：

```json
{
  "priors": [
    {
      "feature": "catalyst_loading",
      "type": "beta",
      "alpha": 2.0,
      "beta": 5.0
    },
    {
      "feature": "reaction_rate",
      "type": "lognormal",
      "mu": 0.0,
      "sigma": 1.0
    },
    {
      "feature": "solvent_type",
      "type": "categorical",
      "categories": ["water", "acetonitrile", "dmf"],
      "probs": [0.5, 0.3, 0.2]
    }
  ]
}
```

### 特征选择扩展机制

**背景**：原特征选择仅使用 Random Forest 重要性排序，是工程启发式而非统计推断。

- **新增排序方法**：
  - **排列重要性**（`--fs-method permutation`）：基于验证集性能下降排序，更稳健但计算较慢；`--permutation-repeats` 控制重复次数
  - **互信息**（`--fs-method mutual_info`）：基于信息论的非线性依赖度量；`--mi-neighbors` 控制近邻数
  - **SHAP 值**（`--fs-method shap`）：基于博弈论的一致性解释，需要 `shap` 可选依赖
- **相关性热力图**：`--correlation-heatmap` 生成 `correlation_heatmap.png`，可视化特征间 Pearson 相关性矩阵，帮助识别多重共线性
- **统一接口**：所有方法通过 `rank_features()` 统一接口返回排序后的特征列表，保持向后兼容

```bash
# 示例 6：使用 SHAP 特征选择 + 相关性热力图
python -m kabo --task co2rr --fs-method shap \
               --correlation-heatmap --iterations 20
```

### PE 查询向量化机制

**背景**：原 `generate_pe_queries` 使用 O(n²) Python 循环穷举候选对，在大候选池时性能瓶颈明显。

- **向量化实现**：使用 PyTorch 张量操作替代 Python 循环，大幅提升大候选池性能
- **池上限控制**：`--pe-pool-cap` 限制候选池大小（默认 50），避免 O(n²) 复杂度爆炸
- **策略选项**：
  - **variance**（默认）：基于 GP 后验方差 + 均值差排序，预热后使用
  - **random**：纯随机采样，适用于冷启动或探索阶段
- **参数验证**：`pe_pool_cap` 必须 ≥ 2，`pe_strategy` 必须为有效选项

```bash
# 示例 7：PE 查询使用随机策略 + 池上限
python -m kabo --task co2rr --kabo-mode \
               --pe-strategy random --pe-pool-cap 30 \
               --expert-prior-file priors/my_prior.json --iterations 20
```

### 电催化 CO₂RR 体系（`ECO2RRTask`）

**背景**：`CO2RRTask` 描述的是**光催化** CO₂ 还原（MOF / 光敏剂 / 牺牲剂）。电催化 CO₂RR 的驱动力是**外加电极电位**而非光激发，描述符落在电极、电解液与电解池工程上，产物以**法拉第效率（FE%）** 而非产率报告。因此它是一个独立的 Task，而不是对光催化 Task 的改写 —— 两者并存，互不影响。

- **19 描述符**（与光催化同构的规模）：

  | 分组 | 描述符 |
  |---|---|
  | 催化剂（6） | `Metal_identity`★ / `Metal_CO_binding_energy` / `Metal_H_binding_energy` / `Metal_d_electron_count`◆ / `Particle_size` / `Roughness_factor` |
  | 电极 · GDE（3） | `Catalyst_loading` / `Ionomer_loading` / `Catalyst_layer_thickness` |
  | 电解液（5） | `Cation`★ / `Cation_charge`◆ / `Electrolyte_concentration` / `Electrolyte_pH` / `Electrolyte_conductivity` |
  | 电解池 · 操作（5） | `Cell_type`★ / `Applied_potential` / `CO2_partial_pressure` / `CO2_flow_rate` / `Temperature` |

  ★ = 类别特征（categorical）　◆ = 整数特征（integer）　其余 14 个为连续特征

- **类别特征**（本项目首个使用 categorical 的 Task）：`Metal_identity`（Cu / Ag / Au / Zn / Sn / Bi）、`Cation`（Li / Na / K / Cs）、`Cell_type`（H-cell / flow-cell / MEA）。声明为 `"categorical"` 后代理模型自动切 `MixedSingleTaskGP`（这些维走 `CategoricalKernel`），避免 GP 把「Ag 和 Au 之间」当成一个真实电极。
  - **序数编码**：类别在 DataFrame / `candidates.csv` 里以其在 `categorical_values()` 中的下标存储（Cu→0, Ag→1 …），设计空间边界相应为 `(0, n-1)`。`ECO2RRTask.encode_categorical()` / `decode_categorical()` 是该映射的唯一定义处，交互提示用它把 `0.0` 还原成 `Cu`。
  - **与连续描述符并存**：本 Task 同时保留 `Metal_identity` 与 `Metal_*_binding_energy` —— 前者捕捉金属特异性（如 Cu 独有的 C–C 偶联能力），后者捕捉 scaling relation，二者信息互补。
  - ⚠ 类别维会让 `--gp-model auto` 永远留在 ExactGP（SVGP 无 Mixed 对应实现），见上文 SVGP 自动路由规则。
- **8 个产物（FE%）**：`FE_CO` / `FE_HCOOH` / `FE_CH4` / `FE_C2H4` / `FE_CH3OH` / `FE_C2H5OH` / `FE_C3H7OH` / `FE_H2`。相比光催化多出 n-丙醇（Cu 在高过电位下有可测的 C3 选择性）；列名前缀 `FE_` 与光催化的 `Y_` 天然区分，同一个 `data/` 目录不会混淆。
- **FE 预算约束**：单次测量的各产物 FE 非负且总和 ≤ 100%（其余为未计电荷）。交互录入时总和超 100% 会告警（不拒绝 —— 真实测量会因标定误差轻微溢出）；演示模式的 `simulate_observation` 则按比例整体缩放以严格守住该上限。
- **HER 惩罚**：`--h2-penalty-weight` 对 `FE_H2` 生效（光催化 Task 打的是 `Y_H2`）。电催化里 HER 与目标反应**共用同一外加电位和质子源**，竞争比光催化更激烈，因此该惩罚在这里更有意义。
- **多目标预设**：`max FE_CO` vs `min FE_H2`。因 FE 总和受限于 100%，每一个百分点的 HER 都直接从碳产物里扣除，帕累托前沿比光催化更陡。
- **动态候选池**：`generate_candidates()` 对类别维做**穷举**而非采样 —— 6 金属 × 4 阳离子 × 3 电解池 = 72 种组合平铺到 n 行，保证小 n 时也不会漏掉某个电极/阳离子组合；连续维走 Sobol，整数维均匀采样。

```bash
# 示例 8：电催化 CO2RR，惩罚 HER
python -m kabo --task eco2rr --target-product CO \
               --h2-penalty-weight 0.3 --iterations 20 --seed 42

# 示例 9：电催化多目标（FE_CO 最大化 vs FE_H2 最小化，使用 Task 预设）
python -m kabo --task eco2rr --multi-objective \
               --iterations 20 --seed 42

# 示例 10：优化 C2H4（Cu 电极的 C2+ 路径）
python -m kabo --task eco2rr --target-product C2H4 \
               --h2-penalty-weight 0.2 --iterations 20
```

> **类别维与采集函数**：`optimize_acqf` 把所有维松弛到连续盒子上求解，因此类别 / 整数维返回后必须snap 回整数网格 —— 否则会出现 `Metal_identity = 2.37` 这种解码不出任何金属的推荐。`SurrogateModel.snap_indices` 统一给出「整数 ∪ 类别」的维索引，`suggest_continuous` / `suggest_continuous_batch` / `suggest_mo_continuous` 三条路径都用它。

### 肽配体电催化 CO₂RR 体系（`PeptideECO2RRTask`）

**背景**：金属中心（Fe）挂氨基酸/短肽配体，扫电位，GC 测气相产物 FE。这个体系的核心问题是「下一个该试哪个残基」，而**把配体编码成类别就永远答不了这个问题**。

- **为什么不用 categorical**：`CategoricalKernel` 只问「相同/不同」，类别之间没有度量；`design_space_bounds` 是 `(0, n-1)` 只覆盖已声明的类别，`generate_candidates()` 也只能枚举这些。**没测过的残基根本没有坐标，BO 永远提不出它**。这是编码的性质，补再多数据也没用。
- **配体如何进入设计空间**：只通过其残基的**平均理化描述符**（`kabo/constants.py` 的 `AMINO_ACID_DESCRIPTORS`）——pI / Kyte-Doolittle 疏水性 / Zamyatnin 体积 / 侧链氢键给体·受体 / pH7 电荷。7 个特征**全部连续，零类别维**，因此代理模型留在普通 `SingleTaskGP`，`--gp-model auto` 数据长大后仍可升 SVGP。
- **边界覆盖全部 20 种天然残基**：由 `aa_descriptor_bounds()` 从描述符表**导出**而非硬编码，保证任何未测残基都落在设计空间内，不会被归一化悄悄裁到隔壁残基的坐标上。
- **候选池枚举 20 种残基**：`generate_candidates()` 枚举全部 20 种天然残基 × Sobol 电位——**这正是 BO 能提出未测残基的机制**。肽配体候选可通过 `candidates.csv` 提供。
- **描述符编码的代价**：`optimize_acqf` 把配体维松弛到连续盒子，返回的是**嵌合体**——对应不到任何真实残基（例如 `Ligand_hbond_donors = 1.1424`，没有残基有 1.14 个氢键给体）。所以离散候选池是主路径；`nearest_residue()` 用于把连续提议解码回最近的真实残基，**并返回距离**（距离大 = 这个"最近残基"并不忠实于采集函数想要的东西）。
- **平均的前提**：残基不单独配位金属（无「主配位残基」）。若某配体族不满足，这个表示需要重做。
- **数据准备**：CSV 存人类可读的 `ligand` 列（`"His-Arg-His"`），用 `task.add_ligand_descriptors(df)` 生成 6 个 `Ligand_*` 数值列。`parse_sequence()` 接受 `His-Arg-His` / `his arg his` / `Gln_Met` / `met`，遇到未知残基**报错而非静默丢弃**（拼错导致肽变短却无声改变平均描述符，是最难查的那类 bug）。

```python
from kabo import KABOOptimizer, PeptideECO2RRTask
import pandas as pd

task = PeptideECO2RRTask()
df = task.add_ligand_descriptors(pd.read_csv("data/raw.csv"), ligand_column="ligand")
df["Applied_potential"] = df.potential_V
df.to_csv("data/peptide_eco2rr.csv", index=False)
```

```bash
# 示例 11：肽配体体系，惩罚 HER
python -m kabo --task peptide --data data/peptide_eco2rr.csv \
               --h2-penalty-weight 0.3 --skip-feature-selection \
               --iterations 10 --seed 42
```

> **描述符可辨识性**：描述符要能用，需要足够多的**不同配体**。4 个配体时，6 个描述符中心化后最多张成秩 3，且在其它特征的趋势之外**没有可证明的信号**。验证方式必须是「留一配体交叉验证」对比一个**去掉描述符的对照**（只留其余特征），而不是对比预测均值的基线——后者会把电位趋势的功劳错记到描述符头上。

### WebUI 多会话机制

**背景**：原 `SessionManager` 为全局单例，仅支持单用户单会话，无法同时运行多个 BO 任务。

- **注册表架构**：`SessionManager` 改为按 `run_id` 键控的字典注册表，支持多并发会话
- **并发控制**：`POST /api/runs` 新增 `allow_concurrent` 参数，设为 `true` 时允许多会话并行
- **会话管理 API**：
  - `GET /api/sessions`：列出所有活跃会话
  - `GET /api/sessions/{run_id}`：获取特定会话详情
  - `POST /api/sessions/{run_id}/answer`：提交 PE 答案
  - `POST /api/sessions/{run_id}/abort`：中止会话
  - `DELETE /api/sessions/{run_id}`：移除会话
- **向后兼容**：保留 `/api/runs/current/*` 端点，单会话模式仍可用
- **自动清理**：终端会话在达到 `max_terminal_sessions`（默认 10）时自动清理最旧的记录

**审计字段**：`run_metadata.json` 追踪 `session_id` 和 `allow_concurrent` 标志。

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
│  │  ├─ co2rr.py              # CO2RRTask（光催化）
│  │  ├─ eco2rr.py             # ECO2RRTask（电催化）
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
