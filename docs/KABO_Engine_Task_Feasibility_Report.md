# KABO_Engine + Task 架构泛化可行性报告

## 1. 报告目标

本报告评估将当前项目从 CO2RR 专用实现，重构为可支持多类仿酶催化体系（例如 OER、ORR、NRR 等）的通用架构之可行性。

目标方案为：

- 将纯数学优化能力封装为底层 KABO_Engine。
- 将体系相关逻辑封装为 Task 接口与具体实现（如 CO2RR_Task、OER_Task）。
- 新增体系时仅新增 Task，而不改动引擎核心。

## 2. 执行结论

结论：可行，且建议实施。

可行性评级：高（8.5/10）。

核心理由：

- 当前代码已经具备模块化基础，GP、采集函数、偏好学习、专家先验等核心算法已独立成模块。
- 体系耦合点主要集中在常量定义、CLI 文案、产物输入输出、目标构造和包命名，边界清晰，可控拆分。
- 主流程已有 Phase 编排结构，天然适合替换为 Engine + Task 协作模型。

## 3. 当前架构评估（基于现仓库）

### 3.1 已具备的通用能力（适合下沉到 KABO_Engine）

以下模块的核心算法逻辑不含体系语义，可直接纳入 Engine：

| 能力 | 当前位置 | 说明 |
|------|---------|------|
| 代理模型拟合 | [surrogate.py](../co2rr_bo/surrogate.py) `SurrogateModel` 类 | SingleTaskGP / ARD Matérn / SpectralMixture 内核配置，基于设计空间边界归一化 |
| 采集函数构建与优化 | [acquisition.py](../co2rr_bo/acquisition.py) `KABOAcquisition` / `build_ucb` / `build_qnei` / `optimize_continuous` | UCB / qNEI / KABO 组合，Z-score 标准化逻辑 |
| 偏好模型 | [preference.py](../co2rr_bo/preference.py) `PreferenceModel` 类 | PairwiseGP Bradley-Terry，含 PE 查询生成（PEBO-style） |
| 专家先验评分 | [knowledge.py](../co2rr_bo/knowledge.py) `ExpertPrior` 类 | JSON 配置驱动的 Gaussian/Uniform 先验评分 |
| 归一化/标准化工具 | [utils.py](../co2rr_bo/utils.py) `normalize_x` / `standardize_y` / `unnormalize_x` | 纯数学工具函数 |
| 离散候选评估与过滤 | [acquisition.py](../co2rr_bo/acquisition.py) `evaluate_discrete_candidates` | 全特征边界预过滤 + GP 归一化评分 |
| β 调度策略 | [optimizer.py](../co2rr_bo/optimizer.py) `_compute_beta_t` | 固定 / theory / theory_strict 三种调度 |

### 3.2 体系耦合点（应迁移到 Task）

以下为代码中直接绑定 CO2RR 体系语义的位置，需迁移至 Task 层：

#### 3.2.1 常量定义

- [constants.py L11-26](../co2rr_bo/constants.py#L11): `ALL_FEATURE_COLUMNS` — 19 个描述符列名，硬编码 CO2RR 氨基酸/卟啉 MOF/光敏剂/溶剂等物理量
- [constants.py L41-67](../co2rr_bo/constants.py#L41): `DESIGN_SPACE_BOUNDS` — 设计空间边界，绑定 CO2RR 体系物理范围
- [constants.py L74-88](../co2rr_bo/constants.py#L74): `PRODUCT_COLUMNS` / `ALL_PRODUCT_COLUMNS` / `PRODUCT_NAMES` — CO2RR 产物 (CO, HCOOH, CH₄, C₂H₄, CH₃OH, C₂H₅OH, H₂) 列定义
- [constants.py L92](../co2rr_bo/constants.py#L92): `DEFAULT_TARGET_PRODUCT` — 默认目标产物 CO

#### 3.2.2 多产物交互输入

- [acquisition.py L504-565](../co2rr_bo/acquisition.py#L504): `prompt_user_input_multiproduct` — 遍历 `PRODUCT_COLUMNS` 字典收集产率，CLI 文案写死 "CO2RR products"
- [acquisition.py L808-846](../co2rr_bo/acquisition.py#L808): `simulate_multiproduct_yields` — Demo 模式模拟产率，硬编码 H₂ 为 HER 副反应 (5-20% 范围)、其他 CO2RR 产物指数分布

#### 3.2.3 目标构造

- [optimizer.py L478-502](../co2rr_bo/optimizer.py#L478): `_build_training_target` — 组合目标中硬编码 `Y_H2` 作为 HER 惩罚，这是 CO2RR 专有的选择性概念

#### 3.2.4 CLI 文案

- [cli.py L2](../co2rr_bo/cli.py#L2): 模块文档字符串 — "CLI entry point for the CO2RR Bayesian Optimization pipeline"
- [cli.py L25-43](../co2rr_bo/cli.py#L25): `argparse` 描述和示例，包含 "CO2RR Products" 节
- [cli.py L197](../co2rr_bo/cli.py#L197): `main()` — 硬编码实例化 `CO2RROptimizer`

#### 3.2.5 类与包命名

- 包名 `co2rr_bo` — 所有 import 路径均绑定此体系名
- [optimizer.py L86](../co2rr_bo/optimizer.py#L86): 类名 `CO2RROptimizer`
- [\_\_init\_\_.py](../co2rr_bo/__init__.py): 包文档和 `__all__` 导出绑定 `CO2RROptimizer`

#### 3.2.6 先验验证脚本

- [scripts/validate_prior.py L26](../scripts/validate_prior.py#L26): `from co2rr_bo.constants import DESIGN_SPACE_BOUNDS` — 直接导入 CO2RR 边界进行校验

#### 3.2.7 可视化

- [feature_selection.py L257](../co2rr_bo/feature_selection.py#L257): `plot_feature_importances` — 图标题硬编码 `"CO2RR Feature Importances"`

#### 3.2.8 候选记录

- [candidate.py L5](../co2rr_bo/candidate.py#L5): `CandidateRecord` 文档字符串 — "all 19 features" 硬写特征数量语义

### 3.3 编排器现状（可直接演进）

现有优化器已按 Phase 组织，具备较好的"编排层"形态：

| 阶段 | 入口方法 | 当前位置 |
|------|---------|---------|
| Phase 1: 特征选择 | `phase1_feature_selection()` | [optimizer.py L383](../co2rr_bo/optimizer.py#L383) |
| Phase 2: 代理模型 | `phase2_fit_surrogate()` | [optimizer.py L451](../co2rr_bo/optimizer.py#L451) |
| Phase 3: 采集+交互 | `phase3_optimize()` | [optimizer.py L507](../co2rr_bo/optimizer.py#L507) |
| 全流程编排 | `run()` | [optimizer.py L985](../co2rr_bo/optimizer.py#L985) |

## 4. 目标架构建议

### 4.1 分层职责

```
┌────────────────────────────────────────┐
│  App / CLI（装配层）                    │
│  - 参数解析、Task 选择、调用 Engine     │
│  - 结果落盘、元数据记录                 │
└────────────┬───────────────────────────┘
             │ 注入 Task
┌────────────┴───────────────────────────┐
│  KABO_Engine（算法层）                  │
│  - 代理模型拟合 (GP)                    │
│  - 采集函数构建 (UCB / qNEI / KABO)    │
│  - 偏好学习 (PairwiseGP)              │
│  - 候选评估、推荐、β 调度              │
│  ⚠ 不包含任何体系关键词                │
└────────────┬───────────────────────────┘
             │ 调用 Task 接口
┌────────────┴───────────────────────────┐
│  Task（领域层）                         │
│  CO2RR_Task / OER_Task / Test_Task     │
│  - 特征集合 & 设计空间边界              │
│  - 目标定义 & 产物列                    │
│  - 先验 schema & 校验规则              │
│  - 交互文案 & 观测字段                  │
└────────────────────────────────────────┘
```

1. **KABO_Engine（算法层）**

   - 输入：数值矩阵 X / y、边界、采集参数、偏好比较对、专家先验评分函数
   - 输出：候选点、采集值、模型状态、运行元数据
   - 不包含任何体系关键词（如 CO2RR、OER、特定产物名）

2. **Task（领域层）**

   - 提供特征集合、目标定义、体系约束、候选可行性规则
   - 提供 JSON 先验 schema 解析与验证
   - 提供观测字段定义与交互输入规则
   - 提供可视化文案（如图表标题中的体系名）

3. **App/CLI（装配层）**

   - 负责参数解析、Task 选择、调用 Engine、结果落盘

### 4.2 推荐接口草案

**TaskBase 建议最小契约：**

```python
from abc import ABC, abstractmethod

class TaskBase(ABC):
    """体系任务基类——定义领域层需要向 Engine 提供的全部信息。"""

    @abstractmethod
    def task_name(self) -> str:
        """体系名称，如 'CO2RR', 'OER'。"""

    @abstractmethod
    def feature_columns(self) -> list[str]:
        """全量特征列名（有序）。"""

    @abstractmethod
    def design_space_bounds(self) -> dict[str, tuple[float, float]]:
        """每个特征的物理设计边界。"""

    @abstractmethod
    def target_columns(self) -> dict[str, str]:
        """产物名 → 列名映射，如 {'CO': 'Y_CO', ...}。"""

    @abstractmethod
    def default_target(self) -> str:
        """默认优化目标的产物名。"""

    @abstractmethod
    def build_training_target(self, df, target_column: str,
                              **kwargs) -> np.ndarray:
        """从 DataFrame 构造 surrogate 训练目标向量。
        允许体系定义复合目标（如 CO2RR 的 H₂ 惩罚）。"""

    @abstractmethod
    def validate_candidates(self, df) -> pd.DataFrame:
        """候选数据完整性校验。"""

    @abstractmethod
    def parse_expert_prior(self, json_obj: dict) -> dict:
        """解析并验证体系专用先验 JSON。"""

    @abstractmethod
    def prompt_observation(self, target_column: str) -> dict[str, float] | None:
        """交互模式下收集实验观测结果。"""

    @abstractmethod
    def simulate_observation(self, target_column: str,
                             y_mean: float, y_std: float) -> dict[str, float]:
        """Demo 模式下模拟实验结果。"""

    def summarize_result(self, df, target_column: str) -> None:
        """打印最终最佳实验结果摘要。"""
```

**KABO_Engine 建议最小契约：**

```python
class KABOEngine:
    """体系无关的贝叶斯优化引擎。"""

    def __init__(self, task: TaskBase, device, ...): ...

    def fit_surrogate(self, X, y, selected_features, bounds, kernel_cfg): ...

    def build_acquisition(self, acq_cfg, preference_state=None,
                          prior_scorer=None): ...

    def suggest(self, acq_func, K, batch_cfg, discrete_pool=None): ...

    def update_preference(self, winner, losers): ...

    def append_observation(self, candidate_record, observation, iteration): ...

    def run(self, n_iterations, interactive): ...
```

## 5. 迁移路线（分阶段）

### 阶段 1：引入 Task 抽象，不改行为（低风险）

- 新增 `TaskBase` 抽象类与 `CO2RR_Task` 具体实现
- 将 `constants.py` 中的 `ALL_FEATURE_COLUMNS`、`DESIGN_SPACE_BOUNDS`、`PRODUCT_COLUMNS` 等通过 `CO2RR_Task` 方法对外暴露
- 优化器改为依赖 `task` 对象而非直接读 CO2RR 常量
- `_build_training_target` 委托给 `task.build_training_target()`

预期结果：现有命令与结果不变，所有测试通过。

### 阶段 2：抽离引擎（中风险）

- 新建 `kabo_engine/` 包（或在现包内新建 `engine.py`），将 `SurrogateModel`、`KABOAcquisition`、`PreferenceModel`、`ExpertPrior` 统一挂到 `KABOEngine` 类
- 优化器降级为编排器，不再包含 GP 拟合 / 采集函数调用的直接实现
- `CandidateRecord` 中的 "19 features" 语义改为 `task.feature_columns()` 动态获取

预期结果：Engine 内无体系语义。

### 阶段 3：领域交互与先验迁移（中风险）

- 将 `prompt_user_input_multiproduct`、`simulate_multiproduct_yields` 迁移到 Task
- 将 `print_recommendations` 中的全特征展示逻辑改为从 Task 获取列名
- `validate_prior.py` 改为从 Task 加载 schema 与 bounds
- `feature_selection.py` 图标题改为从 Task 获取体系名

预期结果：同一引擎可切换 CO2RR / OER 等不同 Task。

### 阶段 4：包重命名与新体系验证（中风险）

- 将包名从 `co2rr_bo` 重命名为通用名（如 `kabo` 或 `catalyst_bo`），保留 `co2rr_bo` 为兼容别名
- 以最小字段集实现 `Test_Task`（或 `OER_Task`），完成端到端 smoke test
- 更新 `run.py`、`__main__.py`、`cli.py` 以支持 `--task` 参数选择体系

预期结果：新增体系无需改动 Engine。

## 6. 关键风险与应对

### 风险 1：交互流程硬编码 CO2RR 产物

- **现状**：`prompt_user_input_multiproduct` 遍历 `PRODUCT_COLUMNS` 字典（7 个 CO2RR 产物），CLI 文案固定为 "CO2RR products"
- **位置**：[acquisition.py L504-565](../co2rr_bo/acquisition.py#L504)
- **对策**：提取为 `task.prompt_observation()` 方法，产物列表、文案、验证规则均由 Task 提供

### 风险 2：Demo 模拟逻辑绑定 CO2RR 产物特性

- **现状**：`simulate_multiproduct_yields` 硬编码 H₂ 为 HER 副反应（5-20% 范围）、其他产物服从指数分布——这是 CO2RR 特有的产物分布假设
- **位置**：[acquisition.py L808-846](../co2rr_bo/acquisition.py#L808)
- **对策**：改为 `task.simulate_observation()`，让每个体系自定义产物模拟策略

### 风险 3：组合目标硬编码 Y_H2

- **现状**：`_build_training_target` 仅支持 `Y_target - weight * Y_H2` 的 CO2RR 选择性概念
- **位置**：[optimizer.py L478-502](../co2rr_bo/optimizer.py#L478)
- **对策**：委托给 `task.build_training_target(df, target_column, **kwargs)`，让体系自定义复合目标（如 OER 可能惩罚 H₂O₂ 而非 H₂）

### 风险 4：全量特征数量假设固定

- **现状**：候选记录文档写死 "all 19 features" 语义
- **位置**：[candidate.py L5](../co2rr_bo/candidate.py#L5)
- **对策**：改为 `task.feature_columns()` 全量语义，CandidateRecord 不再假设固定特征数

### 风险 5：先验校验脚本与边界绑定 CO2RR

- **现状**：校验脚本直接 `from co2rr_bo.constants import DESIGN_SPACE_BOUNDS`
- **位置**：[scripts/validate_prior.py L26](../scripts/validate_prior.py#L26)
- **对策**：改为 `--task` 参数指定体系，按 task_name 加载对应 schema 与 bounds

### 风险 6：可视化标题硬编码 CO2RR

- **现状**：特征重要性图标题硬编码 `"CO2RR Feature Importances"`
- **位置**：[feature_selection.py L257](../co2rr_bo/feature_selection.py#L257)
- **对策**：从 `task.task_name()` 动态生成标题

### 风险 7：包名 `co2rr_bo` 阻碍通用化

- **现状**：所有 import 路径 (`from co2rr_bo.xxx import ...`) 绑定 CO2RR 体系名
- **位置**：所有模块的 import 语句
- **对策**：阶段 4 统一重命名为 `kabo`（或 `catalyst_bo`），同时保留 `co2rr_bo` 作为兼容别名 shim 包

### 风险 8：兼容性回归

- **现状**：用户已有 `run.py` / CLI 使用习惯
- **位置**：[run.py](../run.py)、[cli.py](../co2rr_bo/cli.py)
- **对策**：在 CLI 中新增 `--task` 参数（默认 `co2rr`）；原有 CLI 参数全部保留，行为不变

## 7. 耦合度量化分析

以下统计直接引用 CO2RR 常量的模块级 import 关系，可量化迁移工作量：

| 被引用常量 / 类 | 引用位置（文件数） | 迁移难度 |
|----------------|-------------------|---------|
| `ALL_FEATURE_COLUMNS` | optimizer, acquisition, feature_selection（3 处） | 低：改为 `task.feature_columns()` |
| `DESIGN_SPACE_BOUNDS` | optimizer, acquisition, validate_prior（3 处） | 低：改为 `task.design_space_bounds()` |
| `PRODUCT_COLUMNS` / `PRODUCT_NAMES` | optimizer, acquisition, cli, feature_selection（4 处） | 中：涉及交互逻辑改造 |
| `DEFAULT_TARGET_PRODUCT` | optimizer, cli（2 处） | 低：改为 `task.default_target()` |
| `CO2RROptimizer` (类名) | cli, \_\_init\_\_, run（3 处） | 中：需统一改名 + 兼容别名 |

## 8. 成本评估

预估工期：7 到 10 个工作日。

建议拆分为 4 个 PR：

1. **PR-1**：Task 抽象 + CO2RR_Task 接入（无行为变化）— 2-3 天
2. **PR-2**：KABO_Engine 抽离 + 优化器瘦身 — 2-3 天
3. **PR-3**：领域交互与先验迁移到 Task — 1-2 天
4. **PR-4**：包重命名 + Test_Task 最小实现 + 文档与测试 — 2 天

## 9. 验收标准

1. 新增体系时，仅新增 Task 类与配置文件，不改动 Engine 或 CLI 核心。
2. Engine 层无体系关键词与字段硬编码（可 grep 验证）。
3. 先验解析和候选约束可按 Task 切换。
4. 最少通过：
   - 单元测试：Engine 数学逻辑（GP 拟合、采集函数、偏好模型）
   - 集成测试：CO2RR_Task 端到端流程 1 条（`--non-interactive` 模式）
   - 冒烟测试：Test_Task 端到端流程 1 条
5. 已给出新增 Task 的具体教程（含模板代码与配置文件示例）。
6. 原有 CLI 命令（`python run.py`、`python -m co2rr_bo`）默认行为不变。

## 10. 最终建议

建议立即按"阶段 1 → 阶段 2 → 阶段 3 → 阶段 4"推进。

优先策略：先完成接口与依赖反转（阶段 1），再抽离引擎（阶段 2），然后迁移业务逻辑（阶段 3），最后做包重命名与扩展性验证（阶段 4）。

关键原则：**每个 PR 合入后，`python run.py --non-interactive` 必须跑通，不得破坏现有 CO2RR 功能**。这样可以在不影响现有 CO2RR 生产使用的前提下，逐步获得多体系扩展能力。
