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

## 文档说明
[技术报告-/docs/report.pdf](/docs/report.pdf)

[数据说明-/docs/data.md](/docs/data.md)

[参数说明-/docs/Parameters.md](/docs/Parameters.md)

## 快速开始

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

# 使用谱混合核（CatBOX 启发）
python run.py --kernel-type spectral_mixture --seed 42

# 开启 Knowledge-Augmented Bayesian Optimization (KABO) 模式（基于偏好反馈与专家知识进行联合主导优化）
python run.py --kabo-mode --lambda-p 5.0 --lambda-k 2.0 --expert-prior-file priors.json --seed 42

# 选择性优化（惩罚 HER 副反应）
python run.py --target-product CO --h2-penalty-weight 0.3 --seed 42

# 也可通过模块方式运行
python -m co2rr_bo --non-interactive
```

---


## 流程

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

**KABO 联合采集函数（当启用 `--kabo-mode` 时激活）：**

基于上述 UCB/qNEI 采集评分拓展了 `Preference Learning (偏好学习)` 和 `Knowledge Gradient / Expert Priors (专家先验)`，支持经验约束引导空间搜索：
$$\alpha_{KABO}(x) = \alpha_{base}(x) + \lambda_p \cdot \text{Pref}(x) + \lambda_k \cdot \text{KG}_{\text{expert}}(x)$$
通过历史候选选择的成对比较拟合得到 $\text{Pref}(x)$；依据引入的 `expert_prior_file` JSON 配置推演高斯或均匀距离得分作为 $\text{KG}_{\text{expert}}(x)$。当历史数据极少无法收敛偏好GP时，自动回退保障系统稳定运行。

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

## 编程接口 (API)

除命令行外，该项目也可作为 Python 库直接调用：

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

## 输出文件说明

| 文件 | 说明 |
|------|------|
| `output/feature_importances.png` | 特征重要性柱状图 (含 Top-K 截断线) |
| `output/data_updated.csv` | 包含所有新增实验数据的更新数据集 |
| `output/run_metadata.json` | 运行元数据（参数、seed、β 调度配置、每轮 β_t、特征集合、时间戳等） |

---

## ⚠️ 注意事项

1. **数据量要求**：GP 在 <10 个数据点时不稳定，建议至少 15-20 个初始实验点
2. **特征范围**：确保数值型特征无 NaN 或 Inf 值
3. **β 参数调优**：
   - β < 1.0：偏向利用（exploitation），倾向已知高产率区域
   - β ≈ 2.0：平衡（推荐默认值）
   - β > 3.0：偏向探索（exploration），倾向不确定性高的区域
4. **离散候选集**：`candidates.csv` 应包含与 `data.csv` 相同的特征列（不含产量列）
5. **设备兼容性**：cpu/cuda

---

## 📄 License

MIT License
