# KABOCS 优化建议

本文档基于对项目完整源码（`kabo/`、`webui/`、`scripts/`、`data/`）的阅读与分析，列出可优化方向，按优先级排列。

---

## 一、结构与工程

### 1. 零自动化测试（最紧迫 ◼）

项目完全没有任何单元测试框架（无 `pytest`、`unittest`、`tests/` 目录、CI 管线）。唯一验证方式为手动运行 CLI smoke：

```bash
python -m kabo --task test --non-interactive --iterations 2 --seed 42 \
    --data data/test_data.csv --candidates none --skip-feature-selection
```

`kabo/task/test_task.py` 是 Task 子类而非测试工具。

**建议：**
- 添加 `tests/` 目录和 `pytest` 依赖
- 为核心路径编写单元测试：归一化/反归一化、特征选择、采集函数构造、KABO 联合采集函数、CandidateRecord 构造
- 为 `TestTask` 添加端到端集成测试（非交互模式，断言输出文件存在且格式正确）
- 接入 GitHub Actions CI

---

### 2. `optimizer.py` 和 `acquisition.py` 过于臃肿

两个文件各约 1100 行，职责混杂：

- `phase3_optimize()`（`optimizer.py:509-968`）包含采集构建、PE 查询循环、候选评分、人机交互、数据追加、GP 重拟合，是一个约 300 行的巨型方法
- `acquisition.py` 同时包含采集函数定义（`KABOAcquisition`、`build_ucb`、`build_qnei`）和 CLI 交互逻辑（`prompt_user_candidate_choice`、`prompt_user_manual_candidate`、`prompt_user_nonselected_features`）

**建议：**
- 将 CLI 交互逻辑从 `acquisition.py` 抽出到 `kabo/interaction.py`
- 将 `phase3_optimize` 拆分为独立方法：`_run_single_iteration()`、`_handle_pe_queries()`、`_handle_observation()`
- 将打印推荐/最优结果的函数移到 `kabo/display.py`

---

### 3. 无配置文件和实验追踪

- 所有参数仅通过 `argparse` 传递，无 YAML/TOML 配置文件支持
- `output/run_metadata.json` 已输出运行记录，但未与实验追踪平台（MLflow、Weights & Biases、DVC）集成
- 无 `pyproject.toml` / `setup.cfg` 打包，仅依赖 `requirements.txt`

**建议：**
- 支持 `--config config.yaml` 从文件加载参数
- 可选集成 MLflow 自动记录参数、指标、产出文件
- 添加 `pyproject.toml` 标准化打包

---

### 4. 数据量有限

主数据集 `data/data.csv` 仅 25 行（README 建议至少 15-20），离散候选池 `data/candidates.csv` 仅 12 行。GP 在极小数据集上不稳定。

**建议：**
- 考虑添加数据增强（如对现有数据添加小噪声扰动生成合成扩充点）
- 为极小数据集自动降低 GP 核复杂度或启用先验均值函数

---

## 二、算法层

### 5. 仅单目标优化

当前仅支持组合目标 `Y_target - weight * Y_H2`，不支持真正的 Pareto 前沿多目标优化。催化场景常需权衡产率与选择性。

**建议：**
- 引入 `qExpectedHypervolumeImprovement`（qEHVI）支持多目标 Pareto 优化
- 让 Task 声明多目标列，同时输出 Pareto 前沿图

---

### 6. 仅单点推荐（q=1）

`optimize_acqf` 调用硬编码 `q=1`，每次迭代只推荐一个候选。实际实验室常可并行执行多组条件。

**建议：**
- 支持 `--q-batch > 1` 并行实验推荐
- 使用 BoTorch 的 sequential greedy 或 fantasy model 策略生成批量候选

---

### 7. GP 精确推断复杂度 O(N³)

当前使用 `ExactMarginalLogLikelihood`，Cholesky 分解随训练数据 N³ 增长。25 行数据尚无问题，但扩展到 500+ 行时将成为瓶颈。

**建议：**
- 为大数据集场景提供稀疏 GP（SGP）或随机变分高斯过程（SVGP）选项
- 使用 `gpytorch.models.ApproximateGP` 或 BoTorch 内置的 KISS-GP

---

### 8. 缺失关键采集与收敛策略

- **无约束处理**：无法标记不可行区域。可加入 outcome constraint 或 barrier 函数
- **无早停/收敛检测**：按固定迭代次数运行，缺少基于后验方差衰减、EI 下降或最大产率停滞的自动终止
- **无预热启动**：每次运行从零训练 GP，不能从历史 run 加载代理模型继续优化

**建议：**
- 添加 `--max-stagnation` 参数，目标值连续 N 轮无提升则自动停止
- 支持 `--warm-start-run-id <id>` 加载已有 GP state dict

---

### 9. 专家先验分布有限

`ExpertPrior`（`kabo/knowledge.py`）仅支持 Gaussian 和 Uniform 分布。

**建议：**
- 扩展支持 Beta 分布（适用于浓度/比例参数）、Log-Normal（正值偏斜量）、类别分布
- 支持先验分量之间的软约束（如"特征 A 和 B 之和应 < X"）

---

### 10. 特征选择方法单一

仅使用 Random Forest 重要性排序，是工程启发式而非统计推断。

**建议：**
- 提供 SHAP 值、排列重要性（permutation importance）、互信息（mutual information）作为替代方案
- 输出特征间相关性矩阵热力图以帮助检查多重共线性

---

### 11. KABO 偏好模型可深化

- 偏好学习使用 PairwiseGP + 16 个 MC 后验样本，可考虑更高效的积分方法
- PE 查询策略目前为冷启动随机 / 预热后基于方差+均值差，可探索更多信息论策略
- `generate_pe_queries` 的 O(n²) 穷举搜索在候选池大时可能很慢

---

## 三、数据与交互

### 12. 缺失值处理隐式

- 非选中特征自动填充设计空间中点 `(lo+hi)/2`
- 离散候选 NaN 值直接抛出异常
- 缺少数据质量检查（异常值、重复行、列的物理合理性）

**建议：**
- 添加 `--data-check` 模式，运行基本数据质量审计（异常值检测、缺失值报告、重复实验检测）
- 对缺失值提供可配置策略（中位数/均值插补、拒绝该行）

---

### 13. 候选池生成覆盖有限

CO2RRTask 的 `generate_candidates(n=1000)` 使用 Sobol 序列+整数随机采样，但对 19 维空间 1000 个点的 Sobol 覆盖可能仍然稀疏。整数部分用均匀随机而非拉丁超立方或正交采样。

---

## 四、WebUI

### 14. 单用户单会话

`SessionManager` 是全局单例，同时只能运行一个 BO session。

**建议：**
- 改为按 session_id 管理的多会话架构
- 支持同时启动多个不同 Task 的 BO 运行

---

### 15. 无认证/授权

FastAPI 端点全部开放，`data/` 和 `priors/` 文件可通过浏览器直接编辑，无访问控制。

**建议：**
- 添加可选的 Basic Auth 或 API Key 认证
- 对文件编辑操作添加确认机制

---

### 16. WebUI 无实时可视化

优化过程中缺少 GP 后验均值/方差热力图、采集函数地形图等实时图表。历史仪表盘仅有静态特征重要性图和 β 轨迹。

**建议：**
- 通过 SSE 推送每轮的采集函数等高线图（使用 Base64 编码的 PNG）
- 添加维度缩减投影（PCA/t-SNE）可视化设计空间探索轨迹

---

## 五、运维与可复现

### 17. 无极差化容器化支持

无 Dockerfile 或 Docker Compose 配置，环境配置完全依赖本地 Conda/pip。

### 18. β 调度策略可扩展

当前三种 β 调度策略（`fixed`/`theory`/`theory_strict`），可增加：
- 自适应 β：根据目标值改善速度动态调整
- 周期性 β：周期性在探索与利用之间切换

---

## 优先级建议

| 优先级 | 条目 | 理由 |
|--------|------|------|
| P0 | #1 添加测试 | 无测试即无法安全重构或扩展，风险最高 |
| P1 | #3 配置文件+实验追踪 | 提升可复现性和运维便利性 |
| P1 | #5 多目标 BO | 催化领域核心需求（产率 vs 选择性） |
| P2 | #2 代码拆分 | 降低维护障碍，为后续扩展做准备 |
| P2 | #6 q>1 并行推荐 | 显著提升实验效率 |
| P2 | #8 早停/收敛检测 | 节省计算资源 |
| P3 | #7 稀疏 GP | 数据量扩大后才会成为瓶颈 |
| P3 | #9-#11 算法增强 | 锦上添花 |
| P3 | #14-#16 WebUI 增强 | 需评估用户规模后再决定 |
