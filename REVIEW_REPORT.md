# CO2RR Bayesian Optimization 代码审查报告（第三轮，对照 arXiv:2604.01328v3）

审查日期：2026-04-14

## 一、审查范围

本轮继续按论文算法与实现逐项对照，覆盖：

1. 主实现链路
- run.py -> co2rr_bo/cli.py -> co2rr_bo/optimizer.py

2. 关键模块
- surrogate.py / acquisition.py / feature_selection.py / constants.py

3. 旧实现状态
- 旧版单文件实现已从项目中移除，不再作为运行入口

## 二、总体结论

主实现与论文核心流程已高度一致，且此前报告中的多项问题已经落地修复：

- 已支持 strict training schema（19/19 描述符强约束）
- 已支持 legacy 单目标列 Y 自动探测回退
- 已支持 pre-fill-before-choice（选点前补全连续候选）
- 已支持 manual override（拒绝全部候选并提交替代点）

当前最核心风险已收敛为两类：

1. 理论版 GP-UCB 的 β_t 仍是缩放近似，而非论文公式的逐字复现
2. 连续候选在默认模式下仍可能先以部分配方信息做初选

## 三、与论文一致性（主实现）

### 已确认对齐

1. BO 闭环完整
- fit surrogate -> build acquisition -> optimize/select -> evaluate -> augment data。

2. GP surrogate 设置合理
- SingleTaskGP + ARD Matérn + MLL 拟合。

3. UCB 采集策略落地
- 使用 BoTorch UCB，beta 参数可配，文档已说明参数化形式。

4. Human-in-the-loop 语义完整度较高
- Top-N 候选评审 + 人工选点。
- manual override 分支允许“拒绝全部候选并提交替代点”。
- 多产物观测输入与审计字段回填完整。

5. 训练/候选约束能力完善
- strict_training_schema 可启用。
- 离散候选全特征严格校验与全特征边界过滤可用。

## 四、当前主要发现（按严重级别）

### P0（高优先级）

1. 旧实现分叉问题已解除

现象：
- 项目中的 co2rr_optimizer.py 已移除，官方入口已收敛到 run.py / python -m co2rr_bo。

影响：
- 入口混淆风险已消除。

建议：
- 维持 README 与 CLI 仅展示当前主实现入口。

### P1（中优先级）

2. GP-UCB 理论中的 β_t 调度已实现，但属于工程化缩放近似

现象：
- 当前实现已提供 `--beta-schedule fixed/theory`，`theory` 模式会生成随迭代变化的 β_t。
- 但实现仍保留用户输入的 `beta` 作为缩放因子，因此它不是论文公式的逐字拷贝。

影响：
- 工程运行没有问题，但若目标是严格对齐论文公式，仍存在一层实现差异。

建议：
- 若追求严格理论复现，可进一步提供纯理论模式（例如固定 `beta=1` 或单独的 `beta-scale` 开关）。
- 在文档中明确当前 `theory` 模式是论文思想的工程化实现。

### P2（低优先级）

3. pre-fill-before-choice 默认关闭时，专家仍可能以“部分配方信息”做初选

现象：
- 该能力已实现，但默认值为 False。

影响：
- 默认交互下，连续候选比较时信息完整性略弱于“先补全后选点”模式。

建议：
- 若目标是论文高保真复现，建议在文档中将该参数标注为推荐开启。

## 五、本轮已确认“已修复/已落地”项

1. legacy Y 自动探测回退已实现。
2. strict_training_schema 已实现。
3. manual override 分支已实现。
4. pre_fill_before_choice 模式已实现。
5. 审计字段（full validity / override fields / oob confirmations）链路已打通。
6. 全局随机种子已实现（`--seed` + NumPy/Torch/Python），并写入 `run_metadata.json`。
7. `beta_schedule` / `beta_trace` 已实现并写入 `run_metadata.json`。

## 六、复现度评估（第三轮）

- 论文主流程复现度：9.0/10
- 工程可用性：8.8/10
- 审计可追溯性：8.7/10
- 当前主要短板：理论 β_t 的逐字复现精度与默认交互信息完整性

## 七、建议优先级

1. P1：若要严格对齐论文公式，补充纯理论 β_t 模式或在文档中明确当前为缩放近似。
2. P2：将 pre-fill-before-choice 在 README 中标注为论文复现推荐参数。

---

结论：主实现已达到“可用于论文方法复现与工程运行”的成熟度，下一阶段重点应从“算法补齐”转向“复现治理”（理论公式对齐 + 默认交互体验）。
