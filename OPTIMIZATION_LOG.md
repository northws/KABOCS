# Optimization Log (CO2RR BO)

本文档记录本轮“继续优化”的每一步改动、动机、实现位置与文献依据。

## Step 1: 引入谱混合核选项（已完成）

- 目标：增强代理模型对复杂、多峰催化响应面的建模能力。
- 代码改动：
  - `co2rr_bo/surrogate.py`
    - `SurrogateModel.fit(...)` 新增 `kernel_type` 参数。
    - 支持 `matern` 与 `spectral_mixture` 两种核。
    - `spectral_mixture` 分支中加入 `initialize_from_data(...)` 初始化。
  - `co2rr_bo/optimizer.py`
    - `CO2RROptimizer.__init__` 新增并校验 `kernel_type`。
    - `phase2_fit_surrogate()` 透传 `kernel_type`。
  - `co2rr_bo/cli.py`
    - 新增 CLI 参数 `--kernel-type {matern,spectral_mixture}`。
- 预期收益：在非平稳/多尺度结构明显的催化数据中，通常比单一 Matérn 更有表达力。

文献依据：
- `reference/2505.17393v2.pdf` (CatBOX)
  - 关键词：categorical-continuous BO, spectral mixture kernels, catalysis。

## Step 2: 加入选择性惩罚目标（已完成）

- 目标：让优化更贴近 CO2RR 实验实际，不仅提升目标产物（如 CO），同时抑制 HER 副反应（H2）。
- 方法：将训练目标改为
  - `Y_target - h2_penalty_weight * Y_H2`
  - 当 `h2_penalty_weight = 0` 时保持原单目标行为。
- 代码改动：
  - `co2rr_bo/optimizer.py`
    - 新增参数 `h2_penalty_weight`（含合法性校验）。
    - 新增 `_build_training_target(df)`：按权重生成复合目标。
    - `phase2_fit_surrogate()` 改为通过该函数构建 `Y_raw`。
    - `run_metadata.json` 增加 `h2_penalty_weight` 记录。
  - `co2rr_bo/cli.py`
    - 新增 CLI 参数 `--h2-penalty-weight`。
- 预期收益：减少“目标产物增加但 H2 同时大幅上升”的不理想解，提高选择性。

文献依据：
- `reference/multivariate-bayesian-optimization-of-coo-nanoparticles-for-co2-hydrogenation-catalysis.pdf`
  - 强调多变量/多指标协同优化，不只看单一性能指标。
- `reference/1706.07094v2.pdf`
  - 带噪与约束优化框架支持“在复杂实验条件下同时满足多个目标/约束”。

## Step 3: 增强核函数超参数日志（已完成）

- 目标：保证新核接入后可审计、可复现。
- 代码改动：
  - `co2rr_bo/surrogate.py`
    - `_log_hyperparameters(...)` 按核类型分支：
      - `ScaleKernel(Matern)`：继续输出 output scale 与 ARD length-scales。
      - `SpectralMixtureKernel`：输出 mixture 数量与张量形状。
- 预期收益：调参时更易定位模型不稳定来源。

文献依据：
- `reference/1910.06403v3.pdf` (BoTorch)
  - 强调模块化 BO 组件与可诊断实现。

## Step 4: 文档同步（已完成）

- `README.md` 已补充：
  - 新参数 `--kernel-type`
  - 新参数 `--h2-penalty-weight`
  - 对应运行示例

文献依据：
- `reference/1206.2944v2.pdf`
  - 实用 BO 关注“可操作参数化与可复现实验流程”。

## 推荐运行方式

```bash
python run.py \
  --target-product CO \
  --acq-strategy qnei \
  --qnei-mc-samples 256 \
  --kernel-type spectral_mixture \
  --h2-penalty-weight 0.3 \
  --seed 42
```

说明：
- 若数据量较小，可先用 `--kernel-type matern` 作为稳健基线，再比较 `spectral_mixture`。
- `h2_penalty_weight` 建议从 `0.1 ~ 0.5` 网格搜索。