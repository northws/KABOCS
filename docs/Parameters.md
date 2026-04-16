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
| `--kernel-type` | `matern` | GP 核函数：`matern`（默认）或 `spectral_mixture`（适合复杂多峰响应面） |
| `--h2-penalty-weight` | `0.0` | 若大于 0，优化目标改为 `Y_target - weight * Y_H2`，用于抑制 HER 竞争反应 |
| `--iterations` | `10` | 贝叶斯优化迭代次数 |
| `--non-interactive` | `False` | 非交互演示模式 (自动模拟实验结果) |
| `--skip-feature-selection` | `False` | 跳过 RF 特征筛选，直接使用全部 19 维特征（论文最小闭环模式） |
| `--strict-training-schema` | `False` | 严格训练模式：要求训练数据具备完整 19/19 描述符，否则报错 |
| `--pre-fill-before-choice` | `False` | 选点前补全连续候选的非选中特征，支持按完整 19 维配方比较 |
| `--kabo-mode` | `False` | 启用带有专家领域偏好的 Knowledge-Augmented Bayesian Optimization 模式 |
| `--lambda-p` | `1.0` | 控制历史选择偏好对采集函数的影响比例 |
| `--lambda-k` | `1.0` | 控制专家先验（JSON 指定知识空间）对采集函数的影响比例 |
| `--expert-prior-file`| `None` | (仅 KABO 模式) 定义专家先验知识边界与高斯重心的 JSON 文件路径 |
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
