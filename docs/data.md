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
