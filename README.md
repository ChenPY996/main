# 因果反事实增强的多无人机区域覆盖路径规划 —— 完整复现包

本仓库是论文《基于因果反事实增强的多无人机区域覆盖路径规划》(修订终版) 的完整实验复现材料：
主仿真器、六类区域生成、13 种方法、统计检验、审计器与全部图形脚本，以及附录 B.2/B.5 所述的
QAlloc 学习曲线、PX4 SITL 计时标定与共形风险校验材料。

论文表 2—表 5 与图 4—图 17 均由本仓库代码生成；随包的 `forward-coverage/results/` 是一次完整运行的
原始数据（逐次运行记录、汇总、审计与配置），可直接复核。

## 1. 目录结构

```
forward-coverage/            主实验复现包（对应论文第四节、第五节与附录 B.1/B.3/B.4）
  simulation.py              仿真器：区域/栅格、前向扫描+末端航向 DP、能耗与遥测、13 种方法
  analysis.py                统计：汇总、配对检验、Holm 校正，生成 summary.json
  audit.py                   审计：重演全部状态，校验共同基地、连续性、覆盖、能耗与势函数下降
  figures.py                 绘图：生成 figures/ 下 16 张图
  results/                   一次完整运行的输出
    runs.csv                 主套逐次运行记录（52,512 行）
    runs_swarm.csv           机群规模专项逐次记录（11,760 行）
    summary.json             表 2—表 5、图 8—图 16 的数值来源
    config.json              本次运行的参数与版本记录
    audit.json               审计报告（4,704 状态 / 52,512 次运行，status=passed）
    supplementary.json       图 17 所需的耗时、风险门限与安全—性能数据
    notch.json               凹口跨越代价（13 段直飞 485 m vs 绕行 1 013 m）
    literature_paired.csv, model_paired.csv, scale.csv, risk_gate_sweep.csv, feasibility.*
  figures/                   论文图 4、6—17 的矢量/高分辨率版本（300 dpi）
  figures_prev/, simulation_v5_previous.py
                             作者保留的上一版对照材料（论文未使用，仅备查；
                             上一版 results_prev/ 因体积原因未随仓库发布）
  SHA256SUMS.txt             原包文件校验和

supplementary/               附录 B.2/B.5 的补充材料
  qalloc/                    QAlloc 基线训练环境与学习曲线（图 B.1）
  sitl/                      PX4 SITL 航迹计时标定节点、批量脚本、10 组原始日志（表 6）
    mc_conformal.py          共形覆盖率与 ACI 漂移失配率的蒙特卡洛校验（5.6 节）

docs/
  paper-mapping.md           表格/图 → 脚本与数据字段的对应关系
  seeds.md                   随机种子、数据划分与运行规模
```

## 2. 环境

- Python 3.10+（随包数据由 Python 3.10.16 生成，本仓库在 3.12 上复核通过）
- `numpy`、`scipy`、`matplotlib`（见 `requirements.txt`）
- 无需 GPU；无需额外数据文件

```bash
python -m pip install -r requirements.txt
```

## 3. 复现步骤

### 3.1 主实验（表 2—表 5、图 4—图 17）

```bash
cd forward-coverage
python simulation.py --out results --workers 6     # 主套 4,704 状态 / 52,512 次运行
python analysis.py  --results results              # 生成 summary.json 与各配对检验 CSV
python figures.py                                  # 生成 figures/*.png
python audit.py     --results results              # 重演并审计（约 10 分钟，6 进程）
```

完整规模（表 1 与 5.6 节的 6,384 状态 / 64,272 次运行）还需运行机群规模套：

```bash
python simulation.py --suite swarm --out results --workers 6   # 1,680 状态 / 11,760 次运行
```

两套合计：4,704 + 1,680 = **6,384 状态**，52,512 + 11,760 = **64,272 次算法运行**；
`moves` 列合计 134,243 + 35,042 = **169,285 次接受移交**，与表 1、5.6 节完全一致。

快速先导（不进入论文汇总）：

```bash
python simulation.py --quick --out pilot --workers 6
```

只复现机群规模专项（图 13(d)—(f)、5.5 节规模实验）时，执行上面 `--suite swarm` 一条命令即可。

> 运行时间参考：随包 `results/audit.json` 记录 4,704 状态 / 52,512 次运行为 576 s（6 进程，桌面 CPU）；
> 完整仿真为同一规模。`--workers` 可按本机核数调整。

### 3.2 附录 B.2：QAlloc 学习曲线（图 B.1）

```bash
cd supplementary/qalloc
python train2.py out && python refs.py out && python plot3.py out
```

预期：平台期第 5 轮、最终验证成功率 0.744、随机 0.72、人工轮转 0.853。
`supplementary/qalloc/` 根目录已附一次完整运行的输出，可直接比对。

### 3.3 附录 B.5：SITL 计时标定（表 6）与共形校验（5.6 节）

```bash
cd supplementary/sitl
python mc_conformal.py          # 共形覆盖率约 0.951—0.954；ACI 失配率 0.049 vs 固定乘子 0.121
```

SITL 部分需要 Ubuntu + ROS2 Humble + PX4 v1.16.2 + Gazebo Classic 11.10.2 + MAVROS 2.14.0：

```bash
bash matrix.sh                  # 逐条路径重启仿真并记录到达时刻（需按本机修改 PX4 启停路径）
python sitl_analyze.py runs     # 拟合 v_eff 与 t_r
```

## 4. 复现校验（本仓库实测）

对随包 `results/` 重跑 `analysis.py`，输出与论文表 2、表 3 逐项一致：

| 方法 | 成功率 | 覆盖率 C | WAPE | T_stop |
|---|---|---|---|---|
| Fixed | 28.75% (276/960) | 93.91% | 17.92% | 90.87 |
| BCD | 9.06% (87/960) | 83.27% | 17.81% | 105.53 |
| SSI | 25.10% (241/960) | 89.33% | 17.98% | 99.23 |
| SSI-E | 46.15% (443/960) | 96.52% | 18.07% | 105.15 |
| QAlloc | 48.96% (470/960) | 96.48% | 17.97% | 102.51 |
| Energy | 58.65% (563/960) | 97.66% | 17.96% | 104.60 |
| Model | 60.21% (578/960) | 97.95% | 17.92% | 93.95 |
| **CF** | **77.92% (748/960)** | **98.81%** | **6.24%** | **95.46** |
| NoComm | 77.81% (747/960) | 98.79% | 8.45% | 99.53 |
| NoRisk | 76.25% (732/960) | 98.78% | 6.23% | 93.34 |
| NoSafety | 73.65% (707/960) | 98.70% | 6.23% | 92.39 |
| NoTurn | 65.10% (625/960) | 98.41% | 16.27% | 91.92 |
| NoSplit | 55.21% (530/960) | 96.98% | 6.19% | 107.87 |

`audit.py` 的重演审计结论为 `status=passed`（4,704 状态、52,512 次运行，主套）；两套合计 6,384 状态、
64,272 次运行、169,285 次接受移交，与论文表 1 和 5.6 节一致。检查项包括：
共同外部基地、四邻域连续性、任务所有权守恒、唯一访问、按机扫描前向单调、无 180° 反向、
无重复执行边、前缀/悬停/事件/事件后能量记账、实际预算与覆盖率、预留预算与电池越界、势函数严格下降。

## 5. 结果文件说明

| 文件 | 内容 |
|---|---|
| `results/runs.csv` | 逐次运行：场景、事件、机数、种子、方法、覆盖率、成功、航程、转角、能耗、WAPE、规划耗时、越界标记、扇出指标等 |
| `results/summary.json` | 分方法/事件/区域/敏感性/规模的汇总与配对检验，论文表 2—表 5 的数值来源 |
| `results/config.json` | 参数快照（sigma、noise、battery_factor、comm、drift、margin、delta、util_weight、block_size 等）、种子、方法列表、运行规模 |
| `results/audit.json` | 审计状态、检查项与已知局限 |
| `results/supplementary.json` | 图 17 的耗时中位数/分位、风险门限扫描、安全—性能数据 |
| `results/notch.json` | 凹口直飞与绕行的分段代价 |

## 6. 已知限制（与论文 5.x 节一致）

- 离散网格覆盖模型：只统计被实际到达的目标网格中心，不保证连续传感器足迹覆盖；
- 不包含机间碰撞调度；进场段的非重叠只对名义直线段审计，动态轨迹可能交叉；
- 规划耗时为桌面 CPU 实测，不代表嵌入式平台性能保证；
- 随机性由固定基种子驱动（`BASE_SEED = 314159`），不同平台/版本的浮点差异可能导致个别边界样本不同。

## 7. 引用

若使用本仓库，请引用原论文。论文与仓库的条目对应关系见 `docs/paper-mapping.md`。
