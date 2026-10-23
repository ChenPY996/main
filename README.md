# 因果反事实增强的多无人机区域覆盖路径规划 —— 完整复现包

本仓库是论文《基于因果反事实增强的多无人机区域覆盖路径规划》的完整实验复现材料：
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
  SHA256SUMS.txt             仓库内 forward-coverage 各文件的 SHA-256 校验和

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
