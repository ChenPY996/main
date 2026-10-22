# 论文表格 / 图 → 代码与数据字段对应关系

代码中的内部图号（`figures.py` 注释里的 "Fig 1/2/3…"）与论文最终编号不同，下表按论文编号给出对应关系。

## 一、表格

| 论文位置 | 生成方式 | 数据来源 |
|---|---|---|
| 表 1 参数与仿真矩阵 | `simulation.py` 的 `DEFAULT` 与 `configs()`；`results/config.json` 记录实际取值 | `forward-coverage/results/config.json` |
| 表 2 与基线对比（E1—E4 汇总） | `python analysis.py --results results` | `results/summary.json` → `pooled`；逐次数据 `results/runs.csv` |
| 表 3 消融组安全裕度 | `analysis.py` | `results/summary.json` → `pooled`（成功率/覆盖率/越界率）与 `safety` |
| 表 4 理论结果与验证 | `audit.py` 的重演审计 + `analysis.py` 的配对/边界统计 | `results/audit.json`、`results/summary.json` → `fanout`、`supplementary.json` |
| 表 5 决策耗时与训练开销 | `analysis.py` | `results/supplementary.json` → `methods.*.plan_med / plan_p95`；QAlloc 训练轮数见 `supplementary/qalloc/` |
| 表 6 SITL 计时标定 | ROS2 节点实测 | `supplementary/sitl/runs/*.json`、`sitl.json`、`sitl_analyze.py` |

## 二、图形

| 论文图 | 代码输出（`forward-coverage/figures/`） | 说明 |
|---|---|---|
| 图 1 共同出发点进场与前向接管 | `reallocation.png` | 代码注释 "Fig 3: DP geometry and a factual before/after transfer" |
| 图 2 方法框架示意 | `mechanism.png` | 代码注释 "Fig 1: overview mechanism and information flow" |
| 图 4 六类区域与旋转栅格 | `geometries.png` | 六类多边形、共同基地（黑星）、目标点 |
| 图 6 完整路径 | `paths_geometries.png` | 六类区域的扇出进场与前向航线 |
| 图 7 方法路径对比 | `paths_methods.png` | 同一 G3/E3 下不同方法的轨迹 |
| 图 8 配对差异与 95% 区间 | `paired.png` | `summary.json` → `model_paired` |
| 图 9 各事件成功率/覆盖率/停止时刻 | `events.png` | `summary.json` → `events` |
| 图 10 区域×事件热力图 | `heatmap.png` | `summary.json` → `geometry` |
| 图 11 消融（含预留越界率） | `ablation.png` | `summary.json` → `pooled`、`safety` |
| 图 12 机制诊断 | `mechanism_diagnostic.png` | 个体系数恢复、同事实状态下的预测误差、接受移交次数 |
| 图 13 区域尺度与机群规模 | `scale.png` | `results/scale.csv`、`summary.json` → `scale`、`swarm` |
| 图 14 单因素敏感性 | `sensitivity.png` | `summary.json` → `sensitivity` |
| 图 15 能量校准、排序损失、预算审计 | `bounds.png` | 代码注释 "Fig 4: calibrated quantile and error-ranking bound" |
| 图 16 首航段扇出审计 | `fanout.png` | `summary.json` → `fanout` |
| 图 17 安全—性能权衡等四面板 | `analysis.png` | `results/supplementary.json`、`results/notch.json` |
| 图 B.1 QAlloc 学习曲线 | `supplementary/qalloc/fig_qalloc_curve.png` | `qalloc_curve.csv` + `qalloc_refs.json` |

## 三、关键数值的可核对位置

| 论文数值 | 位置 |
|---|---|
| CF 77.92%（748/960）、WAPE 6.24% | `summary.json` → `pooled.CF` |
| Model 60.21%（578/960） | `summary.json` → `pooled.Model` |
| NoSplit/NoTurn/NoSafety/NoRisk/NoComm 落差 | `summary.json` → `pooled.*` |
| E0—E4 分事件成功率（95.00/85.42/97.92/80.00/48.33） | `summary.json` → `events.<0..4>.CF` |
| 6,384 状态 / 64,272 次运行 | `runs.csv` 52,512 行 + `runs_swarm.csv` 11,760 行 |
| 169,285 次接受移交 | `runs.csv` 与 `runs_swarm.csv` 的 `moves` 列之和 |
| 13 段凹口直飞合计 485 m、绕行 1 013 m | `results/notch.json` |
| 规划耗时中位 11.8 ms、95% 分位 37.7 ms | `results/supplementary.json` → `methods.CF.plan_med / plan_p95` |
