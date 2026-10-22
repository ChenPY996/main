# 因果反事实增强的多无人机区域覆盖路径规划 —— 可复现材料

本仓库提供论文《基于因果反事实增强的多无人机区域覆盖路径规划》配套的**部分**可复现材料：
QAlloc 基线的训练与学习曲线、PX4 SITL 航迹计时标定的脚本与原始日志、以及共形风险筛选的蒙特卡洛校验脚本。

> 说明：论文附录 B.5 声明仓库同时提供主仿真器、实例生成器、统计脚本、逐图数据与轨迹审计器。
> 上述重量级材料尚未随本仓库公开发布，详见第 2 节。本仓库内容与论文的对应关系见第 1 节。

## 1. 目录与论文对应关系

| 目录 / 文件 | 对应论文位置 | 内容 |
|---|---|---|
| `qalloc/` | 附录 B.2、图 B.1 | QAlloc 基线的训练环境、Q-learning 训练脚本、学习曲线与平台期判定 |
| `qalloc/qalloc_curve.csv` | 图 B.1 | 96 轮训练/验证成功率曲线（一次完整运行的输出） |
| `qalloc/qalloc_summary.json` | 附录 B.2 | 平台期轮数、最终验证成功率等汇总 |
| `qalloc/qalloc_refs.json` | 附录 B.2 | 随机选择与人工轮转两个参照策略的成功率 |
| `sitl/sitl_turns.py` | 5.6 节、表 6 | ROS2 + MAVROS 航点执行节点（位置设定点 + 到达判据） |
| `sitl/matrix.sh` | 5.6 节 | 逐条路径重启 SITL 并批量执行的驱动脚本 |
| `sitl/sitl_analyze.py` | 表 6 | 由逐段到达时刻计算 L、A、D、T，并拟合 v_eff 与 t_r |
| `sitl/runs/*.csv` | 5.6 节 | 原始机载状态日志（20 Hz） |
| `sitl/runs/*.json` | 表 6 | 逐段到达记录（长度、到达耗时、是否满足判据） |
| `sitl/mc_conformal.py` | 5.6 节第二段、表 4 | 乘性噪声下共形覆盖率与 ACI 漂移失配率的独立数值校验 |
| `docs/seeds.md` | 附录 B.1 | 几何种子与数据划分清单 |

## 2. 未包含的内容

以下材料**不在**本仓库中，论文中相应描述不代表本仓库已提供：

- 主仿真器（CF / Model / BCD / SSI / SSI-E / QAlloc 的统一评测框架与执行仿真）；
- 六类区域（偏斜五边形、细长四边形、L 形、U 形、C 形、五臂星形）的几何文件与实例生成器；
- 统计脚本、Holm 校正与配对检验脚本，以及图 4—图 17 的逐图数据；
- 轨迹审计器（起飞扇出、非重访、方向单调性、拼接处航向的检查项与输出格式）；
- 除 QAlloc 之外各方法的运行脚本与参数文件。

如需上述材料用于复核，请联系作者。

## 3. 运行环境

- 通用：Python 3.10+，`numpy`、`pandas`、`Pillow`（见 `requirements.txt`）。
- SITL 部分：Ubuntu + ROS2 Humble + PX4 v1.16.2 + Gazebo Classic 11.10.2 + MAVROS 2.14.0，机型 iris，空世界，飞行高度 5 m。

## 4. 复现步骤

### 4.1 QAlloc 学习曲线（图 B.1）

```bash
cd qalloc
python train2.py out        # 训练 96 轮，写 out/qalloc_curve.csv 与 out/qalloc_summary.json
python refs.py  out         # 参照策略成功率 -> out/qalloc_refs.json
python plot3.py out         # 学习曲线图 -> out/fig_qalloc_curve.png
```

预期：平台期出现在第 5 轮附近，最终验证成功率约 0.744；随机选择约 0.72、人工轮转约 0.85。
`qalloc/` 根目录已附一次完整运行的输出，可直接与重新运行的结果比对。

### 4.2 共形风险筛选校验（5.6 节第二段）

```bash
cd sitl
python mc_conformal.py
```

输出两段结果：五类噪声分布下的经验覆盖率（约 0.951—0.954，与标称 0.95 一致）；
分段漂移下 ACI 的长期失配率（约 0.049，固定乘子为 0.121）。

### 4.3 SITL 计时标定（表 6）

需要可用的 PX4 + Gazebo + MAVROS 环境。`matrix.sh` 中的 PX4 启停脚本路径、工作目录与输出目录需按本机环境修改。

```bash
bash matrix.sh              # 逐条路径重启仿真并记录到达时刻
python sitl_analyze.py runs # 汇总 L/A/D/T 并拟合 v_eff、t_r
```

## 5. 数据格式

- `qalloc/qalloc_curve.csv`：`episode, train_succ, val_succ, mean_reward, eps, n_q`（每轮一行）。
- `sitl/runs/m1_*.csv`：`wall, phase, x, y, z, yaw, volt, pct, mode`，约 20 Hz，`wall` 为相对秒。
- `sitl/runs/m1_*.json`：`marks[{seg,len,dt,reached}]`、`L`、`A`、`D`、`path_wall`、`origin`、`alt`。
  其中 `L` 为航迹总长（m）、`A` 为 90° 等效转角数、`D` 为悬停等待时长（s）。

## 6. 引用与许可

若使用本仓库材料，请引用原论文（DOI 待补）。
许可协议尚未指定；在作者明确授权前，默认保留所有权利。
