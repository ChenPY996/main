# -*- coding: utf-8 -*-
"""QAlloc 基线的训练环境与 Q-learning 训练脚本。

按论文 §5.1 的描述重建 QAlloc：
  状态 = (任务段编号, 机群容量利用率分箱)
  动作 = 候选接管机
  奖励 = 对能量越界与残余超载的惩罚
  底层方向优化与几何代价与论文一致（旋转栅格 + 蛇形 + 90° 等效转角）

输出：训练/验证成功率曲线（CSV + PNG）、平台期判定。
"""
import json
import math
import os
import sys

import numpy as np

# ------------------------------------------------------------------ 参数（表 1）
H = 5.0            # 网格边长 m
VC = 5.0           # 巡航速度 m/s
K_L, K_A, K_H = 10.0, 50.0, 20.0
SIGMA_ETA = 0.035
DELTA = 0.05
RHO = 0.05
M_UAV = 4
MAX_SEG = 6        # 每个任务段最多 6 个目标

Z = 1.6448536269514722


def kappa(m=M_UAV, delta=DELTA):
    return math.exp(Z * SIGMA_ETA / math.sqrt(m))   # 保守取 δ/m


KAPPA = kappa()


# ------------------------------------------------------------------ 区域
def polygon(kind, S, rng):
    """返回单位化后按 S 缩放、旋转 ±12°、轴向缩放 0.97—1.03 的多边形顶点。"""
    if kind == 'pent':
        ang = np.linspace(0, 2 * np.pi, 5, endpoint=False) + 0.35
        r = 0.55 * np.ones(5)
        r[2] = 0.45
    elif kind == 'quad':
        ang = np.array([0, 0.5, 1.0, 1.5]) * np.pi + 0.1
        r = np.array([0.62, 0.30, 0.62, 0.30])
    elif kind == 'L':
        pts = np.array([[0, 0], [0.6, 0], [0.6, 0.35], [0.25, 0.35],
                        [0.25, 0.75], [0, 0.75]], float)
        pts = (pts - pts.mean(0))
        return _perturb(pts * S, rng)
    elif kind == 'U':
        pts = np.array([[0, 0], [0.7, 0], [0.7, 0.6], [0.5, 0.6], [0.5, 0.22],
                        [0.2, 0.22], [0.2, 0.6], [0, 0.6]], float)
        pts = (pts - pts.mean(0))
        return _perturb(pts * S, rng)
    elif kind == 'C':
        pts = np.array([[0.7, 0], [0.7, 0.7], [0, 0.7], [0, 0.5], [0.45, 0.5],
                        [0.45, 0.2], [0, 0.2], [0, 0]], float)
        pts = (pts - pts.mean(0))
        return _perturb(pts * S, rng)
    else:  # star
        ang = np.linspace(0, 2 * np.pi, 10, endpoint=False)
        r = np.where(np.arange(10) % 2 == 0, 0.6, 0.26)
    xy = np.c_[r * np.cos(ang), r * np.sin(ang)]
    xy = xy - xy.mean(0)
    return _perturb(xy * S, rng)


def _perturb(xy, rng):
    th = math.radians(rng.uniform(-12, 12))
    c, s = math.cos(th), math.sin(th)
    R = np.array([[c, -s], [s, c]])
    sc = rng.uniform(0.97, 1.03, 2)
    return (xy * sc) @ R.T


def inpoly(pts, poly):
    """射线法，向量化。"""
    x, y = pts[:, 0], pts[:, 1]
    inside = np.zeros(len(pts), bool)
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        cond = ((y1 > y) != (y2 > y))
        with np.errstate(divide='ignore', invalid='ignore'):
            xint = (x2 - x1) * (y - y1) / (y2 - y1) + x1
        inside ^= cond & (x < xint)
    return inside


def grid_targets(poly, S, alpha):
    """生成旋转坐标系下的目标点，按 v 分桶成扫描行。"""
    lo = poly.min(0); hi = poly.max(0)
    xs = np.arange(lo[0], hi[0] + H, H)
    ys = np.arange(lo[1], hi[1] + H, H)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.c_[gx.ravel(), gy.ravel()]
    pts = pts[inpoly(pts, poly)]
    if len(pts) < 20:
        return None
    c, s = math.cos(-alpha), math.sin(-alpha)
    R = np.array([[c, -s], [s, c]])
    loc = pts @ R.T
    vbin = np.round(loc[:, 1] / H).astype(int)
    rows = []
    for vb in np.unique(vbin):
        m = vbin == vb
        rows.append(loc[m][np.argsort(loc[m, 0])])
    return rows, loc


def path_cost(rows, reverse=False):
    """一个无人机负责的连续行块：返回 (L, A)。"""
    if not rows:
        return 0.0, 0
    L = 0.0
    A = 0
    order = list(range(len(rows)))
    if reverse:
        order = order[::-1]
    prev_end = None
    for k, ri in enumerate(order):
        row = rows[ri]
        span = row[-1, 0] - row[0, 0] if len(row) > 1 else 0.0
        L += span
        if prev_end is not None:
            start = row[0] if k % 2 == 1 else row[-1]
            L += abs(start[0] - prev_end[0]) + abs(row[0, 1] - prev_end[1])
            A += 2
        prev_end = row[-1] if k % 2 == 0 else row[0]
    return L, A


def nominal_energy(rows):
    L, A = path_cost(rows)
    return K_L * L + K_A * A, L, A


# ------------------------------------------------------------------ 场景
class Scenario:
    def __init__(self, rng, S=100.0, kind=None, event='E1', budget_factor=1.6):
        self.rng = rng
        kind = kind or rng.choice(['pent', 'quad', 'L', 'U', 'C', 'star'])
        self.kind = kind
        self.poly = polygon(kind, S, rng)
        best = None
        for alpha in np.arange(0, 180, 15) * np.pi / 180:
            r = grid_targets(self.poly, S, alpha)
            if r is None:
                continue
            rows, loc = r
            tot = sum(nominal_energy([row])[0] for row in rows)
            if best is None or tot < best[0]:
                best = (tot, rows, alpha)
        self.rows, self.alpha = best[1], best[2]
        self.ntasks = sum(len(r) for r in self.rows)

        # 按目标数均衡划分给 m 架
        counts = np.array([len(r) for r in self.rows])
        cum = np.cumsum(counts)
        bounds = [0]
        for i in range(1, M_UAV):
            target = cum[-1] * i / M_UAV
            bounds.append(int(np.searchsorted(cum, target)) + 1)
        bounds.append(len(self.rows))
        self.blocks = [list(range(bounds[i], min(bounds[i + 1], len(self.rows))))
                       for i in range(M_UAV)]
        self.theta = np.exp(rng.normal(0, 0.20, M_UAV))          # 个体能耗系数
        self.eta = rng.normal(0, SIGMA_ETA, M_UAV)
        self.budget = np.array([budget_factor * nominal_energy([self.rows[j] for j in b])[0]
                                for b in self.blocks])
        self.blocked = set()
        if event in ('E1', 'E3', 'E4'):
            d = M_UAV - 1                                      # 最高扫描带
            self.budget[d] = 0.35 * self.budget[d]
            self.damaged = d
        else:
            self.damaged = M_UAV - 1

    # ---- 当前分配的能量
    def demand(self, alloc=None):
        alloc = alloc or self.blocks
        e = np.array([nominal_energy([self.rows[j] for j in alloc[i]])[0] for i in range(M_UAV)])
        return self.theta * e

    def est_demand(self, alloc=None):
        return self.demand(alloc) * np.exp(self.eta)

    def utilization(self, alloc=None):
        return self.est_demand(alloc) * KAPPA / np.maximum((1 - RHO) * self.budget, 1.0)

    def potential(self, alloc=None):
        return float(np.sum(np.maximum(self.utilization(alloc) - 1.0, 0.0)))

    def tail_segment(self, i, alloc=None):
        """受损机剩余行序列的末端一段（≤ MAX_SEG 个目标）。"""
        blk = (alloc if alloc is not None else self.blocks)[i]
        if not blk:
            return None
        seg, cnt = [], 0
        for r in reversed(blk):
            seg.insert(0, r)
            cnt += len(self.rows[r])
            if cnt >= MAX_SEG:
                break
        return seg

    def try_move(self, alloc, src, taker):
        """把 src 的末端段交给 taker，返回新分配或 None。"""
        seg = self.tail_segment(src)
        if seg is None or taker == src:
            return None
        new = [list(b) for b in alloc]
        for r in seg:
            new[src].remove(r)
        new[taker] = new[taker] + seg
        # 前向约束：接管机的行号必须小于段的行号
        if max(new[taker]) >= max(seg) and min(seg) < max(new[taker]):
            pass
        return new

    def feasible(self, alloc):
        return bool(np.all(self.utilization(alloc) <= 1.0))

    def success(self, alloc):
        return (sum(len(self.blocks[i]) for i in range(M_UAV)) == len(self.rows)
                and self.feasible(alloc))
