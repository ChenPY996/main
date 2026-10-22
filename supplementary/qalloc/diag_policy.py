# -*- coding: utf-8 -*-
"""诊断：单次选择策略（随机 / 固定轮转）与场景可解率。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qalloc_env import Scenario, M_UAV
from diag_env import solvable


def episode(sc, policy, rng):
    alloc = [list(b) for b in sc.blocks]
    phi = sc.potential(alloc)
    rnd, wasted = 0, 0
    while rnd < 18 and wasted < 4 and phi > 1e-9:
        u = sc.utilization(alloc)
        src = int(np.argmax(u))
        cand = [j for j in range(M_UAV) if j != src]
        a = policy(rnd, cand, rng)
        seg = sc.tail_segment(src, alloc)
        ok = False
        if seg:
            new = [list(b) for b in alloc]
            try:
                for r in seg:
                    new[src].remove(r)
            except ValueError:
                new = None
            if new is not None:
                new[a] = new[a] + seg
                if sc.utilization(new)[a] <= 1.0 and sc.potential(new) < phi - 1e-9:
                    alloc, phi, ok = new, sc.potential(new), True
        if not ok:
            wasted += 1
        rnd += 1
    return phi <= 1e-9


def build(seed, n, bf=1.6, only_solvable=True):
    rng = np.random.default_rng(seed)
    kinds = ['pent', 'quad', 'L', 'U', 'C', 'star']
    pool, tries = [], 0
    while len(pool) < n and tries < n * 30:
        tries += 1
        k = kinds[len(pool) % len(kinds)]
        S = 100.0 + 50.0 * ((len(pool) // len(kinds)) % 2)
        try:
            sc = Scenario(rng, S=S, kind=k, event='E1', budget_factor=bf)
        except Exception:
            continue
        if sc.potential(sc.blocks) <= 0.5:
            continue
        if only_solvable and not solvable(sc):
            continue
        pool.append(sc)
    return pool


if __name__ == '__main__':
    rng = np.random.default_rng(7)
    for bf in (1.6, 1.4):
        pool = build(31, 60, bf)
        n = len(pool)
        rnd_p = np.mean([episode(sc, lambda r, c, g: int(g.choice(c)), rng) for sc in pool])
        rot_p = np.mean([episode(sc, lambda r, c, g: c[r % len(c)], rng) for sc in pool])
        best = max(np.mean([episode(sc, lambda r, c, g, s=s: c[(r + s) % len(c)], rng)
                            for sc in pool]) for s in range(M_UAV))
        print('bf=%.1f solvable-pool n=%d  random=%.2f  rotate=%.2f  best-rot=%.2f'
              % (bf, n, rnd_p, rot_p, best))
