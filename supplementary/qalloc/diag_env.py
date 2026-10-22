# -*- coding: utf-8 -*-
"""诊断：场景可解率（oracle 深度优先）与随机策略成功率。"""
import os
import sys
from functools import lru_cache

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qalloc_env import Scenario, M_UAV


def solvable(sc, depth=6):
    """是否存在一条接管序列使 Phi 归零。"""
    start = tuple(tuple(b) for b in sc.blocks)

    def rec(alloc, phi, d):
        if phi <= 1e-9:
            return True
        if d >= depth:
            return False
        u = sc.utilization([list(a) for a in alloc])
        src = int(np.argmax(u))
        for j in range(M_UAV):
            if j == src:
                continue
            seg = sc.tail_segment(src, [list(a) for a in alloc])
            if not seg:
                continue
            new = [list(a) for a in alloc]
            try:
                for r in seg:
                    new[src].remove(r)
            except ValueError:
                continue
            new[j] = new[j] + seg
            if sc.utilization(new)[j] > 1.0:
                continue
            p2 = sc.potential(new)
            if p2 >= phi - 1e-9:
                continue
            if rec(tuple(tuple(b) for b in new), p2, d + 1):
                return True
        return False

    return rec(start, sc.potential(sc.blocks), 0)


def rand_policy(sc, rng, trials=20):
    ok = 0
    for _ in range(trials):
        alloc = [list(b) for b in sc.blocks]
        phi = sc.potential(alloc)
        r = 0
        while r < 18 and phi > 1e-9:
            u = sc.utilization(alloc)
            src = int(np.argmax(u))
            cand = [j for j in range(M_UAV) if j != src]
            rng.shuffle(cand)
            moved = False
            for j in cand:
                seg = sc.tail_segment(src, alloc)
                new = [list(b) for b in alloc]
                try:
                    for rr in seg:
                        new[src].remove(rr)
                except ValueError:
                    continue
                new[j] = new[j] + seg
                if sc.utilization(new)[j] <= 1.0 and sc.potential(new) < phi - 1e-9:
                    alloc, phi, moved = new, sc.potential(new), True
                    break
            if not moved:
                break
            r += 1
        ok += int(phi <= 1e-9)
    return ok / trials


if __name__ == '__main__':
    rng = np.random.default_rng(5)
    kinds = ['pent', 'quad', 'L', 'U', 'C', 'star']
    for bf in (1.3, 1.6):
        tot, solv, rand = 0, 0, 0.0
        for k in kinds * 3:
            try:
                sc = Scenario(rng, S=100.0, kind=k, event='E1', budget_factor=bf)
            except Exception:
                continue
            if sc.potential(sc.blocks) <= 0.5:
                continue
            tot += 1
            s = solvable(sc)
            solv += int(s)
            rand += rand_policy(sc, rng)
        print('bf=%.1f  n=%d  solvable=%.2f  rand=%.2f' % (bf, tot, solv / max(tot, 1), rand / max(tot, 1)))
