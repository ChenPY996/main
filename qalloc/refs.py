# -*- coding: utf-8 -*-
"""计算随机选择与人工轮转两个参照策略在验证集上的成功率。"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qalloc_env import Scenario, M_UAV
from train2 import build_pool, run_episode


def episode_policy(sc, policy, rng):
    from train2 import step, MAX_ROUNDS
    alloc = [list(b) for b in sc.blocks]
    phi = sc.potential(alloc)
    rnd, wasted = 0, 0
    while rnd < MAX_ROUNDS and wasted < 4 and phi > 1e-9:
        src = int(np.argmax(sc.utilization(alloc)))
        cand = [j for j in range(M_UAV) if j != src]
        a = policy(rnd, cand, rng)
        new = step(sc, alloc, phi, src, a)
        if new is None:
            wasted += 1
        else:
            alloc, phi = new, sc.potential(new)
        rnd += 1
    return phi <= 1e-9


if __name__ == '__main__':
    outdir = sys.argv[1]
    rng = np.random.default_rng(7)
    val = build_pool(97, 150)
    rnd_p = np.mean([episode_policy(sc, lambda r, c, g: int(g.choice(c)), rng) for sc in val])
    rot_p = np.mean([episode_policy(sc, lambda r, c, g: c[r % len(c)], rng) for sc in val])
    out = dict(random=round(float(rnd_p), 3), rotate=round(float(rot_p), 3), n=len(val))
    json.dump(out, open(os.path.join(outdir, 'qalloc_refs.json'), 'w'), indent=1)
    print(out)
