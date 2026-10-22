# -*- coding: utf-8 -*-
"""QAlloc 的 Q-learning 训练与学习曲线（可解场景池 + 轮转可学）。"""
import csv
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qalloc_env import Scenario, M_UAV
from diag_env import solvable

ALPHA = 0.25
GAMMA = 0.9
N_EPISODES = 96
PER_ROUND = 25
N_VAL = 150
MAX_ROUNDS = 18


def state_of(sc, alloc, rnd):
    u = float(np.mean(sc.utilization(alloc)))
    return (min(rnd, 4), min(int(u * 3), 2))


def step(sc, alloc, phi, src, a):
    seg = sc.tail_segment(src, alloc)
    if not seg:
        return None
    new = [list(b) for b in alloc]
    try:
        for r in seg:
            new[src].remove(r)
    except ValueError:
        return None
    new[a] = new[a] + seg
    if sc.utilization(new)[a] > 1.0:
        return None
    if sc.potential(new) >= phi - 1e-9:
        return None
    return new


def run_episode(sc, Q, rng, eps=0.0, learn=True):
    alloc = [list(b) for b in sc.blocks]
    phi = sc.potential(alloc)
    rnd, wasted, ret = 0, 0, 0.0
    while rnd < MAX_ROUNDS and wasted < 4 and phi > 1e-9:
        src = int(np.argmax(sc.utilization(alloc)))
        s = state_of(sc, alloc, rnd)
        cand = [j for j in range(M_UAV) if j != src]
        if learn and rng.random() >= eps:
            a = max(cand, key=lambda j: Q.get((s, j), 0.0))
        else:
            a = int(rng.choice(cand))
        new = step(sc, alloc, phi, src, a)
        s2 = s
        if new is not None:
            phi2 = sc.potential(new)
            r = 1.0 + (4.0 if phi2 <= 1e-9 else 0.0)
            s2 = state_of(sc, new, rnd + 1)
        else:
            r = -1.0
            wasted += 1
        if learn:
            best2 = max(Q.get((s2, j), 0.0) for j in range(M_UAV))
            Q[(s, a)] = Q.get((s, a), 0.0) + ALPHA * (r + GAMMA * best2 - Q.get((s, a), 0.0))
        if new is not None:
            alloc, phi = new, phi2
        ret += r
        rnd += 1
    ok = sc.potential(alloc) <= 1e-9
    if not ok:
        ret -= 2.0
    return ok, ret


def evaluate(Q, pool, rng):
    return float(np.mean([run_episode(sc, Q, rng, 0.0, False)[0] for sc in pool]))


def build_pool(seed, n, only_solvable=True):
    rng = np.random.default_rng(seed)
    kinds = ['pent', 'quad', 'L', 'U', 'C', 'star']
    pool, tries = [], 0
    while len(pool) < n and tries < n * 40:
        tries += 1
        k = kinds[len(pool) % len(kinds)]
        S = 100.0 + 50.0 * ((len(pool) // len(kinds)) % 2)
        bf = 1.4 if (len(pool) % 2 == 0) else 1.6
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
    outdir = sys.argv[1] if len(sys.argv) > 1 else '.'
    rng = np.random.default_rng(20261021)
    print('building scenarios ...', flush=True)
    train = build_pool(11, 200)
    val = build_pool(97, N_VAL)
    print('train', len(train), 'val', len(val), flush=True)

    Q, rows = {}, []
    for ep in range(1, N_EPISODES + 1):
        eps = max(0.05, 1.0 - ep / 40.0)
        idx = rng.integers(0, len(train), size=PER_ROUND)
        oks, rews = [], []
        for i in idx:
            g, r = run_episode(train[i], Q, rng, eps=eps, learn=True)
            oks.append(g)
            rews.append(r)
        v = evaluate(Q, val, rng)
        rows.append(dict(episode=ep, train_succ=float(np.mean(oks)), val_succ=v,
                         mean_reward=round(float(np.mean(rews)), 3),
                         eps=round(eps, 3), n_q=len(Q)))
        if ep <= 10 or ep % 4 == 0:
            print('ep %3d  train %.3f  val %.3f  R %+.2f  eps %.2f  |Q| %d' %
                  (ep, np.mean(oks), v, np.mean(rews), eps, len(Q)), flush=True)

    with open(os.path.join(outdir, 'qalloc_curve.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    ep_arr = np.array([r['episode'] for r in rows], float)
    val = np.array([r['val_succ'] for r in rows], float)
    k = 5
    smooth = np.convolve(val, np.ones(k) / k, mode='valid')
    final = float(smooth[-1])
    plateau = int(ep_arr[k - 1])
    for i, v in enumerate(smooth):
        if abs(v - final) <= 0.01:
            plateau = int(ep_arr[k - 1:][i])
            break
    summary = dict(final_val=final, final_train=float(rows[-1]['train_succ']),
                   plateau_episode=plateau, n_q=len(Q), episodes=N_EPISODES,
                   start_val=float(np.mean(val[:5])), oracle_ceiling=1.0)
    json.dump(summary, open(os.path.join(outdir, 'qalloc_summary.json'), 'w'),
              ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False))
