# -*- coding: utf-8 -*-
"""汇总 SITL 计时数据：由逐段到达时刻计算 L、A、D、T，并拟合 v_eff 与 t_r。"""
import json
import os
import sys
import glob

import pandas as pd

RUNS = sorted(glob.glob(os.path.join(sys.argv[1], '*.json')))
out = {}
for f in RUNS:
    d = json.load(open(f, encoding='utf-8'))
    if 'L' not in d:
        continue
    name = os.path.basename(f).replace('m1_', '').replace('.json', '')
    marks = d['marks']
    T = round(sum(m['dt'] for m in marks), 2)
    out[name] = dict(L=d['L'], A=d['A'], D=d['D'], T=T,
                     reached=all(m['reached'] for m in marks),
                     segs=[m['dt'] for m in marks])

print(json.dumps(out, ensure_ascii=False, indent=1))

# 拟合：T = L / v + A * t_r + c   （D 项单独由悬停对验证）
import numpy as np
rows = [(v['L'], v['A'], v['T']) for v in out.values() if v['D'] == 0]
A_ = np.array([[L, Ad, 1.0] for L, Ad, T in rows])
b_ = np.array([T for L, Ad, T in rows])
coef, res, rank, sv = np.linalg.lstsq(A_, b_, rcond=None)
inv_v, t_r, c = coef
summary = dict(inv_v=float(inv_v), v_eff=float(1.0 / inv_v), t_r=float(t_r), c=float(c),
               n=len(rows), residual=float(res[0]) if len(res) else None)
print(json.dumps(summary, ensure_ascii=False, indent=1))

# 悬停对：D 的线性可加性
hovers = {}
for name, v in out.items():
    if v['D'] > 0:
        base = out.get(name.replace('h%.0f' % v['D'], 'h0'))
        if base:
            hovers[name] = round(v['T'] - base['T'] - v['D'], 2)
print('hover residual (T_D - T_0 - D):', hovers)

out['_fit'] = summary
out['_hover_resid'] = hovers
json.dump(out, open(sys.argv[2], 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('written', sys.argv[2])
