# -*- coding: utf-8 -*-
"""画 QAlloc 学习曲线（Pillow，含旋转纵轴标签）。"""
import csv
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

W, Ht = 1600, 1000
L, R, T, B = 170, 60, 110, 150


def font(sz, bold=False):
    p = r'C:\Windows\Fonts\arialbd.ttf' if bold else r'C:\Windows\Fonts\arial.ttf'
    try:
        return ImageFont.truetype(p, sz)
    except Exception:
        return ImageFont.load_default()


def vtext(base, text, fnt, fill):
    """竖排文字，返回贴图。"""
    tmp = Image.new('RGBA', (600, 80), (255, 255, 255, 0))
    dd = ImageDraw.Draw(tmp)
    dd.text((0, 0), text, font=fnt, fill=fill)
    bbox = tmp.getbbox()
    tmp = tmp.crop(bbox).rotate(90, expand=True)
    return tmp


def main(outdir):
    rows = list(csv.DictReader(open(os.path.join(outdir, 'qalloc_curve.csv'))))
    ep = np.array([float(r['episode']) for r in rows])
    va = np.array([float(r['val_succ']) for r in rows])
    refs = json.load(open(os.path.join(outdir, 'qalloc_refs.json')))
    sm = np.convolve(va, np.ones(5) / 5, mode='valid')
    ep_sm = ep[4:]

    img = Image.new('RGB', (W, Ht), 'white')
    d = ImageDraw.Draw(img)
    f14, f16, f18, f13 = font(26), font(30), font(34, True), font(23)

    x0, x1 = L, W - R
    y0, y1 = Ht - B, T
    lo, hi = 0.60, 0.90
    X = lambda e: x0 + (e - 1) / (ep[-1] - 1) * (x1 - x0)
    Y = lambda v: y1 + (hi - v) / (hi - lo) * (y0 - y1)

    tail = sm[ep_sm >= 20]
    m_, s_ = tail.mean(), tail.std()
    d.rectangle([x0, Y(m_ + s_), x1, Y(m_ - s_)], fill=(230, 239, 250))

    for v in np.arange(lo, hi + 1e-9, 0.05):
        d.line([(x0, Y(v)), (x1, Y(v))], fill=(228, 228, 228), width=2)
        d.text((x0 - 16, Y(v)), '%.2f' % v, font=f13, fill=(55, 55, 55), anchor='rm')
    for e in range(1, 97, 8):
        d.line([(X(e), y0), (X(e), y0 + 8)], fill=(120, 120, 120), width=2)
        d.text((X(e), y0 + 16), str(e), font=f13, fill=(55, 55, 55), anchor='ma')

    d.line([(x0, y0), (x1, y0)], fill=(25, 25, 25), width=3)
    d.line([(x0, y0), (x0, y1)], fill=(25, 25, 25), width=3)

    for val, col, lab, va_ in (
            (refs['rotate'], (70, 145, 85), 'hand-crafted rotation  %.2f' % refs['rotate'], 'ls'),
            (refs['random'], (150, 150, 150), 'random choice  %.2f' % refs['random'], 'ls')):
        y = Y(val)
        for xx in range(x0, x1, 26):
            d.line([(xx, y), (xx + 13, y)], fill=col, width=3)
        d.text((x0 + 10, y + (10 if 'random' in lab else -10)), lab, font=f13, fill=col, anchor=va_)

    for e, v in zip(ep, va):
        d.ellipse([X(e) - 3.2, Y(v) - 3.2, X(e) + 3.2, Y(v) + 3.2], fill=(140, 175, 225))
    d.line([(X(a), Y(b)) for a, b in zip(ep_sm, sm)], fill=(20, 70, 155), width=5)

    d.text((x1 - 10, Y(m_ + s_) - 10), 'plateau band  %.2f ± %.2f' % (m_, s_),
           font=f13, fill=(85, 115, 165), anchor='rs')

    img.paste(vtext(img, 'Success rate', f14, (20, 20, 20)), (24, int((y0 + y1) / 2) - 110))
    d.text(((x0 + x1) / 2, Ht - 56), 'Training rounds', font=f16, fill=(20, 20, 20), anchor='ma')
    d.text(((x0 + x1) / 2, 42),
           'QAlloc learning curve (E1 takeover, m = 4): converged within ~10 rounds',
           font=f18, fill=(20, 20, 20), anchor='ma')

    out = os.path.join(outdir, 'fig_qalloc_curve.png')
    img.save(out, dpi=(300, 300))
    print('saved', out, 'plateau %.3f +- %.3f' % (m_, s_))


if __name__ == '__main__':
    main(sys.argv[1])
