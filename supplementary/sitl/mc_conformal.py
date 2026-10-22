# -*- coding: utf-8 -*-
"""独立数值校验：乘性误差下 (i) 分裂共形分位数的覆盖率，
(ii) 自适应共形推断(ACI) 在漂移下的长期失配率。

模型与论文式(20)-(21) 同构：
    e_meas = theta * e_base * exp(eta),   theta_hat = e_meas/e_base = theta*exp(eta)
    R = e/e_hat = theta/theta_hat = exp(-eta)     （可由遥测直接算出）
保证形式：Pr[ e(a) <= kappa * e_hat(a) ] >= 1-delta  <=>  Pr[ R <= kappa ] >= 1-delta
"""
import numpy as np

RNG = np.random.default_rng(20261021)
DELTA = 0.05
CONF = 1.0 - DELTA
SIGMA_ASSUMED = 0.035
N_CAL = 200
REPS = 10000


def sample_eta(kind, size, rng):
    if kind == 'gauss':
        return rng.normal(0.0, 0.035, size)
    if kind == 'laplace':
        return rng.laplace(0.0, 0.035 / np.sqrt(2), size)
    if kind == 't3':
        return rng.standard_t(3, size) * (0.035 / np.sqrt(3.0))
    if kind == 'mixture':
        comp = rng.random(size) < 0.9
        out = np.where(comp, rng.normal(0.0, 0.02, size), rng.normal(0.0, 0.093, size))
        return out
    if kind == 'exp':
        # 零均值、方差 0.035^2 的平移指数分布（右偏，95% 分位高于高斯）
        return 0.035 - rng.exponential(0.035, size)
    if kind == 'drift':
        seg = (np.arange(size) // 200) % 2
        sd = np.where(seg == 0, 0.02, 0.08)
        return rng.normal(0.0, sd, size)
    raise ValueError(kind)


def conformal_quantile(r, delta):
    """分裂共形分位数：第 ceil((n+1)(1-delta)) 个顺序统计量。"""
    n = len(r)
    k = int(np.ceil((n + 1) * (1.0 - delta)))
    k = min(max(k, 1), n)
    return np.sort(r)[k - 1]


def coverage(kind, reps=REPS):
    cov_cf, cov_g = [], []
    kappa_g = np.exp(1.6448536269514722 * SIGMA_ASSUMED)   # z_{0.95} * sigma
    for _ in range(reps):
        eta_cal = sample_eta(kind, N_CAL, RNG)
        r_cal = np.exp(-eta_cal)
        kappa = conformal_quantile(r_cal, DELTA)
        r_test = np.exp(-sample_eta(kind, 1, RNG))[0]
        cov_cf.append(r_test <= kappa)
        cov_g.append(r_test <= kappa_g)
    return float(np.mean(cov_cf)), float(np.mean(cov_g))


def aci(kind='drift', T=4000, gamma=0.01, window=200, alpha0=DELTA):
    """自适应共形：alpha_{t+1} = alpha_t + gamma(delta - err_t)。"""
    eta = sample_eta(kind, T + window, RNG)
    r = np.exp(-eta)
    alpha = alpha0
    errs = []
    kappa_fixed = np.exp(1.6448536269514722 * SIGMA_ASSUMED)
    errs_fixed = []
    for t in range(T):
        cal = r[t:t + window]
        kappa = conformal_quantile(cal, alpha)
        e = 1.0 if r[window + t] > kappa else 0.0
        errs.append(e)
        alpha = alpha + gamma * (DELTA - e)
        alpha = min(max(alpha, 1e-3), 1 - 1e-3)
        errs_fixed.append(1.0 if r[window + t] > kappa_fixed else 0.0)
    return np.mean(errs), np.mean(errs_fixed)


if __name__ == '__main__':
    print('=== 覆盖率（标称 1-delta = %.3f, n_cal=%d, reps=%d） ===' % (CONF, N_CAL, REPS))
    for kind in ('gauss', 'laplace', 't3', 'mixture', 'exp'):
        cf, gm = coverage(kind)
        print('%-9s conformal=%.4f   gaussian(assumed sigma=%.3f)=%.4f' % (kind, cf, SIGMA_ASSUMED, gm))

    print()
    print('=== ACI 在分段漂移下的长期失配率（delta=%.2f, T=4000, gamma=0.01） ===' % DELTA)
    aci_rate, fixed_rate = aci()
    print('ACI=%.4f   fixed-kappa=%.4f' % (aci_rate, fixed_rate))

    print()
    print('=== 额外：静止段（gauss）上的 ACI ===')
    aci_g, fixed_g = aci('gauss')
    print('ACI=%.4f   fixed-kappa=%.4f' % (aci_g, fixed_g))
