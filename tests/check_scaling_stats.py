# Unit tests for analog_common.linear_fit_nlr and window_noise_cov.
#  1. Independent noise (scalar variance): mean nlr ~1, ~5% of p < 0.05.
#  2. Curved signal: nlr >> 1; the fitted rate recovers the true slope.
#  3. Realistic noise: white annual noise at two cells, averaged over the
#     ACTUAL AR6 GWL windows of several models (overlapping windows and a
#     shared 1991-2020 baseline). With the covariance from window_noise_cov,
#     mean nlr ~1 and ~5% rejections. With the naive scalar noise model the
#     test is miscalibrated (shown for reference).
# Usage: python tests/check_scaling_stats.py
import sys
import os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import analog_config as cfg
from analog_common import linear_fit_nlr, window_noise_cov

rng = np.random.default_rng(0)
K = 2
g = np.array([1.5, 2.0, 3.0, 4.0])

# ---- 1 and 2: independent noise ----
n = 200_000
noise_var = 2.0 / 20 / 18
kd = np.full(n, K)
beta_true = rng.normal(0, 0.5, (n, K))
alpha_true = -0.9 * beta_true
lin = (alpha_true[:, None, :] + beta_true[:, None, :] * g[None, :, None]
       + rng.normal(0, np.sqrt(noise_var), (n, len(g), K)))
rate, nlr, p, _, _ = linear_fit_nlr(lin, g, noise_var, kd)
rate_err = np.median(np.abs(rate - np.linalg.norm(beta_true, axis=1)))
print(f'1 independent : mean nlr {nlr.mean():.3f} (expect 1), p<0.05 {np.mean(p < 0.05):.3f}, '
      f'median |rate err| {rate_err:.3f}')
assert abs(nlr.mean() - 1) < 0.02 and abs(np.mean(p < 0.05) - 0.05) < 0.005

curv = lin + 0.15 * (g[None, :, None] - 2.5)**2 * np.sign(beta_true[:, None, :])
_, nlr_c, p_c, _, _ = linear_fit_nlr(curv, g, noise_var, kd)
print(f'2 curved      : mean nlr {nlr_c.mean():.2f} (expect >> 1), p<0.05 {np.mean(p_c < 0.05):.3f}')
assert nlr_c.mean() > 3

# ---- 3: realistic overlapping windows ----
keys = ['gwl1p5', 'gwl2', 'gwl3', 'gwl4']
models = [m for m in cfg.MODELS if all(m in cfg.GWL_WINDOWS[k] for k in keys)]
base = (cfg.CLIM_START, cfg.CLIM_END)
years = np.arange(1950, 2101)
npair = 20_000
M = len(models)


def window_mean(series, w):
    sel = (years >= w[0]) & (years <= w[1])
    return series[..., sel].mean(axis=-1)


V = np.zeros((npair, len(g), K))
C = np.zeros((len(g), len(g)))
for m in models:
    wins = [cfg.GWL_WINDOWS[k][m] for k in keys]
    C += 2.0 * window_noise_cov(wins, base)          # two cells per pair
    for cell_sign in (+1, -1):                       # change_a - change_f
        ann = rng.normal(0, 1, (npair, K, len(years)))
        b = window_mean(ann, base)
        for li, w in enumerate(wins):
            V[:, li, :] += cell_sign * (window_mean(ann, w) - b)
V /= M
C /= M**2
V += (alpha_true[:npair, None, :] + beta_true[:npair, None, :] * g[None, :, None])
kd3 = np.full(npair, K)
_, nlr3, p3, _, _ = linear_fit_nlr(V, g, C, kd3)
print(f'3 AR6 windows ({M} models): exact cov  mean nlr {nlr3.mean():.3f}, p<0.05 {np.mean(p3 < 0.05):.3f}')
naive = 2.0 / 20 / M
_, nlr3n, p3n, _, _ = linear_fit_nlr(V, g, naive, kd3)
print(f'                       naive scalar mean nlr {nlr3n.mean():.3f}, p<0.05 {np.mean(p3n < 0.05):.3f}')
assert abs(nlr3.mean() - 1) < 0.03 and abs(np.mean(p3 < 0.05) - 0.05) < 0.01
print('PASS')
