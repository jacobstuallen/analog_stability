# Unit test for analog_common.linear_fit_nlr. Under a linear signal plus
# white noise of the stated variance, nlr should average ~1 and ~5% of pairs
# should have p < 0.05. A curved signal should push nlr well above 1, and
# the fitted rate should recover the true slope.
# Usage: python tests/check_scaling_stats.py
import sys
import os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from analog_common import linear_fit_nlr

rng = np.random.default_rng(0)
n, K = 200_000, 2
g = np.array([1.5, 2.0, 3.0, 4.0])
noise_var = 2.0 / 20 / 18
kd = np.full(n, K)
beta_true = rng.normal(0, 0.5, (n, K))
alpha_true = -0.9 * beta_true
noise = rng.normal(0, np.sqrt(noise_var), (n, len(g), K))

lin = alpha_true[:, None, :] + beta_true[:, None, :] * g[None, :, None] + noise
rate, nlr, p, _, beta = linear_fit_nlr(lin, g, noise_var, kd)
rate_err = np.median(np.abs(rate - np.linalg.norm(beta_true, axis=1)))
print(f'linear : mean nlr {nlr.mean():.3f} (expect 1), frac p<0.05 {np.mean(p < 0.05):.3f} '
      f'(expect 0.05), median |rate err| {rate_err:.3f}')
assert abs(nlr.mean() - 1) < 0.02 and abs(np.mean(p < 0.05) - 0.05) < 0.005

curv = lin + 0.15 * (g[None, :, None] - 2.5)**2 * np.sign(beta_true[:, None, :])
_, nlr_c, p_c, _, _ = linear_fit_nlr(curv, g, noise_var, kd)
print(f'curved : mean nlr {nlr_c.mean():.2f} (expect >> 1), frac p<0.05 {np.mean(p_c < 0.05):.3f}')
assert nlr_c.mean() > 3

kd1 = np.ones(n)
lin1 = lin.copy(); lin1[:, :, 1] = 0.0          # one retained dimension
_, nlr1, p1, _, _ = linear_fit_nlr(lin1, g, noise_var, kd1)
print(f'1 dim  : mean nlr {nlr1.mean():.3f}, frac p<0.05 {np.mean(p1 < 0.05):.3f}')
assert abs(nlr1.mean() - 1) < 0.02
print('PASS')
