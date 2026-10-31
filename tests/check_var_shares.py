# Unit test: per-variable Mahalanobis shares sum exactly to D^2 and agree
# with the PC-whitened distance, including a truncated PC.
# Usage: python tests/check_var_shares.py
import sys
import os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from analog_common import mahal_var_shares, pc_distance

rng = np.random.default_rng(1)
n = 5000
sd = rng.uniform(0.3, 3, (n, 2))
rho = rng.uniform(-0.9, 0.9, n)
rho[:100] = 0.995                       # near-singular: one PC dropped
R = np.zeros((n, 2, 2)); R[:, 0, 0] = R[:, 1, 1] = 1; R[:, 0, 1] = R[:, 1, 0] = rho
ev, ec = np.linalg.eigh(R)
diff = rng.normal(0, 2, (n, 2))
c, d2 = mahal_var_shares(diff, sd, ev, ec)
d, m = pc_distance(diff, sd, ev, ec)
print(f'max |sum c - D^2| {np.abs(c.sum(1) - d2).max():.2e}; max |D^2 - pc_distance^2| {np.abs(d2 - d**2).max():.2e}')
print(f'PC dropped in {np.sum(m < 2)} cells')
z = diff / sd
c0, _ = mahal_var_shares(diff, sd, np.ones((n, 2)), np.tile(np.eye(2), (n, 1, 1)))
print(f'uncorrelated case: shares == z^2: {np.allclose(c0, z**2)}')
assert np.allclose(c.sum(1), d2) and np.allclose(d2, d**2) and np.allclose(c0, z**2)
print('PASS')
