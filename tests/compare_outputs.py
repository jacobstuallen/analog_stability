# Compare two pipeline output directories variable by variable.
# Usage: python tests/compare_outputs.py <dir_a> <dir_b>
import xarray as xr, numpy as np, sys
a, b = sys.argv[1], sys.argv[2]
for f in ['analog_historical_sigma.nc','analog_sets.nc','analog_retention.nc','analog_stability_gwl.nc']:
    A = xr.open_dataset(f'{a}/{f}'); B = xr.open_dataset(f'{b}/{f}')
    bad = []
    for v in A.data_vars:
        x, y = A[v].values, B[v].values
        if x.shape != y.shape or not np.allclose(x, y, equal_nan=True, atol=1e-6): bad.append(v)
    print(f, 'MATCH' if not bad else f'DIFF {bad}', '| attrs differ:', [k for k in A.attrs if A.attrs[k] != B.attrs.get(k)])
for f in ['per_model_biome_sigma.npz','analog_historical_dist_summary.npz']:
    A = np.load(f'{a}/{f}'); B = np.load(f'{b}/{f}')
    ok = set(A.files) == set(B.files) and all(np.array_equal(A[k], B[k]) or np.allclose(A[k].astype(float), B[k].astype(float), equal_nan=True) for k in A.files if A[k].dtype.kind != 'U')
    print(f, 'MATCH' if ok else 'DIFF')
