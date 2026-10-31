# Sanity check for delta mode: at GWL0 (the 1991-2020 baseline window) the
# future equals ERA5, so every gated analog is retained, local departure and
# pair divergence are zero, and the future pair distance equals the
# historical one.
# Run analog.retention.py with ANALOG_GWLS including gwl0 first.
# Usage: python tests/check_gwl0.py <out_dir>
import sys
import numpy as np
import xarray as xr

d = sys.argv[1]
r = xr.open_dataset(f'{d}/analog_retention.nc')
s = xr.open_dataset(f'{d}/analog_sets.nc')
assert r.attrs['future_mode'] == 'delta', 'run in delta mode'

ok = s.n_analogs.values > 0
ret = r.retention_gwl0.sel(Dt=1.25).values[ok]   # gate is sigma <= 1
dep = np.nanmax(np.abs(r.departure_gwl0.values))
div = np.nanmax(np.abs(r.divergence_gwl0.values))
print(f'retention GWL0 at Dt=1.25: min {np.nanmin(ret):.4f} (expect 1)')
print(f'departure / divergence GWL0: max {dep:.2e} / {div:.2e} (expect 0)')
assert np.nanmin(ret) > 0.999 and dep < 1e-5 and div < 1e-5
print('PASS')
