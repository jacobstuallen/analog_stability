# Sanity check for delta mode: at GWL0 (the 1991-2020 baseline window) the
# future equals ERA5, so every gated analog must be retained and the
# stability sigma must equal the historical best-analog sigma.
# Run retention + stability with ANALOG_GWLS=gwl0,gwl2,gwl3 first.
# Usage: python tests/check_gwl0.py <out_dir>
import sys
import numpy as np
import xarray as xr

d = sys.argv[1]
r = xr.open_dataset(f'{d}/analog_retention.nc')
s = xr.open_dataset(f'{d}/analog_sets.nc')
st = xr.open_dataset(f'{d}/analog_stability_gwl.nc')
h = xr.open_dataset(f'{d}/analog_historical_sigma.nc')
assert r.attrs['future_mode'] == 'delta', 'run in delta mode'

ok = s.n_analogs.values > 0
ret = r.retention_gwl0.sel(Dt=1.25).values[ok]   # gate is sigma <= 1
a, b = st.sigma_gwl0_mean.values, h.sigma_best.values
m = np.isfinite(a) & np.isfinite(b)
dmax = float(np.abs(a[m] - b[m]).max())

print(f'retention GWL0 at Dt=1.25: min {np.nanmin(ret):.4f} (expect 1)')
print(f'stability GWL0 vs historical sigma: max |diff| {dmax:.2e} (expect 0)')
assert np.nanmin(ret) > 0.999 and dmax < 1e-5
print('PASS')
