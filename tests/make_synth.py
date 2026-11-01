# Synthetic test data for the analog pipeline (small grid, 3 models).
# Usage: python tests/make_synth.py <out_root>
import numpy as np, xarray as xr, os, sys
root = sys.argv[1]
rng = np.random.default_rng(1)
lat = np.arange(-20, 20, 0.5) + 0.25    # 80
lon = np.arange(0, 80, 0.5) + 0.25      # 160
LA, LO = np.meshgrid(lat, lon, indexing='ij')
biome = np.where(LO < 40, 1.0, 14.0)
biome[(LA > 15) & (LO > 70)] = 4.0       # small biome (~ 200 cells)
os.makedirs(f'{root}/biomes', exist_ok=True)
xr.Dataset({'BIOME_ID': (['lat', 'lon'], biome)}, coords={'lat': lat, 'lon': lon}).to_netcdf(f'{root}/biomes/biomes.analog.gridded.nc')
os.makedirs(f'{root}/CCKP/ERA5', exist_ok=True)
yrs = np.arange(1950, 2025)
t = xr.date_range('1950-01-01', periods=len(yrs), freq='YS')
txx_m = 30 + 10*np.cos(np.radians(LA*3)) + 0.05*LO
rx_m = 50 + 40*np.sin(np.radians(LO*4))**2 + LA
def ser(mean, sd, trend):
    n = len(yrs)
    return mean[None] + trend*(yrs-1990)[:, None, None] + sd*rng.standard_normal((n,)+mean.shape)
txx = ser(txx_m, 1.0, 0.03)
rx = np.clip(ser(rx_m, 8.0, 0.1), 1, None)
rx[:, 0, 0] = np.nan  # a NaN cell
for tok, arr in [('txx', txx), ('rx5day', rx)]:
    xr.Dataset({f'timeseries-{tok}-annual-mean': (['time', 'lat', 'lon'], arr)},
               coords={'time': t, 'lat': lat, 'lon': lon}).to_netcdf(
        f'{root}/CCKP/ERA5/timeseries-{tok}-annual-mean_era5-x0.25_era5-x0.25-historical_timeseries_mean_1950-2024.nc')
fy_all = np.arange(1950, 2101)
for mi, (model, ens) in enumerate([('access-cm2', 'r1i1p1f1'), ('canesm5', 'r1i1p1f1'), ('miroc6', 'r1i1p1f1')]):
    bias_t = rng.normal(0, 1, txx_m.shape) + mi
    bias_r = rng.normal(0, 5, rx_m.shape)
    for tok, m, sd, tr, b in [('txx', txx_m, 1.0, 0.04 + 0.0005*LA, bias_t), ('rx5day', rx_m, 8.0, 0.2, bias_r)]:
        # model = truth + fixed per-model bias + spatially varying trend + noise
        arr = m[None] + b[None] + tr*(fy_all-1990)[:, None, None] + sd*rng.standard_normal((len(fy_all),)+m.shape)
        if tok == 'rx5day': arr = np.clip(arr, 1, None)
        for exp, y0, y1 in [('historical', 1950, 2014), ('ssp585', 2015, 2100)]:
            sl = (fy_all >= y0) & (fy_all <= y1)
            d = f'{root}/CCKP/CMIP6/{tok}/{model}-{ens}-{exp}'; os.makedirs(d, exist_ok=True)
            xr.Dataset({f'timeseries-{tok}-annual-mean': (['time', 'lat', 'lon'], arr[sl])},
                       coords={'time': xr.date_range(f'{y0}-01-01', periods=int(sl.sum()), freq='YS'),
                               'lat': lat, 'lon': lon}).to_netcdf(
                f'{d}/timeseries-{tok}-annual-mean_cmip6-x0.25_{model}-{ens}-{exp}_timeseries_mean_{y0}-{y1}.nc')

# ---- means variable set (tas, pr): separate RNG so the extremes files above
# stay identical to earlier versions. Spatial patterns differ from TXx/Rx5day
# so the two analog sets are not the same places.
rng2 = np.random.default_rng(2)
tas_m = 20 + 8*np.cos(np.radians(LA*4)) - 0.04*LO
pr_m = 900 + 600*np.cos(np.radians(LO*3))**2 + 10*LA
def ser2(mean, sd, trend):
    return mean[None] + trend*(yrs-1990)[:, None, None] + sd*rng2.standard_normal((len(yrs),)+mean.shape)
tas = ser2(tas_m, 0.5, 0.025)
pr = np.clip(ser2(pr_m, 120.0, 0.5), 10, None)
for tok, arr in [('tas', tas), ('pr', pr)]:
    xr.Dataset({f'timeseries-{tok}-annual-mean': (['time', 'lat', 'lon'], arr)},
               coords={'time': t, 'lat': lat, 'lon': lon}).to_netcdf(
        f'{root}/CCKP/ERA5/timeseries-{tok}-annual-mean_era5-x0.25_era5-x0.25-historical_timeseries_mean_1950-2024.nc')
for mi, (model, ens) in enumerate([('access-cm2', 'r1i1p1f1'), ('canesm5', 'r1i1p1f1'), ('miroc6', 'r1i1p1f1')]):
    for tok, m, sd, tr, b in [('tas', tas_m, 0.5, 0.03 + 0.0004*LA, rng2.normal(0, 1, tas_m.shape)),
                              ('pr', pr_m, 120.0, 0.8 - 0.02*LA, rng2.normal(0, 60, pr_m.shape))]:
        arr = m[None] + b[None] + tr*(fy_all-1990)[:, None, None] + sd*rng2.standard_normal((len(fy_all),)+m.shape)
        if tok == 'pr': arr = np.clip(arr, 10, None)
        for exp, y0, y1 in [('historical', 1950, 2014), ('ssp585', 2015, 2100)]:
            sl = (fy_all >= y0) & (fy_all <= y1)
            d = f'{root}/CCKP/CMIP6/{tok}/{model}-{ens}-{exp}'; os.makedirs(d, exist_ok=True)
            xr.Dataset({f'timeseries-{tok}-annual-mean': (['time', 'lat', 'lon'], arr[sl])},
                       coords={'time': xr.date_range(f'{y0}-01-01', periods=int(sl.sum()), freq='YS'),
                               'lat': lat, 'lon': lon}).to_netcdf(
                f'{d}/timeseries-{tok}-annual-mean_cmip6-x0.25_{model}-{ens}-{exp}_timeseries_mean_{y0}-{y1}.nc')
print('ok')
