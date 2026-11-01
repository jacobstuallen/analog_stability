#!/bin/python
#
# Do the CCKP (bias-adjusted) CMIP6 models have the same interannual
# variability (ICV) as ERA5?
#
# Both noise floors in this project (retention divergence floor, scaling-test
# null) assume each model's year-to-year noise equals the ERA5 ICV used as
# the ruler. Quantile mapping to ERA5 makes that plausible but not certain,
# especially for tails (TXx, Rx5day) and in complex terrain.
#
# For each model and variable: detrended 1991-2020 SD (historical 1991-2014 +
# SSP5-8.5 2015-2020; same QC, log1p and detrending as ERA5) divided by the
# ERA5 SD. Writes the median ratio over models per cell (used by
# analog.scaling.py to scale its noise model) plus the 10th/90th percentiles,
# and prints biome medians.
#
# Writes: model_icv_ratio.nc

import numpy as np
import xarray as xr
import os
import warnings

import analog_config as cfg
from analog_common import load_biomes, load_era5_icv, cckp_ts_path, _read_years, ensure_out_dir
from icv_utils import detrend_icv

VKEYS = cfg.VKEYS

biome_grid, bmin, bmax = load_biomes()
clim_mean, icv_sd, _, _, lat_arr, lon_arr = load_era5_icv(bmin, bmax)

ratios = {vk: [] for vk in VKEYS}
models_used = []
for model, ens in cfg.MODELS.items():
    paths = {(vk, e): cckp_ts_path(cfg.VARIABLES[vk]['cckp_token'], model, ens, e)
             for vk in VKEYS for e in ('historical', cfg.SCENARIO)}
    if not all(os.path.exists(p) for p in paths.values()):
        print(f'  {model}: missing files, skipping')
        continue
    print(f'  {model}...', end=' ', flush=True)
    for vk in VKEYS:
        v = cfg.VARIABLES[vk]
        name = f'timeseries-{v["cckp_token"]}-annual-mean'
        arr = np.concatenate([
            _read_years(paths[(vk, 'historical')], name, cfg.CLIM_START, cfg.MODEL_HIST_END, bmin, bmax),
            _read_years(paths[(vk, cfg.SCENARIO)], name, cfg.MODEL_HIST_END + 1, cfg.CLIM_END, bmin, bmax)])
        arr[arr > v['qc_max']] = np.nan
        if v['log']:
            arr = np.log1p(arr)
        _, sd_m, _ = detrend_icv(arr)
        with np.errstate(invalid='ignore', divide='ignore'):
            ratios[vk].append((sd_m / icv_sd[vk]).astype(np.float32))
    models_used.append(model)
    print('OK')

ds = xr.Dataset(coords={'lat': lat_arr, 'lon': lon_arr})
with warnings.catch_warnings():
    warnings.simplefilter('ignore', RuntimeWarning)
    for vk in VKEYS:
        stk = np.stack(ratios[vk], axis=0)
        ds[f'ratio_{vk}'] = (['lat', 'lon'], np.nanmedian(stk, axis=0))
        ds[f'ratio_{vk}_p10'] = (['lat', 'lon'], np.nanpercentile(stk, 10, axis=0))
        ds[f'ratio_{vk}_p90'] = (['lat', 'lon'], np.nanpercentile(stk, 90, axis=0))
ds['biome_id'] = (['lat', 'lon'], biome_grid.astype(np.float32))
ds.attrs['method'] = (
    'Model / ERA5 detrended 1991-2020 ICV SD per cell (hist 1991-2014 + '
    f'{cfg.SCENARIO} 2015-2020; same QC, log1p and detrending as ERA5). '
    'ratio_<var>: median over models; _p10/_p90 across models.')
ds.attrs['models'] = ', '.join(models_used)
ensure_out_dir(cfg.OUT_DIR)
ds.to_netcdf(cfg.MODEL_ICV_FILE)
print(f'\nSaved: {cfg.MODEL_ICV_FILE}')

print(f'\n--- Model / ERA5 ICV ratio, biome medians ({len(models_used)} models) ---')
print('    1 = models match ERA5. Noise variance scales with ratio^2.')
print(f'{"biome":<52}' + ''.join(f' {vk:>8}' for vk in VKEYS))
for b, bname in cfg.BIOMES.items():
    bm = biome_grid == b
    if bm.sum() < cfg.MIN_BIOME_CELLS:
        continue
    print(f'{bname:<52}' + ''.join(f' {np.nanmedian(ds[f"ratio_{vk}"].values[bm]):>8.2f}' for vk in VKEYS))
print('\nDone.')
