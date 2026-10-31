#!/bin/python
# author: (drafted for jsallen@ucar.edu)
#
# ANALOG RETENTION under global warming levels. The core scientific payload.
#
# Question: for each focal cell, do its present-day within-biome analogs (the
# diverse top-N set in analog_sets.nc) REMAIN good analogs under warming?
# i.e. can a place that shares your extremes today still inform your
# adaptation tomorrow, or does the analog relationship break?
#
# Metric (confirmed design): future-state resemblance with a FIXED RULER.
#   For focal f, analog a, model, GWL:
#     future_f = future climatology at f (per-model 20yr GWL window)
#     future_a = future climatology at a
#     In the default delta mode (analog_config.FUTURE_MODE) each future is
#     ERA5 1991-2020 + the model's own change, so model bias cancels and a
#     pair diverges only through differing changes.
#     sigma_fut = dissimilarity(future_a, future_f), scored in f's HISTORICAL
#                 ICV PC space (same Mahony whitening as selection/method 1).
#   The historical ICV is the fixed ruler: holding it constant between present
#   and future isolates whether the CLIMATES diverged (not whether variability
#   changed). An analog "survives" at threshold Dt if sigma_fut < Dt.
#
# Because Dt is not defensibly a single number, we SWEEP it:
#   retention(Dt) = fraction of the cell's N analogs with sigma_fut < Dt.
# Stored as a curve over a Dt grid, per cell, per GWL (model-aggregated).
#
# Two attribution diagnostics explain WHY retention fails, per analog pair:
#   magnitude : |delta_a| vs |delta_f|, change-magnitude ratio in std space
#               (delta = future - historical). Does the analog change as MUCH?
#   direction : cosine of angle between delta_f and delta_a in std space.
#               Does the analog change the same WAY? (the key one: an analog
#               moving orthogonally is no longer a preview of your future.)
# A failed-retention cell with HIGH direction congruence = "we moved together
# but the fixed ruler flags us" (mild); LOW congruence = "we genuinely
# diverged" (the real signal). This lets retention failure be attributed.
#
# Ensemble: per-model then aggregate (median + p10/p90 across models),
# consistent with method 1.
#
# Reads:  analog_sets.nc  (diverse analog sets + metadata)
# Writes: analog_retention.nc  (retention curves + magnitude/direction maps)
#
# Variable set is the same CONFIG list as method 1 (TXx, Rx5day now).

import numpy as np
import xarray as xr
import os
import time

import analog_config as cfg
from analog_common import (load_biomes, load_era5_icv, load_future_gwl,
                           grid_index_lookup, dt_grid, pc_sigma,
                           focal_eig as _focal_eig)

# ==============================
# Configuration (edit in analog_config.py; aliased here)
# ==============================
sets_file = cfg.SETS_FILE
out_dir = cfg.OUT_DIR
outfile = cfg.RETENTION_FILE
min_biome_cells = cfg.MIN_BIOME_CELLS
scenario = cfg.SCENARIO
DT_GRID = dt_grid()        # Dt sweep grid (sigma units)
DT_SLICES = cfg.DT_SLICES  # representative Dt slices for maps
VKEYS, K = cfg.VKEYS, cfg.K
biomes = cfg.BIOMES
GWLS = cfg.GWLS

# ==============================
# Load ERA5 ICV + biome + analog sets
# ==============================
print("Loading biome mask and analog sets...")
biome_grid, bmin, bmax = load_biomes()

sets_ds = xr.open_dataset(sets_file).sortby('lat')
SET_LAT = sets_ds['analog_lat'].values     # (nlat,nlon,N)
SET_LON = sets_ds['analog_lon'].values
N_SET = SET_LAT.shape[2]
N_FOUND = sets_ds['n_analogs'].values

clim_mean, icv_sd, icv_series, icv_corr, lat_arr, lon_arr = load_era5_icv(bmin, bmax)
nlat, nlon = len(lat_arr), len(lon_arr)

# ==============================
# Load CMIP6 future GWL climatologies per model
# ==============================
# future_ens[g][vk]: (n_models_g, nlat, nlon); models_used[g]: model names
future_ens, models_used = load_future_gwl(bmin, bmax, GWLS, clim_mean=clim_mean)

# lookup analog lat/lon value -> grid index
lat_to_i, lon_to_j = grid_index_lookup(lat_arr, lon_arr)

# ==============================
# Output arrays
# ==============================
nDt = len(DT_GRID)
out = {}
for g in GWLS:
    # retention curve: (nlat, nlon, nDt) ensemble-median fraction retained
    out[f'retention_{g}'] = np.full((nlat, nlon, nDt), np.nan, np.float32)
    out[f'retention_{g}_p10'] = np.full((nlat, nlon, nDt), np.nan, np.float32)
    out[f'retention_{g}_p90'] = np.full((nlat, nlon, nDt), np.nan, np.float32)
    # attribution maps (ensemble-median over models, mean over analogs)
    out[f'magratio_{g}'] = np.full((nlat, nlon), np.nan, np.float32)   # |da|/|df|
    out[f'dircong_{g}']  = np.full((nlat, nlon), np.nan, np.float32)   # cos angle
    # mean future sigma across the analog set (context for the curve)
    out[f'meansigma_{g}'] = np.full((nlat, nlon), np.nan, np.float32)


def focal_eig(vlat, vlon):
    """Focal-cell ICV eigendecomposition (see analog_common.focal_eig)."""
    return _focal_eig(vlat, vlon, clim_mean, icv_sd, icv_corr)

# ==============================
# Process biome by biome
# ==============================
for b, bname in biomes.items():
    t0 = time.time()
    bm = (biome_grid == b)
    clat, clon = np.where(bm)
    if len(clat) < min_biome_cells:
        print(f'\nBiome {b:2d} ({bname}): {len(clat)} cells, SKIPPING'); continue

    # valid focal cells: finite ICV + at least one analog
    mu0 = np.column_stack([clim_mean[vk][clat, clon] for vk in VKEYS])
    sd0 = np.column_stack([icv_sd[vk][clat, clon] for vk in VKEYS])
    valid = (np.all(np.isfinite(mu0), axis=1) & np.all(np.isfinite(sd0), axis=1)
             & np.all(sd0 > 0, axis=1) & (N_FOUND[clat, clon] > 0))
    if valid.sum() < min_biome_cells:
        print(f'\nBiome {b:2d} ({bname}): {valid.sum()} valid, SKIPPING'); continue
    vlat, vlon = clat[valid], clon[valid]
    n = len(vlat)
    print(f'\nBiome {b:2d} ({bname}): {n} valid cells')

    mu, sd, evals, evecs = focal_eig(vlat, vlon)

    # resolve analog grid indices for this biome's focal cells: (n, N_SET)
    ali = np.full((n, N_SET), -1, int)
    alj = np.full((n, N_SET), -1, int)
    nfound = N_FOUND[vlat, vlon]
    for r in range(N_SET):
        alv = SET_LAT[vlat, vlon, r]; aov = SET_LON[vlat, vlon, r]
        for c in range(n):
            if np.isfinite(alv[c]) and np.isfinite(aov[c]):
                ali[c, r] = lat_to_i.get(round(float(alv[c]), 4), -1)
                alj[c, r] = lon_to_j.get(round(float(aov[c]), 4), -1)

    for g in GWLS:
        # focal future vector (n,K) per model, and focal delta (n,K) per model
        fut_f = np.stack([future_ens[g][vk][:, vlat, vlon] for vk in VKEYS], axis=-1)  # (M,n,K)
        delta_f = fut_f - mu[None]                                                     # (M,n,K)

        # accumulators across models
        n_models = len(models_used[g])
        ret_curve_m = np.full((n_models, n, nDt), np.nan, np.float32)
        magr_m = np.full((n_models, n), np.nan, np.float32)
        dirc_m = np.full((n_models, n), np.nan, np.float32)
        msig_m = np.full((n_models, n), np.nan, np.float32)

        for mi in range(n_models):
            # per-analog future sigma + attribution, then aggregate over analogs
            sig_a = np.full((n, N_SET), np.nan)     # future sigma per analog
            mag_a = np.full((n, N_SET), np.nan)     # |da|/|df| per analog
            dir_a = np.full((n, N_SET), np.nan)     # cos angle per analog
            # focal delta magnitude in std space (n,)
            df_std = delta_f[mi] / sd               # (n,K)
            df_norm = np.sqrt(np.nansum(df_std**2, axis=1))
            for r in range(N_SET):
                ok = (ali[:, r] >= 0) & (alj[:, r] >= 0) & (r < nfound)
                if ok.sum() == 0:
                    continue
                ii = ali[ok, r]; jj = alj[ok, r]
                # analog future vector (m,K) and analog historical vector
                fa = np.column_stack([future_ens[g][vk][mi, ii, jj] for vk in VKEYS])
                ha = np.column_stack([clim_mean[vk][ii, jj] for vk in VKEYS])
                # retention: future_a vs future_f in focal ICV
                diff = fa - fut_f[mi][ok]            # (m,K)
                sig_a[ok, r] = pc_sigma(diff, sd[ok], evals[ok], evecs[ok])
                # attribution: analog delta in focal-std space
                da_std = (fa - ha) / sd[ok]
                da_norm = np.sqrt(np.nansum(da_std**2, axis=1))
                with np.errstate(invalid='ignore', divide='ignore'):
                    mag_a[ok, r] = da_norm / df_norm[ok]
                    dot = np.nansum(da_std * df_std[ok], axis=1)
                    dir_a[ok, r] = dot / (da_norm * df_norm[ok])

            # retention curve: fraction of found analogs with sigma < Dt
            for di, Dt in enumerate(DT_GRID):
                below = (sig_a < Dt)
                # count only over found analogs
                cnt = np.nansum(np.where(np.isfinite(sig_a), below, 0), axis=1)
                ret_curve_m[mi, :, di] = cnt / np.maximum(nfound, 1)
            magr_m[mi] = np.nanmean(mag_a, axis=1)
            dirc_m[mi] = np.nanmean(dir_a, axis=1)
            msig_m[mi] = np.nanmean(sig_a, axis=1)

        out[f'retention_{g}'][vlat, vlon, :]     = np.nanmedian(ret_curve_m, axis=0)
        out[f'retention_{g}_p10'][vlat, vlon, :] = np.nanpercentile(ret_curve_m, 10, axis=0)
        out[f'retention_{g}_p90'][vlat, vlon, :] = np.nanpercentile(ret_curve_m, 90, axis=0)
        out[f'magratio_{g}'][vlat, vlon] = np.nanmedian(magr_m, axis=0)
        out[f'dircong_{g}'][vlat, vlon]  = np.nanmedian(dirc_m, axis=0)
        out[f'meansigma_{g}'][vlat, vlon] = np.nanmedian(msig_m, axis=0)

        # quick read at the DT_SLICES
        slc = []
        for Dt in DT_SLICES:
            di = int(np.argmin(np.abs(DT_GRID - Dt)))
            slc.append(f'Dt={Dt}:{np.nanmean(out[f"retention_{g}"][vlat,vlon,di]):.2f}')
        print(f'  {g}: mean retention {" ".join(slc)} | dir={np.nanmean(out[f"dircong_{g}"][vlat,vlon]):.2f}')
    print(f'  done | {time.time()-t0:.0f}s')

# ==============================
# Save
# ==============================
os.makedirs(out_dir, exist_ok=True)
ds_out = xr.Dataset(coords={'lat': lat_arr, 'lon': lon_arr, 'Dt': DT_GRID})
for k, arr in out.items():
    if arr.ndim == 3:
        ds_out[k] = (['lat', 'lon', 'Dt'], arr)
    else:
        ds_out[k] = (['lat', 'lon'], arr)
ds_out['biome_id'] = (['lat', 'lon'], biome_grid.astype(np.float32))
ds_out['n_analogs'] = (['lat', 'lon'], N_FOUND.astype(np.float32))
ds_out.attrs['method'] = (
    'Analog retention under GWLs. Future-state resemblance with fixed '
    'historical-ICV ruler: per focal cell, fraction of its diverse top-N '
    'within-biome analogs (analog_sets.nc) whose FUTURE climatology stays '
    'within Dt sigma of the focal cell FUTURE climatology, swept over Dt. '
    'Per-model (SSP5-8.5 GWL windows, see gwl_source) then ensemble median/p10/p90. '
    'Attribution: magratio (|delta_a|/|delta_f|) and dircong (cos angle of '
    'delta_a vs delta_f) in focal-std space, mean over the analog set.')
ds_out.attrs['icv_definition'] = cfg.ICV_DEFINITION
ds_out.attrs['variables'] = ', '.join(VKEYS)
ds_out.attrs['scenario'] = scenario
ds_out.attrs['gwl_source'] = cfg.GWL_SOURCE
ds_out.attrs['model_set'] = cfg.MODEL_SET
for g in GWLS:
    ds_out.attrs[f'models_{g}'] = ', '.join(models_used[g])
    ds_out.attrs[f'n_models_{g}'] = len(models_used[g])
ds_out.attrs['future_mode'] = cfg.FUTURE_MODE
ds_out.attrs['future_definition'] = (
    'delta: ERA5 1991-2020 mean + (model GWL window - model 1991-2020 mean; '
    'historical 1991-2014 + scenario 2015-2020), log1p space for log variables. '
    'raw: model GWL window as-is.')
ds_out.attrs['dt_grid'] = ', '.join(f'{d}' for d in DT_GRID)
ds_out.to_netcdf(outfile)
print(f'\nSaved: {outfile}')

# ==============================
# Per-biome summary at the Dt slices
# ==============================
print('\n--- Per-biome ensemble-median retention ---')
hdr = f'{"biome":<46}'
for g in GWLS:
    for Dt in DT_SLICES:
        hdr += f' {g[3:]:>3}@{Dt:<3}'
for g in GWLS:
    hdr += f' {"dir" + g[3:]:>6}'
print(hdr)
for b, bname in biomes.items():
    bm = (biome_grid == b) & (N_FOUND > 0)
    if bm.sum() == 0:
        continue
    row = f'{bname:<46}'
    for g in GWLS:
        for Dt in DT_SLICES:
            di = int(np.argmin(np.abs(DT_GRID - Dt)))
            row += f' {np.nanmean(out[f"retention_{g}"][bm, di]):>7.2f}'
    for g in GWLS:
        row += f' {np.nanmean(out[f"dircong_{g}"][bm]):>6.2f}'
    print(row)

print('\nDone.')
