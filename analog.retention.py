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
import warnings

import analog_config as cfg
from analog_common import (load_biomes, load_era5_icv, load_future_gwl,
                           grid_index_lookup, dt_grid, pc_sigma, pc_whiten,
                           sigma_by_dims, mahal_var_shares, window_noise_cov,
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

# Forced response (noise floor, plan step 8): append the multi-model-mean
# future as one extra pseudo-member at index n_models. Per-model statistics
# use members [0, n_models); *_mmm_* outputs come from the mean. The mean
# keeps the shared forced pattern and cuts interannual sampling noise by
# ~sqrt(n_models); it also smooths model-to-model pattern differences, so
# mmm divergence is a lower bound on any single model's.
for g in GWLS:
    for vk in VKEYS:
        arr = future_ens[g][vk]
        future_ens[g][vk] = np.concatenate([arr, np.nanmean(arr, axis=0)[None]], axis=0)

# Analytic noise floor of |divergence| if there were NO forced divergence:
# white annual noise with ERA5 ICV at both cells, each model's actual window
# and the 1991-2020 baseline (window_noise_cov). Median |v| = sqrt(var) *
# sqrt(median chi2_K). Per model (median over models) and for the mean.
from scipy.stats import chi2 as _chi2
_chimed = np.sqrt(_chi2.median(K))
NOISE_FLOOR = {}
for g in GWLS:
    _var = np.array([2.0 * window_noise_cov([cfg.GWL_WINDOWS[g][m]],
                                            (cfg.CLIM_START, cfg.CLIM_END))[0, 0]
                     for m in models_used[g]])
    NOISE_FLOOR[g] = (float(np.sqrt(np.median(_var)) * _chimed),
                      float(np.sqrt(_var.sum() / len(_var)**2) * _chimed))

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
    # mean future sigma across the analog set (context for the curve)
    out[f'meansigma_{g}'] = np.full((nlat, nlon), np.nan, np.float32)
    # departure vs divergence (all whitened in focal ICV PC space, unsaturated
    # Mahalanobis units unless noted). Future pair difference = h + v, with
    # h = historical difference and v = change_a - change_f (pair divergence).
    out[f'departure_{g}'] = np.full((nlat, nlon), np.nan, np.float32)        # |change_f|
    out[f'departure_sigma_{g}'] = np.full((nlat, nlon), np.nan, np.float32)  # same, as sigma
    out[f'divergence_{g}'] = np.full((nlat, nlon), np.nan, np.float32)       # |v|, median over analogs
    out[f'comove_{g}'] = np.full((nlat, nlon), np.nan, np.float32)           # 1-|v|^2/(|ca|^2+|cf|^2)
    out[f'futdist_{g}'] = np.full((nlat, nlon), np.nan, np.float32)          # |h+v|, median over analogs
    # attribution of divergence (replaces magratio/dircong, issue C):
    # share of divergence^2 from each variable (exact Mahalanobis split)
    for vk in VKEYS:
        out[f'share_{vk}_div_{g}'] = np.full((nlat, nlon), np.nan, np.float32)
    # ICV-free views (issue B): per-variable divergence in physical units, and
    # divergence/departure measured with ONE global ICV ruler for every cell
    for vk in VKEYS:
        out[f'divabs_{vk}_{g}'] = np.full((nlat, nlon), np.nan, np.float32)
    out[f'divcommon_{g}'] = np.full((nlat, nlon), np.nan, np.float32)
    out[f'depcommon_{g}'] = np.full((nlat, nlon), np.nan, np.float32)

# Forced-response (multi-model-mean) versions of the main outputs
MMM_KEYS = (['departure', 'departure_sigma', 'divergence', 'comove', 'futdist',
             'divcommon', 'depcommon']
            + [f'share_{vk}_div' for vk in VKEYS] + [f'divabs_{vk}' for vk in VKEYS])
for g in GWLS:
    out[f'retention_mmm_{g}'] = np.full((nlat, nlon, nDt), np.nan, np.float32)
    for k in MMM_KEYS:
        out[f'{k}_mmm_{g}'] = np.full((nlat, nlon), np.nan, np.float32)

# Global ICV ruler: land median of each variable's ICV SD over analyzed biomes.
_land = np.isin(biome_grid, list(biomes.keys()))
COMMON_SD = np.array([np.nanmedian(icv_sd[vk][_land]) for vk in VKEYS])
# Physical units for divabs: identity for linear variables; 100 x log-ratio
# (~ percent change difference) for log1p variables.
ABS_SCALE = np.array([100.0 if cfg.VARIABLES[vk]['log'] else 1.0 for vk in VKEYS])
ABS_UNITS = {vk: ('% (100 x log1p difference)' if cfg.VARIABLES[vk]['log'] else 'native units')
             for vk in VKEYS}
print('Common ICV ruler: ' + ', '.join(f'{vk}={v:.4g}' for vk, v in zip(VKEYS, COMMON_SD)))


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
        nm1 = n_models + 1                         # + multi-model mean at index n_models
        ret_curve_m = np.full((nm1, n, nDt), np.nan, np.float32)
        msig_m = np.full((nm1, n), np.nan, np.float32)
        dep_m = np.full((nm1, n), np.nan, np.float32)
        deps_m = np.full((nm1, n), np.nan, np.float32)
        div_m = np.full((nm1, n), np.nan, np.float32)
        com_m = np.full((nm1, n), np.nan, np.float32)
        fdist_m = np.full((nm1, n), np.nan, np.float32)
        share_m = np.full((nm1, n, K), np.nan, np.float32)
        dabs_m = np.full((nm1, n, K), np.nan, np.float32)
        divc_m = np.full((nm1, n), np.nan, np.float32)
        depc_m = np.full((nm1, n), np.nan, np.float32)

        for mi in range(nm1):
            # per-analog future sigma + attribution, then aggregate over analogs
            sig_a = np.full((n, N_SET), np.nan)     # future sigma per analog
            # focal change whitened in its own ICV PC space -> local departure
            wf, keepf = pc_whiten(delta_f[mi], sd, evals, evecs)        # (n,K)
            wf2 = np.sum(wf**2, axis=1)
            dep = np.sqrt(wf2)
            dep_m[mi] = dep
            deps_m[mi] = sigma_by_dims(dep, keepf.sum(axis=1))
            div_a = np.full((n, N_SET), np.nan)      # |v| per analog
            com_a = np.full((n, N_SET), np.nan)      # co-movement per analog
            fd_a = np.full((n, N_SET), np.nan)       # |h+v| per analog
            sh_a = np.full((n, N_SET, K), np.nan)    # per-variable share of |v|^2
            ab_a = np.full((n, N_SET, K), np.nan)    # |change_a - change_f| physical
            dc_a = np.full((n, N_SET), np.nan)       # divergence, common ruler
            depc_m[mi] = np.sqrt(np.sum((delta_f[mi] / COMMON_SD)**2, axis=1))
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
                # departure / divergence decomposition (whitened, focal space)
                wa, _ = pc_whiten(fa - ha, sd[ok], evals[ok], evecs[ok])    # analog change
                wh, _ = pc_whiten(ha - mu[ok], sd[ok], evals[ok], evecs[ok])  # historical diff
                v = wa - wf[ok]
                v2 = np.sum(v**2, axis=1)
                div_a[ok, r] = np.sqrt(v2)
                fd_a[ok, r] = np.sqrt(np.sum((wh + v)**2, axis=1))
                with np.errstate(invalid='ignore', divide='ignore'):
                    com_a[ok, r] = 1.0 - v2 / (np.sum(wa**2, axis=1) + wf2[ok])
                # attribution and ICV-free views of the raw change mismatch
                dv = (fa - ha) - delta_f[mi][ok]                             # (m,K) raw
                c, d2 = mahal_var_shares(dv, sd[ok], evals[ok], evecs[ok])
                with np.errstate(invalid='ignore', divide='ignore'):
                    sh_a[ok, r] = c / d2[:, None]
                ab_a[ok, r] = np.abs(dv) * ABS_SCALE
                dc_a[ok, r] = np.sqrt(np.sum((dv / COMMON_SD)**2, axis=1))

            # retention curve: fraction of found analogs with sigma < Dt
            for di, Dt in enumerate(DT_GRID):
                below = (sig_a < Dt)
                # count only over found analogs
                cnt = np.nansum(np.where(np.isfinite(sig_a), below, 0), axis=1)
                ret_curve_m[mi, :, di] = cnt / np.maximum(nfound, 1)
            msig_m[mi] = np.nanmean(sig_a, axis=1)
            with warnings.catch_warnings():      # cells with no finite analog -> NaN
                warnings.simplefilter('ignore', RuntimeWarning)
                div_m[mi] = np.nanmedian(div_a, axis=1)
                com_m[mi] = np.nanmedian(com_a, axis=1)
                fdist_m[mi] = np.nanmedian(fd_a, axis=1)
                share_m[mi] = np.nanmedian(sh_a, axis=1)
                dabs_m[mi] = np.nanmedian(ab_a, axis=1)
                divc_m[mi] = np.nanmedian(dc_a, axis=1)

        M_ = n_models                               # per-model members: [:M_]; mean: [M_]
        out[f'retention_{g}'][vlat, vlon, :]     = np.nanmedian(ret_curve_m[:M_], axis=0)
        out[f'retention_{g}_p10'][vlat, vlon, :] = np.nanpercentile(ret_curve_m[:M_], 10, axis=0)
        out[f'retention_{g}_p90'][vlat, vlon, :] = np.nanpercentile(ret_curve_m[:M_], 90, axis=0)
        out[f'meansigma_{g}'][vlat, vlon] = np.nanmedian(msig_m[:M_], axis=0)
        out[f'retention_mmm_{g}'][vlat, vlon, :] = ret_curve_m[M_]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            for k, arr in [('departure', dep_m), ('departure_sigma', deps_m),
                           ('divergence', div_m), ('comove', com_m), ('futdist', fdist_m),
                           ('divcommon', divc_m), ('depcommon', depc_m)]:
                out[f'{k}_{g}'][vlat, vlon] = np.nanmedian(arr[:M_], axis=0)
                out[f'{k}_mmm_{g}'][vlat, vlon] = arr[M_]
            for kk, vk in enumerate(VKEYS):
                out[f'share_{vk}_div_{g}'][vlat, vlon] = np.nanmedian(share_m[:M_, :, kk], axis=0)
                out[f'divabs_{vk}_{g}'][vlat, vlon] = np.nanmedian(dabs_m[:M_, :, kk], axis=0)
                out[f'share_{vk}_div_mmm_{g}'][vlat, vlon] = share_m[M_, :, kk]
                out[f'divabs_{vk}_mmm_{g}'][vlat, vlon] = dabs_m[M_, :, kk]

        # quick read at the DT_SLICES
        slc = []
        for Dt in DT_SLICES:
            di = int(np.argmin(np.abs(DT_GRID - Dt)))
            slc.append(f'Dt={Dt}:{np.nanmean(out[f"retention_{g}"][vlat,vlon,di]):.2f}')
        med = {k: np.nanmedian(out[f'{k}_{g}'][vlat, vlon]) for k in ['departure', 'divergence', 'comove']}
        print(f'  {g}: mean retention {" ".join(slc)} | departure {med["departure"]:.2f} '
              f'divergence {med["divergence"]:.2f} comove {med["comove"]:.2f}')
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
    'Departure/divergence (whitened focal ICV PC space, Mahalanobis units): '
    'departure=|change_f| (focal future vs own past); divergence=|change_a - change_f|; '
    'comove=1-divergence^2/(|change_a|^2+|change_f|^2) (1 identical change, 0 unrelated, '
    '<0 opposite); futdist=|future_a - future_f|. '
    'Attribution: share_<var>_div = exact per-variable share of divergence^2 '
    '(z_k (S+ z)_k / D^2). ICV-free: divabs_<var> = |change_a - change_f| in physical '
    'units; divcommon/depcommon = Euclidean divergence/departure standardized by one '
    'global land-median ICV per variable (no per-cell ICV). '
    'Per model: median over analogs; then ensemble median.')
for kk, vk in enumerate(VKEYS):
    ds_out.attrs[f'common_icv_sd_{vk}'] = float(COMMON_SD[kk])
    ds_out.attrs[f'divabs_units_{vk}'] = ABS_UNITS[vk]
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
ds_out.attrs['mmm_definition'] = (
    '*_mmm_*: same metrics from the multi-model-mean future (forced response; '
    'sampling noise reduced ~sqrt(n_models); model pattern differences averaged out, '
    'so mmm divergence is a lower bound on single-model divergence).')
for g in GWLS:
    ds_out.attrs[f'noise_floor_div_{g}'] = NOISE_FLOOR[g][0]
    ds_out.attrs[f'noise_floor_div_mmm_{g}'] = NOISE_FLOOR[g][1]
ds_out.attrs['noise_floor_definition'] = (
    'Median |divergence| expected from white interannual noise alone (ERA5 ICV at '
    'both cells, actual GWL windows and 1991-2020 baseline): per model (median over '
    'models) and for the multi-model mean.')
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
    hdr += f' {VKEYS[0][:3] + "sh" + g[3:]:>8}'
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
        row += f' {np.nanmedian(out[f"share_{VKEYS[0]}_div_{g}"][bm]):>8.2f}'
    print(row)

warnings.simplefilter('ignore', RuntimeWarning)   # comove is NaN at gwl0 (no change)
print('\n--- Per-biome departure vs divergence (cell medians) ---')
print(f'{"biome":<46}' + ''.join(f' {"dep" + g[3:]:>7} {"div" + g[3:]:>7} {"com" + g[3:]:>7}' for g in GWLS))
for b, bname in biomes.items():
    bm = (biome_grid == b) & (N_FOUND > 0)
    if bm.sum() == 0 or not np.isfinite(out[f'departure_{GWLS[0]}'][bm]).any():
        continue
    row = f'{bname:<46}'
    for g in GWLS:
        for k in ['departure', 'divergence', 'comove']:
            row += f' {np.nanmedian(out[f"{k}_{g}"][bm]):>7.2f}'
    print(row)


# ICV-denominator test (issue B): if the latitude pattern in divergence is
# mostly the per-cell ICV ruler, it weakens in divcommon / divabs.
gl = GWLS[-1]
print(f'\n--- Per-biome divergence: per-cell ICV vs one global ruler vs physical units ({gl}) ---')
print(f'{"biome":<46} {"div":>7} {"divcom":>7}' + ''.join(f' {"abs_" + vk[:5]:>10}' for vk in VKEYS)
      + ''.join(f' {"sh_" + vk[:5]:>9}' for vk in VKEYS))
for b, bname in biomes.items():
    bm = (biome_grid == b) & (N_FOUND > 0)
    if bm.sum() == 0 or not np.isfinite(out[f'divergence_{gl}'][bm]).any():
        continue
    row = f'{bname:<46} {np.nanmedian(out[f"divergence_{gl}"][bm]):>7.2f} {np.nanmedian(out[f"divcommon_{gl}"][bm]):>7.2f}'
    row += ''.join(f' {np.nanmedian(out[f"divabs_{vk}_{gl}"][bm]):>10.2f}' for vk in VKEYS)
    row += ''.join(f' {np.nanmedian(out[f"share_{vk}_div_{gl}"][bm]):>9.2f}' for vk in VKEYS)
    print(row)
print('  divabs units: ' + '; '.join(f'{vk}: {ABS_UNITS[vk]}' for vk in VKEYS))

print('\n--- Noise floor: per-model vs multi-model-mean (forced) divergence, biome medians ---')
print('    floor = median |divergence| from interannual noise alone')
print(f'{"":<46}' + ''.join(f' {"div" + g[3:]:>8} {"mmm" + g[3:]:>8}' for g in GWLS))
print(f'{"noise floor":<46}' + ''.join(f' {NOISE_FLOOR[g][0]:>8.2f} {NOISE_FLOOR[g][1]:>8.2f}' for g in GWLS))
for b, bname in biomes.items():
    bm = (biome_grid == b) & (N_FOUND > 0)
    if bm.sum() == 0 or not np.isfinite(out[f'divergence_{GWLS[-1]}'][bm]).any():
        continue
    print(f'{bname:<46}' + ''.join(f' {np.nanmedian(out[f"divergence_{g}"][bm]):>8.2f}'
                                   f' {np.nanmedian(out[f"divergence_mmm_{g}"][bm]):>8.2f}' for g in GWLS))
di1 = int(np.argmin(np.abs(DT_GRID - 1.0)))
print('\n--- Retention at Dt=1: per-model median vs multi-model mean, biome means ---')
print(f'{"biome":<46}' + ''.join(f' {"ret" + g[3:]:>8} {"mmm" + g[3:]:>8}' for g in GWLS))
for b, bname in biomes.items():
    bm = (biome_grid == b) & (N_FOUND > 0)
    if bm.sum() == 0 or not np.isfinite(out[f'retention_{GWLS[-1]}'][bm, di1]).any():
        continue
    print(f'{bname:<46}' + ''.join(f' {np.nanmean(out[f"retention_{g}"][bm, di1]):>8.2f}'
                                   f' {np.nanmean(out[f"retention_mmm_{g}"][bm, di1]):>8.2f}' for g in GWLS))

print('\nDone.')
