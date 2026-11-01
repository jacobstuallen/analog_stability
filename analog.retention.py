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
                           area_weights, wmean, wmedian,
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
W = area_weights(lat_arr, nlon)      # cos(lat) weights for every biome statistic

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
with warnings.catch_warnings():                 # all-NaN cells -> NaN mean
    warnings.simplefilter('ignore', RuntimeWarning)
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

# Model agreement (step 5). Per pair: percent of models retaining it within
# Dt (uint8, 255 = no analog / no data). Per cell: fractions of the analog set
# that are robustly retained / uncertain / robustly lost.
def _dt_tag(Dt):
    return f'dt{Dt:.1f}'.replace('.', 'p')


AGREE = cfg.ROBUST_AGREE
pair_out = {}
# Share of the model spread in future pair distance that internal variability
# alone would produce (Hawkins & Sutton-style split of the 'uncertain' tier):
# fnoise = noise_var / var_across_models(|future_a - future_f|), capped at 1.
# noise_var per whitened dim from window_noise_cov (two cells, each model's
# window and the baseline), mean over models; scaled per pair by the squared
# model/ERA5 ICV ratio if model_icv_ratio.nc exists.
for g in GWLS:
    out[f'fnoise_{g}'] = np.full((nlat, nlon), np.nan, np.float32)
NOISE_VAR_DIST = {g: float(np.mean([2.0 * window_noise_cov([cfg.GWL_WINDOWS[g][m]],
                                                           (cfg.CLIM_START, cfg.CLIM_END))[0, 0]
                                    for m in models_used[g]])) for g in GWLS}
ICV_R2 = None
if os.path.exists(cfg.MODEL_ICV_FILE):
    _r = xr.open_dataset(cfg.MODEL_ICV_FILE).sortby('lat')
    ICV_R2 = np.nanmean(np.stack([_r[f'ratio_{vk}'].values**2 for vk in VKEYS]), axis=0)
    ICV_R2 = np.where(np.isfinite(ICV_R2), ICV_R2, 1.0)
    print(f'Noise fraction scaled by model/ERA5 ICV ratio^2 from {cfg.MODEL_ICV_FILE}')
for g in GWLS:
    for Dt in DT_SLICES:
        t = _dt_tag(Dt)
        pair_out[f'pairconf_{g}_{t}'] = np.full((nlat, nlon, N_SET), 255, np.uint8)
        for k in ['robret', 'uncert', 'roblost']:
            out[f'{k}_{g}_{t}'] = np.full((nlat, nlon), np.nan, np.float32)

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

        # per-pair model counts for the agreement tiers (per-model members only)
        ret_cnt = np.zeros((len(DT_SLICES), n, N_SET))
        fd_sum = np.zeros((n, N_SET)); fd_sq = np.zeros((n, N_SET)); fd_n = np.zeros((n, N_SET))
        fin_cnt = np.zeros((n, N_SET))

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
                # denominator: pairs with a finite result (missing data is
                # neither retained nor lost); no finite pair -> NaN
                nfin = np.sum(np.isfinite(sig_a), axis=1)
                with np.errstate(invalid='ignore', divide='ignore'):
                    ret_curve_m[mi, :, di] = np.where(nfin > 0, cnt / np.maximum(nfin, 1), np.nan)
            msig_m[mi] = np.nanmean(sig_a, axis=1)
            if mi < n_models:
                fdf = np.isfinite(fd_a)
                fd_sum += np.where(fdf, fd_a, 0.0); fd_sq += np.where(fdf, fd_a**2, 0.0); fd_n += fdf
                fin = np.isfinite(sig_a)
                fin_cnt += fin
                for si, Dt in enumerate(DT_SLICES):
                    ret_cnt[si] += fin & (sig_a < Dt)
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
        # model agreement: pairs with no finite model result are excluded
        # (not counted as lost)
        # noise fraction of the across-model spread in future pair distance
        with np.errstate(invalid='ignore', divide='ignore'):
            fd_var = (fd_sq - fd_sum**2 / np.maximum(fd_n, 1)) / np.maximum(fd_n - 1, 1)
            nv = NOISE_VAR_DIST[g] * np.ones((n, N_SET))
            if ICV_R2 is not None:
                r2a = np.where(ali >= 0, ICV_R2[np.maximum(ali, 0), np.maximum(alj, 0)], np.nan)
                nv = nv * 0.5 * (ICV_R2[vlat, vlon][:, None] + r2a)
            fn = np.where(fd_n >= 3, np.minimum(nv / fd_var, 1.0), np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            out[f'fnoise_{g}'][vlat, vlon] = np.nanmedian(fn, axis=1)
        has_pair = fin_cnt > 0
        npair = has_pair.sum(axis=1)
        for si, Dt in enumerate(DT_SLICES):
            t = _dt_tag(Dt)
            with np.errstate(invalid='ignore', divide='ignore'):
                conf = np.where(has_pair, ret_cnt[si] / np.maximum(fin_cnt, 1), np.nan)
            pc = np.full(conf.shape, 255, np.uint8)
            pc[has_pair] = np.round(100 * conf[has_pair]).astype(np.uint8)
            pair_out[f'pairconf_{g}_{t}'][vlat, vlon, :] = pc
            rob_r = np.sum(has_pair & (conf >= AGREE - 1e-9), axis=1)
            rob_l = np.sum(has_pair & (conf <= 1 - AGREE + 1e-9), axis=1)
            with np.errstate(invalid='ignore', divide='ignore'):
                den = np.where(npair > 0, npair, np.nan)
                out[f'robret_{g}_{t}'][vlat, vlon] = rob_r / den
                out[f'roblost_{g}_{t}'][vlat, vlon] = rob_l / den
                out[f'uncert_{g}_{t}'][vlat, vlon] = (npair - rob_r - rob_l) / den
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
            slc.append(f'Dt={Dt}:{wmean(out[f"retention_{g}"][vlat,vlon,di], W[vlat, vlon]):.2f}')
        med = {k: wmedian(out[f'{k}_{g}'][vlat, vlon], W[vlat, vlon]) for k in ['departure', 'divergence', 'comove']}
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
ds_out = ds_out.assign_coords(rank=np.arange(N_SET))
for k, arr in pair_out.items():
    ds_out[k] = (['lat', 'lon', 'rank'], arr)
    ds_out[k].attrs.update({'units': 'percent of models', 'missing': 255,
                            'description': 'percent of models in which this analog pair stays within Dt'})
ds_out['biome_id'] = (['lat', 'lon'], biome_grid.astype(np.float32))
ds_out['n_analogs'] = (['lat', 'lon'], N_FOUND.astype(np.float32))
ds_out.attrs['fnoise_definition'] = (
    'fnoise_<gwl>: median over analogs of min(1, noise_var / var across models of the '
    'future pair distance); noise_var from white interannual noise (window_noise_cov, '
    'mean over models), x model/ERA5 ICV ratio^2 if available. ~1: model spread is '
    'internal variability; ~0: model disagreement.')
ds_out.attrs['retention_denominator'] = 'pairs with a finite result (missing data excluded, not counted as lost)'
ds_out.attrs['agreement_definition'] = (
    f'pairconf_<gwl>_<dt>: percent of models (per-model members, not the mean) in which '
    f'the pair is retained (sigma_fut < Dt); 255 = no analog. robret/uncert/roblost: '
    f'fraction of a cell\'s analogs retained by >= {AGREE:.0%} of models / in between / '
    f'retained by <= {1 - AGREE:.0%}; pairs without data excluded. Map agreement = 1 - uncert.')
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
ds_out.attrs['variable_set'] = cfg.VARSET
ds_out.attrs['analog_sets_file'] = cfg.SETS_FILE
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
enc = {k: {'zlib': True, 'complevel': 4} for k in ds_out.data_vars}
ds_out.to_netcdf(outfile, encoding=enc)
print(f'\nSaved: {outfile}')

# ==============================
# Per-biome summary at the Dt slices (all biome statistics cos(lat) area-weighted)
# ==============================
print('\n--- Per-biome ensemble-median retention, area-weighted mean ---')
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
            row += f' {wmean(out[f"retention_{g}"][bm, di], W[bm]):>7.2f}'
    for g in GWLS:
        row += f' {wmedian(out[f"share_{VKEYS[0]}_div_{g}"][bm], W[bm]):>8.2f}'
    print(row)

warnings.simplefilter('ignore', RuntimeWarning)   # comove is NaN at gwl0 (no change)
print('\n--- Per-biome departure vs divergence (area-weighted medians) ---')
print(f'{"biome":<46}' + ''.join(f' {"dep" + g[3:]:>7} {"div" + g[3:]:>7} {"com" + g[3:]:>7}' for g in GWLS))
for b, bname in biomes.items():
    bm = (biome_grid == b) & (N_FOUND > 0)
    if bm.sum() == 0 or not np.isfinite(out[f'departure_{GWLS[0]}'][bm]).any():
        continue
    row = f'{bname:<46}'
    for g in GWLS:
        for k in ['departure', 'divergence', 'comove']:
            row += f' {wmedian(out[f"{k}_{g}"][bm], W[bm]):>7.2f}'
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
    row = f'{bname:<46} {wmedian(out[f"divergence_{gl}"][bm], W[bm]):>7.2f} {wmedian(out[f"divcommon_{gl}"][bm], W[bm]):>7.2f}'
    row += ''.join(f' {wmedian(out[f"divabs_{vk}_{gl}"][bm], W[bm]):>10.2f}' for vk in VKEYS)
    row += ''.join(f' {wmedian(out[f"share_{vk}_div_{gl}"][bm], W[bm]):>9.2f}' for vk in VKEYS)
    print(row)
print('  divabs units: ' + '; '.join(f'{vk}: {ABS_UNITS[vk]}' for vk in VKEYS))

print('\n--- Noise floor: per-model vs multi-model-mean (forced) divergence, area-weighted medians ---')
print('    floor = median |divergence| from interannual noise alone')
print(f'{"":<46}' + ''.join(f' {"div" + g[3:]:>8} {"mmm" + g[3:]:>8}' for g in GWLS))
print(f'{"noise floor":<46}' + ''.join(f' {NOISE_FLOOR[g][0]:>8.2f} {NOISE_FLOOR[g][1]:>8.2f}' for g in GWLS))
for b, bname in biomes.items():
    bm = (biome_grid == b) & (N_FOUND > 0)
    if bm.sum() == 0 or not np.isfinite(out[f'divergence_{GWLS[-1]}'][bm]).any():
        continue
    print(f'{bname:<46}' + ''.join(f' {wmedian(out[f"divergence_{g}"][bm], W[bm]):>8.2f}'
                                   f' {wmedian(out[f"divergence_mmm_{g}"][bm], W[bm]):>8.2f}' for g in GWLS))
di1 = int(np.argmin(np.abs(DT_GRID - 1.0)))
print('\n--- Retention at Dt=1: per-model median vs multi-model mean, area-weighted biome means ---')
print(f'{"biome":<46}' + ''.join(f' {"ret" + g[3:]:>8} {"mmm" + g[3:]:>8}' for g in GWLS))
for b, bname in biomes.items():
    bm = (biome_grid == b) & (N_FOUND > 0)
    if bm.sum() == 0 or not np.isfinite(out[f'retention_{GWLS[-1]}'][bm, di1]).any():
        continue
    print(f'{bname:<46}' + ''.join(f' {wmean(out[f"retention_{g}"][bm, di1], W[bm]):>8.2f}'
                                   f' {wmean(out[f"retention_mmm_{g}"][bm, di1], W[bm]):>8.2f}' for g in GWLS))

for Dt in DT_SLICES:
    t = _dt_tag(Dt)
    print(f'\n--- Reliability tiers at Dt={Dt}: share of analogs robustly retained / uncertain / '
          f'robustly lost (>= {AGREE:.0%} model agreement), area-weighted biome means ---')
    print(f'{"biome":<46}' + ''.join(f' {g[3:] + " ret/unc/lost":>18}' for g in GWLS))
    for b, bname in biomes.items():
        bm = (biome_grid == b) & (N_FOUND > 0)
        if bm.sum() == 0 or not np.isfinite(out[f'robret_{GWLS[-1]}_{t}'][bm]).any():
            continue
        row = f'{bname:<46}'
        for g in GWLS:
            row += '      {:.2f}/{:.2f}/{:.2f}'.format(*[wmean(out[f'{k}_{g}_{t}'][bm], W[bm])
                                                       for k in ['robret', 'uncert', 'roblost']])
        print(row)

print('\n--- Noise fraction of model spread in future pair distance (area-weighted medians) ---')
print('    1 = spread across models is what internal variability alone gives; 0 = model disagreement')
print(f'{"biome":<46}' + ''.join(f' {"fn" + g[3:]:>7}' for g in GWLS if g != 'gwl0'))
for b, bname in biomes.items():
    bm = (biome_grid == b) & (N_FOUND > 0)
    if bm.sum() == 0 or not any(np.isfinite(out[f'fnoise_{g}'][bm]).any() for g in GWLS):
        continue
    print(f'{bname:<46}' + ''.join(f' {wmedian(out[f"fnoise_{g}"][bm], W[bm]):>7.2f}' for g in GWLS if g != 'gwl0'))

# Item Q: how close do the historical analogs sit to the sigma <= 1 gate?
SET_SIG = sets_ds['analog_sigma'].values          # (nlat,nlon,N)
with warnings.catch_warnings():
    warnings.simplefilter('ignore', RuntimeWarning)
    cell_med = np.nanmedian(SET_SIG, axis=2)
    cell_hi = np.nanmean(np.where(np.isfinite(SET_SIG), SET_SIG > 0.75, np.nan), axis=2)
print('\n--- Historical analog quality (analog_sets.nc), area-weighted ---')
print(f'{"biome":<46} {"med sigma_hist":>14} {"share > 0.75":>13}')
for b, bname in biomes.items():
    bm = (biome_grid == b) & (N_FOUND > 0)
    if bm.sum() == 0:
        continue
    print(f'{bname:<46} {wmedian(cell_med[bm], W[bm]):>14.2f} {wmean(cell_hi[bm], W[bm]):>13.2f}')

print('\nDone.')
