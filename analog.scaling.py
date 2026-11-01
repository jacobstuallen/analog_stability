#!/bin/python
#
# LINEAR SCALING TEST of analog-pair divergence with global warming level.
#
# For focal f, analog a, model m, warming level g (delta mode):
#   future pair difference = historical difference + v_m(g),
#   v_m(g) = change_a(g) - change_f(g), whitened in f's ICV PC space.
# Pattern scaling (e.g. Seneviratne et al. 2016) predicts v(g) ~ alpha + beta*g:
# pairs drift apart at a fixed rate per degree of global warming.
#
# Per pair we fit that line to the MULTI-MODEL MEAN v(g) at GWL1.5, 2, 3, 4
# (the same models at every level: ANALOG_MODEL_SET is forced to 'common').
#   div_rate : |beta|, pair divergence in whitened ICV units per deg C
#              (Mahalanobis distance units, not sigma percentiles)
#   nlr      : residual / its expected value under linear + noise (~1 if
#              linear; a MEAN-based ratio). The noise covariance across GWLs
#              is exact for white annual noise given each model's actual GWL
#              windows and the shared 1991-2020 baseline (window_noise_cov),
#              in ERA5 ICV units at both cells, averaged over models. If
#              model_icv_ratio.nc exists (diag.model.icv.py), each pair's
#              noise is scaled by the models' squared ICV ratio.
#   frac_nonlin : fraction of a cell's analog pairs with p < 0.05
#                 (0.05 expected if every pair scales linearly)
#   nl_frac  : SIZE of the nonlinearity: noise-corrected RMS residual per GWL,
#              sqrt(max(RSS - E[RSS], 0) / L), over the linear change across
#              the GWL range, |beta| * (gmax - gmin). 0 = linear.
#   vnorm_<gwl> : median over analogs of |mean v(g)|, divergence at each GWL
# Cell values are medians over the cell's analog set.
#
# Reads:  analog_sets.nc + ERA5 + CCKP CMIP6 (delta mode)
# Writes: analog_scaling.nc

import numpy as np
import xarray as xr
import os
import time
import warnings

import analog_config as cfg
from analog_common import (load_biomes, load_era5_icv, load_future_gwl,
                           grid_indices, focal_eig, pc_whiten, linear_fit_nlr,
                           window_noise_cov)

# ==============================
# Configuration
# ==============================
GWLS_SCALE = ['gwl1p5', 'gwl2', 'gwl3', 'gwl4']
G = np.array([cfg.GWL_KEYS[g] for g in GWLS_SCALE])
WINDOW_YEARS = 20           # GWL window length (all AR6 windows are 20 yr)
P_CRIT = 0.05
FOCAL_CHUNK = 20000         # focal cells per batch (memory)
outfile = cfg.SCALING_FILE
VKEYS, K = cfg.VKEYS, cfg.K
biomes = cfg.BIOMES

if cfg.FUTURE_MODE != 'delta':
    raise RuntimeError('scaling test requires FUTURE_MODE=delta')
if cfg.GWL_SOURCE != 'ar6':
    raise RuntimeError('scaling test needs GWL1.5-4 (ANALOG_GWL_SOURCE=ar6)')

# ==============================
# Load
# ==============================
print('Loading biome mask and analog sets...')
biome_grid, bmin, bmax = load_biomes()
sets_ds = xr.open_dataset(cfg.SETS_FILE).sortby('lat')
SET_LAT = sets_ds['analog_lat'].values     # (nlat,nlon,N)
SET_LON = sets_ds['analog_lon'].values
N_SET = SET_LAT.shape[2]
N_FOUND = sets_ds['n_analogs'].values

clim_mean, icv_sd, icv_series, icv_corr, lat_arr, lon_arr = load_era5_icv(bmin, bmax)
nlat, nlon = len(lat_arr), len(lon_arr)

future_ens, models_used = load_future_gwl(bmin, bmax, GWLS_SCALE, clim_mean=clim_mean,
                                          model_set='common')
models = models_used[GWLS_SCALE[0]]
M = len(models)
# Noise covariance across GWLs of one whitened dimension of the ensemble-mean
# pair divergence (two cells, M independent models, actual window overlaps).
noise_cov = sum(2.0 * window_noise_cov([cfg.GWL_WINDOWS[g][m] for g in GWLS_SCALE],
                                       (cfg.CLIM_START, cfg.CLIM_END))
                for m in models) / M**2
print(f'  {M} common models; noise variance by GWL (ensemble mean, per whitened dim): '
      + ', '.join(f'{x:.4f}' for x in np.diag(noise_cov)))

# Optional: model/ERA5 ICV ratio (median over models) per cell and variable.
icv_ratio2 = None
if os.path.exists(cfg.MODEL_ICV_FILE):
    _r = xr.open_dataset(cfg.MODEL_ICV_FILE).sortby('lat')
    icv_ratio2 = np.stack([_r[f'ratio_{vk}'].values**2 for vk in VKEYS], axis=0)  # (K,nlat,nlon)
    print(f'  scaling noise by model/ERA5 ICV ratio^2 from {cfg.MODEL_ICV_FILE}')
else:
    print('  no model ICV ratio file: noise assumes model ICV = ERA5 ICV')

# per-model change fields (future - ERA5 clim), (M, nlat, nlon) per gwl, var
change = {g: {vk: future_ens[g][vk] - clim_mean[vk][None] for vk in VKEYS}
          for g in GWLS_SCALE}
del future_ens

# ==============================
# Outputs
# ==============================
out = {k: np.full((nlat, nlon), np.nan, np.float32)
       for k in ['div_rate', 'nlr', 'frac_nonlin', 'nl_frac', 'n_pairs']}
for g in GWLS_SCALE:
    out[f'vnorm_{g}'] = np.full((nlat, nlon), np.nan, np.float32)
biome_rows = {}

# ==============================
# Biome by biome, in focal chunks
# ==============================
for b, bname in biomes.items():
    t0 = time.time()
    clat, clon = np.where(biome_grid == b)
    if len(clat) < cfg.MIN_BIOME_CELLS:
        print(f'\nBiome {b:2d} ({bname}): {len(clat)} cells, SKIPPING'); continue
    mu0 = np.column_stack([clim_mean[vk][clat, clon] for vk in VKEYS])
    sd0 = np.column_stack([icv_sd[vk][clat, clon] for vk in VKEYS])
    valid = (np.all(np.isfinite(mu0), axis=1) & np.all(np.isfinite(sd0), axis=1)
             & np.all(sd0 > 0, axis=1) & (N_FOUND[clat, clon] > 0))
    if valid.sum() < cfg.MIN_BIOME_CELLS:
        print(f'\nBiome {b:2d} ({bname}): {valid.sum()} valid, SKIPPING'); continue
    vlat_all, vlon_all = clat[valid], clon[valid]
    print(f'\nBiome {b:2d} ({bname}): {len(vlat_all)} valid cells')
    pair_rate, pair_nlr, pair_p, pair_nlf = [], [], [], []

    for cs in range(0, len(vlat_all), FOCAL_CHUNK):
        vlat = vlat_all[cs:cs + FOCAL_CHUNK]; vlon = vlon_all[cs:cs + FOCAL_CHUNK]
        n = len(vlat)
        _, sd, evals, evecs = focal_eig(vlat, vlon, clim_mean, icv_sd, icv_corr)

        # analog grid indices (n, N_SET); -1 = no analog
        ai = grid_indices(SET_LAT[vlat, vlon, :], lat_arr)
        aj = grid_indices(SET_LON[vlat, vlon, :], lon_arr)
        nf = N_FOUND[vlat, vlon]
        has = (ai >= 0) & (aj >= 0) & (np.arange(N_SET)[None, :] < nf[:, None])

        # flatten pairs: focal params repeated per analog slot
        P = n * N_SET
        fi = np.repeat(vlat, N_SET); fj = np.repeat(vlon, N_SET)
        sdp = np.repeat(sd, N_SET, axis=0)
        evp = np.repeat(evals, N_SET, axis=0)
        ecp = np.repeat(evecs, N_SET, axis=0)
        aip = np.where(has, ai, 0).ravel(); ajp = np.where(has, aj, 0).ravel()
        hasp = has.ravel()

        vsum = np.zeros((P, len(G), K))
        vcnt = np.zeros((P, len(G)))
        keep = None
        for li, g in enumerate(GWLS_SCALE):
            for mi in range(M):
                dfoc = np.column_stack([change[g][vk][mi, fi, fj] for vk in VKEYS])
                dana = np.column_stack([change[g][vk][mi, aip, ajp] for vk in VKEYS])
                w, keep = pc_whiten(dana - dfoc, sdp, evp, ecp)
                ok = hasp & np.all(np.isfinite(w), axis=1)
                vsum[ok, li] += w[ok]
                vcnt[ok, li] += 1
        full = hasp & np.all(vcnt == M, axis=1)        # every model, every level
        vbar = vsum / np.maximum(vcnt, 1)[:, :, None]
        kd = keep.sum(axis=1)
        full &= kd > 0

        rate = np.full(P, np.nan); nlr = np.full(P, np.nan); pv = np.full(P, np.nan)
        nlf = np.full(P, np.nan)
        nscale = None
        if icv_ratio2 is not None:
            # mean over variables and the two cells of the squared ICV ratio
            # (approximation: ignores the rotation into PCs)
            rf = icv_ratio2[:, fi, fj]; ra = icv_ratio2[:, aip, ajp]
            nscale = np.nanmean(np.concatenate([rf, ra], axis=0), axis=0)
            nscale = np.where(np.isfinite(nscale), nscale, 1.0)[full]
        if full.any():
            r_, n_, p_, _, _, rss_, erss_ = linear_fit_nlr(vbar[full], G, noise_cov, kd[full],
                                                           nscale, return_rss=True)
            rate[full], nlr[full], pv[full] = r_, n_, p_
            with np.errstate(invalid='ignore', divide='ignore'):
                nlf[full] = (np.sqrt(np.maximum(rss_ - erss_, 0.0) / len(G))
                             / (r_ * (G.max() - G.min())))
        vnorm = np.where(full[:, None], np.linalg.norm(vbar, axis=2), np.nan)   # (P, L)

        rate = rate.reshape(n, N_SET); nlr = nlr.reshape(n, N_SET)
        pv = pv.reshape(n, N_SET); vnorm = vnorm.reshape(n, N_SET, len(G))
        nlf = nlf.reshape(n, N_SET)
        npair = np.sum(np.isfinite(rate), axis=1)
        with warnings.catch_warnings():          # all-NaN rows -> NaN, quietly
            warnings.simplefilter('ignore', RuntimeWarning)
            out['div_rate'][vlat, vlon] = np.nanmedian(rate, axis=1)
            out['nlr'][vlat, vlon] = np.nanmedian(nlr, axis=1)
            out['nl_frac'][vlat, vlon] = np.nanmedian(nlf, axis=1)
            fnl = np.nansum(pv < P_CRIT, axis=1) / np.maximum(npair, 1)
            out['frac_nonlin'][vlat, vlon] = np.where(npair > 0, fnl, np.nan)
            for li, g in enumerate(GWLS_SCALE):
                out[f'vnorm_{g}'][vlat, vlon] = np.nanmedian(vnorm[:, :, li], axis=1)
        out['n_pairs'][vlat, vlon] = npair
        fin = np.isfinite(rate)
        pair_rate.append(rate[fin]); pair_nlr.append(nlr[fin]); pair_p.append(pv[fin])
        pair_nlf.append(nlf[fin])

    pr = np.concatenate(pair_rate); pn = np.concatenate(pair_nlr); pp = np.concatenate(pair_p)
    pf = np.concatenate(pair_nlf)
    if pr.size:
        biome_rows[b] = (np.median(pr), np.median(pn), np.mean(pp < P_CRIT), pr.size,
                         np.nanmedian(pf), np.nanpercentile(pf, 90))
        print(f'  pairs {pr.size}: median rate {np.median(pr):.3f}/degC, median nlr '
              f'{np.median(pn):.2f}, frac p<{P_CRIT} {np.mean(pp < P_CRIT):.3f} | {time.time()-t0:.0f}s')

# ==============================
# Save
# ==============================
os.makedirs(cfg.OUT_DIR, exist_ok=True)
ds_out = xr.Dataset(coords={'lat': lat_arr, 'lon': lon_arr})
for k, arr in out.items():
    ds_out[k] = (['lat', 'lon'], arr)
ds_out['biome_id'] = (['lat', 'lon'], biome_grid.astype(np.float32))
ds_out['div_rate'].attrs['units'] = 'whitened ICV units per deg C global warming'
ds_out.attrs['method'] = (
    'Per analog pair, divergence vector v(g) = change_a - change_f (delta mode), '
    'whitened in focal ICV PC space; multi-model mean fitted as alpha + beta*g over '
    f'GWLs {", ".join(GWLS_SCALE)}. div_rate=|beta|; nlr=RSS/E[RSS|linear+noise], noise '
    f'covariance exact for white annual noise given each model window and the 1991-2020 '
    f'baseline (ERA5 ICV units, two cells, M models); frac_nonlin = fraction of pairs with '
    f'Satterthwaite p<{P_CRIT}. Cell values: median over the analog set.')
ds_out.attrs['models'] = ', '.join(models)
ds_out.attrs['variable_set'] = cfg.VARSET
ds_out.attrs['n_models'] = M
ds_out.attrs['noise_var_by_gwl'] = ', '.join(f'{x:.5f}' for x in np.diag(noise_cov))
ds_out.attrs['icv_ratio_applied'] = int(icv_ratio2 is not None)
# Reference for the MEDIAN nlr under perfect linearity (nlr is mean-normalized;
# its median is below 1 because the scaled chi-square is right-skewed).
from scipy.stats import chi2 as _chi2
_X = np.column_stack([np.ones(len(G)), G])
_P = np.eye(len(G)) - _X @ np.linalg.solve(_X.T @ _X, _X.T)
_A = _P @ noise_cov @ _P
_dof = K * np.trace(_A)**2 / np.trace(_A @ _A)
ds_out.attrs['nlr_null_median'] = float(_chi2.median(_dof) / _dof)
ds_out.attrs['gwl_levels'] = ', '.join(f'{x}' for x in G)
ds_out.attrs['gwl_source'] = cfg.GWL_SOURCE
ds_out.attrs['future_mode'] = cfg.FUTURE_MODE
ds_out.attrs['scenario'] = cfg.SCENARIO
ds_out.attrs['icv_definition'] = cfg.ICV_DEFINITION
ds_out.to_netcdf(outfile)
print(f'\nSaved: {outfile}')

print(f'\n--- Per-biome linear scaling ({M} common models) ---')
print(f'    median nlr if perfectly linear (this noise model): {ds_out.attrs["nlr_null_median"]:.2f}; '
      f'p<0.05 fraction if linear: {P_CRIT}')
print(f'{"biome":<52} {"rate/degC":>9} {"nlr":>6} {"p<.05":>6} {"nlfrac":>7} {"nl p90":>7} {"pairs":>8}')
for b, (r_, n_, f_, c_, nf_, nf9_) in biome_rows.items():
    print(f'{biomes[b]:<52} {r_:>9.3f} {n_:>6.2f} {f_:>6.3f} {nf_:>7.2f} {nf9_:>7.2f} {c_:>8d}')
print('    nlfrac: noise-corrected nonlinear residual / linear change over GWL range (pair median, 90th pct)')
print('\nDone.')
