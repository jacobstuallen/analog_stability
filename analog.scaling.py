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
#   nlr      : residual / analytic noise floor. ~1 if linear. Noise assumes
#              each 20-yr model mean has ICV-sized white noise at both cells
#              (variance 1/20 per whitened dim per cell; model ICV ~ ERA5 ICV
#              after bias adjustment), divided by the number of models.
#              Overlapping GWL windows correlate the noise and lower the
#              residual, so the test is conservative.
#   frac_nonlin : fraction of a cell's analog pairs with p < 0.05
#                 (0.05 expected if every pair scales linearly)
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
                           grid_indices, focal_eig, pc_whiten, linear_fit_nlr)

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
noise_var = 2.0 / WINDOW_YEARS / M
print(f'  {M} common models; noise variance per whitened dim of the ensemble mean: {noise_var:.4f}')

# per-model change fields (future - ERA5 clim), (M, nlat, nlon) per gwl, var
change = {g: {vk: future_ens[g][vk] - clim_mean[vk][None] for vk in VKEYS}
          for g in GWLS_SCALE}
del future_ens

# ==============================
# Outputs
# ==============================
out = {k: np.full((nlat, nlon), np.nan, np.float32)
       for k in ['div_rate', 'nlr', 'frac_nonlin', 'n_pairs']}
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
    pair_rate, pair_nlr, pair_p = [], [], []

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
        if full.any():
            r_, n_, p_, _, _ = linear_fit_nlr(vbar[full], G, noise_var, kd[full])
            rate[full], nlr[full], pv[full] = r_, n_, p_
        vnorm = np.where(full[:, None], np.linalg.norm(vbar, axis=2), np.nan)   # (P, L)

        rate = rate.reshape(n, N_SET); nlr = nlr.reshape(n, N_SET)
        pv = pv.reshape(n, N_SET); vnorm = vnorm.reshape(n, N_SET, len(G))
        npair = np.sum(np.isfinite(rate), axis=1)
        with warnings.catch_warnings():          # all-NaN rows -> NaN, quietly
            warnings.simplefilter('ignore', RuntimeWarning)
            out['div_rate'][vlat, vlon] = np.nanmedian(rate, axis=1)
            out['nlr'][vlat, vlon] = np.nanmedian(nlr, axis=1)
            fnl = np.nansum(pv < P_CRIT, axis=1) / np.maximum(npair, 1)
            out['frac_nonlin'][vlat, vlon] = np.where(npair > 0, fnl, np.nan)
            for li, g in enumerate(GWLS_SCALE):
                out[f'vnorm_{g}'][vlat, vlon] = np.nanmedian(vnorm[:, :, li], axis=1)
        out['n_pairs'][vlat, vlon] = npair
        fin = np.isfinite(rate)
        pair_rate.append(rate[fin]); pair_nlr.append(nlr[fin]); pair_p.append(pv[fin])

    pr = np.concatenate(pair_rate); pn = np.concatenate(pair_nlr); pp = np.concatenate(pair_p)
    if pr.size:
        biome_rows[b] = (np.median(pr), np.median(pn), np.mean(pp < P_CRIT), pr.size)
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
    f'GWLs {", ".join(GWLS_SCALE)}. div_rate=|beta|; nlr=RSS/(noise_var*dof) with '
    f'noise_var=2/{WINDOW_YEARS}/M; frac_nonlin = fraction of pairs with chi-square '
    f'p<{P_CRIT}. Cell values: median over the analog set.')
ds_out.attrs['models'] = ', '.join(models)
ds_out.attrs['n_models'] = M
ds_out.attrs['noise_var'] = noise_var
# Reference for the MEDIAN nlr under perfect linearity with independent noise:
# median(chi2_dof)/dof, below 1 because chi-square is right-skewed. Window
# overlap pushes the real null lower still.
from scipy.stats import chi2 as _chi2
_dof = (len(G) - 2) * K
ds_out.attrs['nlr_null_median'] = float(_chi2.median(_dof) / _dof)
ds_out.attrs['gwl_levels'] = ', '.join(f'{x}' for x in G)
ds_out.attrs['gwl_source'] = cfg.GWL_SOURCE
ds_out.attrs['future_mode'] = cfg.FUTURE_MODE
ds_out.attrs['scenario'] = cfg.SCENARIO
ds_out.attrs['icv_definition'] = cfg.ICV_DEFINITION
ds_out.to_netcdf(outfile)
print(f'\nSaved: {outfile}')

print(f'\n--- Per-biome linear scaling ({M} common models) ---')
print(f'    median nlr if perfectly linear (independent noise): {ds_out.attrs["nlr_null_median"]:.2f}; '
      f'p<0.05 fraction if linear: {P_CRIT}')
print(f'{"biome":<52} {"rate/degC":>9} {"nlr":>6} {"p<.05":>6} {"pairs":>8}')
for b, (r_, n_, f_, c_) in biome_rows.items():
    print(f'{biomes[b]:<52} {r_:>9.3f} {n_:>6.2f} {f_:>6.3f} {c_:>8d}')
print('\nDone.')
