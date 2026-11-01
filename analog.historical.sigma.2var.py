#!/bin/python
# author: jsallen@ucar.edu and Claude
#
# Historical climate analogs via sigma dissimilarity (Mahony et al. 2017).
# (TXx, Rx5day)
#
# NOTE: sigma values are only comparable across products built on the SAME
# variable set. Do not compare a K=2 sigma to a K=3 sigma.
#

# NOTE: Definitions: 
# ------------
#  FOCAL cell : a single gridpoint of intereste
#  ICV : See mahony et al. 2017, this is interannual climate variability 
#        simply, a standard deviation of an annual metric in this case

# NOTE: Process: 
# --------
# For each FOCAL cell j: build its local ICV correlation matrix from the
# 1991-2020 annual timeseries, rotate candidate climatological-mean vectors
# into j's ICV principal components, whiten, drop PCs with eigenvalue <
# pc_var_min, take standardized Euclidean (= Mahalanobis) distance, find the
# minimum-distance within-biome analog, convert to sigma via chi(df=M).
# Candidates scored against FOCAL cell's ICV (Mahony asymmetric Dji).
#
# Outputs (focal-cell grid): sigma_best, dmaha_best, analog_lat/lon, m_dims,
# biome_id. Sidecar npz: per-biome pairwise distance distribution percentiles
# for Dt calibration.

# NOTE: Limitations and subjective choices: 
# -----------------------------------
# This analysis has been tailored to our specfic research questions: 
#     - will a climate analog today remain one in the future? – 
#     - what regions globally have stable analogs and vice versa? -

# 1. There's a deliberate distance exclusion built in. For our questions,
#    we don't want to consider an analog if it's right next door to the
#    focal cell, because then it should "probably" experience the same
#    climate change signal and that's not really the point...
#
# 2. ICV comes from ERA5, not from the CMIP6 models. Across-model spread in
#    Rx5day ICV is large, so a per-model ICV would leak model scatter into
#    the metric. (The CMIP6 data are CCKP 0.25 deg downscaled, bias
#    adjusted to ERA5; they are NOT NASA NEX-GDDP.)

import numpy as np
import xarray as xr
import os
import time

import analog_config as cfg
from analog_common import (load_biomes, load_era5_icv, haversine_km,
                           sigma_from_distance)

# Settings live in analog_config.py; aliased here so the body reads as before.
out_dir = cfg.OUT_DIR
clim_start, clim_end = cfg.CLIM_START, cfg.CLIM_END
pc_var_min = cfg.PC_VAR_MIN          # drop PCs that explain too little variance
min_biome_cells = cfg.MIN_BIOME_CELLS  # was 5000 here (500 elsewhere); now unified
focal_chunk = 1000  # batch size for vector distance calc (smaller == less memory)
sigma_cap = cfg.SIGMA_CAP
BUFFER_KM = cfg.BUFFER_KM            # trivial-neighbor exclusion, shared with set selection
VKEYS, K = cfg.VKEYS, cfg.K
biomes = cfg.BIOMES
dist_pctiles = [5, 10, 15, 20, 25, 50, 75, 90, 95]

# ==============================
# Load ERA5 ICV + biome mask
# ==============================
print("Loading biome mask and ERA5 annual timeseries...")
biome_grid, bmin, bmax = load_biomes()
clim_mean, icv_sd, icv_series, icv_corr, lat_arr, lon_arr = load_era5_icv(bmin, bmax)
nlat, nlon = len(lat_arr), len(lon_arr)

# ==============================
# Output arrays
# ==============================
sigma_best = np.full((nlat, nlon), np.nan, np.float32)
dmaha_best = np.full((nlat, nlon), np.nan, np.float32)
analog_lat = np.full((nlat, nlon), np.nan, np.float32)
analog_lon = np.full((nlat, nlon), np.nan, np.float32)
m_dims_map = np.full((nlat, nlon), np.nan, np.float32)
biome_dist_summary = {}


for b, bname in biomes.items():
    t0 = time.time()
    bmask = (biome_grid == b)
    cell_lats, cell_lons = np.where(bmask)
    if len(cell_lats) < min_biome_cells:
        print(f'\nBiome {b:2d} ({bname}): {len(cell_lats)} cells, SKIPPING')
        continue

    mu = np.column_stack([clim_mean[vk][cell_lats, cell_lons] for vk in VKEYS])
    sd = np.column_stack([icv_sd[vk][cell_lats, cell_lons] for vk in VKEYS])
    valid = np.all(np.isfinite(mu), axis=1) & np.all(np.isfinite(sd), axis=1) \
            & np.all(sd > 0, axis=1)
    for (i, j), grid in icv_corr.items():
        valid &= np.isfinite(grid[cell_lats, cell_lons])
    if valid.sum() < min_biome_cells:
        print(f'\nBiome {b:2d} ({bname}): {valid.sum()} valid, SKIPPING')
        continue

    vlat, vlon = cell_lats[valid], cell_lons[valid]
    mu, sd = mu[valid], sd[valid]
    n = len(vlat)
    print(f'\nBiome {b:2d} ({bname}): {n} valid cells')

    A = mu

    R = np.zeros((n, K, K))
    for kk in range(K):
        R[:, kk, kk] = 1.0
    for (i, j), grid in icv_corr.items():
        rij = grid[vlat, vlon]
        R[:, i, j] = rij; R[:, j, i] = rij
    evals, evecs = np.linalg.eigh(R) # here be the PCs!!!

    best_sigma = np.full(n, np.nan)
    best_dmaha = np.full(n, np.nan)
    best_idx   = np.full(n, -1, dtype=np.int64)
    m_dims     = np.full(n, np.nan)
    dist_sample = []

    # Geographic coordinates of every valid cell in this biome, for the buffer.
    cell_lat = lat_arr[vlat]
    cell_lon = lon_arr[vlon]

    for cs in range(0, n, focal_chunk):
        ce = min(cs + focal_chunk, n)
        c = ce - cs
        d = np.empty((c, n))
        m_chunk = np.empty(c, dtype=int)
        for r in range(c): # oh lawd
            fj = cs + r
            diff = (A - mu[fj][None, :]) / sd[fj][None, :]
            proj = diff @ evecs[fj] # @ is same as np.matmul in py
            keep = evals[fj] >= pc_var_min
            with np.errstate(invalid='ignore', divide='ignore'):
                contrib = np.where(keep[None, :], proj**2 / evals[fj][None, :], 0.0)
            d[r] = np.sqrt(np.nansum(contrib, axis=1))
            m_chunk[r] = int(keep.sum())

            # NEAR-NEIGHBOR EXCLUSION. This town aint big enough for the both of us.
            # Without this, the nearest same-biome cell is a few grid points away, 
            # its climatology is essentially
            # identical, D_maha ~ 0, and sigma collapses to 0 everywhere. That
            # is not an analog we really care about. Mask everything
            # inside BUFFER_KM (same 250 km decorrelation result used by
            # analog_set_selection.py, so the two products are consistent).
            dkm = haversine_km(lat_arr[vlat[fj]], lon_arr[vlon[fj]],
                               cell_lat, cell_lon)
            d[r, dkm <= BUFFER_KM] = np.inf

        # A focal cell in a small/fragmented biome may have NO same-biome
        # candidate outside the buffer -> whole row is inf. Record as NaN
        # rather than silently taking an inf.
        jmin = np.argmin(d, axis=1)
        dmin = d[np.arange(c), jmin]
        no_cand = ~np.isfinite(dmin)
        if no_cand.any():
            jmin[no_cand] = -1
            dmin[no_cand] = np.nan

        s_chunk = np.full(c, np.nan)
        for mv in np.unique(m_chunk):
            if mv < 1:
                continue
            sel = (m_chunk == mv) & np.isfinite(dmin)
            if sel.any():
                s_chunk[sel] = sigma_from_distance(dmin[sel], int(mv))

        best_idx[cs:ce]   = jmin
        best_dmaha[cs:ce] = dmin
        best_sigma[cs:ce] = s_chunk
        m_dims[cs:ce]     = m_chunk

        if cs == 0:
            finite = d[np.isfinite(d)]
            if finite.size > 5_000_000:
                finite = np.random.default_rng(0).choice(finite, 5_000_000, replace=False)
            dist_sample.append(finite)

    sigma_best[vlat, vlon] = best_sigma.astype(np.float32)
    dmaha_best[vlat, vlon] = best_dmaha.astype(np.float32)
    m_dims_map[vlat, vlon] = m_dims.astype(np.float32)
    vb = best_idx >= 0
    analog_lat[vlat[vb], vlon[vb]] = lat_arr[vlat[best_idx[vb]]].astype(np.float32)
    analog_lon[vlat[vb], vlon[vb]] = lon_arr[vlon[best_idx[vb]]].astype(np.float32)

    if dist_sample:
        ds_all = np.concatenate(dist_sample)
        m_med = int(np.nanmedian(m_dims[m_dims >= 1])) if np.any(m_dims >= 1) else 1
        sig_all = sigma_from_distance(ds_all, max(m_med, 1))
        biome_dist_summary[b] = {
            'dmaha_pctiles': np.percentile(ds_all, dist_pctiles).astype(np.float32),
            'sigma_pctiles': np.percentile(sig_all, dist_pctiles).astype(np.float32),
            'pctile_levels': np.array(dist_pctiles, dtype=np.float32),
        }

    print(f'  best-analog sigma: median={np.nanmedian(best_sigma):.3f} '
          f'p90={np.nanpercentile(best_sigma, 90):.3f} '
          f'| retained dims median={np.nanmedian(m_dims):.1f} '
          f'| {time.time()-t0:.0f}s')

# alright alright alright - let's save some data to loook at
# -----------------------------------------------------------
os.makedirs(out_dir, exist_ok=True)
print('\nSaving netCDF...')
ds_out = xr.Dataset(coords={'lat': lat_arr, 'lon': lon_arr})
ds_out['sigma_best'] = (['lat', 'lon'], sigma_best)
ds_out['dmaha_best'] = (['lat', 'lon'], dmaha_best)
ds_out['analog_lat'] = (['lat', 'lon'], analog_lat)
ds_out['analog_lon'] = (['lat', 'lon'], analog_lon)
ds_out['m_dims']     = (['lat', 'lon'], m_dims_map)
ds_out['biome_id']   = (['lat', 'lon'], biome_grid.astype(np.float32))
ds_out.attrs['method'] = ('Sigma dissimilarity, Mahony et al. 2017. Per-focal-cell ICV '
    'correlation, batched PC rotation, var<0.01 truncation, chi sigma. '
    'Candidates scored against focal-cell ICV (asymmetric Dji).')
ds_out.attrs['variables'] = ', '.join(VKEYS)
ds_out.attrs['variable_set'] = cfg.VARSET
ds_out.attrs['icv_period'] = f'{clim_start}-{clim_end}'
ds_out.attrs['buffer_km'] = BUFFER_KM
ds_out.attrs['sigma_convention'] = ('half-normal: sigma = norm.ppf(0.5 + chi.cdf(D,df=m)/2); 0 = identical, monotone increasing, never negative.')
ds_out.attrs['icv_definition'] = cfg.ICV_DEFINITION
ds_out.attrs['note'] = 'Historical only. No Dt threshold (deferred to stability step).'
outfile = cfg.HIST_SIGMA_FILE
ds_out.to_netcdf(outfile)
print(f'Saved: {outfile}')

np.savez(os.path.join(out_dir, 'analog_historical_dist_summary.npz'),
         **{f'biome_{b}_{k}': v for b, d in biome_dist_summary.items()
            for k, v in d.items() if isinstance(v, np.ndarray)})
print(f"Saved distance summaries: {os.path.join(out_dir, 'analog_historical_dist_summary.npz')}")

print('\n--- Per-biome best-analog sigma (historical) ---')
for b, bname in biomes.items():
    bm = (biome_grid == b) & np.isfinite(sigma_best)
    if bm.sum() == 0:
        continue
    v = sigma_best[bm]
    print(f'  {bname:<55s}: median={np.median(v):.3f} '
          f'p90={np.percentile(v, 90):.3f} n={bm.sum()}')
print('\nDone.')
