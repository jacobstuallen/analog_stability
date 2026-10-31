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
# 2. The NASA-NEX-CMIP6 downscaled climat emodel's ICV is bad... I (Jacob) 
#    have figures showing this. SO I use ERA5 based ICV instead. I don't 
#    have good quantitative info on the difference in using GCM ICV versus
#    ERA5 ICV, but the resulting analog maps looked mostly similar (good sign). 
#    Biggest changes came from precip. 

import numpy as np
import xarray as xr
import os
import time
from scipy.stats import chi, norm
from icv_utils import detrend_icv, build_icv_corr

# User variables to edit/play with
# --------------------------------
data_root = '/glade/campaign/ral/risc/jsallen/TNC'
biome_file = '/glade/u/home/jsallen/projects/tnc_2025/analogs/biomes/biomes.analog.gridded.nc'
out_dir = f'{data_root}/analog_stability'

clim_start = 1991
clim_end   = 2020
pc_var_min = 0.01 # remove pcs which don't explain enough variance (noise)
min_biome_cells = 5000 # require this many cells to werk on a biome
focal_chunk = 1000  # batch size for vector distance calc - smaller == less memory, maybe longer processing
sigma_cap = 8.0     # Subjective (?) and would change with different #s of vars
BUFFER_KM = 250.0   # trivial-neighbor exclusion. MUST match
                    # analog_set_selection.py or the historical sigma map
                    # and the analog sets are built on different rules.

# ---- Variable configuration (annual extremes) ----
# Rx5day is log1p-transformed before the ICV is computed, so its detrending
# and standardization both happen in log space 
# There's an option to add in a theoretically reasonable maximum value
VARIABLES = {
    'txx': {
        'era5_ts':  f'{data_root}/CCKP/ERA5/timeseries-txx-annual-mean_era5-x0.25_era5-x0.25-historical_timeseries_mean_1950-2024.nc',
        'era5_var': 'timeseries-txx-annual-mean',
        'log': False, 'qc_max': 70.0,
    },
    'rx5day': {
        'era5_ts':  f'{data_root}/CCKP/ERA5/timeseries-rx5day-annual-mean_era5-x0.25_era5-x0.25-historical_timeseries_mean_1950-2024.nc',
        'era5_var': 'timeseries-rx5day-annual-mean',
        'log': True, 'qc_max': 3000.0,
    },
}
VKEYS = list(VARIABLES.keys())
K = len(VKEYS)

biomes = {
    1:'Boreal Forests/Taiga', 2:'Deserts & Xeric Shrublands',
    3:'Flooded Grasslands & Savannas', 4:'Mangroves',
    5:'Mediterranean Forests, Woodlands & Scrub', 6:'Montane Grasslands & Shrublands',
    7:'Rock and Ice', 8:'Temperate Broadleaf & Mixed Forests',
    9:'Temperate Conifer Forests', 10:'Temperate Grasslands, Savannas & Shrublands',
    11:'Tropical & Subtropical Coniferous Forests', 12:'Tropical & Subtropical Dry Broadleaf Forests',
    13:'Tropical & Subtropical Grasslands, Savannas & Shrublands',
    14:'Tropical & Subtropical Moist Broadleaf Forests', 15:'Tundra',
}
dist_pctiles = [5, 10, 15, 20, 25, 50, 75, 90, 95]

# ==============================
# Load ERA5 ICV + biome mask
# ==============================
print(f"Variables: {VKEYS} (K={K})")
print("Loading biome mask and ERA5 annual timeseries...")
biome_ds = xr.open_dataset(biome_file).sortby('lat')
bmin = float(biome_ds.lat.min()); bmax = float(biome_ds.lat.max())
biome_grid = biome_ds['BIOME_ID'].values

clim_mean, icv_sd, icv_series = {}, {}, {}
lat_arr = lon_arr = None
T = None
for vk in VKEYS:
    cfg = VARIABLES[vk]
    ds = xr.open_dataset(cfg['era5_ts']).sortby('lat').sel(lat=slice(bmin, bmax))
    ts = ds[cfg['era5_var']].sel(time=slice(str(clim_start), str(clim_end)))
    if lat_arr is None:
        lat_arr = ts.lat.values; lon_arr = ts.lon.values
        nlat, nlon = len(lat_arr), len(lon_arr)
        T = ts.sizes['time']
        print(f"  Grid: {nlat} x {nlon}, ICV years: {T}")
    arr = ts.values.astype(np.float64)
    arr[arr > cfg['qc_max']] = np.nan
    if cfg['log']:
        arr = np.log1p(arr)
    # Detrended ERA5 ICV: per-cell linear trend removed before taking the SD,
    # so the 1991-2020 forced signal is not counted as interannual variability.
    # clim_mean is unchanged by detrending (intercept == period mean).
    clim_mean[vk], icv_sd[vk], icv_series[vk] = detrend_icv(arr)
    ds.close()

# Per-cell ICV correlation matrix entries (off-diagonals only).
# Per-cell ICV cross-variable correlation, computed from the DETRENDED anomaly
# series with a pairwise-finite year count (ddof=2), consistent with icv_sd.
# Using undetrended anomalies here would credit a shared forced trend as shared
# interannual variability, inflating the off-diagonal and rotating the PC axes.
icv_corr = build_icv_corr(VKEYS, icv_sd, icv_series)

# ==============================
# Output arrays
# ==============================
sigma_best = np.full((nlat, nlon), np.nan, np.float32)
dmaha_best = np.full((nlat, nlon), np.nan, np.float32)
analog_lat = np.full((nlat, nlon), np.nan, np.float32)
analog_lon = np.full((nlat, nlon), np.nan, np.float32)
m_dims_map = np.full((nlat, nlon), np.nan, np.float32)
biome_dist_summary = {}


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance (km). Inputs in degrees, broadcastable."""
    R = 6371.0
    p1 = np.radians(lat1); p2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dlmb/2)**2
    return 2*R*np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def sigma_from_distance(d, m):
    """Mahony sigma dissimilarity, HALF-NORMAL convention: 0 = identical,
    grows monotonically positive with distance.

    p = chi.cdf(D, df=m) is the probability an ICV draw falls WITHIN distance
    D; the sigma level is the two-sided normal quantile enclosing that same
    probability, i.e. norm.ppf(0.5 + p/2). Never negative.

    NOTE: a previous version used norm.ppf(p) clipped at 0, which returns
    EXACTLY 0 for every analog with D < 1.177 (df=2), silently zeroing the
    entire map. Do not reintroduce that."""
    p = np.clip(chi.cdf(d, df=m), 0.0, 1.0 - 1e-15)
    return np.clip(norm.ppf(0.5 + p/2), 0.0, sigma_cap)


# Slow loop through each biome 
# ---------------------------- 
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
ds_out.attrs['icv_period'] = f'{clim_start}-{clim_end}'
ds_out.attrs['buffer_km'] = BUFFER_KM
ds_out.attrs['sigma_convention'] = ('half-normal: sigma = norm.ppf(0.5 + chi.cdf(D,df=m)/2); 0 = identical, monotone increasing, never negative.')
ds_out.attrs['icv_definition'] = ('ERA5 1991-2020 annual index series, per grid cell, LINEARLY DETRENDED before taking the SD (ddof=2). Fixed observational ruler: the same ICV is applied to historical and to every CMIP6 model/GWL; ICV is never recomputed per model. Cross-variable ICV correlations are computed from the same detrended anomalies. Focal-cell convention (Mahony 2017): D_ji is scored in focal cell j ICV PC space and is asymmetric.')
ds_out.attrs['note'] = 'Historical only. No Dt threshold (deferred to stability step).'
outfile = os.path.join(out_dir, 'analog_historical_sigma.nc')
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
