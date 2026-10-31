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
#     future_f = CMIP6 future climatology at f (per-model 20yr GWL window)
#     future_a = CMIP6 future climatology at a
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
from scipy.stats import chi, norm
from icv_utils import detrend_icv, build_icv_corr

# ==============================
# Configuration  (mirrors method 1)
# ==============================
data_root = '/glade/campaign/ral/risc/jsallen/TNC'
cckp_root = f'{data_root}/CCKP/CMIP6'
biome_file = '/glade/u/home/jsallen/projects/tnc_2025/analogs/biomes/biomes.analog.gridded.nc'
sets_file = f'{data_root}/analog_stability/analog_sets.nc'
out_dir = f'{data_root}/analog_stability'
outfile = os.path.join(out_dir, 'analog_retention.nc')

clim_start, clim_end = 1991, 2020
pc_var_min = 0.01
min_biome_cells = 500
sigma_cap = 8.0
scenario = 'ssp585'

# Dt sweep grid (sigma units). retention(Dt) evaluated at each.
DT_GRID = np.round(np.arange(0.5, 4.01, 0.25), 2)
# representative Dt slices to also store as standalone maps (communication)
DT_SLICES = [1.0, 2.0]

VARIABLES = {
    'txx': {
        'era5_ts':  f'{data_root}/CCKP/ERA5/timeseries-txx-annual-mean_era5-x0.25_era5-x0.25-historical_timeseries_mean_1950-2024.nc',
        'era5_var': 'timeseries-txx-annual-mean',
        'cckp_token': 'txx', 'log': False, 'qc_max': 70.0,
    },
    'rx5day': {
        'era5_ts':  f'{data_root}/CCKP/ERA5/timeseries-rx5day-annual-mean_era5-x0.25_era5-x0.25-historical_timeseries_mean_1950-2024.nc',
        'era5_var': 'timeseries-rx5day-annual-mean',
        'cckp_token': 'rx5day', 'log': True, 'qc_max': 3000.0,
    },
}
VKEYS = list(VARIABLES.keys())
K = len(VKEYS)

models = {
    'access-cm2':'r1i1p1f1', 'access-esm1-5':'r1i1p1f1', 'bcc-csm2-mr':'r1i1p1f1',
    'canesm5':'r1i1p1f1', 'cmcc-esm2':'r1i1p1f1', 'cnrm-cm6-1':'r1i1p1f2',
    'cnrm-esm2-1':'r1i1p1f2', 'ec-earth3':'r1i1p1f1', 'ec-earth3-veg-lr':'r1i1p1f1',
    'fgoals-g3':'r3i1p1f1', 'gfdl-cm4':'r1i1p1f1', 'gfdl-esm4':'r1i1p1f1',
    'giss-e2-1-g':'r1i1p1f2', 'hadgem3-gc31-ll':'r1i1p1f3', 'hadgem3-gc31-mm':'r1i1p1f3',
    'inm-cm4-8':'r1i1p1f1', 'inm-cm5-0':'r1i1p1f1', 'ipsl-cm6a-lr':'r1i1p1f1',
    'kace-1-0-g':'r1i1p1f1', 'kiost-esm':'r1i1p1f1', 'miroc-es2l':'r1i1p1f2',
    'miroc6':'r1i1p1f1', 'mpi-esm1-2-hr':'r1i1p1f1', 'mpi-esm1-2-lr':'r1i1p1f1',
    'mri-esm2-0':'r1i1p1f1', 'nesm3':'r1i1p1f1', 'noresm2-lm':'r1i1p1f1',
    'noresm2-mm':'r1i1p1f1', 'taiesm1':'r1i1p1f1', 'ukesm1-0-ll':'r1i1p1f2',
}

GWL = {
    'access-cm2':(2024,2043,2043,2062), 'access-esm1-5':(2027,2046,2048,2067),
    'bcc-csm2-mr':(2030,2049,2052,2071), 'canesm5':(2019,2038,2036,2055),
    'cmcc-esm2':(2033,2052,2049,2068), 'cnrm-cm6-1':(2029,2048,2048,2067),
    'cnrm-esm2-1':(2029,2048,2048,2067), 'ec-earth3':(2025,2044,2046,2065),
    'ec-earth3-veg-lr':(2029,2048,2047,2066), 'gfdl-cm4':(2030,2049,2048,2067),
    'gfdl-esm4':(2038,2057,2061,2080), 'giss-e2-1-g':(2028,2047,2052,2071),
    'hadgem3-gc31-ll':(2020,2039,2039,2058), 'hadgem3-gc31-mm':(2023,2042,2037,2056),
    'inm-cm4-8':(2037,2056,2060,2079), 'inm-cm5-0':(2035,2054,2063,2082),
    'ipsl-cm6a-lr':(2027,2046,2045,2064), 'kace-1-0-g':(2020,2039,2041,2060),
    'kiost-esm':(2037,2056,2062,2081), 'miroc-es2l':(2033,2052,2056,2075),
    'miroc6':(2037,2056,2061,2080), 'mpi-esm1-2-hr':(2039,2058,2064,2083),
    'mpi-esm1-2-lr':(2040,2059,2063,2082), 'mri-esm2-0':(2029,2048,2053,2072),
    'nesm3':(2026,2045,2046,2065), 'noresm2-lm':(2041,2060,2063,2082),
    'taiesm1':(2021,2040,2037,2056), 'ukesm1-0-ll':(2019,2038,2034,2053),
}

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
GWLS = ['gwl2', 'gwl3']


# ==============================
# Helpers
# ==============================
def cckp_ts_path(token, model, ensemble, scen):
    folder = f'{model}-{ensemble}-{scen}'
    fname = (f'timeseries-{token}-annual-mean_cmip6-x0.25_'
             f'{model}-{ensemble}-{scen}_timeseries_mean_2015-2100.nc')
    return os.path.join(cckp_root, token, folder, fname)


def load_ts_window(filepath, varname, ys, ye, lat_min, lat_max):
    ds = xr.open_dataset(filepath).sortby('lat').sel(lat=slice(lat_min, lat_max))
    da = ds[varname] if varname in ds else ds[
        [v for v in ds.data_vars if 'bnds' not in v and ds[v].ndim >= 2][0]]
    w = da.sel(time=slice(str(ys), str(ye))).mean(dim='time').values
    ds.close()
    if w.ndim == 3:
        w = w.squeeze()
    return w


def sigma_from_distance(d, m):
    """Mahony sigma, half-normal convention (0 = identical, grows positive)."""
    p = np.clip(chi.cdf(d, df=m), 0.0, 1.0 - 1e-15)
    return np.clip(norm.ppf(0.5 + p/2), -sigma_cap, sigma_cap)


# ==============================
# Load ERA5 ICV + biome + analog sets
# ==============================
print(f"Variables: {VKEYS} (K={K})")
print("Loading biome mask and analog sets...")
biome_ds = xr.open_dataset(biome_file).sortby('lat')
bmin = float(biome_ds.lat.min()); bmax = float(biome_ds.lat.max())
biome_grid = biome_ds['BIOME_ID'].values

sets_ds = xr.open_dataset(sets_file).sortby('lat')
SET_LAT = sets_ds['analog_lat'].values     # (nlat,nlon,N)
SET_LON = sets_ds['analog_lon'].values
N_SET = SET_LAT.shape[2]
N_FOUND = sets_ds['n_analogs'].values

print("Loading ERA5 ICV per variable...")
clim_mean = {}; icv_sd = {}; icv_series = {}
lat_arr = lon_arr = None
for vk in VKEYS:
    cfg = VARIABLES[vk]
    ds = xr.open_dataset(cfg['era5_ts']).sortby('lat').sel(lat=slice(bmin, bmax))
    ts = ds[cfg['era5_var']].sel(time=slice(str(clim_start), str(clim_end)))
    if lat_arr is None:
        lat_arr = ts.lat.values; lon_arr = ts.lon.values
        nlat, nlon = len(lat_arr), len(lon_arr); T = ts.sizes['time']
    arr = ts.values.astype(np.float64)
    arr[arr > cfg['qc_max']] = np.nan
    if cfg['log']:
        arr = np.log1p(arr)
    # Detrended ERA5 ICV: per-cell linear trend removed before taking the SD,
    # so the 1991-2020 forced signal is not counted as interannual variability.
    # clim_mean is unchanged by detrending (intercept == period mean).
    clim_mean[vk], icv_sd[vk], icv_series[vk] = detrend_icv(arr)
    ds.close()

# Per-cell ICV cross-variable correlation, computed from the DETRENDED anomaly
# series with a pairwise-finite year count (ddof=2), consistent with icv_sd.
# Using undetrended anomalies here would credit a shared forced trend as shared
# interannual variability, inflating the off-diagonal and rotating the PC axes.
icv_corr = build_icv_corr(VKEYS, icv_sd, icv_series)

# ==============================
# Load CMIP6 future GWL climatologies per model
# ==============================
print("\nLoading CMIP6 future GWL climatologies...")
future_stack = {g: {vk: [] for vk in VKEYS} for g in GWLS}
models_used = []
for model, ens in models.items():
    if model not in GWL:
        print(f'  {model}: no SSP5-8.5 GWL window, skipping'); continue
    g2s, g2e, g3s, g3e = GWL[model]
    wins = {'gwl2': (g2s, g2e), 'gwl3': (g3s, g3e)}
    paths = {vk: cckp_ts_path(VARIABLES[vk]['cckp_token'], model, ens, scenario)
             for vk in VKEYS}
    missing = [vk for vk, p in paths.items() if not os.path.exists(p)]
    if missing:
        print(f'  {model}: missing {missing}, skipping'); continue
    print(f'  {model} (GWL2 {g2s}-{g2e}, GWL3 {g3s}-{g3e})...', end=' ', flush=True)
    tmp = {g: {} for g in GWLS}
    for vk in VKEYS:
        cfg = VARIABLES[vk]
        for g in GWLS:
            ys, ye = wins[g]
            w = load_ts_window(paths[vk], f'timeseries-{cfg["cckp_token"]}-annual-mean',
                               ys, ye, bmin, bmax)
            w[w > cfg['qc_max']] = np.nan
            if cfg['log']:
                w = np.log1p(w)
            tmp[g][vk] = w
    for g in GWLS:
        for vk in VKEYS:
            future_stack[g][vk].append(tmp[g][vk])
    models_used.append(model)
    print('OK')

n_models = len(models_used)
print(f'  Loaded {n_models} models')
future_ens = {g: {vk: np.stack(future_stack[g][vk], axis=0) for vk in VKEYS}
              for g in GWLS}   # (n_models, nlat, nlon)

# lookup analog lat/lon value -> grid index
lat_to_i = {round(float(v), 4): i for i, v in enumerate(lat_arr)}
lon_to_j = {round(float(v), 4): j for j, v in enumerate(lon_arr)}

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
    """Batched eigendecomposition of each focal cell's historical ICV
    correlation matrix. Returns mu (n,K), sd (n,K), evals (n,K), evecs (n,K,K)."""
    n = len(vlat)
    mu = np.column_stack([clim_mean[vk][vlat, vlon] for vk in VKEYS])
    sd = np.column_stack([icv_sd[vk][vlat, vlon] for vk in VKEYS])
    R = np.zeros((n, K, K))
    for kk in range(K):
        R[:, kk, kk] = 1.0
    for (i, j), grid in icv_corr.items():
        rij = grid[vlat, vlon]
        R[:, i, j] = rij; R[:, j, i] = rij
    evals, evecs = np.linalg.eigh(R)
    return mu, sd, evals, evecs


def pc_sigma(diff, sd, evals, evecs):
    """Sigma dissimilarity of a standardized difference vector, in focal PC
    space. diff (n,K) is (vec - focal_mu); we scale by sd, rotate, whiten.
    Returns sigma (n,)."""
    dstd = diff / sd
    proj = np.einsum('nk,nkm->nm', dstd, evecs)
    keep = evals >= pc_var_min
    with np.errstate(invalid='ignore', divide='ignore'):
        contrib = np.where(keep, proj**2 / evals, 0.0)
    d = np.sqrt(np.nansum(contrib, axis=1))
    mdims = keep.sum(axis=1)
    s = np.full(len(diff), np.nan)
    for mv in np.unique(mdims):
        if mv < 1: continue
        sel = mdims == mv
        s[sel] = sigma_from_distance(d[sel], int(mv))
    return s


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
    'Per-model (SSP5-8.5, Tye 2025 GWL windows) then ensemble median/p10/p90. '
    'Attribution: magratio (|delta_a|/|delta_f|) and dircong (cos angle of '
    'delta_a vs delta_f) in focal-std space, mean over the analog set.')
ds_out.attrs['icv_definition'] = (
    'ERA5 1991-2020 annual index series, per grid cell, LINEARLY DETRENDED before taking the SD (ddof=2). Fixed observational ruler: the same ICV is applied to historical and to every CMIP6 model/GWL; ICV is never recomputed per model. Cross-variable ICV correlations are computed from the same detrended anomalies. Focal-cell convention (Mahony 2017): D_ji is scored in focal cell j ICV PC space and is asymmetric.')
ds_out.attrs['variables'] = ', '.join(VKEYS)
ds_out.attrs['n_models'] = n_models
ds_out.attrs['models'] = ', '.join(models_used)
ds_out.attrs['scenario'] = scenario
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
        hdr += f' {g[-1]}@{Dt:>3}'
hdr += f' {"dir2":>6} {"dir3":>6}'
print(hdr)
for b, bname in biomes.items():
    bm = (biome_grid == b) & (N_FOUND > 0)
    if bm.sum() == 0:
        continue
    row = f'{bname:<46}'
    for g in GWLS:
        for Dt in DT_SLICES:
            di = int(np.argmin(np.abs(DT_GRID - Dt)))
            row += f' {np.nanmean(out[f"retention_{g}"][bm, di]):>5.2f}'
    d2 = np.nanmean(out['dircong_gwl2'][bm]); d3 = np.nanmean(out['dircong_gwl3'][bm])
    row += f' {d2:>6.2f} {d3:>6.2f}'
    print(row)

print('\nDone.')
