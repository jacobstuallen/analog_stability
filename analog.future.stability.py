#!/bin/python
# Analog stability under global warming levels (Methods 2.4-2.5).
#
# Builds on analog_historical_sigma.py. For each grid cell we:
#   1. Take its FUTURE climate vector at GWL2 and GWL3 (per-model 20-year
#      mean over each model's GWL crossing window, Tye et al. 2025
#      Appendix B, ensemble-mean periods, SSP5-8.5).
#   2. Standardize that future vector by the cell's OWN historical
#      1991-2020 ERA5 ICV (s_jk), preserving the metric in historical
#      units (Methods 2.5).
#   3. Measure sigma dissimilarity from the future vector to the
#      HISTORICAL climatology of its best historical within-biome analog,
#      scored in the FOCAL cell's historical ICV (Mahony convention, same
#      as the historical script).
#
# Primary diagnostic produced here (per cell, per GWL):
#   sigma_gwl2, sigma_gwl3 : sigma dissimilarity of the future cell to its
#                            historical best analog. The analog stability
#                            trajectory (Methods 2.5 diagnostic 1).
#   Plus ensemble 10th/90th percentile bounds across models.
#
# DEFERRED (need Dt): the retained/shifted/novel classification
# (diagnostic 2) and biome fractional-area retention (diagnostic 3).
# Those are added once Dt is calibrated.
#
# Variable set is a CONFIG list. Two variables now (TXx, Rx5day);
# extends to {TXx, WBT, CDD, Rx5day} by editing VARIABLES and adding
# the corresponding ERA5 ICV files + CMIP6 paths.

import numpy as np
import xarray as xr
import os
import time
from scipy.stats import chi, norm
from icv_utils import detrend_icv, build_icv_corr

# ==============================
# Configuration
# ==============================
data_root = '/glade/campaign/ral/risc/jsallen/TNC'
cckp_root = f'{data_root}/CCKP/CMIP6'
biome_file = '/glade/u/home/jsallen/projects/tnc_2025/analogs/biomes/biomes.analog.gridded.nc'
hist_analog_file = f'{data_root}/analog_stability/analog_historical_sigma.nc'
out_dir = f'{data_root}/analog_stability'

clim_start, clim_end = 1991, 2020
pc_var_min = 0.01
min_biome_cells = 500
focal_chunk = 1000
sigma_cap = 8.0
scenario = 'ssp585'

# ---- Variable configuration ----
# Each entry: key used internally -> dict with ERA5 annual timeseries file,
# ERA5 timeseries varname, CCKP index token (folder/filename), and whether
# to log1p-transform. Extend this list to go to 4 variables.
VARIABLES = {
    'txx': {
        'era5_ts':  f'{data_root}/CCKP/ERA5/timeseries-txx-annual-mean_era5-x0.25_era5-x0.25-historical_timeseries_mean_1950-2024.nc',
        'era5_var': 'timeseries-txx-annual-mean',
        'cckp_token': 'txx',
        'log': False,
        'qc_max': 70.0,
    },
    'rx5day': {
        'era5_ts':  f'{data_root}/CCKP/ERA5/timeseries-rx5day-annual-mean_era5-x0.25_era5-x0.25-historical_timeseries_mean_1950-2024.nc',
        'era5_var': 'timeseries-rx5day-annual-mean',
        'cckp_token': 'rx5day',
        'log': True,
        'qc_max': 3000.0,
    },
    # To extend to 4 variables, add e.g.:
    # 'wbt': {... era5 annual wbt ts ..., 'cckp_token':'wbt', 'log':False, 'qc_max':45.0},
    # 'cdd': {... era5 annual cdd ts ..., 'cckp_token':'cdd', 'log':False, 'qc_max':400.0},
}
VKEYS = list(VARIABLES.keys())
K = len(VKEYS)

# ---- CMIP6 ensemble members (model -> ensemble variant) ----
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

# ---- GWL windows, SSP5-8.5, ensemble-mean periods (Tye 2025 Appendix B) ----
# model -> (gwl2_start, gwl2_end, gwl3_start, gwl3_end)
# Models with no published SSP5-8.5 window (fgoals-g3, noresm2-mm) are
# absent here and will be skipped with a logged warning.
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
    """CCKP CMIP6 future ANNUAL TIMESERIES file for a model+scenario.
    GWL windows require annual timeseries (we average over the window),
    not the pre-computed fixed-period climatologies."""
    folder = f'{model}-{ensemble}-{scen}'
    fname = (f'timeseries-{token}-annual-mean_cmip6-x0.25_'
             f'{model}-{ensemble}-{scen}_timeseries_mean_2015-2100.nc')
    return os.path.join(cckp_root, token, folder, fname)


def load_ts_window(filepath, varname, ys, ye, lat_min, lat_max):
    """Load a CCKP annual timeseries, return the (lat,lon) mean over
    [ys,ye] inclusive."""
    ds = xr.open_dataset(filepath).sortby('lat').sel(lat=slice(lat_min, lat_max))
    da = ds[varname] if varname in ds else ds[
        [v for v in ds.data_vars if 'bnds' not in v and ds[v].ndim >= 2][0]]
    w = da.sel(time=slice(str(ys), str(ye))).mean(dim='time').values
    ds.close()
    if w.ndim == 3:
        w = w.squeeze()
    return w


def sigma_from_distance(d, m):
    p = np.clip(chi.cdf(d, df=m), 0.0, 1.0 - 1e-15)
    # HALF-NORMAL convention (matches set_selection / retention /
    # historical). norm.ppf(p) would emit NEGATIVE sigma for D < 1.177
    # (df=2), which is meaningless for a dissimilarity and corrupts
    # any mean/median taken over it.
    return np.clip(norm.ppf(0.5 + p/2), 0.0, sigma_cap)


# ==============================
# Load ERA5 ICV per variable + historical climatology + biome + analogs
# ==============================
print(f"Variables: {VKEYS} (K={K})")
print("Loading biome mask and historical analog assignments...")
biome_ds = xr.open_dataset(biome_file).sortby('lat')
bmin = float(biome_ds.lat.min()); bmax = float(biome_ds.lat.max())
biome_grid = biome_ds['BIOME_ID'].values

hist_ds = xr.open_dataset(hist_analog_file).sortby('lat')
analog_lat_grid = hist_ds['analog_lat'].values
analog_lon_grid = hist_ds['analog_lon'].values

print("Loading ERA5 ICV for each variable...")
clim_mean = {}   # variable -> (nlat,nlon) 1991-2020 climatological mean
icv_sd    = {}   # variable -> (nlat,nlon) ICV std
icv_series = {}  # variable -> (T,nlat,nlon) anomaly series (for covariance)
lat_arr = lon_arr = None
for vk in VKEYS:
    cfg = VARIABLES[vk]
    ds = xr.open_dataset(cfg['era5_ts']).sortby('lat').sel(lat=slice(bmin, bmax))
    ts = ds[cfg['era5_var']].sel(time=slice(str(clim_start), str(clim_end)))
    if lat_arr is None:
        lat_arr = ts.lat.values; lon_arr = ts.lon.values
        nlat, nlon = len(lat_arr), len(lon_arr)
        T = ts.sizes['time']
    arr = ts.values.astype(np.float64)
    arr[arr > cfg['qc_max']] = np.nan
    if cfg['log']:
        arr = np.log1p(arr)
    # Detrended ERA5 ICV: per-cell linear trend removed before taking the SD,
    # so the 1991-2020 forced signal is not counted as interannual variability.
    # clim_mean is unchanged by detrending (intercept == period mean).
    clim_mean[vk], icv_sd[vk], icv_series[vk] = detrend_icv(arr)
    ds.close()

# Per-cell ICV correlation matrix entries (K x K). For K=2 just one off-diag.
# Store as dict of (vi,vj) -> (nlat,nlon) correlation.
# Per-cell ICV cross-variable correlation, computed from the DETRENDED anomaly
# series with a pairwise-finite year count (ddof=2), consistent with icv_sd.
# Using undetrended anomalies here would credit a shared forced trend as shared
# interannual variability, inflating the off-diagonal and rotating the PC axes.
icv_corr = build_icv_corr(VKEYS, icv_sd, icv_series)

# ==============================
# Load CMIP6 future GWL climatologies per model -> ensemble mean per GWL
# ==============================
print("\nLoading CMIP6 future GWL climatologies...")
# future_mean[gwl][vk] accumulates list of per-model (nlat,nlon) arrays
future_stack = {g: {vk: [] for vk in VKEYS} for g in GWLS}
models_used = []
for model, ens in models.items():
    if model not in GWL:
        print(f'  {model}: no SSP5-8.5 GWL window, skipping')
        continue
    g2s, g2e, g3s, g3e = GWL[model]
    wins = {'gwl2': (g2s, g2e), 'gwl3': (g3s, g3e)}
    # check files exist for all variables
    paths = {vk: cckp_ts_path(VARIABLES[vk]['cckp_token'], model, ens, scenario)
             for vk in VKEYS}
    missing = [vk for vk, p in paths.items() if not os.path.exists(p)]
    if missing:
        print(f'  {model}: missing {missing}, skipping')
        continue
    print(f'  {model} (GWL2 {g2s}-{g2e}, GWL3 {g3s}-{g3e})...', end=' ', flush=True)
    ok = True
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

# Ensemble: per GWL, per variable -> (n_models, nlat, nlon)
future_ens = {g: {vk: np.stack(future_stack[g][vk], axis=0) for vk in VKEYS}
              for g in GWLS}

# ==============================
# Build a fast lookup from (lat,lon) value -> grid index for analog cells
# ==============================
lat_to_i = {round(float(v), 4): i for i, v in enumerate(lat_arr)}
lon_to_j = {round(float(v), 4): j for j, v in enumerate(lon_arr)}

# ==============================
# Output arrays
# ==============================
out = {}
for g in GWLS:
    out[f'sigma_{g}_mean'] = np.full((nlat, nlon), np.nan, np.float32)
    out[f'sigma_{g}_p10']  = np.full((nlat, nlon), np.nan, np.float32)
    out[f'sigma_{g}_p90']  = np.full((nlat, nlon), np.nan, np.float32)
    # D_maha is the UNSATURATED metric. sigma = norm.ppf(0.5 + chi.cdf(D)/2)
    # compresses violently in the tail and pins at sigma_cap (8.0): a pair at
    # D=8 and a pair at D=21 both report sigma 8.0. Carry D_maha through so the
    # divergence signal survives where sigma flatlines. This is the quantity to
    # use for the supplemental figure.
    out[f'dmaha_{g}_mean'] = np.full((nlat, nlon), np.nan, np.float32)
    out[f'dmaha_{g}_p10']  = np.full((nlat, nlon), np.nan, np.float32)
    out[f'dmaha_{g}_p90']  = np.full((nlat, nlon), np.nan, np.float32)
    # Fraction of ensemble members whose sigma hits the cap at this cell.
    # If this is large, sigma is uninformative there and D_maha must be used.
    out[f'capfrac_{g}']    = np.full((nlat, nlon), np.nan, np.float32)

# Per-model biome-median sigma, for inter-model boxplots.
# biome_id -> {gwl -> array of length n_models of spatial-median sigma}
per_model_biome = {}

# ==============================
# Process biome by biome
# ==============================
for b, bname in biomes.items():
    t0 = time.time()
    bm = (biome_grid == b)
    clat, clon = np.where(bm)
    if len(clat) < min_biome_cells:
        print(f'\nBiome {b:2d} ({bname}): {len(clat)} cells, SKIPPING')
        continue

    # focal cell historical ICV quantities
    mu = np.column_stack([clim_mean[vk][clat, clon] for vk in VKEYS])     # (n,K)
    sd = np.column_stack([icv_sd[vk][clat, clon] for vk in VKEYS])        # (n,K)
    valid = np.all(np.isfinite(mu), axis=1) & np.all(np.isfinite(sd), axis=1) \
            & np.all(sd > 0, axis=1)
    # require analog assignment too
    al = analog_lat_grid[clat, clon]; ao = analog_lon_grid[clat, clon]
    valid &= np.isfinite(al) & np.isfinite(ao)
    if valid.sum() < min_biome_cells:
        print(f'\nBiome {b:2d} ({bname}): {valid.sum()} valid, SKIPPING')
        continue

    vlat, vlon = clat[valid], clon[valid]
    mu, sd = mu[valid], sd[valid]
    al, ao = al[valid], ao[valid]
    n = len(vlat)
    print(f'\nBiome {b:2d} ({bname}): {n} valid cells')

    # historical climatology vector of each cell's BEST ANALOG.
    # Map analog lat/lon to grid indices, then pull clim_mean there.
    ai = np.array([lat_to_i.get(round(float(v), 4), -1) for v in al])
    aj = np.array([lon_to_j.get(round(float(v), 4), -1) for v in ao])
    good = (ai >= 0) & (aj >= 0)
    analog_vec = np.full((n, K), np.nan)
    for kk, vk in enumerate(VKEYS):
        cm = clim_mean[vk]
        analog_vec[good, kk] = cm[ai[good], aj[good]]

    # focal ICV correlation per cell (K x K). For K=2 just rho.
    # Precompute the eigendecomposition of each cell's correlation matrix.
    # General K: build (n,K,K) corr, eigh batched.
    R = np.zeros((n, K, K))
    for kk in range(K):
        R[:, kk, kk] = 1.0
    for (i, j), grid in icv_corr.items():
        rij = grid[vlat, vlon]
        R[:, i, j] = rij; R[:, j, i] = rij
    evals, evecs = np.linalg.eigh(R)   # (n,K), (n,K,K) ascending

    # future ensemble vectors at this biome's cells: (n_models, n, K) per GWL
    fut = {}
    for g in GWLS:
        stk = np.stack([future_ens[g][vk][:, vlat, vlon] for vk in VKEYS], axis=-1)
        fut[g] = stk   # (n_models, n, K)

    # ---- distance: future focal vector vs analog's historical vector,
    #      in focal cell's ICV PC space ----
    # standardized anomaly of a vector x wrt focal ICV: (x - mu)/sd
    # but Mahony scales the DIFFERENCE (future_focal - analog_hist) by sd,
    # then rotates into focal PCs and whitens. Equivalent to standardizing
    # both by focal mu,sd and taking PC distance. We use the difference form.
    for g in GWLS:
        sig_models = np.full((n_models, n), np.nan)
        dm_models  = np.full((n_models, n), np.nan)   # unsaturated D_maha
        for mi in range(n_models):
            diff = (fut[g][mi] - analog_vec) / sd     # (n,K) standardized diff
            # rotate into focal PCs: project diff onto evecs
            # proj_k = diff . evec_k  ; whiten by sqrt(eval_k); drop eval<min
            proj = np.einsum('nk,nkm->nm', diff, evecs)  # (n,K) in PC basis
            keep = evals >= pc_var_min                   # (n,K)
            with np.errstate(invalid='ignore', divide='ignore'):
                contrib = np.where(keep, proj**2 / evals, 0.0)
            d = np.sqrt(np.nansum(contrib, axis=1))      # (n,)
            mdims = keep.sum(axis=1)                      # (n,)
            s = np.full(n, np.nan)
            for mv in np.unique(mdims):
                if mv < 1: continue
                sel = mdims == mv
                s[sel] = sigma_from_distance(d[sel], int(mv))
            sig_models[mi] = s
            dm_models[mi]  = d

        out[f'sigma_{g}_mean'][vlat, vlon] = np.nanmean(sig_models, axis=0).astype(np.float32)
        out[f'sigma_{g}_p10'][vlat, vlon]  = np.nanpercentile(sig_models, 10, axis=0).astype(np.float32)
        out[f'sigma_{g}_p90'][vlat, vlon]  = np.nanpercentile(sig_models, 90, axis=0).astype(np.float32)

        out[f'dmaha_{g}_mean'][vlat, vlon] = np.nanmean(dm_models, axis=0).astype(np.float32)
        out[f'dmaha_{g}_p10'][vlat, vlon]  = np.nanpercentile(dm_models, 10, axis=0).astype(np.float32)
        out[f'dmaha_{g}_p90'][vlat, vlon]  = np.nanpercentile(dm_models, 90, axis=0).astype(np.float32)

        # how often does sigma pin at the cap here?
        atcap = (sig_models >= sigma_cap - 1e-6) & np.isfinite(sig_models)
        nfin  = np.sum(np.isfinite(sig_models), axis=0)
        with np.errstate(invalid='ignore', divide='ignore'):
            out[f'capfrac_{g}'][vlat, vlon] = np.where(
                nfin > 0, atcap.sum(axis=0) / np.maximum(nfin, 1), np.nan
            ).astype(np.float32)

        # per-model spatial-median sigma AND D_maha for this biome
        per_model_biome.setdefault(b, {})[g] = np.nanmedian(sig_models, axis=1)
        per_model_biome[b][f'{g}_dmaha'] = np.nanmedian(dm_models, axis=1)

    em2 = np.nanmean(out['sigma_gwl2_mean'][vlat, vlon])
    em3 = np.nanmean(out['sigma_gwl3_mean'][vlat, vlon])
    dm2 = np.nanmean(out['dmaha_gwl2_mean'][vlat, vlon])
    dm3 = np.nanmean(out['dmaha_gwl3_mean'][vlat, vlon])
    cf2 = np.nanmean(out['capfrac_gwl2'][vlat, vlon])
    cf3 = np.nanmean(out['capfrac_gwl3'][vlat, vlon])
    print(f'  sigma : GWL2={em2:6.2f}  GWL3={em3:6.2f}   (cap={sigma_cap})')
    print(f'  D_maha: GWL2={dm2:6.2f}  GWL3={dm3:6.2f}   (unsaturated)')
    print(f'  frac at sigma cap: GWL2={cf2:5.1%}  GWL3={cf3:5.1%} | {time.time()-t0:.0f}s')

# ==============================
# Save
# ==============================
os.makedirs(out_dir, exist_ok=True)
ds_out = xr.Dataset(coords={'lat': lat_arr, 'lon': lon_arr})
for k, arr in out.items():
    ds_out[k] = (['lat', 'lon'], arr)
ds_out['biome_id'] = (['lat', 'lon'], biome_grid.astype(np.float32))
ds_out.attrs['method'] = ('Analog stability under GWLs (Methods 2.4-2.5). Future GWL '
    'climatology (per-model 20yr window, Tye 2025 Appendix B, SSP5-8.5 ens-mean periods) '
    'standardized by historical 1991-2020 ERA5 ICV; sigma dissimilarity to historical '
    'best within-biome analog in focal-cell ICV PC space (Mahony 2017).')
ds_out.attrs['variables'] = ', '.join(VKEYS)
ds_out.attrs['n_models'] = n_models
ds_out.attrs['models'] = ', '.join(models_used)
ds_out.attrs['scenario'] = scenario
ds_out.attrs['icv_definition'] = ('ERA5 1991-2020 annual index series, per grid cell, LINEARLY DETRENDED before taking the SD (ddof=2). Fixed observational ruler: the same ICV is applied to historical and to every CMIP6 model/GWL; ICV is never recomputed per model. Cross-variable ICV correlations are computed from the same detrended anomalies. Focal-cell convention (Mahony 2017): D_ji is scored in focal cell j ICV PC space and is asymmetric.')
ds_out.attrs['sigma_saturation'] = (
    'sigma = norm.ppf(0.5 + chi.cdf(D,df=m)/2) saturates at sigma_cap=%s. Above '
    'D_maha ~6 the map is uninformative: distinct separations collapse to the same '
    'sigma. Use dmaha_* for any quantitative divergence analysis; capfrac_* gives '
    'the fraction of ensemble members pinned at the cap per cell.' % sigma_cap)
ds_out.attrs['note'] = ('Continuous sigma trajectory only (diagnostic 1). '
    'Retained/shifted/novel classification deferred pending Dt.')
outfile = os.path.join(out_dir, 'analog_stability_gwl.nc')
ds_out.to_netcdf(outfile)
print(f'\nSaved: {outfile}')

# Sidecar: per-model biome-median sigma, for inter-model boxplots.
# Keys: 'biome_{b}_{gwl}' -> array length n_models; plus 'models'.
save_kv = {'models': np.array(models_used)}
for b, gd in per_model_biome.items():
    for g, arr in gd.items():
        save_kv[f'biome_{b}_{g}'] = arr.astype(np.float32)
np.savez(os.path.join(out_dir, 'per_model_biome_sigma.npz'), **save_kv)
print(f"Saved: {os.path.join(out_dir, 'per_model_biome_sigma.npz')}")

# ==============================
# Per-biome summary
# ==============================
print('\n--- Per-biome ensemble-mean sigma to historical analog ---')
print(f'{"biome":<52} {"GWL2":>7} {"GWL3":>7}')
for b, bname in biomes.items():
    bm = (biome_grid == b)
    s2 = out['sigma_gwl2_mean'][bm]; s3 = out['sigma_gwl3_mean'][bm]
    s2 = s2[np.isfinite(s2)]; s3 = s3[np.isfinite(s3)]
    if s2.size == 0: continue
    print(f'{bname:<52} {np.median(s2):>7.2f} {np.median(s3):>7.2f}')
print('\nDone.')
