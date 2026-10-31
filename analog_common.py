# analog_common.py
#
# Shared helpers for the analog scripts: loading the biome grid, the ERA5 ICV
# ruler and CMIP6 GWL climatologies, and computing Mahony (2017) distances in
# a focal cell's ICV PC space. Constants live in analog_config.py.

import os
import numpy as np
import xarray as xr
from scipy.stats import chi, norm

import analog_config as cfg
from icv_utils import detrend_icv, build_icv_corr


# ==============================
# Distance / sigma
# ==============================
def sigma_from_distance(d, m):
    """Mahony sigma dissimilarity, HALF-NORMAL convention: 0 = identical,
    monotone increasing, never negative. p = chi.cdf(D, df=m) is the chance
    an ICV draw falls within D; sigma is the two-sided normal quantile that
    encloses the same probability. Clipped to [0, SIGMA_CAP].

    Do not use norm.ppf(p): it is negative for D < 1.177 (df=2)."""
    p = np.clip(chi.cdf(d, df=m), 0.0, 1.0 - 1e-15)
    return np.clip(norm.ppf(0.5 + p / 2), 0.0, cfg.SIGMA_CAP)


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance (km). Inputs in degrees, broadcastable."""
    R = 6371.0
    p1 = np.radians(lat1); p2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2)**2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2)**2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def focal_eig(vlat, vlon, clim_mean, icv_sd, icv_corr):
    """Batched eigendecomposition of each focal cell's ICV correlation matrix.
    Returns mu (n,K), sd (n,K), evals (n,K) ascending, evecs (n,K,K)."""
    n = len(vlat)
    K = cfg.K
    mu = np.column_stack([clim_mean[vk][vlat, vlon] for vk in cfg.VKEYS])
    sd = np.column_stack([icv_sd[vk][vlat, vlon] for vk in cfg.VKEYS])
    R = np.zeros((n, K, K))
    for kk in range(K):
        R[:, kk, kk] = 1.0
    for (i, j), grid in icv_corr.items():
        rij = grid[vlat, vlon]
        R[:, i, j] = rij; R[:, j, i] = rij
    evals, evecs = np.linalg.eigh(R)
    return mu, sd, evals, evecs


def pc_distance(diff, sd, evals, evecs):
    """Mahalanobis distance of raw difference vectors in each focal cell's
    ICV PC space. diff, sd (n,K); evals (n,K); evecs (n,K,K).
    Returns D (n,) and retained dimensions m (n,)."""
    dstd = diff / sd
    proj = np.einsum('nk,nkm->nm', dstd, evecs)
    keep = evals >= cfg.PC_VAR_MIN
    with np.errstate(invalid='ignore', divide='ignore'):
        contrib = np.where(keep, proj**2 / evals, 0.0)
    d = np.sqrt(np.nansum(contrib, axis=1))
    return d, keep.sum(axis=1)


def sigma_by_dims(d, mdims):
    """sigma_from_distance applied per retained-dimension group."""
    s = np.full(len(d), np.nan)
    for mv in np.unique(mdims):
        if mv < 1:
            continue
        sel = (mdims == mv) & np.isfinite(d)
        s[sel] = sigma_from_distance(d[sel], int(mv))
    return s


def pc_sigma(diff, sd, evals, evecs):
    """Sigma dissimilarity of raw difference vectors in focal PC space."""
    d, mdims = pc_distance(diff, sd, evals, evecs)
    return sigma_by_dims(d, mdims)


# ==============================
# Loading
# ==============================
def load_biomes():
    """Returns biome_grid (nlat,nlon), lat_min, lat_max."""
    ds = xr.open_dataset(cfg.BIOME_FILE).sortby('lat')
    bmin = float(ds.lat.min()); bmax = float(ds.lat.max())
    grid = ds['BIOME_ID'].values
    ds.close()
    return grid, bmin, bmax


def load_era5_icv(bmin, bmax, verbose=True):
    """Load ERA5 annual series for every variable over CLIM_START-CLIM_END,
    QC-mask, log1p where configured, detrend.
    Returns clim_mean, icv_sd, icv_series (dicts by variable), icv_corr,
    lat_arr, lon_arr."""
    if verbose:
        print(f'Variables: {cfg.VKEYS} (K={cfg.K})')
        print('Loading ERA5 ICV per variable...')
    clim_mean, icv_sd, icv_series = {}, {}, {}
    lat_arr = lon_arr = None
    for vk in cfg.VKEYS:
        v = cfg.VARIABLES[vk]
        ds = xr.open_dataset(v['era5_ts']).sortby('lat').sel(lat=slice(bmin, bmax))
        ts = ds[v['era5_var']].sel(time=slice(str(cfg.CLIM_START), str(cfg.CLIM_END)))
        if lat_arr is None:
            lat_arr = ts.lat.values; lon_arr = ts.lon.values
            if verbose:
                print(f'  Grid: {len(lat_arr)} x {len(lon_arr)}, ICV years: {ts.sizes["time"]}')
        arr = ts.values.astype(np.float64)
        arr[arr > v['qc_max']] = np.nan
        if v['log']:
            arr = np.log1p(arr)
        # Detrended ICV: per-cell linear trend removed before the SD so the
        # 1991-2020 forced signal is not counted as variability. clim_mean is
        # unchanged by detrending (intercept == period mean).
        clim_mean[vk], icv_sd[vk], icv_series[vk] = detrend_icv(arr)
        ds.close()
    # Cross-variable ICV correlation from the DETRENDED anomalies (ddof=2).
    icv_corr = build_icv_corr(cfg.VKEYS, icv_sd, icv_series)
    return clim_mean, icv_sd, icv_series, icv_corr, lat_arr, lon_arr


def cckp_ts_path(token, model, ensemble, scen):
    """CCKP CMIP6 future annual timeseries file for one model + scenario."""
    folder = f'{model}-{ensemble}-{scen}'
    fname = (f'timeseries-{token}-annual-mean_cmip6-x0.25_'
             f'{model}-{ensemble}-{scen}_timeseries_mean_2015-2100.nc')
    return os.path.join(cfg.CCKP_CMIP6, token, folder, fname)


def load_ts_window(filepath, varname, ys, ye, lat_min, lat_max):
    """Mean over [ys, ye] inclusive of a CCKP annual timeseries -> (nlat,nlon)."""
    ds = xr.open_dataset(filepath).sortby('lat').sel(lat=slice(lat_min, lat_max))
    da = ds[varname] if varname in ds else ds[
        [v for v in ds.data_vars if 'bnds' not in v and ds[v].ndim >= 2][0]]
    w = da.sel(time=slice(str(ys), str(ye))).mean(dim='time').values
    ds.close()
    if w.ndim == 3:
        w = w.squeeze()
    return w


def load_future_gwl(bmin, bmax, gwls=None, verbose=True):
    """Per-model GWL-window climatologies for every variable.

    A model is used only if it has a window for EVERY requested GWL and a
    file for every variable, so all GWLs share one model list.

    Returns future_ens[gwl][vk] -> (n_models, nlat, nlon), models_used."""
    gwls = gwls or cfg.GWLS
    if verbose:
        print('\nLoading CMIP6 future GWL climatologies...')
    stack = {g: {vk: [] for vk in cfg.VKEYS} for g in gwls}
    models_used = []
    for model, ens in cfg.MODELS.items():
        no_win = [g for g in gwls if model not in cfg.GWL_WINDOWS[g]]
        if no_win:
            if verbose:
                print(f'  {model}: no {scen_label()} window for {no_win}, skipping')
            continue
        paths = {vk: cckp_ts_path(cfg.VARIABLES[vk]['cckp_token'], model, ens, cfg.SCENARIO)
                 for vk in cfg.VKEYS}
        missing = [vk for vk, p in paths.items() if not os.path.exists(p)]
        if missing:
            if verbose:
                print(f'  {model}: missing {missing}, skipping')
            continue
        if verbose:
            wtxt = ', '.join(f'{g.upper()} {cfg.GWL_WINDOWS[g][model][0]}-'
                             f'{cfg.GWL_WINDOWS[g][model][1]}' for g in gwls)
            print(f'  {model} ({wtxt})...', end=' ', flush=True)
        for vk in cfg.VKEYS:
            v = cfg.VARIABLES[vk]
            for g in gwls:
                ys, ye = cfg.GWL_WINDOWS[g][model]
                w = load_ts_window(paths[vk], f'timeseries-{v["cckp_token"]}-annual-mean',
                                   ys, ye, bmin, bmax)
                w[w > v['qc_max']] = np.nan
                if v['log']:
                    w = np.log1p(w)
                stack[g][vk].append(w)
        models_used.append(model)
        if verbose:
            print('OK')
    if verbose:
        print(f'  Loaded {len(models_used)} models')
    future_ens = {g: {vk: np.stack(stack[g][vk], axis=0) for vk in cfg.VKEYS}
                  for g in gwls}
    return future_ens, models_used


def scen_label():
    return cfg.SCENARIO.upper().replace('SSP585', 'SSP5-8.5')


def grid_index_lookup(lat_arr, lon_arr):
    """Dicts mapping rounded lat/lon values to grid indices."""
    lat_to_i = {round(float(v), 4): i for i, v in enumerate(lat_arr)}
    lon_to_j = {round(float(v), 4): j for j, v in enumerate(lon_arr)}
    return lat_to_i, lon_to_j


def dt_grid():
    return np.round(np.arange(cfg.DT_GRID_START, cfg.DT_GRID_STOP + 0.01,
                              cfg.DT_GRID_STEP), 2)
