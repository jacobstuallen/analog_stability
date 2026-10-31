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


def pc_whiten(diff, sd, evals, evecs):
    """Raw difference vectors -> whitened coordinates in each focal cell's
    ICV PC space (unit ICV variance per retained PC). Dropped PCs are 0.
    diff, sd (n,K); evals (n,K); evecs (n,K,K). Returns w (n,K), keep (n,K)."""
    dstd = diff / sd
    proj = np.einsum('nk,nkm->nm', dstd, evecs)
    keep = evals >= cfg.PC_VAR_MIN
    with np.errstate(invalid='ignore', divide='ignore'):
        w = np.where(keep, proj / np.sqrt(np.where(keep, evals, 1.0)), 0.0)
    return w, keep


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


def linear_fit_nlr(V, g, noise_var, kdims):
    """Per-pair linear fit of divergence vectors against global warming level.

    V (n,L,K): divergence vectors at L warming levels (whitened units).
    g (L,): warming levels (deg C). noise_var: expected noise variance per
    retained dimension of each V entry (e.g. 2/20/M for an M-model mean of
    20-yr windows). kdims (n,): retained dimensions per pair.

    Fits V = alpha + beta*g per dimension (free intercept absorbs the
    baseline and its noise). Returns
      rate (n,): |beta|, divergence per deg C
      nlr  (n,): RSS / (noise_var * dof), ~1 if linear and noise is as assumed
      pval (n,): chi-square p-value of RSS under linear + noise, dof=(L-2)*kdims
      alpha, beta (n,K)
    Overlapping windows correlate the noise and LOWER the residual, so the
    test is conservative: nlr > 1 is evidence of nonlinearity."""
    from scipy.stats import chi2
    g = np.asarray(g, float)
    L = len(g)
    gc = g - g.mean()
    vbar = V.mean(axis=1)                                   # (n,K)
    beta = np.einsum('l,nlk->nk', gc, V - vbar[:, None, :]) / np.sum(gc**2)
    alpha = vbar - beta * g.mean()
    resid = V - (alpha[:, None, :] + beta[:, None, :] * g[None, :, None])
    rss = np.sum(resid**2, axis=(1, 2))
    dof = (L - 2) * np.asarray(kdims, float)
    with np.errstate(invalid='ignore', divide='ignore'):
        nlr = rss / (noise_var * dof)
        pval = chi2.sf(rss / noise_var, dof)
    rate = np.sqrt(np.sum(beta**2, axis=1))
    return rate, nlr, pval, alpha, beta


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
    """CCKP CMIP6 annual timeseries file for one model + experiment.
    scen = 'historical' (1950-2014) or a scenario such as 'ssp585' (2015-2100)."""
    folder = f'{model}-{ensemble}-{scen}'
    years = '1950-2014' if scen == 'historical' else '2015-2100'
    fname = (f'timeseries-{token}-annual-mean_cmip6-x0.25_'
             f'{model}-{ensemble}-{scen}_timeseries_mean_{years}.nc')
    return os.path.join(cfg.CCKP_CMIP6, token, folder, fname)


def _read_years(filepath, varname, ys, ye, lat_min, lat_max):
    """Annual values for [ys, ye] inclusive -> (T, nlat, nlon), or None if
    the file has no years in range."""
    ds = xr.open_dataset(filepath).sortby('lat').sel(lat=slice(lat_min, lat_max))
    da = ds[varname] if varname in ds else ds[
        [v for v in ds.data_vars if 'bnds' not in v and ds[v].ndim >= 2][0]]
    da = da.sel(time=slice(str(ys), str(ye)))
    arr = da.values.astype(np.float64) if da.sizes['time'] > 0 else None
    ds.close()
    return arr


def model_window_mean(vk, model, ens, ys, ye, lat_min, lat_max):
    """Model climatology over [ys, ye] for variable vk, spanning the
    historical and scenario files as needed. QC and log1p are applied to each
    YEAR before averaging, the same order as the ERA5 climatology (mean of
    log1p, not log1p of mean)."""
    v = cfg.VARIABLES[vk]
    varname = f'timeseries-{v["cckp_token"]}-annual-mean'
    parts = []
    if ys <= cfg.MODEL_HIST_END:
        parts.append(_read_years(cckp_ts_path(v['cckp_token'], model, ens, 'historical'),
                                 varname, ys, min(ye, cfg.MODEL_HIST_END), lat_min, lat_max))
    if ye > cfg.MODEL_HIST_END:
        parts.append(_read_years(cckp_ts_path(v['cckp_token'], model, ens, cfg.SCENARIO),
                                 varname, max(ys, cfg.MODEL_HIST_END + 1), ye, lat_min, lat_max))
    arr = np.concatenate([p for p in parts if p is not None], axis=0)
    nexp = ye - ys + 1
    if arr.shape[0] != nexp:
        raise ValueError(f'{model} {vk}: {arr.shape[0]} years in {ys}-{ye}, expected {nexp}')
    arr[arr > v['qc_max']] = np.nan
    if v['log']:
        arr = np.log1p(arr)
    with np.errstate(invalid='ignore'):
        return np.nanmean(arr, axis=0)


def load_future_gwl(bmin, bmax, gwls=None, clim_mean=None, mode=None, verbose=True,
                    model_set=None):
    """Per-model GWL-window climatologies for every variable.

    mode 'delta' (default, cfg.FUTURE_MODE):
        future = ERA5 clim_mean + (model GWL window - model CLIM_START-CLIM_END)
        Needs clim_mean (from load_era5_icv). The model baseline spans
        historical + scenario runs to cover exactly the ERA5 years. In log
        space for log variables, so Rx5day changes are multiplicative.
    mode 'raw':
        future = model GWL window (model biases included).

    Ensembles can differ by GWL: a model enters a GWL if it has a window
    there (cfg.MODEL_SET='per_gwl'), or only if it has windows for ALL
    requested GWLs (model_set='common'; default cfg.MODEL_SET).

    Returns future_ens[gwl][vk] -> (n_models_gwl, nlat, nlon) and
    models_used[gwl] -> list of model names (same order as axis 0)."""
    gwls = gwls or cfg.GWLS
    mode = mode or cfg.FUTURE_MODE
    model_set = model_set or cfg.MODEL_SET
    if mode not in ('delta', 'raw'):
        raise ValueError(f'unknown FUTURE_MODE {mode!r}')
    if mode == 'delta' and clim_mean is None:
        raise ValueError("mode='delta' needs clim_mean from load_era5_icv")
    if verbose:
        print(f'\nLoading CMIP6 GWL climatologies (mode={mode}, '
              f'windows={cfg.GWL_SOURCE}, models={model_set})...')
    stack = {g: {vk: [] for vk in cfg.VKEYS} for g in gwls}
    models_used = {g: [] for g in gwls}
    for model, ens in cfg.MODELS.items():
        mg = [g for g in gwls if model in cfg.GWL_WINDOWS[g]]
        if model_set == 'common' and len(mg) < len(gwls):
            mg = []
        if not mg:
            if verbose:
                print(f'  {model}: no {scen_label()} window for the requested GWLs, skipping')
            continue
        need_hist = mode == 'delta' or any(cfg.GWL_WINDOWS[g][model][0] <= cfg.MODEL_HIST_END
                                           for g in mg)
        exps = [cfg.SCENARIO] + (['historical'] if need_hist else [])
        missing = [(vk, e) for vk in cfg.VKEYS for e in exps
                   if not os.path.exists(cckp_ts_path(cfg.VARIABLES[vk]['cckp_token'], model, ens, e))]
        if missing:
            if verbose:
                print(f'  {model}: missing {missing}, skipping')
            continue
        if verbose:
            wtxt = ', '.join(f'{g.upper()} {cfg.GWL_WINDOWS[g][model][0]}-'
                             f'{cfg.GWL_WINDOWS[g][model][1]}' for g in mg)
            print(f'  {model} ({wtxt})...', end=' ', flush=True)
        for vk in cfg.VKEYS:
            base = (model_window_mean(vk, model, ens, cfg.CLIM_START, cfg.CLIM_END, bmin, bmax)
                    if mode == 'delta' else None)
            for g in mg:
                ys, ye = cfg.GWL_WINDOWS[g][model]
                w = model_window_mean(vk, model, ens, ys, ye, bmin, bmax)
                if mode == 'delta':
                    w = clim_mean[vk] + (w - base)
                stack[g][vk].append(w)
        for g in mg:
            models_used[g].append(model)
        if verbose:
            print('OK')
    if verbose:
        for g in gwls:
            print(f'  {g}: {len(models_used[g])} models')
    empty = [g for g in gwls if not models_used[g]]
    if empty:
        raise RuntimeError(f'no CMIP6 models loaded for {empty}; check paths and windows')
    future_ens = {g: {vk: np.stack(stack[g][vk], axis=0) for vk in cfg.VKEYS}
                  for g in gwls}
    return future_ens, models_used


def scen_label():
    return cfg.SCENARIO.upper().replace('SSP585', 'SSP5-8.5')


def grid_indices(vals, axis_vals, tol=1e-3):
    """Vectorized value -> index lookup on a 1-D grid axis. Returns -1 where
    vals is NaN or not within tol of a grid value."""
    order = np.argsort(axis_vals)
    srt = axis_vals[order]
    v = np.asarray(vals, float)
    out = np.full(v.shape, -1, int)
    ok = np.isfinite(v)
    pos = np.clip(np.searchsorted(srt, v[ok]), 1, len(srt) - 1)
    left, right = srt[pos - 1], srt[pos]
    pick = np.where(np.abs(v[ok] - left) <= np.abs(v[ok] - right), pos - 1, pos)
    hit = np.abs(srt[pick] - v[ok]) <= tol
    tmp = np.where(hit, order[pick], -1)
    out[ok] = tmp
    return out


def grid_index_lookup(lat_arr, lon_arr):
    """Dicts mapping rounded lat/lon values to grid indices."""
    lat_to_i = {round(float(v), 4): i for i, v in enumerate(lat_arr)}
    lon_to_j = {round(float(v), 4): j for j, v in enumerate(lon_arr)}
    return lat_to_i, lon_to_j


def dt_grid():
    return np.round(np.arange(cfg.DT_GRID_START, cfg.DT_GRID_STOP + 0.01,
                              cfg.DT_GRID_STEP), 2)
