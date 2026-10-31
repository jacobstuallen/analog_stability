#!/bin/python
# Analog stability under global warming levels (Methods 2.4-2.5).
#
# Builds on analog_historical_sigma.py. For each grid cell we:
#   1. Take its FUTURE climate vector at GWL2 and GWL3 (per-model 20-year
#      mean over each model's GWL crossing window, Tye et al. 2025
#      Appendix B, SSP5-8.5). Default delta mode: ERA5 1991-2020 + model
#      change, so model bias does not enter (analog_config.FUTURE_MODE).
#   2. Standardize that future vector by the cell's OWN historical
#      1991-2020 ERA5 ICV (s_jk), preserving the metric in historical
#      units (Methods 2.5).
#   3. Measure sigma dissimilarity from the future vector to the
#      HISTORICAL climatology of its best historical within-biome analog,
#      scored in the FOCAL cell's historical ICV (Mahony convention, same
#      as the historical script).
#
# Primary diagnostic produced here (per cell, per GWL):
#   sigma_<gwl> (each GWL in GWLS) : sigma dissimilarity of the future cell to its
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

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import analog_config as cfg  # noqa: E402
from analog_common import (load_biomes, load_era5_icv, load_future_gwl,
                           grid_index_lookup, sigma_from_distance)

# ==============================
# Configuration (edit in analog_config.py; aliased here)
# ==============================
hist_analog_file = cfg.HIST_SIGMA_FILE
out_dir = cfg.OUT_DIR
pc_var_min = cfg.PC_VAR_MIN
min_biome_cells = cfg.MIN_BIOME_CELLS
focal_chunk = 1000
sigma_cap = cfg.SIGMA_CAP
scenario = cfg.SCENARIO
VKEYS, K = cfg.VKEYS, cfg.K
biomes = cfg.BIOMES
GWLS = cfg.GWLS

# ==============================
# Load ERA5 ICV per variable + historical climatology + biome + analogs
# ==============================
print("Loading biome mask and historical analog assignments...")
biome_grid, bmin, bmax = load_biomes()

hist_ds = xr.open_dataset(hist_analog_file).sortby('lat')
analog_lat_grid = hist_ds['analog_lat'].values
analog_lon_grid = hist_ds['analog_lon'].values

clim_mean, icv_sd, icv_series, icv_corr, lat_arr, lon_arr = load_era5_icv(bmin, bmax)
nlat, nlon = len(lat_arr), len(lon_arr)

# ==============================
# Load CMIP6 future GWL climatologies per model
# ==============================
# future_ens[g][vk]: (n_models_g, nlat, nlon); models_used[g]: model names
future_ens, models_used = load_future_gwl(bmin, bmax, GWLS, clim_mean=clim_mean)

# ==============================
# Build a fast lookup from (lat,lon) value -> grid index for analog cells
# ==============================
lat_to_i, lon_to_j = grid_index_lookup(lat_arr, lon_arr)

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
        n_models = len(models_used[g])
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

    for lab, key, fmt in [('sigma ', 'sigma_{}_mean', '6.2f'), ('D_maha', 'dmaha_{}_mean', '6.2f'),
                          ('at cap', 'capfrac_{}', '6.1%')]:
        vals = '  '.join(f'{g.upper()}={np.nanmean(out[key.format(g)][vlat, vlon]):{fmt}}' for g in GWLS)
        print(f'  {lab}: {vals}')
    print(f'  {time.time()-t0:.0f}s')

# ==============================
# Save
# ==============================
os.makedirs(out_dir, exist_ok=True)
ds_out = xr.Dataset(coords={'lat': lat_arr, 'lon': lon_arr})
for k, arr in out.items():
    ds_out[k] = (['lat', 'lon'], arr)
ds_out['biome_id'] = (['lat', 'lon'], biome_grid.astype(np.float32))
ds_out.attrs['method'] = ('Analog stability under GWLs (Methods 2.4-2.5). Future GWL '
    'climatology (per-model 20yr window, SSP5-8.5, see gwl_source) '
    'standardized by historical 1991-2020 ERA5 ICV; sigma dissimilarity to historical '
    'best within-biome analog in focal-cell ICV PC space (Mahony 2017).')
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
ds_out.attrs['icv_definition'] = cfg.ICV_DEFINITION
ds_out.attrs['sigma_saturation'] = (
    'sigma = norm.ppf(0.5 + chi.cdf(D,df=m)/2) saturates at sigma_cap=%s. Above '
    'D_maha ~6 the map is uninformative: distinct separations collapse to the same '
    'sigma. Use dmaha_* for any quantitative divergence analysis; capfrac_* gives '
    'the fraction of ensemble members pinned at the cap per cell.' % sigma_cap)
ds_out.attrs['note'] = ('Continuous sigma trajectory only (diagnostic 1). '
    'Retained/shifted/novel classification deferred pending Dt.')
outfile = cfg.STABILITY_FILE
ds_out.to_netcdf(outfile)
print(f'\nSaved: {outfile}')

# Sidecar: per-model biome-median sigma, for inter-model boxplots.
# Keys: 'biome_{b}_{gwl}' -> array length n_models_gwl; 'models_{gwl}' -> names.
save_kv = {f'models_{g}': np.array(models_used[g]) for g in GWLS}
for b, gd in per_model_biome.items():
    for g, arr in gd.items():
        save_kv[f'biome_{b}_{g}'] = arr.astype(np.float32)
np.savez(cfg.STABILITY_NPZ, **save_kv)
print(f'Saved: {cfg.STABILITY_NPZ}')

# ==============================
# Per-biome summary
# ==============================
print('\n--- Per-biome ensemble-mean sigma to historical analog ---')
print(f'{"biome":<52}' + ''.join(f' {g.upper():>7}' for g in GWLS))
for b, bname in biomes.items():
    bm = (biome_grid == b)
    if not np.isfinite(out[f'sigma_{GWLS[0]}_mean'][bm]).any():
        continue
    print(f'{bname:<52}' + ''.join(f' {np.nanmedian(out[f"sigma_{g}_mean"][bm]):>7.2f}' for g in GWLS))
print('\nDone.')
