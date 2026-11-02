#!/bin/python
#
# NULL BASELINE for the means-vs-extremes cross-evaluation.
#
# Question: analogs chosen on MEAN climate are extremes analogs only 19-54%
# of the time (cross run, GWL0). Is that better than picking a RANDOM place
# in the same biome?
#
# For a seeded sample of focal cells in each biome, draw random same-biome
# cells beyond BUFFER_KM and score them in the ACTIVE variable space
# (ANALOG_VARSET; default extremes) exactly as real analogs are scored:
# ERA5 1991-2020 climatology, sigma in the focal cell's ICV PC space.
# Report the share with sigma < Dt for several Dt, next to the share for
# means-selected analogs (cross run retention_mmm_gwl0, which is historical
# ERA5) and the lift (means-selected / random).
#
# Lift ~1: mean-climate matching carries ~no information about the extremes
# regime. Lift >> 1: it helps, but (per the cross run) still misses most.
#
# Historical ERA5 only; no models. All biome statistics cos(lat) weighted.
# Env: ANALOG_CROSS_DIR (default <DATA_ROOT>/analog_stability_cross)
# Writes: random_baseline_<varset>.nc in OUT_DIR

import os
import time
import warnings
import numpy as np
import xarray as xr

import analog_config as cfg
from analog_common import (load_biomes, load_era5_icv, focal_eig, pc_distance,
                           sigma_by_dims, haversine_km, area_weights, wmean)

N_FOCAL = 4000          # focal cells sampled per biome (all if fewer)
N_RAND = 300            # random candidates drawn per focal cell
DTS = [1.0, 1.5, 2.0, 3.0]
CHUNK = 500
SEED = 20261102
CRO = os.environ.get('ANALOG_CROSS_DIR', f'{cfg.DATA_ROOT}/analog_stability_cross')

warnings.simplefilter('ignore', RuntimeWarning)
rng = np.random.default_rng(SEED)

biome_grid, bmin, bmax = load_biomes()
clim_mean, icv_sd, _, icv_corr, lat_arr, lon_arr = load_era5_icv(bmin, bmax)
W = area_weights(lat_arr, len(lon_arr))
VK = cfg.VKEYS

cross = None
if os.path.exists(f'{CRO}/analog_retention.nc'):
    cross = xr.open_dataset(f'{CRO}/analog_retention.nc').sortby('lat')
    if cross.attrs.get('variable_set') not in (None, cfg.VARSET):
        print(f'Cross run is in {cross.attrs.get("variable_set")!r} space, this run is '
              f'{cfg.VARSET!r}: printing the random baseline only')
        cross = None
if cross is not None:
    if not np.allclose(cross.lat.values, lat_arr) or len(cross.lon) != len(lon_arr):
        raise SystemExit('cross file grid does not match ERA5 grid')
    print(f'Cross run: {CRO} (analogs from {cross.attrs.get("analog_sets_file")})')
else:
    print('Random baseline only (no cross run in this variable space)')

rand_share = {Dt: np.full(biome_grid.shape, np.nan, np.float32) for Dt in DTS}
rows = {}
for b, bname in cfg.BIOMES.items():
    t0 = time.time()
    ci, cj = np.where(biome_grid == b)
    ok = np.ones(len(ci), bool)
    for vk in VK:
        ok &= np.isfinite(clim_mean[vk][ci, cj]) & np.isfinite(icv_sd[vk][ci, cj]) & (icv_sd[vk][ci, cj] > 0)
    for grid in icv_corr.values():
        ok &= np.isfinite(grid[ci, cj])
    ci, cj = ci[ok], cj[ok]
    if len(ci) < cfg.MIN_BIOME_CELLS:
        continue
    pick = rng.choice(len(ci), size=min(N_FOCAL, len(ci)), replace=False)
    fi, fj = ci[pick], cj[pick]
    clat_all, clon_all = lat_arr[ci], lon_arr[cj]
    shares = {Dt: np.full(len(fi), np.nan) for Dt in DTS}
    for cs in range(0, len(fi), CHUNK):
        a, z = fi[cs:cs + CHUNK], fj[cs:cs + CHUNK]
        n = len(a)
        mu, sd, ev, ec = focal_eig(a, z, clim_mean, icv_sd, icv_corr)
        cand = rng.integers(0, len(ci), size=(n, N_RAND))
        dkm = haversine_km(lat_arr[a][:, None], lon_arr[z][:, None], clat_all[cand], clon_all[cand])
        far = dkm > cfg.BUFFER_KM
        cvec = np.stack([clim_mean[vk][ci[cand], cj[cand]] for vk in VK], axis=-1)   # (n,R,K)
        diff = (cvec - mu[:, None, :]).reshape(-1, cfg.K)
        rep = lambda x: np.repeat(x, N_RAND, axis=0)
        d, m = pc_distance(diff, rep(sd), rep(ev), rep(ec))
        sig = sigma_by_dims(d, m).reshape(n, N_RAND)
        sig = np.where(far, sig, np.nan)
        nfar = np.sum(np.isfinite(sig), axis=1)
        for Dt in DTS:
            shares[Dt][cs:cs + n] = np.where(nfar > 0, np.nansum(sig < Dt, axis=1) / np.maximum(nfar, 1), np.nan)
    for Dt in DTS:
        rand_share[Dt][fi, fj] = shares[Dt]
    wf = W[fi, fj]
    r = {Dt: wmean(shares[Dt], wf) for Dt in DTS}
    mshare = {}
    if cross is not None:
        bm = (biome_grid == b) & (cross.n_analogs.values > 0)
        for Dt in DTS:
            mshare[Dt] = wmean(cross.retention_mmm_gwl0.sel(Dt=Dt).values[bm], W[bm])
    rows[b] = (r, mshare, len(fi))
    print(f'  {bname[:40]:<40} {len(fi)} focal cells x {N_RAND} random | {time.time() - t0:.0f}s', flush=True)

out = xr.Dataset(coords={'lat': lat_arr, 'lon': lon_arr})
for Dt in DTS:
    out[f'random_share_dt{Dt:.1f}'.replace('.', 'p')] = (['lat', 'lon'], rand_share[Dt])
out['biome_id'] = (['lat', 'lon'], biome_grid.astype(np.float32))
out.attrs.update({'variable_set': cfg.VARSET, 'n_focal_per_biome': N_FOCAL, 'n_random': N_RAND,
                  'seed': SEED, 'buffer_km': cfg.BUFFER_KM,
                  'method': 'Share of random same-biome cells beyond buffer_km with historical '
                            'sigma < Dt, scored in the focal cell ICV PC space (ERA5 1991-2020). '
                            'Sampled focal cells only; others NaN.'})
os.makedirs(cfg.OUT_DIR, exist_ok=True)
fout = os.path.join(cfg.OUT_DIR, f'random_baseline_{cfg.VARSET}.nc')
out.to_netcdf(fout)

print(f'\n--- Share of analogs within Dt in {cfg.VARSET} space ({", ".join(VK)}), area-weighted ---')
print('    random = random same-biome cell beyond the buffer; means-sel = means-selected analogs (cross, GWL0)')
hdr = f'{"biome":<46}' + ''.join(f' {"rand" + str(Dt):>8}' for Dt in DTS)
if cross is not None:
    hdr += ''.join(f' {"mean" + str(Dt):>8}' for Dt in DTS) + f' {"lift@1":>7} {"lift@2":>7}'
print(hdr)
for b, (r, ms, nf) in rows.items():
    row = f'{cfg.BIOMES[b]:<46}' + ''.join(f' {r[Dt]:>8.2f}' for Dt in DTS)
    if ms:
        row += ''.join(f' {ms[Dt]:>8.2f}' for Dt in DTS)
        row += f' {ms[1.0] / max(r[1.0], 1e-6):>7.1f} {ms[2.0] / max(r[2.0], 1e-6):>7.1f}'
    print(row)
print('    lift = means-selected share / random share (1 = no better than random)')
print(f'\nSaved: {fout}\nDone.')
