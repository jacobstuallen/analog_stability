#!/bin/python
#
# MEANS vs EXTREMES: side-by-side comparison of two complete pipeline runs.
#
#   extremes : TXx, Rx5day   (ANALOG_VARSET=extremes, default output folder)
#   means    : tas, pr       (ANALOG_VARSET=means, <DATA_ROOT>/analog_stability_means)
#   cross    : OPTIONAL. Retention scored on EXTREMES using the MEANS-selected
#              analogs (ANALOG_VARSET=extremes, ANALOG_SETS_FILE=<means sets>,
#              ANALOG_OUT_DIR=<DATA_ROOT>/analog_stability_cross). Answers:
#              are analogs chosen on mean climate also analogs for extremes,
#              today (GWL0) and under warming?
#
# Both sets use K=2 variables, so sigma and Dt are on the same scale.
# All biome statistics are cos(lat) area-weighted.
#
# Env overrides: ANALOG_EXT_DIR, ANALOG_MEANS_DIR, ANALOG_CROSS_DIR
# Writes: compare_varsets.nc (per-cell analog-set overlap) in the means folder

import os
import warnings
import numpy as np
import xarray as xr

import analog_config as cfg
from analog_common import area_weights, wmean, wmedian, haversine_km, ensure_out_dir

EXT = os.environ.get('ANALOG_EXT_DIR', f'{cfg.DATA_ROOT}/analog_stability')
MEA = os.environ.get('ANALOG_MEANS_DIR', f'{cfg.DATA_ROOT}/analog_stability_means')
CRO = os.environ.get('ANALOG_CROSS_DIR', f'{cfg.DATA_ROOT}/analog_stability_cross')
OVERLAP_KM = cfg.BUFFER_KM        # "same place" = within the neighbor buffer
CHUNK = 50000

warnings.simplefilter('ignore', RuntimeWarning)

se = xr.open_dataset(f'{EXT}/analog_sets.nc').sortby('lat')
sm = xr.open_dataset(f'{MEA}/analog_sets.nc').sortby('lat')
re_ = xr.open_dataset(f'{EXT}/analog_retention.nc').sortby('lat')
rm = xr.open_dataset(f'{MEA}/analog_retention.nc').sortby('lat')
rc = xr.open_dataset(f'{CRO}/analog_retention.nc').sortby('lat') if os.path.exists(
    f'{CRO}/analog_retention.nc') else None
for name, ds in [('extremes', re_), ('means', rm)]:
    print(f'{name:>8}: {ds.attrs.get("variables")} | future_mode {ds.attrs.get("future_mode")} '
          f'| gwl_source {ds.attrs.get("gwl_source")}')
print(f'   cross: {"found " + CRO if rc is not None else "not run"}')

lat = se.lat.values; lon = se.lon.values
W = area_weights(lat, len(lon))
biome = se.biome_id.values
GW = [g for g in cfg.GWL_KEYS if f'retention_mmm_{g}' in re_ and f'retention_mmm_{g}' in rm]
DT1 = 1.0
t1 = 'dt1p0'

# ---- 1. geographic overlap of the two analog sets ----
# overlap_e2m: share of a cell's extremes analogs that have a means analog
# within OVERLAP_KM (and vice versa). 1 = same places; 0 = different places.
la_e, lo_e = se.analog_lat.values, se.analog_lon.values
la_m, lo_m = sm.analog_lat.values, sm.analog_lon.values
ov_em = np.full(biome.shape, np.nan, np.float32)
ov_me = np.full(biome.shape, np.nan, np.float32)
ii, jj = np.where((se.n_analogs.values > 0) & (sm.n_analogs.values > 0))
for cs in range(0, len(ii), CHUNK):
    i, j = ii[cs:cs + CHUNK], jj[cs:cs + CHUNK]
    ae, oe, am, om = la_e[i, j], lo_e[i, j], la_m[i, j], lo_m[i, j]   # (n,N)
    d = haversine_km(ae[:, :, None], oe[:, :, None], am[:, None, :], om[:, None, :])  # (n,Ne,Nm)
    d = np.where(np.isfinite(d), d, np.inf)
    hit_e = (d.min(axis=2) <= OVERLAP_KM)
    hit_m = (d.min(axis=1) <= OVERLAP_KM)
    ne = np.isfinite(ae).sum(axis=1); nm = np.isfinite(am).sum(axis=1)
    ov_em[i, j] = np.where(ne > 0, (hit_e & np.isfinite(ae)).sum(axis=1) / np.maximum(ne, 1), np.nan)
    ov_me[i, j] = np.where(nm > 0, (hit_m & np.isfinite(am)).sum(axis=1) / np.maximum(nm, 1), np.nan)

out = xr.Dataset(coords={'lat': lat, 'lon': lon})
out['overlap_ext_in_means'] = (['lat', 'lon'], ov_em)
out['overlap_means_in_ext'] = (['lat', 'lon'], ov_me)
out['biome_id'] = (['lat', 'lon'], biome.astype(np.float32))
out.attrs['method'] = (f'Share of one set\'s analogs with an analog of the other set within '
                       f'{OVERLAP_KM:.0f} km (extremes = {EXT}, means = {MEA}).')
ensure_out_dir(MEA)
out.to_netcdf(f'{MEA}/compare_varsets.nc')


def rows(title, cols, fn):
    print(f'\n--- {title} ---')
    print(f'{"biome":<46}' + ''.join(f' {c:>11}' for c in cols))
    for b, bname in cfg.BIOMES.items():
        bm = (biome == b) & (se.n_analogs.values > 0) & (sm.n_analogs.values > 0)
        if bm.sum() < cfg.MIN_BIOME_CELLS:
            continue
        vals = fn(bm)
        print(f'{bname:<46}' + ''.join(f' {v:>11}' for v in vals))


rows(f'Analog-set overlap (same place = within {OVERLAP_KM:.0f} km), area-weighted means',
     ['ext in mean', 'mean in ext'],
     lambda bm: [f'{wmean(ov_em[bm], W[bm]):.2f}', f'{wmean(ov_me[bm], W[bm]):.2f}'])

rows(f'Forced (mmm) retention at Dt={DT1}: extremes / means',
     [g.upper() for g in GW],
     lambda bm: [f'{wmean(re_[f"retention_mmm_{g}"].sel(Dt=DT1).values[bm], W[bm]):.2f}/'
                 f'{wmean(rm[f"retention_mmm_{g}"].sel(Dt=DT1).values[bm], W[bm]):.2f}' for g in GW])

for g in [x for x in ['gwl2', 'gwl4'] if x in GW]:
    rows(f'Reliability tiers {g.upper()} Dt={DT1} (ret/unc/lost): extremes | means',
         ['extremes', 'means'],
         lambda bm, g=g: ['/'.join(f'{wmean(d[f"{k}_{g}_{t1}"].values[bm], W[bm]):.2f}'
                                   for k in ['robret', 'uncert', 'roblost']) for d in (re_, rm)])

gl = GW[-1]
rows(f'{gl.upper()}: co-movement, forced divergence, departure (area-weighted medians): extremes / means',
     ['comove', 'div mmm', 'departure'],
     lambda bm: [f'{wmedian(re_[f"{k}_{gl}"].values[bm], W[bm]):.2f}/'
                 f'{wmedian(rm[f"{k}_{gl}"].values[bm], W[bm]):.2f}'
                 for k in ['comove', 'divergence_mmm', 'departure']])

if rc is not None:
    GC = [g for g in ['gwl0'] + GW if f'retention_mmm_{g}' in rc]
    rows(f'CROSS: means-selected analogs scored on extremes, forced retention at Dt={DT1} '
         f'(GWL0 = share that are extremes analogs today)',
         [g.upper() for g in GC],
         lambda bm: [f'{wmean(rc[f"retention_mmm_{g}"].sel(Dt=DT1).values[bm], W[bm]):.2f}'
                     for g in GC])

print(f'\nSaved: {MEA}/compare_varsets.nc')
print('Done.')
