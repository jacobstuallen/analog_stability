import numpy as np, xarray as xr, json
import os


import os
import json
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import colormaps as cmaps

data_root = '/glade/campaign/ral/risc/jsallen/TNC'
out_dir = f'{data_root}/analog_stability'
ret_file = os.path.join(out_dir, 'analog_retention.nc')
sets_file = os.path.join(out_dir, 'analog_sets.nc')

# Written by ecoregion_shp_to_reference.py. Must be on the SAME grid as
# analog_retention.nc.
#eco_file = '/glade/u/home/jsallen/projects/tnc_2025/analogs/biomes/ecoregions.analog.gridded.nc'
eco_file = '/glade/work/jsallen/projects/tnc_2025/draft_03/ecoregion/ecoregions.analog.gridded.nc'

N_FULL = 10           # full analog set size
S_FULL = 500.0        # full spacing (cells below this were relaxed)
DT_STRICT, DT_LOOSE = 1.0, 2.0

MIN_COVERAGE = 0.50   # ecoregion needs >=50% of its land cells passing `keep`
MIN_CELLS = 10        # and >=10 land cells total
EXCLUDE_BIOMES = [98, 99]   # Inland Water, Rock & Ice: not analog pools


import numpy as np, xarray as xr

ds  = xr.open_dataset(ret_file)
eco = xr.open_dataset(eco_file, mask_and_scale=False)

for name, d in [('retention', ds), ('ecoregion', eco)]:
    la, lo = d['lat'].values, d['lon'].values
    print(f'{name:10s} lat[0]={la[0]:8.3f} lat[-1]={la[-1]:8.3f} '
          f'n={len(la):4d} {"ASC" if la[0] < la[-1] else "DESC"}')
    print(f'{" ":10s} lon[0]={lo[0]:8.3f} lon[-1]={lo[-1]:8.3f} '
          f'n={len(lo):4d} {"0-360" if lo.max() > 180 else "-180/180"}')

print('\nlat identical:', np.allclose(ds.lat.values, eco.lat.values))
print('lon identical:', np.allclose(ds.lon.values, eco.lon.values))


ds = xr.open_dataset(ret_file)
eco_ds = xr.open_dataset(eco_file, mask_and_scale=False)

biome_id = ds['biome_id'].values
n_analogs = ds['n_analogs'].values
s_used = xr.open_dataset(sets_file)['s_used_km'].values

eco_raw = eco_ds['ECO_ID'].values
eco_id = np.where(np.isfinite(eco_raw) & (eco_raw > 0), eco_raw, 0).astype(np.int64)
eco_b_raw = eco_ds['BIOME_ID'].values
eco_biome = np.where(np.isfinite(eco_b_raw) & (eco_b_raw > 0), eco_b_raw, 0).astype(np.int64)

keep = (n_analogs >= 10) & (s_used >= 500.0)
land = np.isfinite(biome_id) & (biome_id > 0)

print('retention land cells     :', land.sum())
print('ecoregion cells (>0)     :', (eco_id > 0).sum())
print('overlap                  :', ((eco_id > 0) & land).sum())
print('eco but NOT retention-land:', ((eco_id > 0) & ~land).sum())
print('retention-land but no eco :', ((eco_id == 0) & land).sum())
print()
print('keep rate on retention land:', keep[land].mean())

eco_valid = (eco_id > 0) & land
for b in (98, 99):
    eco_valid &= (eco_biome != b)
print('eco_valid cells          :', eco_valid.sum())
print('keep rate on eco_valid   :', keep[eco_valid].mean())
print()

# distribution of per-ecoregion coverage
lat = ds['lat'].values
w = np.broadcast_to(np.cos(np.deg2rad(lat))[:, None], eco_id.shape).copy()
nbin = int(eco_id.max()) + 1
fe, fv, fk, fw = eco_id.ravel(), eco_valid.ravel(), (keep & eco_valid).ravel(), w.ravel()
wt = np.bincount(fe[fv], weights=fw[fv], minlength=nbin)
wk = np.bincount(fe[fk], weights=fw[fk], minlength=nbin)
nt = np.bincount(fe[fv], minlength=nbin)
cov = np.where(wt > 0, wk / wt, np.nan)

ids = np.unique(eco_id[eco_valid]); ids = ids[ids > 0]
c = cov[ids]
print('n ecoregions             :', len(ids))
for thr in [0.1, 0.25, 0.5, 0.75, 0.9]:
    print(f'  coverage >= {thr:.2f}      : {(c >= thr).sum():4d}')
print('coverage percentiles     :',
      np.round(np.nanpercentile(c, [5, 25, 50, 75, 95]), 3))
print('n>=10 cells              :', (nt[ids] >= 10).sum())
