#!/bin/python
#
# Ecoregion-aggregated analog retention maps.
#
# Same figure as retention_maps.png (2 GWL x 2 Dt), but every grid cell is
# replaced by the area-weighted mean retention of the ecoregion it belongs to.
#
# AGGREGATION IS POST-HOC. The sigma-dissimilarity / analog selection is still
# done at 0.25 deg on a per-cell basis. We only average the DERIVED retention
# metric up to the ecoregion. Aggregating the climate indices before the analog
# calculation would destroy the local ICV that sigma dissimilarity is defined
# against.
#
# TWO SCREENS, both must pass, else the ecoregion is drawn grey:
#   1. COVERAGE: >= MIN_COVERAGE of the ecoregion's land cells pass the
#      per-cell `keep` mask (n_analogs >= N_FULL and s_used >= S_FULL).
#   2. SIZE: the ecoregion has >= MIN_CELLS land cells at 0.25 deg. Guards
#      against a "67% covered" ecoregion whose mean rests on 2 cells.
#
# Grey therefore means "we do not trust the aggregate here", NOT "no data".
# Coverage is a selection-quality screen: cells can fail it because their
# analogs were marginal to begin with, which is a different failure mode from
# analogs that were good and then diverged.

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

# ------------------------------------------------------------------
# Load
# ------------------------------------------------------------------
print(f"Loading {ret_file} ...")
ds = xr.open_dataset(ret_file)
lat = ds['lat'].values
lon = ds['lon'].values
Dt = ds['Dt'].values
biome_id = ds['biome_id'].values
n_analogs = ds['n_analogs'].values

sets = xr.open_dataset(sets_file)
s_used = sets['s_used_km'].values

eco_ds = xr.open_dataset(eco_file, mask_and_scale=False)

# Align to the analog grid. The ecoregion file was rasterized against a
# descending-lat reference; the retention grid is ascending.
if eco_ds['lat'].values[0] > eco_ds['lat'].values[-1]:
    eco_ds = eco_ds.isel(lat=slice(None, None, -1))


eco_id_raw = eco_ds['ECO_ID'].values
eco_biome_raw = eco_ds['BIOME_ID'].values

# Coerce to int with 0 = "no ecoregion". Handles both the int32 fill=-1 case
# and the case where CF decoding already turned the fill into NaN (float).
eco_id = np.where(np.isfinite(eco_id_raw) & (eco_id_raw > 0),
                  eco_id_raw, 0).astype(np.int64)
eco_biome = np.where(np.isfinite(eco_biome_raw) & (eco_biome_raw > 0),
                     eco_biome_raw, 0).astype(np.int64)

eco_lookup = {int(k): v for k, v in json.loads(eco_ds.attrs['eco_lookup']).items()}

if eco_id.shape != biome_id.shape:
    raise ValueError(
        f"Grid mismatch: ecoregions {eco_id.shape} vs retention {biome_id.shape}. "
        "Rerun ecoregion_shp_to_reference.py against the analog reference grid."
    )

# per-cell masks, identical definition to plot.analog.retention.maps.py
keep = (n_analogs >= N_FULL) & (s_used >= S_FULL)
land = np.isfinite(biome_id) & (biome_id > 0)

# ecoregion domain: valid ecoregion, on the analog land mask, in a usable biome
eco_valid = (eco_id > 0) & land
for b in EXCLUDE_BIOMES:
    eco_valid &= (eco_biome != b)

# cos(lat) area weights. Ecoregions such as Boreal/Tundra span large latitude
# ranges; an unweighted mean would overweight their poleward cells.
w2d = np.broadcast_to(np.cos(np.deg2rad(lat))[:, None], eco_id.shape).copy()

# ------------------------------------------------------------------
# Ecoregion screens
# ------------------------------------------------------------------
ids = np.unique(eco_id[eco_valid])
ids = ids[ids > 0]

flat_eco = eco_id.ravel()
flat_valid = eco_valid.ravel()
flat_keep = (keep & eco_valid).ravel()
flat_w = w2d.ravel()

nbin = int(eco_id.max()) + 1

# weighted cell counts per ecoregion
w_total = np.bincount(flat_eco[flat_valid], weights=flat_w[flat_valid], minlength=nbin)
w_kept = np.bincount(flat_eco[flat_keep], weights=flat_w[flat_keep], minlength=nbin)
n_total = np.bincount(flat_eco[flat_valid], minlength=nbin)   # unweighted, for MIN_CELLS

with np.errstate(invalid='ignore', divide='ignore'):
    coverage = np.where(w_total > 0, w_kept / w_total, np.nan)

passes = np.zeros(nbin, dtype=bool)
for e in ids:
    e = int(e)
    passes[e] = (n_total[e] >= MIN_CELLS) and (coverage[e] >= MIN_COVERAGE)

print(f"Ecoregions in domain: {len(ids)}")
print(f"  fail MIN_CELLS={MIN_CELLS}:      "
      f"{sum(1 for e in ids if n_total[int(e)] < MIN_CELLS)}")
print(f"  fail MIN_COVERAGE={MIN_COVERAGE}: "
      f"{sum(1 for e in ids if n_total[int(e)] >= MIN_CELLS and coverage[int(e)] < MIN_COVERAGE)}")
print(f"  passing both:                    {int(passes[ids.astype(int)].sum())}")


def aggregate_to_ecoregion(field2d):
    """Area-weighted ecoregion mean of a per-cell field, painted back onto the
    grid. Only cells passing `keep` contribute to the mean. Ecoregions failing
    either screen are returned as NaN so they draw as the grey backdrop."""
    valid = flat_keep & np.isfinite(field2d.ravel())
    f = field2d.ravel()

    wsum = np.bincount(flat_eco[valid], weights=(f[valid] * flat_w[valid]), minlength=nbin)
    wcnt = np.bincount(flat_eco[valid], weights=flat_w[valid], minlength=nbin)

    with np.errstate(invalid='ignore', divide='ignore'):
        means = np.where(wcnt > 0, wsum / wcnt, np.nan)
    means[~passes] = np.nan

    out = np.full(field2d.shape, np.nan, dtype=float)
    sel = eco_valid
    out[sel] = means[eco_id[sel]]
    return out


def di_of(dt_val):
    return int(np.argmin(np.abs(Dt - dt_val)))


# ------------------------------------------------------------------
# Plot: identical styling to plot.analog.retention.maps.py
# ------------------------------------------------------------------
def plot_background(ax):
    ax.set_extent([-180, 180, -60, 90])
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.4, zorder=3)
    ax.add_feature(cfeature.OCEAN, facecolor='whitesmoke', zorder=2)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.7, zorder=3)
    return ax


def add_map(ax, field, cmap, vmin, vmax, title, greylayer=True):
    if greylayer:
        grey = np.where(land, 0.85, np.nan)
        ax.pcolormesh(lon, lat, grey, cmap=ListedColormap(['0.85']),
                      vmin=0, vmax=1, transform=ccrs.PlateCarree(), zorder=1)
    cm = plt.get_cmap(cmap).copy() if isinstance(cmap, str) else cmap
    pm = ax.pcolormesh(lon, lat, field, cmap=cm, vmin=vmin, vmax=vmax,
                       transform=ccrs.PlateCarree(), zorder=2)
    plot_background(ax)
    ax.set_title(title, fontsize=10)
    return pm


proj = dict(projection=ccrs.Robinson())

fig, axes = plt.subplots(2, 2, figsize=(11, 5.5), subplot_kw=proj)
panels = [
    ('gwl2', DT_STRICT, axes[0, 0]), ('gwl2', DT_LOOSE, axes[0, 1]),
    ('gwl3', DT_STRICT, axes[1, 0]), ('gwl3', DT_LOOSE, axes[1, 1]),
]

let = ['a)', 'b)', 'c)', 'd)']
for i, (g, dt_val, ax) in enumerate(panels):
    cell = ds[f'retention_{g}'].values[:, :, di_of(dt_val)]
    fld = aggregate_to_ecoregion(cell)
    pm = add_map(ax, fld, cmaps.precip_diff_12lev, 0.0, 1.0,
                 f'{let[i]} {g.upper()}  retention at Dt={dt_val}')

fig.subplots_adjust(top=0.95, bottom=0.13, left=0.05, right=0.95,
                    hspace=0.14, wspace=0.08)
cax = fig.add_axes([0.30, 0.08, 0.40, 0.016])
plt.colorbar(pm, cax=cax, orientation='horizontal',
             label=r'Fraction of analogs retained (ecoregion mean)')

f1 = os.path.join(out_dir, 'retention_maps_ecoregion.png')
fig.savefig(f1, dpi=400, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {f1}')

# ------------------------------------------------------------------
# console: which ecoregions were dropped, and why
# ------------------------------------------------------------------
print(f'\n--- ecoregions excluded (grey) ---')
dropped = [(int(e), n_total[int(e)], coverage[int(e)])
           for e in ids if not passes[int(e)]]
dropped.sort(key=lambda t: t[2] if np.isfinite(t[2]) else -1)
for e, n, cov in dropped[:25]:
    why = []
    if n < MIN_CELLS:
        why.append(f'n={n}')
    if not (cov >= MIN_COVERAGE):
        why.append(f'cov={cov:.0%}')
    print(f"  {eco_lookup[e]['eco_name'][:44]:<44} "
          f"[{eco_lookup[e]['biome_name'][:26]:<26}]  {', '.join(why)}")
if len(dropped) > 25:
    print(f'  ... and {len(dropped) - 25} more')

print(f'\n--- ecoregion coverage by biome (% of ecoregions retained) ---')
bl = {int(k): v for k, v in json.loads(eco_ds.attrs['biome_lookup']).items()}
for b in sorted(bl):
    if b in EXCLUDE_BIOMES:
        continue
    ein = [int(e) for e in ids if eco_lookup[int(e)]['biome_id'] == b]
    if not ein:
        continue
    npass = sum(passes[e] for e in ein)
    print(f'  {bl[b][:44]:<44} {npass:>3d}/{len(ein):<3d} '
          f'({100.0 * npass / len(ein):>5.1f}%)')

print('\nDone.')
