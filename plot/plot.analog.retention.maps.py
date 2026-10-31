#!/bin/python
#
# Retention + attribution MAPS from analog_retention.nc.
# Three figures:
#   (1) retention_maps.png   : retention at GWL2 & GWL3, strict (Dt=1) and
#       loose (Dt=2) thresholds. The headline "where do analogs survive".
#   (2) attribution_maps.png : magratio and dircong at GWL3, side by side, so
#       low-retention regions can be read as magnitude- vs direction-driven.
#   (3) retention_curves.png : retention(Dt) curves per biome, the honest
#       threshold-free object (no single Dt assumed).
#
# COVERAGE MASK: cells where selection was compromised (n_analogs < N_FULL or
# s_used < 500 km) are drawn in light grey, NOT colored. This separates the
# retention signal (analogs that diverged) from selection-stage scarcity
# (analogs that were marginal to begin with) -- the confound we flagged.
# Set MASK_LOW_COVERAGE=False to show everything.

import numpy as np
import xarray as xr
import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import colormaps as cmaps
HAVE_CARTOPY = True

data_root = '/glade/campaign/ral/risc/jsallen/TNC'
out_dir = f'{data_root}/analog_stability'
ret_file = os.path.join(out_dir, 'analog_retention.nc')
sets_file = os.path.join(out_dir, 'analog_sets.nc')

MASK_LOW_COVERAGE = True
N_FULL = 10           # full analog set size
S_FULL = 500.0        # full spacing (cells below this were relaxed)
DT_STRICT, DT_LOOSE = 1.0, 2.0

print(f"Loading {ret_file} ...")
ds = xr.open_dataset(ret_file)
lat = ds['lat'].values; lon = ds['lon'].values
Dt = ds['Dt'].values
biome_id = ds['biome_id'].values
n_analogs = ds['n_analogs'].values

sets = xr.open_dataset(sets_file)
s_used = sets['s_used_km'].values

# coverage mask: True = KEEP (well covered)
if MASK_LOW_COVERAGE:
    keep = (n_analogs >= N_FULL) & (s_used >= S_FULL)
else:
    keep = np.isfinite(n_analogs) & (n_analogs > 0)
land = np.isfinite(biome_id) & (biome_id > 0)


def di_of(dt_val):
    return int(np.argmin(np.abs(Dt - dt_val)))


def masked_field(arr2d):
    """Return arr with non kept land cells set NaN (so they draw as the
    'bad' grey under-color), ocean stays fully transparent."""
    out = np.full_like(arr2d, np.nan, dtype=float)
    sel = keep & np.isfinite(arr2d)
    out[sel] = arr2d[sel]
    return out

def plot_background(ax):
    """ Cartopy ax settings helper """
    xticks = np.arange(-180,180,60)
    yticks = np.arange(-60,90,30)
    #ax.set_yticks(yticks, crs=pcrs)
    #ax.set_xticks(xticks, crs=pcrs)
    ax.set_extent([-180,180,-60,90])
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.4, zorder=3)
    ax.add_feature(cfeature.OCEAN, facecolor='whitesmoke', zorder=2)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.7, zorder=3)
    return(ax)

def add_map(ax, field, cmap, vmin, vmax, title, cbar_label, greylayer=True):

    #ax.set_global()
    #ax.add_feature(cfeature.OCEAN, facecolor='white', zorder=0)

    if greylayer:
        # light-grey backdrop on land so masked land reads as "covered but
        # excluded", distinct from ocean
        grey = np.where(land, 0.85, np.nan)
        ax.pcolormesh(lon, lat, grey, cmap=ListedColormap(['0.85']),
                      vmin=0, vmax=1, transform=ccrs.PlateCarree(), zorder=1)
    cm = plt.get_cmap(cmap).copy()
    pm = ax.pcolormesh(lon, lat, field, cmap=cm, vmin=vmin, vmax=vmax,
                       transform=ccrs.PlateCarree(), zorder=2)
    #ax.add_feature(cfeature.COASTLINE, lw=0.3, zorder=3)
    plot_background(ax)

    ax.set_title(title, fontsize=10)
    #cb = plt.colorbar(pm, ax=ax, shrink=0.65, pad=0.02)
    #cb.set_label(cbar_label, fontsize=8)
    return(pm)

proj = dict(projection=ccrs.Robinson()) if HAVE_CARTOPY else {}

# ==============================
# FIG 1: retention maps (2 GWL x 2 Dt)
# ==============================
fig, axes = plt.subplots(2, 2, figsize=(11, 6), subplot_kw=proj)
panels = [
    ('gwl2', DT_STRICT, axes[0, 0]), ('gwl2', DT_LOOSE, axes[0, 1]),
    ('gwl3', DT_STRICT, axes[1, 0]), ('gwl3', DT_LOOSE, axes[1, 1]),
]

i = 0
let = ['a)','b)','c)','d)']
for g, dt_val, ax in panels:
    fld = masked_field(ds[f'retention_{g}'].values[:, :, di_of(dt_val)])
    pm = add_map(ax, fld, cmaps.precip_diff_12lev, 0.0, 1.0,
            f'{let[i]} {g.upper()}  retention at Dt={dt_val}',
            'fraction of analogs retained')

    fig.subplots_adjust(
        top=0.95, bottom=0.13,
        left=0.05, right=0.95,
        hspace=0.18, wspace=0.08)

    cax = fig.add_axes([0.30,0.08,0.40,0.016])
    cbar = plt.colorbar(pm, cax=cax, orientation='horizontal', label=r'Fraction of analogs retained')
    i+=1

#fig.suptitle('Analog retention: fraction of present-day analogs still analogs under warming\n'
#             '(grey = low-coverage cells excluded)', fontsize=12)

#fig.tight_layout()
f1 = os.path.join(out_dir, 'retention_maps.png')
fig.savefig(f1, dpi=400, bbox_inches='tight'); plt.close(fig)
print(f'Saved: {f1}')

# ==============================
# FIG 2: attribution at GWL3 (magnitude vs direction)
# ==============================
fig, axes = plt.subplots(1, 2, figsize=(15, 4.5), subplot_kw=proj)
# magratio: clip to a readable range; ~1 means analog changes as much as focal
magr = ds['magratio_gwl3'].values.copy()
magr = np.clip(magr, 0, 2)
add_map(axes[0], masked_field(magr), 'PuOr', 0.0, 2.0,
        'GWL3 magnitude ratio  |dA|/|dF|  (1 = same pace)',
        '|delta_analog| / |delta_focal|')
# dircong: -1..1, 1 = same direction
add_map(axes[1], masked_field(ds['dircong_gwl3'].values), 'RdBu', -1.0, 1.0,
        'GWL3 direction congruence  cos(dF,dA)  (1 = same way)',
        'cosine of change-vector angle')
fig.suptitle('Why retention fails: magnitude vs direction of change (GWL3)', fontsize=12)
fig.tight_layout()
f2 = os.path.join(out_dir, 'attribution_maps.png')
fig.savefig(f2, dpi=130, bbox_inches='tight'); plt.close(fig)
print(f'Saved: {f2}')

# ==============================
# FIG 3: retention(Dt) curves per biome (threshold-free)
# ==============================
biomes = {
    1:'Boreal', 2:'Deserts', 3:'Flooded Grasslands', 5:'Mediterranean',
    6:'Montane Grass', 7:'Rock & Ice', 8:'Temperate Broadleaf',
    9:'Temperate Conifer', 10:'Temperate Grass', 11:'Trop Coniferous',
    12:'Trop Dry Broadleaf', 13:'Trop Grasslands', 14:'Trop Moist Broadleaf',
    15:'Tundra',
}
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
cmap = plt.get_cmap('tab20')
for ax, g in zip(axes, ['gwl2', 'gwl3']):
    ret = ds[f'retention_{g}'].values   # (nlat,nlon,nDt)
    for ci, (b, bn) in enumerate(biomes.items()):
        bm = (biome_id == b) & keep
        if bm.sum() < 50:
            continue
        curve = np.nanmean(ret[bm], axis=0)   # mean over cells -> (nDt,)
        ax.plot(Dt, curve, color=cmap(ci % 20), lw=1.4, label=bn)
    ax.set_title(f'{g.upper()}'); ax.set_xlabel('Dt (sigma threshold)')
    ax.set_ylim(0, 1); ax.grid(alpha=0.3)
axes[0].set_ylabel('mean fraction retained')
axes[1].legend(fontsize=7, ncol=1, loc='lower right')
fig.suptitle('Retention vs threshold, per biome (well-covered cells only)', fontsize=12)
fig.tight_layout()
f3 = os.path.join(out_dir, 'retention_curves.png')
fig.savefig(f3, dpi=400, bbox_inches='tight'); plt.close(fig)
print(f'Saved: {f3}')

# ==============================
# console: how much area is masked per biome
# ==============================
print('\n--- coverage retained per biome (% of land cells kept) ---')
for b, bn in biomes.items():
    bm = (biome_id == b) & land
    if bm.sum() == 0:
        continue
    pct = 100.0 * np.mean(keep[bm])
    print(f'  {bn:<22} {pct:>5.1f}% kept')

print('\nDone.')
