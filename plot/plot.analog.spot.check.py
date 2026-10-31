#!/bin/python
#
# SPOT-CHECK / read-back for analog_sets.nc. Two jobs:
#   (1) Map a handful of focal cells with their selected analog sets, to eyeball
#       that analogs are plausible climate twins AND genuinely spread out (this
#       catches indexing/coordinate bugs the coverage table cannot see).
#   (2) Summarize the s_used distribution per biome, so we know how much real
#       spread we got vs how often relaxation kicked in.
#
# Focal examples are chosen for contrast: the Texas cell, plus one
# each from a homogeneous biome, a patchy biome, and Rock&Ice (worst coverage).

import numpy as np
import xarray as xr
import os
import matplotlib
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

data_root = '/glade/campaign/ral/risc/jsallen/TNC'
out_dir = f'{data_root}/analog_stability'
sets_file = os.path.join(out_dir, 'analog_sets.nc')

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

# focal examples: (label, target_lat, target_lon). We snap to the nearest cell
# that actually has analogs
examples = [
    ('a) Texas ',        31.5, -99.0),
    ('b) Central Australia',-24.0, 134.0),
    ('c) Spain',      40.0,  -4.0),
    ('d) SE Asia',       3.0, 113.0),
]

print(f"Loading {sets_file} ...")
ds = xr.open_dataset(sets_file)
lat = ds['lat'].values; lon = ds['lon'].values
alat = ds['analog_lat'].values   # (nlat,nlon,N)
alon = ds['analog_lon'].values
asig = ds['analog_sigma'].values
nfound = ds['n_analogs'].values
sused = ds['s_used_km'].values
effn = ds['eff_n'].values
biome_id = ds['biome_id'].values
N = alat.shape[2]


def nearest_with_analogs(tlat, tlon):
    """Nearest cell to (tlat,tlon) that actually has >=1 analog."""
    i0 = np.argmin(np.abs(lat - tlat))
    j0 = np.argmin(np.abs(lon - tlon))
    if nfound[i0, j0] > 0:
        return i0, j0
    # spiral outward to find a populated neighbour
    for r in range(1, 25):
        for di in range(-r, r+1):
            for dj in range(-r, r+1):
                ii, jj = i0+di, j0+dj
                if 0 <= ii < len(lat) and 0 <= jj < len(lon) and nfound[ii, jj] > 0:
                    return ii, jj
    return i0, j0

pcrs = ccrs.Robinson()
tcrs = ccrs.PlateCarree()

def plot_background(ax):
    """ Cartopy ax helper """
    xticks = np.arange(-180,180,60)
    yticks = np.arange(-60,90,30)
    #ax.set_yticks(yticks, crs=pcrs)
    #ax.set_xticks(xticks, crs=pcrs)
    ax.set_extent([-180,180,-60,90])
    ax.add_feature(cfeature.COASTLINE.with_scale('110m'), linewidth=0.4)
    ax.add_feature(cfeature.OCEAN, facecolor='whitesmoke', zorder=2)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, alpha=0.7)
    return(ax)

# example analog-set maps
# -----------------------
nex = len(examples)
pcrs = ccrs.Robinson()
fig, axarr = plt.subplots(
        nrows=2, ncols=2, 
        figsize=(11, 6), 
        subplot_kw={'projection': pcrs})

fig.subplots_adjust(
        top=0.95, bottom=0.13, 
        left=0.05, right=0.95,
        hspace=0.18, wspace=0.08)

axlist = axarr.flatten()

for e, (label, tlat, tlon) in enumerate(examples):
    ax = axlist[e]
    fi, fj = nearest_with_analogs(tlat, tlon)
    flat, flon = lat[fi], lon[fj]
    m = int(nfound[fi, fj])
    bid = int(biome_id[fi, fj]) if np.isfinite(biome_id[fi, fj]) else -1
    set_lat = alat[fi, fj, :m]; set_lon = alon[fi, fj, :m]; set_sig = asig[fi, fj, :m]

    sc = ax.scatter(set_lon, set_lat, c=set_sig, cmap='viridis_r',
                    s=50, edgecolor='k', lw=0.5, vmin=0, vmax=1.0,
                    transform=tcrs, zorder=5)
    ax.scatter([flon], [flat], marker='*', s=200, c='red',
               edgecolor='k', lw=0.6, transform=tcrs, zorder=6)

    bname = biomes.get(bid, f'biome {bid}')
    ax.set_title(f'{label}  | {bname}\n'
                 f'n={m}  s_used={sused[fi,fj]:.0f}km  effN={effn[fi,fj]:.1f}',
                 fontsize=9)

    plot_background(ax)

cax = fig.add_axes([0.30,0.08,0.40,0.016])
cbar = plt.colorbar(sc, cax=cax, orientation='horizontal', label=r'Historical $\sigma$')

f1 = os.path.join(out_dir, 'analog_sets_spotcheck_maps.png')
fig.savefig(f1, dpi=400); plt.show()
print(f'Saved: {f1}')

# also print the Texas set to console for an exact-number sanity read
fi, fj = nearest_with_analogs(*examples[0][1:])
print(f'\n{examples[0][0]} focal ({lat[fi]:.2f},{lon[fj]:.2f}):')
for r in range(int(nfound[fi, fj])):
    print(f'  analog {r+1:2d}: ({alat[fi,fj,r]:7.2f},{alon[fi,fj,r]:8.2f})  '
          f'sigma={asig[fi,fj,r]:.3f}')

# JOB 2: s_used distribution per biome
# -------------------------------------
print('\n--- s_used distribution per biome (% of populated cells at each step) ---')
steps = [500.0, 400.0, 300.0, 250.0]
print(f'{"biome":<48} ' + ' '.join(f'{s:>6.0f}' for s in steps) + '   med_effN')
rows = []
for b, bname in biomes.items():
    bm = (biome_id == b) & (nfound > 0)
    if bm.sum() == 0:
        continue
    su = sused[bm]
    pct = [100.0*np.mean(np.isclose(su, s)) for s in steps]
    me = np.nanmedian(effn[bm])
    rows.append((b, pct, me))
    print(f'{bname:<48} ' + ' '.join(f'{p:>5.1f}%' for p in pct) + f'   {me:>6.1f}')

# bar figure of s_used composition
fig, ax = plt.subplots(figsize=(11, 6))
labels = [biomes[b][:26] for b, _, _ in rows]
bottoms = np.zeros(len(rows))
colors = ['#2c7bb6', '#abd9e9', '#fdae61', '#d7191c']
for si, s in enumerate(steps):
    vals = np.array([r[1][si] for r in rows])
    ax.barh(labels, vals, left=bottoms, color=colors[si], label=f'{s:.0f} km')
    bottoms += vals
ax.set_xlabel('% of populated cells'); ax.set_xlim(0, 100)
ax.legend(title='spacing used', loc='lower right', fontsize=8)
ax.set_title('Spacing actually used per biome (blue = full 500 km spread; '
             'red = relaxed to 250 km floor)', fontsize=10)
ax.invert_yaxis()
fig.tight_layout()
f2 = os.path.join(out_dir, 'analog_sets_spacing_composition.png')
fig.savefig(f2, dpi=400); plt.close(fig)
print(f'\nSaved: {f2}')

print('\nDone.')
