"""
Three-panel diagnostic:
  (a) Biomes
  (b) Ecoregions, with biome fill + ecoregion boundaries overlaid
  (c) log10(number of 0.25 deg cells per ecoregion), which sets the floor on
      which ecoregions can support a fraction-based analog statistic.

Panel (b) is the one that answers the nesting question directly: the ecoregion
outlines should never cut across a biome color boundary.
"""

import json
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.font_manager as font_manager

font_dir = '/glade/work/jsallen/conda-envs/earth/fonts/Avenir-Medium.otf'
font_manager.fontManager.addfont(font_dir)
prop = font_manager.FontProperties(fname=font_dir)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = prop.get_name()

pcrs = ccrs.Robinson()
tcrs = ccrs.PlateCarree()

PATH = '/glade/u/home/jsallen/projects/tnc_2025/analogs/biomes/'
FILE = 'ecoregions.analog.gridded.nc'

#ds = xr.open_dataset(PATH + FILE)
ds = xr.open_dataset(FILE)
eco = ds['ECO_ID']
biome = ds['BIOME_ID']

eco_lookup = {int(k): v for k, v in json.loads(ds.attrs['eco_lookup']).items()}
biome_lookup = {int(k): v for k, v in json.loads(ds.attrs['biome_lookup']).items()}

lat = ds.lat.values
lon = ds.lon.values
X, Y = np.meshgrid(lon, lat)

eco_m = np.where(eco.values > 0, eco.values, np.nan)
biome_m = np.where(biome.values > 0, biome.values, np.nan)

# WWF_MHTNUM is NOT contiguous: 1-14 are the vegetated biomes, then 98 = Inland
# Water and 99 = Rock and Ice. Those two are excluded from the analog analysis
# (Rock and Ice alone is ~25% of the land grid and is not an analog pool), but
# they are drawn in grey here so the coverage gaps are visible.
biome_colors = {
    1:  '#228B22',   # Tropical & Subtropical Moist Broadleaf Forests
    2:  '#8FAF4D',   # Tropical & Subtropical Dry Broadleaf Forests
    3:  '#3D6B3D',   # Tropical & Subtropical Coniferous Forests
    4:  '#4A7C59',   # Temperate Broadleaf & Mixed Forests
    5:  '#1F5E3D',   # Temperate Conifer Forests
    6:  '#2D5016',   # Boreal Forests/Taiga
    7:  '#E8954A',   # Tropical & Subtropical Grasslands, Savannas & Shrublands
    8:  '#D4A837',   # Temperate Grasslands, Savannas & Shrublands
    9:  '#6BCDCD',   # Flooded Grasslands & Savannas
    10: '#9B6B9E',   # Montane Grasslands & Shrublands
    11: '#8B7FA8',   # Tundra
    12: '#C17E5D',   # Mediterranean Forests, Woodlands & Scrub
    13: '#E8A965',   # Deserts & Xeric Shrublands
    14: '#1B4D3E',   # Mangroves
    98: '#3A4A5A',   # Inland Water   (excluded from analysis)
    99: '#6E6E76',   # Rock and Ice   (excluded from analysis)
}

EXCLUDE_BIOMES = [98, 99]

biome_ids = sorted(biome_lookup.keys())

# Map the non-contiguous IDs onto contiguous positions 0..N-1 so the colormap
# indexes correctly. Do NOT feed raw IDs to BoundaryNorm.
biome_pos = {b: i for i, b in enumerate(biome_ids)}
biome_field = np.full(biome_m.shape, np.nan)
for b, i in biome_pos.items():
    biome_field[biome.values == b] = i

cmap_biome = ListedColormap([biome_colors[b] for b in biome_ids])
norm_biome = BoundaryNorm(np.arange(len(biome_ids) + 1) - 0.5, cmap_biome.N)


def basemap(ax, title):
    ax.set_global()
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidths=0.4)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'),
                   linewidths=0.25, alpha=0.4)
    ax.set_title(title,  loc='left')
    return ax


#plt.style.use('dark_background')
fig = plt.figure(figsize=(10, 6))
fig.subplots_adjust(left=0.04, right=0.96, top=0.96, bottom=0.08, hspace=0.10, wspace=0.10)

# ---- (a) Biomes -------------------------------------------------------------
ax1 = fig.add_subplot(221, projection=pcrs)
basemap(ax1, f"(a) Biomes (WWF major habitat types, n={len(biome_ids)})")
ax1.pcolormesh(X, Y, biome_field, cmap=cmap_biome, norm=norm_biome,
               transform=tcrs, shading='nearest', rasterized=True)

# ---- (b) Ecoregions over biome fill ----------------------------------------
# Boundaries are drawn as contour lines on the ECO_ID field. Any change in
# ECO_ID between adjacent cells produces an edge, so this traces every
# ecoregion outline without needing the vector geometry.
ax2 = fig.add_subplot(222, projection=pcrs)
basemap(ax2, f"(b) Ecoregion boundaries over biome fill (n={len(eco_lookup)})")
ax2.pcolormesh(X, Y, biome_field, cmap=cmap_biome, norm=norm_biome,
               transform=tcrs, shading='nearest', rasterized=True, alpha=0.75)

# Edge detection on the ecoregion index
e = np.where(np.isnan(eco_m), -1, eco_m)
edge = np.zeros_like(e, dtype=bool)
edge[:, :-1] |= (e[:, :-1] != e[:, 1:])
edge[:-1, :] |= (e[:-1, :] != e[1:, :])
edge &= (e > 0)
edge_plot = np.where(edge, 1.0, np.nan)

ax2.pcolormesh(X, Y, edge_plot, cmap=ListedColormap(['#000000']),
               transform=tcrs, shading='nearest', rasterized=True, alpha=0.85)

# ---- (c) Cells per ecoregion ------------------------------------------------
ax3 = fig.add_subplot(223, projection=pcrs)

ids, counts = np.unique(eco.values[eco.values > 0], return_counts=True)
count_map = dict(zip(ids.tolist(), counts.tolist()))
size_field = np.full(eco_m.shape, np.nan)
for i, c in count_map.items():
    size_field[eco.values == i] = np.log10(c)

basemap(ax3, "(c) log$_{10}$(0.25$\\degree$ cells per ecoregion)")
cf = ax3.pcolormesh(X, Y, size_field, cmap='magma', vmin=1, vmax=4,
                    transform=tcrs, shading='nearest', rasterized=True)

cax = fig.add_axes([0.11, 0.08, 0.30, 0.012])
cb = plt.colorbar(cf, cax=cax, orientation='horizontal')
cb.set_label('log$_{10}$(n cells)', fontsize=9)
cb.ax.tick_params(labelsize=8)

# ---- Legend -----------------------------------------------------------------
handles = [Patch(facecolor=biome_colors[b], edgecolor='none') for b in biome_ids]
labels = [biome_lookup[b] for b in biome_ids]
fig.legend(handles=handles, labels=labels, loc='upper left',
           bbox_to_anchor=(0.55, 0.53), ncols=1, fontsize=8,
           labelspacing=0.6, columnspacing=0.8, frameon=False)

plt.savefig('biomes.ecoregions.overview.png', dpi=400, bbox_inches='tight')
plt.show()
plt.close()

# ---- Diagnostics printed for the aggregation decision -----------------------
print(f"Ecoregions on grid: {len(count_map)} / {len(eco_lookup)} in shapefile")
print(f"  < 5 cells:  {(counts < 5).sum()}")
print(f"  < 10 cells: {(counts < 10).sum()}")
print(f"  < 25 cells: {(counts < 25).sum()}")
print(f"  median: {np.median(counts):.0f}   max: {counts.max()}")

print("\nSmallest ecoregions on the grid:")
for i in np.argsort(counts)[:10]:
    eid = int(ids[i])
    print(f"  {counts[i]:5d}  {eco_lookup[eid]['eco_name']}"
          f"  [{eco_lookup[eid]['biome_name']}]")

print("\nEcoregions per biome (n polys / n grid cells):")
for b in biome_ids:
    n = sum(1 for e in eco_lookup.values() if e['biome_id'] == b)
    ncell = int((biome.values == b).sum())
    flag = '  [EXCLUDED]' if b in EXCLUDE_BIOMES else ''
    print(f"  {b:2d}  {n:4d}  {ncell:7d}  {biome_lookup[b]}{flag}")

# Ecoregions that survive a minimum-cell-count screen. Below ~10 cells a
# fraction-retained statistic is not meaningful; these get masked in the
# aggregated analog figures.
MIN_CELLS = 10
keep = [int(i) for i, c in zip(ids, counts)
        if c >= MIN_CELLS and eco_lookup[int(i)]['biome_id'] not in EXCLUDE_BIOMES]
print(f"\nEcoregions passing MIN_CELLS={MIN_CELLS} and biome screen: "
      f"{len(keep)} / {len(eco_lookup)}")
