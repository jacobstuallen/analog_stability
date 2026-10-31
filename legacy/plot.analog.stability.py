#!/bin/python
# Summary figure: analog stability under global warming levels.
#
# Layout:
#   Top row:    two global maps, ensemble-mean sigma dissimilarity to the
#               historical best within-biome analog, at GWL2 and GWL3.
#   Bottom:     per-biome comparison. For each biome, the median ensemble-mean
#               sigma at GWL2 and GWL3 (points), ordered by GWL3 magnitude,
#               with horizontal bars showing the biome's 10th-90th ensemble
#               spread (median across cells of the per-cell p10/p90).
#
# Reads analog_stability_gwl.nc produced by analog_stability_gwl.py.

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib.lines import Line2D
import cartopy.crs as ccrs
import cartopy.feature as cfeature

data_root = '/glade/campaign/ral/risc/jsallen/TNC'
infile = f'{data_root}/analog_stability/analog_stability_gwl.nc'
outfile = './figures/figure_stability_summary.png'

vmax = 4.0   # sigma colour ceiling; 2sigma=moderate, 4sigma=extreme (Mahony)

biomes = {
    1:'Boreal Forests/Taiga', 2:'Deserts & Xeric Shrublands',
    3:'Flooded Grasslands & Savannas', 4:'Mangroves',
    5:'Mediterranean Forests', 6:'Montane Grasslands & Shrublands',
    7:'Rock and Ice', 8:'Temperate Broadleaf & Mixed Forests',
    9:'Temperate Conifer Forests', 10:'Temperate Grasslands & Savannas',
    11:'Trop. & Subtrop. Coniferous Forests', 12:'Trop. & Subtrop. Dry Broadleaf Forests',
    13:'Trop. & Subtrop. Grasslands & Savannas',
    14:'Trop. & Subtrop. Moist Broadleaf Forests', 15:'Tundra',
}

# ---- font (from your scripts) ----
try:
    fdir = '/glade/work/jsallen/conda-envs/earth/fonts/Avenir-Medium.otf'
    font_manager.fontManager.addfont(fdir)
    prop = font_manager.FontProperties(fname=fdir)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = prop.get_name()
except Exception:
    pass
plt.rcParams['axes.linewidth'] = 0.6
plt.rcParams['xtick.major.width'] = 0.6
plt.rcParams['ytick.major.width'] = 0.6

ds = xr.open_dataset(infile)
lat = ds['lat'].values; lon = ds['lon'].values
s2 = ds['sigma_gwl2_mean'].values
s3 = ds['sigma_gwl3_mean'].values
s2_p10 = ds['sigma_gwl2_p10'].values; s2_p90 = ds['sigma_gwl2_p90'].values
s3_p10 = ds['sigma_gwl3_p10'].values; s3_p90 = ds['sigma_gwl3_p90'].values
bio = ds['biome_id'].values
LON, LAT = np.meshgrid(lon, lat)

n_models = ds.attrs.get('n_models', '?')
varlist = ds.attrs.get('variables', '')

# ---- per-biome medians + spread ----
rows = []
for b, name in biomes.items():
    m = (bio == b)
    v2 = s2[m]; v3 = s3[m]
    v2 = v2[np.isfinite(v2)]; v3 = v3[np.isfinite(v3)]
    if v2.size == 0:
        continue
    rows.append({
        'name': name,
        'med2': np.median(v2), 'med3': np.median(v3),
        'p10_2': np.nanmedian(s2_p10[m]), 'p90_2': np.nanmedian(s2_p90[m]),
        'p10_3': np.nanmedian(s3_p10[m]), 'p90_3': np.nanmedian(s3_p90[m]),
    })
rows.sort(key=lambda r: r['med3'])   # ascending; largest at top after plotting

# ==============================
# Figure
# ==============================
fig = plt.figure(figsize=(11, 10))
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15],
                      hspace=0.20, wspace=0.08,
                      left=0.06, right=0.94, top=0.95, bottom=0.07)

cmap = 'magma'
proj = ccrs.Miller()

def draw_map(ax, field, title):
    ax.add_feature(cfeature.LAND, facecolor='#e8e8e8', zorder=0)
    sc = ax.scatter(LON[np.isfinite(field)], LAT[np.isfinite(field)],
                    c=field[np.isfinite(field)], s=0.6, cmap=cmap,
                    vmin=0, vmax=vmax, transform=ccrs.PlateCarree(), zorder=1)
    ax.add_feature(cfeature.COASTLINE, lw=0.25, zorder=2)
    ax.set_global()
    ax.set_title(title, loc='left', fontsize=11)
    return sc

axA = fig.add_subplot(gs[0, 0], projection=proj)
scA = draw_map(axA, s2, 'a) Analog dissimilarity at GWL2 (+2°C)')
axB = fig.add_subplot(gs[0, 1], projection=proj)
scB = draw_map(axB, s3, 'b) Analog dissimilarity at GWL3 (+3°C)')

cax = fig.add_axes([0.30, 0.590, 0.40, 0.013])
cb = fig.colorbar(scB, cax=cax, orientation='horizontal', extend='max')
cb.set_label(r'$\sigma$ dissimilarity to historical analog', fontsize=9)
cb.ax.tick_params(labelsize=8)

# ---- bottom: per-biome comparison ----
axC = fig.add_subplot(gs[1, :])
y = np.arange(len(rows))
c2, c3 = '#f4a582', '#b2182b'

for i, r in enumerate(rows):
    # spread bars (10-90 ensemble) behind the points
    axC.plot([r['p10_2'], r['p90_2']], [i-0.13, i-0.13], color=c2,
             lw=1.0, alpha=0.5, zorder=1)
    axC.plot([r['p10_3'], r['p90_3']], [i+0.13, i+0.13], color=c3,
             lw=1.0, alpha=0.5, zorder=1)
    # connector
    axC.plot([r['med2'], r['med3']], [i-0.13, i+0.13], color='#999999',
             lw=0.7, alpha=0.6, zorder=2)

axC.scatter([r['med2'] for r in rows], y-0.13, s=42, c=c2, zorder=3,
            edgecolors='white', linewidths=0.5, label='GWL2 (+2°C)')
axC.scatter([r['med3'] for r in rows], y+0.13, s=42, c=c3, zorder=3,
            edgecolors='white', linewidths=0.5, label='GWL3 (+3°C)')

# reference lines at 1, 2, 4 sigma
for xv, lab in [(1, r'1$\sigma$'), (2, r'2$sigma$'), (4, r'4$sigma$')]:
    axC.axvline(xv, color='k', lw=0.5, ls=':', alpha=0.4, zorder=0)
    axC.text(xv, len(rows)-0.3, lab, fontsize=7, alpha=0.5, ha='center')

axC.set_yticks(y)
axC.set_yticklabels([r['name'] for r in rows], fontsize=8)
axC.set_ylim(-0.6, len(rows)-0.4)
axC.set_xlabel(r'$\sigma$ dissimilarity to historical best analog (biome median)', fontsize=9)
axC.set_title('c) Biome analog stability, ordered by GWL3 dissimilarity '
              '(bars: 10-90th ensemble spread)', loc='left', fontsize=11)
axC.legend(fontsize=8, loc='lower right', framealpha=0.9)
axC.grid(axis='x', alpha=0.25)
axC.tick_params(labelsize=8)

fig.suptitle(f'Analog stability under global warming {varlist} '
             f'({n_models} CMIP6 models, SSP5-8.5)', fontsize=12, y=0.985)

import os
os.makedirs('./figures', exist_ok=True)
plt.savefig(outfile, dpi=300, bbox_inches='tight')
print(f'Saved: {outfile}')
plt.close()
