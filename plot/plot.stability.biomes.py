#!/bin/python
# Figure 2: per-biome analog dissimilarity, GWL2 -> GWL3 dumbbell chart.

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib.lines import Line2D

data_root = '/glade/campaign/ral/risc/jsallen/TNC'
infile = f'{data_root}/analog_stability/analog_stability_gwl.nc'
outfile = './figures/figure_stability_biomes.png'

# clip the displayed spread at 0 (signed sigma below 0 = sub-1sigma dense-pool
# cells; negative values are honest but read oddly on a dissimilarity axis).
clip_zero = True

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
s2 = ds['sigma_gwl2_mean'].values
s3 = ds['sigma_gwl3_mean'].values
s3_p10 = ds['sigma_gwl3_p10'].values
s3_p90 = ds['sigma_gwl3_p90'].values
bio = ds['biome_id'].values
n_models = ds.attrs.get('n_models', '?')
varlist = ds.attrs.get('variables', '')

rows = []
for b, name in biomes.items():
    m = (bio == b)
    v2 = s2[m]; v3 = s3[m]
    v2 = v2[np.isfinite(v2)]; v3 = v3[np.isfinite(v3)]
    if v2.size == 0:
        continue
    p10 = np.nanmedian(s3_p10[m]); p90 = np.nanmedian(s3_p90[m])
    if clip_zero:
        p10 = max(p10, 0.0)
    rows.append({'name': name, 'med2': np.median(v2), 'med3': np.median(v3),
                 'p10': p10, 'p90': p90, 'jump': np.median(v3)-np.median(v2)})
rows.sort(key=lambda r: r['med3'])

y = np.arange(len(rows))
c2, c3 = '#7fb3d5', '#922b21'
band = '#922b21'

fig, ax = plt.subplots(figsize=(8.5, 7))

for i, r in enumerate(rows):
    # GWL3 ensemble spread as a light band
    ax.plot([r['p10'], r['p90']], [i, i], color=band, lw=4, alpha=0.15,
            solid_capstyle='round', zorder=1)
    # connector GWL2 -> GWL3
    ax.plot([r['med2'], r['med3']], [i, i], color='#bbbbbb', lw=1.4, zorder=2)

ax.scatter([r['med2'] for r in rows], y, s=55, c=c2, zorder=3,
           edgecolors='white', linewidths=0.6)
ax.scatter([r['med3'] for r in rows], y, s=55, c=c3, zorder=3,
           edgecolors='white', linewidths=0.6)

for xv in [1, 2, 3]:
    ax.axvline(xv, color='k', lw=0.5, ls=':', alpha=0.3, zorder=0)

ax.set_yticks(y)
ax.set_yticklabels([r['name'] for r in rows], fontsize=9)
ax.set_ylim(-0.6, len(rows)-0.4)
ax.set_xlim(left=0)
ax.set_xlabel(r'\$sigma$ dissimilarity to historical best analog (biome median)', fontsize=10)
ax.set_title(f'Biome analog stability, GWL2 - GWL3  ({varlist}, '
             f'{n_models} models)\nband: GWL3 10-90th ensemble spread',
             loc='left', fontsize=11)

legend = [Line2D([0],[0], marker='o', color='w', markerfacecolor=c2,
                 markersize=9, label='GWL2 (+2°C)'),
          Line2D([0],[0], marker='o', color='w', markerfacecolor=c3,
                 markersize=9, label='GWL3 (+3°C)')]
ax.legend(handles=legend, fontsize=9, loc='lower right', framealpha=0.9)
ax.grid(axis='x', alpha=0.2)
ax.tick_params(labelsize=9)

plt.tight_layout()
import os
os.makedirs('./figures', exist_ok=True)
plt.savefig(outfile, dpi=300, bbox_inches='tight')
print(f'Saved: {outfile}')
plt.close()
