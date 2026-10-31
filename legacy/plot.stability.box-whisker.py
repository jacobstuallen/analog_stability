#!/bin/python
# Biome analog stability: inter-model spread as box-and-whisker.
#
# For each biome, the distribution ACROSS the 27 CMIP6 models of that
# biome's spatial-median sigma dissimilarity, at GWL2 and GWL3.
# Box = IQR, line = median, whiskers = 5th-95th percentile across models.
#
# Reads per_model_biome_sigma.npz produced by analog_stability_gwl.py.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib.patches import Patch

data_root = '/glade/campaign/ral/risc/jsallen/TNC'
infile = f'{data_root}/analog_stability/per_model_biome_sigma.npz'
outfile = './figures/figure_stability_biomes_box.png'

whisker_lo, whisker_hi = 5, 95

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

f = np.load(infile, allow_pickle=True)
models = f['models_gwl3'] if 'models_gwl3' in f.files else f['models']   # npz keys per GWL since step 2a
n_models = len(models)

def stats(arr):
    """Return dict of box stats from a 1D array across models."""
    a = arr[np.isfinite(arr)]
    return {
        'med': np.median(a),
        'q1': np.percentile(a, 25), 'q3': np.percentile(a, 75),
        'wlo': np.percentile(a, whisker_lo), 'whi': np.percentile(a, whisker_hi),
    }

rows = []
for b, name in biomes.items():
    k2, k3 = f'biome_{b}_gwl2', f'biome_{b}_gwl3'
    if k2 not in f or k3 not in f:
        continue
    rows.append({'name': name, 'g2': stats(f[k2]), 'g3': stats(f[k3]),
                 'order': np.median(f[k3][np.isfinite(f[k3])])})
rows.sort(key=lambda r: r['order'])

c2, c3 = '#7fb3d5', '#c0392b'

fig, ax = plt.subplots(figsize=(9, 7.5))

def draw_box(ax, ypos, st, color, h=0.32):
    # whiskers
    ax.plot([st['wlo'], st['whi']], [ypos, ypos], color=color, lw=1.0, zorder=2)
    for x in (st['wlo'], st['whi']):
        ax.plot([x, x], [ypos-h*0.4, ypos+h*0.4], color=color, lw=1.0, zorder=2)
    # box
    ax.add_patch(plt.Rectangle((st['q1'], ypos-h/2), st['q3']-st['q1'], h,
                               facecolor=color, edgecolor=color, alpha=0.35,
                               lw=1.0, zorder=3))
    # median
    ax.plot([st['med'], st['med']], [ypos-h/2, ypos+h/2], color=color,
            lw=2.0, zorder=4)

for i, r in enumerate(rows):
    draw_box(ax, i+0.18, r['g2'], c2)
    draw_box(ax, i-0.18, r['g3'], c3)

for xv in [1, 2, 3]:
    ax.axvline(xv, color='k', lw=0.5, ls=':', alpha=0.3, zorder=0)

ax.set_yticks(np.arange(len(rows)))
ax.set_yticklabels([r['name'] for r in rows], fontsize=9)
ax.set_ylim(-0.7, len(rows)-0.3)

# x-limits from actual whisker extent so nothing clips
all_hi = max(max(r['g2']['whi'], r['g3']['whi']) for r in rows)
all_lo = min(min(r['g2']['wlo'], r['g3']['wlo']) for r in rows)
ax.set_xlim(min(0, all_lo*1.05), all_hi*1.08)

ax.set_xlabel('\u03c3 dissimilarity to historical best analog (biome spatial median)',
              fontsize=10)
ax.set_title(f'Biome analog stability: inter-model spread ({n_models} CMIP6 models, SSP5-8.5)\n'
             f'box: IQR, whiskers: {whisker_lo}\u2013{whisker_hi}th percentile across models',
             loc='left', fontsize=11)

ax.legend(handles=[Patch(facecolor=c2, alpha=0.5, label='GWL2 (+2\u00b0C)'),
                   Patch(facecolor=c3, alpha=0.5, label='GWL3 (+3\u00b0C)')],
          fontsize=9, loc='lower right', framealpha=0.9)
ax.grid(axis='x', alpha=0.2)
ax.tick_params(labelsize=9)

plt.tight_layout()
import os
os.makedirs('./figures', exist_ok=True)
plt.savefig(outfile, dpi=300, bbox_inches='tight')
print(f'Saved: {outfile}')
plt.close()
