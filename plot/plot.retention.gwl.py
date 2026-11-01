#!/bin/python
#
# Retention vs global warming level, one small panel per biome.
#
# In each panel, for Dt = 1 sigma (dark) and Dt = 2 sigma (light):
#   solid line + dots : forced response (retention from the multi-model-mean
#                       future, retention_mmm_<gwl>)
#   dashed line       : ensemble median of per-model retention (retention_<gwl>)
#   band              : area-weighted mean of each cell's 10th-90th percentile
#                       across models. NOTE: this is the typical per-cell model
#                       range, not a percentile of the biome mean.
# All values are cos(lat) area-weighted over cells with >= 1 analog.
# GWL4 has fewer models than GWL1.5-3 (x-axis labels give n).
import numpy as np
import matplotlib.pyplot as plt
import plot_common as pcm
from plot_common import wmean, gwl_label

pcm.setup_style()
R = pcm.Retention()
ds = R.ds
blist = R.biomes()
x = np.array([pcm.cfg.GWL_KEYS[g] for g in R.gwls])

COL = {pcm.DT_STRICT: '#08519c', pcm.DT_LOOSE: '#6baed6'}


def biome_series(var, bm, k):
    return np.array([wmean(ds[var.format(g=g)].values[:, :, k][bm], R.W[bm]) for g in R.gwls])


ncol = 4
nrow = int(np.ceil(len(blist) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 2.1 * nrow), sharex=True, sharey=True)
axes = axes.ravel()
rows = []
for ax, (b, name, bm) in zip(axes, blist):
    for dt in (pcm.DT_STRICT, pcm.DT_LOOSE):
        k = R.di(dt)
        mmm = biome_series('retention_mmm_{g}', bm, k)
        med = biome_series('retention_{g}', bm, k)
        p10 = biome_series('retention_{g}_p10', bm, k)
        p90 = biome_series('retention_{g}_p90', bm, k)
        c = COL[dt]
        ax.fill_between(x, p10, p90, color=c, alpha=0.18, lw=0)
        ax.plot(x, med, color=c, lw=1.0, ls='--')
        ax.plot(x, mmm, color=c, lw=1.6, marker='o', ms=3)
        rows.append((name, dt, mmm, med, p10, p90))
    ax.set_title(name, fontsize=8.5, loc='left')
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3, lw=0.5)
    ax.tick_params(labelsize=7.5)
for ax in axes[len(blist):]:
    ax.set_visible(False)
xt = [f'{gwl_label(g)}\n(n={R.n_models(g)})' for g in R.gwls]
for ax in axes[:len(blist)]:
    ax.set_xticks(x)
    ax.set_xticklabels(xt, fontsize=6.5)
    ax.xaxis.set_tick_params(labelbottom=True)
for r in range(nrow):
    axes[r * ncol].set_ylabel('Fraction retained', fontsize=8)

# legend in the first unused slot (or below the figure)
h = [plt.Line2D([], [], color=COL[pcm.DT_STRICT], lw=1.6, marker='o', ms=3, label='Forced response (MMM)'),
     plt.Line2D([], [], color=COL[pcm.DT_STRICT], lw=1.0, ls='--', label='Per-model median'),
     plt.Rectangle((0, 0), 1, 1, color=COL[pcm.DT_STRICT], alpha=0.18, label='Per-cell model 10-90%'),
     plt.Line2D([], [], color=COL[pcm.DT_STRICT], lw=4, label=f'D$_t$ = {pcm.DT_STRICT:g}$\\sigma$'),
     plt.Line2D([], [], color=COL[pcm.DT_LOOSE], lw=4, label=f'D$_t$ = {pcm.DT_LOOSE:g}$\\sigma$')]
if len(blist) < len(axes):
    axes[len(blist)].set_visible(True)
    axes[len(blist)].axis('off')
    axes[len(blist)].legend(handles=h, loc='center', fontsize=7.5, frameon=False)
else:
    fig.legend(handles=h, loc='lower center', ncol=5, fontsize=7.5, frameon=False)
fig.tight_layout()
pcm.savefig(fig, 'retention_gwl_biomes.png')

print('\n--- area-weighted retention: biome, Dt, then per GWL  mmm / median [p10-p90] ---')
print('    GWLs: ' + ', '.join(R.gwls))
for name, dt, mmm, med, p10, p90 in rows:
    cells = '  '.join(f'{a:.2f}/{m:.2f} [{l:.2f}-{u:.2f}]' for a, m, l, u in zip(mmm, med, p10, p90))
    print(f'  {name:<24} Dt={dt:g}  {cells}')
