#!/bin/python
#
# Retention + attribution maps, and retention(Dt) curves, from
# analog_retention.nc. Central estimate set by ANALOG_HEADLINE (plot_common).
#
#   retention_maps.png   (SI): central retention at a low and a high GWL,
#       Dt=1 and Dt=2. The main-text map is plot.retention.tiers.py.
#   attribution_maps.png (main): at ATTR_GWL, forced (mmm) pair divergence
#       on the local-ICV ruler vs one global ICV ruler (same color scale), and
#       the TXx share of divergence^2. If the tropical maximum survives the
#       global ruler, it is not an artifact of narrow local ICV (issue B).
#   retention_curves.png (SI): retention(Dt) per biome at every GWL.
#
# COVERAGE MASK: cells with compromised selection (n_analogs < N_ANALOGS or
# spacing relaxed below S_TRY[0]) are grey on maps. Curves use all cells with
# >= 1 analog, area-weighted (issue I).
import numpy as np
import matplotlib.pyplot as plt
import colormaps as cmaps
import plot_common as pcm
from plot_common import wmean, wmedian, gwl_label

pcm.setup_style()
R = pcm.Retention(mask_low_coverage=True)
ds = R.ds

ATTR_GWL = 'gwl4' if 'gwl4' in R.gwls else R.gwls[-1]
MAP_GWLS = [g for g in ('gwl2', 'gwl4') if g in R.gwls] or [R.gwls[0], R.gwls[-1]]

# ==============================
# FIG 1 (SI): central retention maps, 2 GWL x 2 Dt
# ==============================
fig = plt.figure(figsize=(11, 3.1 * len(MAP_GWLS) + 0.6))
gs = fig.add_gridspec(len(MAP_GWLS), 2, hspace=0.12, wspace=0.05,
                      top=0.95, bottom=0.12, left=0.03, right=0.97)
let = iter('abcdefgh')
for r, g in enumerate(MAP_GWLS):
    for c, dt in enumerate((pcm.DT_STRICT, pcm.DT_LOOSE)):
        ax = pcm.map_axes(fig, gs[r, c])
        pm = pcm.add_map(ax, R, R.masked(R.central(g, dt)), cmaps.precip_diff_12lev, 0, 1,
                         f'{next(let)}) {gwl_label(g)}, D$_t$ = {dt:g}$\\sigma$')
cax = fig.add_axes([0.30, 0.06, 0.40, 0.018])
fig.colorbar(pm, cax=cax, orientation='horizontal',
             label=f'Fraction of analogs retained ({R.central_label()})')
pcm.savefig(fig, 'retention_maps.png')

# ==============================
# FIG 2: attribution at ATTR_GWL (forced response)
# ==============================
div_loc = ds[f'divergence_mmm_{ATTR_GWL}'].values
div_glb = ds[f'divcommon_mmm_{ATTR_GWL}'].values
share = ds[f'share_txx_div_mmm_{ATTR_GWL}'].values
vmax = float(np.nanpercentile(np.r_[div_loc[R.keep], div_glb[R.keep]], 98))
floor = ds.attrs.get(f'noise_floor_div_mmm_{ATTR_GWL}', np.nan)

fig = plt.figure(figsize=(16, 4.4))
gs = fig.add_gridspec(1, 3, wspace=0.05, top=0.92, bottom=0.18, left=0.02, right=0.98)
ax = pcm.map_axes(fig, gs[0, 0])
pm0 = pcm.add_map(ax, R, R.masked(div_loc), 'viridis', 0, vmax,
                  f'a) Pair divergence, local ICV ruler ({gwl_label(ATTR_GWL)})')
ax = pcm.map_axes(fig, gs[0, 1])
pcm.add_map(ax, R, R.masked(div_glb), 'viridis', 0, vmax,
            f'b) Pair divergence, one global ICV ruler ({gwl_label(ATTR_GWL)})')
cax = fig.add_axes([0.10, 0.12, 0.46, 0.025])
cb = fig.colorbar(pm0, cax=cax, orientation='horizontal', extend='max')
cb.set_label(f'Median divergence over analogs (ICV units; noise floor {floor:.2f})')
ax = pcm.map_axes(fig, gs[0, 2])
pm2 = pcm.add_map(ax, R, R.masked(share), 'PuOr_r', 0, 1,
                  f'c) TXx share of divergence$^2$ ({gwl_label(ATTR_GWL)})')
cax = fig.add_axes([0.72, 0.12, 0.22, 0.025])
fig.colorbar(pm2, cax=cax, orientation='horizontal',
             label='0 = all Rx5day, 1 = all TXx')
pcm.savefig(fig, 'attribution_maps.png')

# ==============================
# FIG 3 (SI): retention(Dt) curves per biome, every GWL
# ==============================
blist = R.biomes()
cmap = plt.get_cmap('tab20')
fig, axes = plt.subplots(1, len(R.gwls), figsize=(4.2 * len(R.gwls), 4.4), sharey=True)
axes = np.atleast_1d(axes)
for ax, g in zip(axes, R.gwls):
    ret = R.central(g)
    for ci, (b, name, bm) in enumerate(blist):
        curve = [wmean(ret[bm, k], R.W[bm]) for k in range(len(R.Dt))]
        ax.plot(R.Dt, curve, color=cmap(ci % 20), lw=1.3, label=name)
    for dt in (pcm.DT_STRICT, pcm.DT_LOOSE):
        ax.axvline(dt, color='0.5', lw=0.6, ls=':')
    ax.set_title(f'{gwl_label(g)} ({R.n_models(g)} models)', fontsize=10)
    ax.set_xlabel(r'D$_t$ ($\sigma$)')
    ax.set_ylim(0, 1.02); ax.grid(alpha=0.3)
axes[0].set_ylabel('Area-weighted mean fraction retained')
axes[-1].legend(fontsize=6.5, loc='lower right')
fig.suptitle(f'Retention vs threshold by biome: {R.central_label()}', fontsize=11)
fig.tight_layout()
pcm.savefig(fig, 'retention_curves.png')

# ==============================
# console
# ==============================
print(f'\n--- {ATTR_GWL} forced divergence, area-weighted median: local vs global ruler ---')
for b, name, bm in blist:
    print(f'  {name:<24} {wmedian(div_loc[bm], R.W[bm]):6.2f} {wmedian(div_glb[bm], R.W[bm]):6.2f}'
          f'   TXx share {wmedian(share[bm], R.W[bm]):.2f}')
print('\n--- coverage kept per biome (% of land cells) ---')
for b, name in pcm.BIOME_SHORT.items():
    bm = (R.biome_id == b) & R.land
    if bm.sum():
        print(f'  {name:<24} {100 * np.mean(R.keep[bm]):5.1f}%')
print('\nDone.')
