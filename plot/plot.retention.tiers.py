#!/bin/python
#
# Headline figures: where analogs hold, and how sure we are.
#
#   retention_tier_maps.png: central retention (ANALOG_HEADLINE) at Dt=1 for
#       a low and a high GWL. Dots mark 2.5 deg blocks where most kept
#       cells have >= ROBLOST_MAJORITY of their analogs ROBUSTLY LOST (lost in
#       >= 80% of models, i.e. retained by <= 20%). Model agreement, as in the
#       IPCC AR6 convention; uncertain and robustly retained shares are in the bars.
#   retention_tier_bars.png: per biome and GWL, area-weighted fraction of
#       analogs robustly retained / uncertain / robustly lost (80% model
#       agreement; robret_/uncert_/roblost_), at Dt=1 and Dt=2. Right column:
#       fnoise, the share of model spread explained by internal variability
#       (~1 = noise, ~0 = model disagreement). Area-weighted median.
import numpy as np
import matplotlib.pyplot as plt
import plot_common as pcm
from plot_common import wmean, wmedian, gwl_label, dt_tag

pcm.setup_style()
R = pcm.Retention()
ds = R.ds
MAP_GWLS = [g for g in ('gwl2', 'gwl3') if g in R.gwls] or [R.gwls[0], R.gwls[-1]]
ROBLOST_MAJORITY = 0.5   # share of a cell's analogs robustly lost
STIPPLE_BLOCK = max(1, int(round(2.5 / abs(float(R.lat[1] - R.lat[0])))))

# ==============================
# FIG A: maps
# ==============================
dt = pcm.DT_STRICT
cmap_ret, norm_ret = pcm.retention_cmap_norm()
fig = plt.figure(figsize=(6.5, 3.05 * len(MAP_GWLS) + 1.2))
gs = fig.add_gridspec(len(MAP_GWLS), 1, hspace=0.1, top=0.97, bottom=0.17, left=0.01, right=0.99)
for r, g in enumerate(MAP_GWLS):
    ax = pcm.map_axes(fig, gs[r, 0])
    pm = pcm.add_map(ax, R, R.masked(R.central(g, dt)), cmap_ret, None, None,
                     f'{"abcd"[r]}) {gwl_label(g)}, D$_t$ = {dt:g}$\\sigma$ ({R.n_models(g)} models)',
                     norm=norm_ret)
    lost = ds[f'roblost_{g}_{dt_tag(dt)}'].values
    pcm.coarse_stipple(ax, R, np.nan_to_num(lost) >= ROBLOST_MAJORITY, block=STIPPLE_BLOCK)
cax = fig.add_axes([0.12, 0.115, 0.76, 0.018])
pcm.retention_colorbar(fig, cax, pm, f'Fraction of analogs retained, {R.central_label()}')
leg_h = [plt.Line2D([], [], ls='', marker='o', ms=3.5, mfc='k', mec='white', mew=0.4,
                    label=f'\u2265{ROBLOST_MAJORITY:.0%} of analogs robustly lost (lost in \u226580% of models)'),
         plt.Rectangle((0, 0), 1, 1, fc=pcm.COVERAGE_GREY, ec='none', label=pcm.COVERAGE_LABEL)]
fig.legend(handles=leg_h, loc='upper center', bbox_to_anchor=(0.5, 0.045), ncol=1, fontsize=7.5,
           frameon=False, handlelength=1.2, labelspacing=0.4)
pcm.savefig(fig, 'retention_tier_maps.png')

# ==============================
# FIG B: stacked tier bars + fnoise
# ==============================
blist = R.biomes()
G = len(R.gwls)
TIER_COL = {'robret': 'darkslategray', 'uncert': '#d9d9d9', 'roblost': 'indianred'}
TIER_LAB = {'robret': 'Robustly retained', 'uncert': 'Uncertain', 'roblost': 'Robustly lost'}
gap = 0.8
ypos, ylab, yb = [], [], []
y = 0.0
for b, name, bm in blist:
    yb.append((name, y + (G - 1) / 2))
    for g in R.gwls:
        ypos.append(y); ylab.append(gwl_label(g)); y += 1
    y += gap
ypos = -np.array(ypos)

fig, axes = plt.subplots(1, 3, figsize=(7, 8.5), sharey=True,
                         gridspec_kw=dict(width_ratios=[1, 1, 0.28], wspace=0.08,
                                          left=0.22, right=0.98, top=0.96, bottom=0.1))
axes[0].set_ylim(ypos.min() - 0.7, ypos.max() + 0.7)   # no empty band above/below the bars
table = []
for ax, dt in zip(axes[:2], (pcm.DT_STRICT, pcm.DT_LOOSE)):
    i = 0
    for b, name, bm in blist:
        for g in R.gwls:
            fr = {t: wmean(ds[f'{t}_{g}_{dt_tag(dt)}'].values[bm], R.W[bm]) for t in TIER_COL}
            left = 0.0
            for t, c in TIER_COL.items():
                ax.barh(ypos[i], fr[t], left=left, height=0.85, color=c, lw=0)
                left += fr[t]
            table.append((name, g, dt, fr))
            i += 1
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(['0', '0.25', '0.5', '0.75', '1'])
    ax.set_title(f'{"ab"[int(dt != pcm.DT_STRICT)]}) D$_t$ = {dt:g}$\\sigma$', loc='left', fontsize=9)
    ax.set_xlabel('Area-weighted fraction of analogs', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(axis='x', alpha=0.3, lw=0.5)
axes[0].set_yticks(ypos)
axes[0].set_yticklabels(ylab, fontsize=6)
for name, yc in yb:
    axes[0].text(-0.17, -yc, name, transform=axes[0].get_yaxis_transform(), ha='right',
                 va='center', fontsize=7.5)

# fnoise column
axf = axes[2]
cmap = plt.get_cmap('cividis')
i = 0
fn_rows = []
for b, name, bm in blist:
    for g in R.gwls:
        v = wmedian(ds[f'fnoise_{g}'].values[bm], R.W[bm])
        axf.barh(ypos[i], 1, height=0.85, color=cmap(v) if np.isfinite(v) else 'white', lw=0)
        axf.text(0.5, ypos[i], f'{v:.2f}' if np.isfinite(v) else 'n/a', ha='center',
                 va='center', fontsize=5.5,
                 color='white' if (np.isfinite(v) and v < 0.5) else 'black')
        fn_rows.append((name, g, v))
        i += 1
axf.set_xlim(0, 1); axf.set_xticks([])
axf.set_title('c) f$_{noise}$', loc='left', fontsize=9)
axf.set_xlabel('noise share\nof model spread', fontsize=7)
for s in axf.spines.values():
    s.set_visible(False)
h = [plt.Rectangle((0, 0), 1, 1, color=c, label=TIER_LAB[t]) for t, c in TIER_COL.items()]
fig.legend(handles=h, loc='lower center', ncol=3, fontsize=8, frameon=False,
           bbox_to_anchor=(0.6, 0.0))
pcm.savefig(fig, 'retention_tier_bars.png')

print('\n--- tiers (ret/unc/lost), area-weighted mean over cells with analogs ---')
for name, g, dt, fr in table:
    print(f'  {name:<24} {g:<7} Dt={dt:g}  {fr["robret"]:.2f}/{fr["uncert"]:.2f}/{fr["roblost"]:.2f}')
print('\n--- fnoise, area-weighted median ---')
for name, g, v in fn_rows:
    print(f'  {name:<24} {g:<7} {v:.2f}')
