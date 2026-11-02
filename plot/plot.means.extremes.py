#!/bin/python
#
# MAIN-TEXT FIGURE: means vs extremes analog selection.
# "How analogs are chosen matters more than how far climate warms."
#
# All shares are the fraction of analog pairs within Dt = 1 sigma in EXTREMES
# space (TXx, Rx5day), cos(lat) area-weighted over cells with both analog sets.
#
# (a) Ladder by biome, sorted by lift (means-selected / random):
#       open grey circle : random same-biome cell beyond 250 km, today (ERA5)
#       orange square    : means-selected analogs, today (cross run, GWL0)
#       blue triangle    : extremes-selected analogs at GWL2 (central estimate)
#       navy circle      : extremes-selected analogs at GWL4 (central estimate)
#                          (GWL2 / GWL4 markers offset vertically so they never hide)
#       black tick       : extremes-selected analogs today (~1 by the sigma ceiling;
#                          a reference, not a result)
#     Dotted line separates lift < 2 from lift >= 2.
# (b-e) Share vs GWL for four biomes, one lettered panel each: extremes-selected (blue) vs means-selected
#     scored on extremes (orange, cross run); band = area-weighted mean of the
#     per-cell model p10-p90; dashed grey = random baseline (today).
#     Both lines use the extremes model ensemble (cross is scored on TXx/Rx5day).
#     Shared x label under the bottom row. Legends for (a) and (b-e) sit
#     along the bottom of the figure.
# Example focal-cell maps: plot.means.extremes.examples.py (SI).
#
# Writes: means_extremes.png (+ prints the numbers behind every panel)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import plot_common as pcm
from plot_common import cfg, gwl_label
import means_extremes_common as mec
from means_extremes_common import C_RAND, C_MEANS, C_EXT2, C_EXT4, SIG, DEG, TIMES, gwl_tex

DT = pcm.DT_STRICT
PANEL_B = [14, 1, 8, 2]          # tropical moist, boreal, temperate broadleaf, deserts
# shorter titles so '(b) name' fits over a narrow panel
PANEL_B_NAME = {14: 'Trop. moist forest', 1: 'Boreal', 8: 'Temp. broadleaf', 2: 'Deserts/xeric'}
EXT_DY = 0.17                    # vertical offset of GWL2 (up) / GWL4 (down) markers
LIFT_SPLIT = 2.0

pcm.setup_style()
M = mec.MeansExtremes()
rows = M.biome_rows(DT)
if not rows:
    raise SystemExit('no biomes to plot')

ext0 = M.ext0(DT)
crossg = {g: M.ret(M.rc, g, DT) for g in ['gwl0'] + M.gwls}
extg = {'gwl0': ext0}
extg.update({g: M.ret(M.re, g, DT) for g in M.gwls})

fig = plt.figure(figsize=(7.2, 5.4))
top = GridSpec(1, 2, figure=fig, width_ratios=[1.12, 1.0], wspace=0.42)

# ------------------------------------------------------------------
# (a) ladder
# ------------------------------------------------------------------
ax = fig.add_subplot(top[0])
y = np.arange(len(rows))
print(f'\n--- (a) share within Dt={DT:g} in extremes space ---')
print(f'{"biome":<24} {"random":>7} {"means0":>7} {"ext0":>6} {"extGWL2":>8} {"extGWL4":>8} {"lift":>5}')
for yi, r in zip(y, rows):
    bm = r['bm']
    e0 = M.bmean(ext0, bm)
    e2 = M.bmean(extg['gwl2'], bm) if 'gwl2' in extg else np.nan
    e4 = M.bmean(extg['gwl4'], bm) if 'gwl4' in extg else np.nan
    vals = [v for v in (r['rand'], r['m0'], e2, e4) if np.isfinite(v)]
    ax.plot([min(vals), max(vals)], [yi, yi], color='0.8', lw=1.0, zorder=1)
    ax.plot(e0, yi, marker='|', ms=9, mew=1.2, color='k', zorder=2)
    ax.plot(r['rand'], yi, 'o', ms=5.5, mfc='white', mec=C_RAND, mew=1.3, zorder=3)
    ax.plot(r['m0'], yi, 's', ms=5.5, color=C_MEANS, mec='white', mew=0.6, zorder=4)
    if np.isfinite(e2):
        ax.plot(e2, yi - EXT_DY, '^', ms=6, color=C_EXT2, mec='white', mew=0.6, zorder=4)
    if np.isfinite(e4):
        ax.plot(e4, yi + EXT_DY, 'o', ms=5.5, color=C_EXT4, mec='white', mew=0.6, zorder=5)
    ax.text(1.04, yi, f'{TIMES}{r["lift"]:.1f}', va='center', ha='left', fontsize=7,
            transform=ax.get_yaxis_transform(), color='0.25')
    print(f'{r["name"]:<24} {r["rand"]:>7.2f} {r["m0"]:>7.2f} {e0:>6.2f} {e2:>8.2f} {e4:>8.2f} {r["lift"]:>5.1f}')
nsplit = sum(r['lift'] < LIFT_SPLIT for r in rows)
if 0 < nsplit < len(rows):
    ax.axhline(nsplit - 0.5, color='0.5', lw=0.6, ls=':')
ax.text(1.04, -1.0, 'lift', ha='left', va='center', fontsize=7, color='0.25',
        transform=ax.get_yaxis_transform())
ax.set_yticks(y)
ax.set_yticklabels([r['name'] for r in rows], fontsize=7.5)
ax.set_ylim(len(rows) - 0.5, -1.6)
ax.set_xlim(0, 1.03)
ax.set_xlabel(f'Share of pairs within {DT:g}{SIG} in extremes space', fontsize=8)
ax.tick_params(axis='x', labelsize=7.5)
ax.grid(axis='x', alpha=0.3, lw=0.5)
ax.set_title('(a) Analog validity for extremes', loc='left', fontsize=8.5)
g2, g4 = gwl_tex('gwl2'), gwl_tex('gwl4')
handles = [
    Line2D([], [], ls='', marker='o', mfc='white', mec=C_RAND, mew=1.3, ms=5.5, label='Random, same biome (today)'),
    Line2D([], [], ls='', marker='s', color=C_MEANS, ms=5.5, label='Means-selected (today)'),
    Line2D([], [], ls='', marker='^', color=C_EXT2, ms=6, label=f'Extremes-selected ({g2})'),
    Line2D([], [], ls='', marker='o', color=C_EXT4, ms=5.5, label=f'Extremes-selected ({g4})'),
    Line2D([], [], ls='', marker='|', color='k', ms=8, mew=1.2, label='Extremes-selected (today)'),
]
ax_a = ax

# ------------------------------------------------------------------
# (b) share vs GWL, four biomes
# ------------------------------------------------------------------
pb = [b for b in PANEL_B if any(r['b'] == b for r in rows)]
sub = top[1].subgridspec(2, 2, hspace=0.45, wspace=0.12)
xg = np.array([0.0] + [cfg.GWL_KEYS[g] for g in M.gwls])
r1 = M.rand(DT)
print(f'\n--- (b) share within Dt={DT:g} vs GWL (0, {", ".join(gwl_label(g) for g in M.gwls)}) ---')
axes_b = []
for k, b in enumerate(pb):
    axb = fig.add_subplot(sub[k // 2, k % 2], sharey=axes_b[0] if axes_b else None)
    axes_b.append(axb)
    r = next(rr for rr in rows if rr['b'] == b)
    bm = r['bm']
    ex = np.array([M.bmean(extg[g], bm) for g in ['gwl0'] + M.gwls])
    cr = np.array([M.bmean(crossg[g], bm) for g in ['gwl0'] + M.gwls])
    for ds, col, ser in [(M.re, C_EXT4, ex), (M.rc, C_MEANS, cr)]:
        lo = [M.ret_pct(ds, g, DT, 10) for g in M.gwls]
        hi = [M.ret_pct(ds, g, DT, 90) for g in M.gwls]
        if all(a is not None for a in lo + hi):
            lo = [ser[0]] + [M.bmean(a, bm) for a in lo]
            hi = [ser[0]] + [M.bmean(a, bm) for a in hi]
            axb.fill_between(xg, lo, hi, color=col, alpha=0.15, lw=0)
    axb.axhline(r['rand'], color=C_RAND, lw=1.0, ls='--', zorder=1)
    axb.plot(xg, ex, '-o', color=C_EXT4, lw=1.6, ms=3, zorder=3)
    axb.plot(xg, cr, '-s', color=C_MEANS, lw=1.6, ms=3, zorder=3)
    axb.set_title(f"({chr(98 + k)}) {PANEL_B_NAME.get(b, r['name'])}", loc='left', fontsize=8)
    axb.set_ylim(0, 1.05)
    axb.set_xticks(xg)
    axb.set_xticklabels(['0'] + [f'{cfg.GWL_KEYS[g]:g}' for g in M.gwls], fontsize=6.5)
    axb.tick_params(axis='y', labelsize=6.5)
    axb.grid(alpha=0.3, lw=0.5)
    if k % 2:
        plt.setp(axb.get_yticklabels(), visible=False)
    print(f'{r["name"]:<24} extremes ' + ' '.join(f'{v:.2f}' for v in ex)
          + ' | means-sel ' + ' '.join(f'{v:.2f}' for v in cr) + f' | random {r["rand"]:.2f}')
    #if k == 0:
        # selection gap (today) vs warming loss (to the last GWL)
        #axb.annotate('', xy=(0.12, ex[0]), xytext=(0.12, cr[0]),
        #             arrowprops=dict(arrowstyle='<->', lw=0.8, color='0.2', shrinkA=0, shrinkB=0))
        #axb.text(0.25, 0.5 * (ex[0] + cr[0]), 'selection', fontsize=6, va='center', color='0.2')
        #axb.annotate('', xy=(xg[-1] - 0.12, ex[-1]), xytext=(xg[-1] - 0.12, ex[0]),
        #             arrowprops=dict(arrowstyle='<->', lw=0.8, color='0.2', shrinkA=0, shrinkB=0))
        #axb.text(xg[-1] - 0.05, ex[-1] - 0.05, 'warming', fontsize=6, va='top',
        #         ha='right', color='0.2')
if axes_b:
    axes_b[0].set_ylabel(f'Share within {DT:g}{SIG}', fontsize=7)
    axes_b[2 if len(axes_b) > 2 else 0].set_ylabel(f'Share within {DT:g}{SIG}', fontsize=7)
    hb = [Line2D([], [], color=C_EXT4, marker='o', ms=3, lw=1.6, label='Extremes-selected'),
          Line2D([], [], color=C_MEANS, marker='s', ms=3, lw=1.6, label='Means-selected'),
          Line2D([], [], color=C_RAND, ls='--', lw=1.0, label='Random (today)')]
    if 'gwl4' in M.gwls:
        print("  n models: " + ', '.join(f'{gwl_label(g)} {M.n_models(g)}' for g in M.gwls))

# shared x label under the bottom row of (b)
bl = axes_b[-2:] if len(axes_b) >= 2 else axes_b
p0, p1 = bl[0].get_position(), bl[-1].get_position()
xc = 0.5 * (p0.x0 + p1.x1)
fig.text(xc, p0.y0 - 0.055, f'Global warming level ({DEG})', ha='center', va='top', fontsize=7)
# legends for both blocks along the bottom of the figure
pa = ax_a.get_position()
y_leg = min(pa.y0, p0.y0) - 0.10
fig.legend(handles=handles, fontsize=6.2, loc='upper center', bbox_to_anchor=(0.5 * (pa.x0 + pa.x1), y_leg),
           ncol=2, frameon=False, handletextpad=0.3, columnspacing=0.8)
if axes_b:
    fig.legend(handles=hb, fontsize=6.2, loc='upper center', bbox_to_anchor=(xc, y_leg),
               ncol=1, frameon=False, handlelength=1.6, handletextpad=0.4)

pcm.savefig(fig, 'means_extremes.png')
print('Done.')
