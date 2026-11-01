#!/bin/python
#
# Why analogs persist: pairs co-move.
#
#   departure_divergence.png
#   a) Per biome, area-weighted median local DEPARTURE (|change of the focal
#      cell|, focal ICV units) vs pair DIVERGENCE (|change_a - change_f|),
#      per-model ensemble median, one point per GWL joined into a trajectory.
#      Points far below the 1:1 line = both cells changed, but together.
#   b) Pair divergence vs GWL per biome: forced response (mmm, solid) and
#      per-model median (dashed). Grey lines: the analytic noise floors
#      (median |divergence| from interannual noise alone), per model (upper)
#      and mmm (lower), from the file attributes.
# Colors match retention_curves.png (same biome order).
import numpy as np
import matplotlib.pyplot as plt
import plot_common as pcm
from plot_common import wmedian, gwl_label

pcm.setup_style()
R = pcm.Retention()
ds = R.ds
blist = R.biomes()
x = np.array([pcm.cfg.GWL_KEYS[g] for g in R.gwls])
cmap = plt.get_cmap('tab20')


def series(var, bm):
    return np.array([wmedian(ds[var.format(g=g)].values[bm], R.W[bm]) for g in R.gwls])


fig, (axa, axb) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw=dict(wspace=0.25))
rows = []
for ci, (b, name, bm) in enumerate(blist):
    c = cmap(ci % 20)
    dep = series('departure_{g}', bm)
    div = series('divergence_{g}', bm)
    divm = series('divergence_mmm_{g}', bm)
    com = series('comove_{g}', bm)
    axa.plot(dep, div, color=c, lw=1.0, alpha=0.8)
    axa.scatter(dep, div, color=c, s=8 + 10 * (x - x.min()), zorder=3, label=name)
    axb.plot(x, divm, color=c, lw=1.5, marker='o', ms=3)
    axb.plot(x, div, color=c, lw=0.9, ls='--')
    rows.append((name, dep, div, divm, com))

lim = np.nanmax([np.nanmax(np.r_[r[1], r[2]]) for r in rows]) * 1.05
axa.plot([0, lim], [0, lim], color='0.4', lw=0.8, ls=':')
axa.text(lim * 0.97, lim * 0.97, '1:1', ha='right', va='top', fontsize=7, color='0.4')
axa.set_xlim(0, lim)
axa.set_ylim(0, lim)
axa.set_aspect('equal')
axa.set_xlabel('Local departure |$\\Delta_f$| (ICV units)')
axa.set_ylabel('Pair divergence |$\\Delta_a - \\Delta_f$| (ICV units)')
axa.set_title('a) Departure vs divergence (per-model median)', loc='left', fontsize=9.5)
axa.grid(alpha=0.3, lw=0.5)
axa.legend(fontsize=6.3, ncol=2, loc='upper left', frameon=False,
           title='marker size = GWL', title_fontsize=6.5)

fl = np.array([ds.attrs.get(f'noise_floor_div_{g}', np.nan) for g in R.gwls], float)
flm = np.array([ds.attrs.get(f'noise_floor_div_mmm_{g}', np.nan) for g in R.gwls], float)
axb.plot(x, fl, color='0.55', lw=2.5, alpha=0.6, ls='--')
axb.plot(x, flm, color='0.55', lw=2.5, alpha=0.6)
axb.set_xticks(x)
axb.set_xticklabels([f'{gwl_label(g)}\n(n={R.n_models(g)})' for g in R.gwls], fontsize=8)
axb.set_ylim(0, None)
axb.set_ylabel('Pair divergence (ICV units)')
axb.set_title('b) Divergence vs warming level', loc='left', fontsize=9.5)
axb.grid(alpha=0.3, lw=0.5)
h = [plt.Line2D([], [], color='k', lw=1.5, marker='o', ms=3, label='Forced response (MMM)'),
     plt.Line2D([], [], color='k', lw=0.9, ls='--', label='Per-model median'),
     plt.Line2D([], [], color='0.55', lw=2.5, alpha=0.6, label='Noise floor, MMM'),
     plt.Line2D([], [], color='0.55', lw=2.5, alpha=0.6, ls='--', label='Noise floor, per model')]
axb.legend(handles=h, fontsize=7, loc='lower right', frameon=False)
pcm.savefig(fig, 'departure_divergence.png')

print('\n--- area-weighted medians per GWL (' + ', '.join(R.gwls) + ') ---')
print('    departure | divergence per-model | divergence mmm | comove')
for name, dep, div, divm, com in rows:
    f = lambda a: ' '.join(f'{v:5.2f}' for v in a)  # noqa: E731
    print(f'  {name:<24} {f(dep)} | {f(div)} | {f(divm)} | {f(com)}')
