#!/bin/python
#
# Why analogs persist: pairs co-move.
#
#   departure_divergence.png
#   a) Per biome, area-weighted median local DEPARTURE (|change of the focal
#      cell|, focal ICV units) vs pair DIVERGENCE (|change_a - change_f|), from
#      the FORCED response (multi-model-mean future), one point per GWL joined
#      into a trajectory (marker size grows with GWL). Dotted 1:1 = the analog
#      did not change at all; faint lines = divergence/departure 0.25 and 0.5.
#      Points far below 1:1 = both cells changed, but together. The forced
#      response is used because per-model values sit on a ~0.47 noise floor.
#   b) Forced divergence vs GWL with the analytic MMM noise floor.
#   c) Per-model median divergence vs GWL with the per-model noise floor:
#      what a single model (one member, 20-yr windows) sees. Same y axis as b.
#   b, c highlight HIGHLIGHT biomes (the four in means_extremes.png); others
#   are thin grey. Colors match panel a.
#
# Model set: the common-model retention file (analog_retention_common.nc,
# every GWL uses the models that reach GWL4) is used when it exists, so the
# GWL3 -> GWL4 step is not an ensemble change. Otherwise the per-GWL file, with
# n shown under the x axis. Env ANALOG_DD_SOURCE=per_gwl forces the per-GWL file.
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
import plot_common as pcm
from plot_common import cfg, wmedian
from means_extremes_common import gwl_tex

HIGHLIGHT = [14, 1, 8, 2]        # tropical moist, boreal, temperate broadleaf, deserts
SHORT = {14: 'Trop. moist', 1: 'Boreal', 8: 'Temp. broadleaf', 2: 'Deserts/xeric'}
RATIOS = [0.25, 0.5]
LABEL_A = {7: 'Rock and ice', 5: 'Mediterranean'}   # direct labels in (a): steepest trajectories
# dark outline on highlighted lines in (b, c): keeps the light boreal tan readable
# in print without changing the shared tab20b colors (a darker tan collides with
# the temperate golds: validator dE < 8 against #bd9e39 / #e7ba52)
OUTLINE = [pe.Stroke(linewidth=2.6, foreground='0.35'), pe.Normal()]

pcm.setup_style()
R = pcm.Retention()
blist = R.biomes()

# ---- data source: common models if available AND complete ----
NEED = ['departure_mmm_{g}', 'divergence_mmm_{g}', 'divergence_{g}', 'comove_mmm_{g}']


def usable_gwls(d):
    return [g for g in cfg.GWL_KEYS if all(v.format(g=g) in d for v in NEED)]


ds, src = R.ds, f'per-GWL models ({cfg.RETENTION_FILE})'
f_common = os.path.join(cfg.OUT_DIR, 'analog_retention_common.nc')
if os.path.exists(f_common) and os.environ.get('ANALOG_DD_SOURCE', 'common') == 'common':
    dc = xr.open_dataset(f_common)
    gc = usable_gwls(dc)
    if len(dc.lat) != len(R.lat) or not np.allclose(dc.lat.values, R.lat) \
            or len(dc.lon) != len(R.lon) or not np.allclose(dc.lon.values, R.lon):
        print(f'  {f_common}: grid mismatch, not used')
    elif len(gc) < 2:
        missing = sorted({v.format(g=g) for g in cfg.GWL_KEYS for v in NEED} - set(dc.data_vars))
        print(f'  {f_common}: missing {len(missing)} needed variables '
              f'(e.g. {", ".join(missing[:4])}); probably written before the mmm outputs '
              f'(step 8). Rerun analog.retention.py with ANALOG_MODEL_SET=common to use it.')
    else:
        ds, src = dc, f'common models ({f_common})'
use_common = ds is not R.ds
print(f'Source: {src}')
gwls = usable_gwls(ds)
if not gwls:
    raise SystemExit(f'no GWL has all of {NEED} in {src}')
x = np.array([cfg.GWL_KEYS[g] for g in gwls])
nmod = [int(ds.attrs.get(f'n_models_{g}', -1)) for g in gwls]

cmap = plt.get_cmap('tab20b')
color = {b: cmap(ci % 20) for ci, (b, _, _) in enumerate(blist)}


def series(var, bm):
    return np.array([wmedian(ds[var.format(g=g)].values[bm], R.W[bm]) for g in gwls])


rows = []
for b, name, bm in blist:
    rows.append(dict(b=b, name=name,
                     dep=series('departure_mmm_{g}', bm), divm=series('divergence_mmm_{g}', bm),
                     div=series('divergence_{g}', bm), com=series('comove_mmm_{g}', bm)))

fig, axes = plt.subplots(1, 3, figsize=(11, 4.5),
                         gridspec_kw=dict(width_ratios=[1.25, 1, 1], wspace=0.32))
axa, axb, axc = axes

# ------------------------------------------------------------------
# (a) forced departure vs forced divergence
# ------------------------------------------------------------------
xmax = np.nanmax([np.nanmax(r['dep']) for r in rows]) * 1.08
ymax = np.nanmax([np.nanmax(r['divm']) for r in rows]) * 1.15
axa.plot([0, xmax], [0, xmax], color='0.4', lw=0.8, ls=':')
for q in RATIOS:
    axa.plot([0, xmax], [0, q * xmax], color='0.75', lw=0.6, zorder=0)


def ratio_label(q, txt, col):
    # label just inside the axes where the line y = q x leaves the plot
    xt = min(xmax * 0.97, 0.94 * ymax / q)
    axa.text(xt + 0.01 * xmax, q * xt, txt, fontsize=6.5, color=col, ha='left', va='center')


ratio_label(1.0, '1:1', '0.4')
for q in RATIOS:
    ratio_label(q, f'{q:g}', '0.5')
sizes = 8 + 20 * (x - x.min())
for r in rows:
    c = color[r['b']]
    axa.plot(r['dep'], r['divm'], color=c, lw=1.0, alpha=0.8)
    axa.scatter(r['dep'], r['divm'], color=c, s=sizes, zorder=3, edgecolor='white', lw=0.4)
    #if r['b'] in LABEL_A:
    #    axa.annotate(LABEL_A[r['b']], xy=(r['dep'][-1], r['divm'][-1]), xytext=(6, 3),
    #                 textcoords='offset points', fontsize=6.5, color='0.15', ha='left', va='bottom')
axa.set_xlim(0, xmax)
axa.set_ylim(0, ymax)
axa.set_xlabel(r'Local departure $|\Delta_f|$ (ICV units)', fontsize=8)
axa.set_ylabel(r'Pair divergence $|\Delta_a - \Delta_f|$ (ICV units)', fontsize=8)
axa.set_title('(a) Departure vs divergence (forced)', loc='left', fontsize=8.5)

# ------------------------------------------------------------------
# (b) forced and (c) per-model divergence vs GWL
# ------------------------------------------------------------------
fl = np.array([ds.attrs.get(f'noise_floor_div_{g}', np.nan) for g in gwls], float)
flm = np.array([ds.attrs.get(f'noise_floor_div_mmm_{g}', np.nan) for g in gwls], float)
yhi = np.nanmax([np.nanmax(np.r_[r['div'], r['divm']]) for r in rows] + list(fl)) * 1.08


def direct_labels(ax, items, ymin_gap):
    """items: [(y, text, color)] at the right edge; nudged apart vertically."""
    items = sorted(items, key=lambda t: t[0])
    ys = [it[0] for it in items]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + ymin_gap)
    for (y0, txt, c), y in zip(items, ys):
        ax.annotate(txt, xy=(x[-1], y0), xytext=(x[-1] + 0.3, y), fontsize=6.5, color='0.15',
                    va='center', ha='left', annotation_clip=False,
                    arrowprops=dict(arrowstyle='-', color=c, lw=0.6, shrinkA=0, shrinkB=2))


for ax, key, floor, title in [(axb, 'divm', flm, '(b) Forced response'),
                              (axc, 'div', fl, '(c) Per-model median')]:
    hl = []
    for r in rows:
        if r['b'] not in HIGHLIGHT:
            ax.plot(x, r[key], color='0.75', lw=0.8, zorder=1)
    for r in rows:
        if r['b'] in HIGHLIGHT:
            c = color[r['b']]
            ax.plot(x, r[key], color=c, lw=1.5, zorder=3, path_effects=OUTLINE)
            ax.plot(x, r[key], ls='', marker='o', ms=3.5, color=c, mec='0.35', mew=0.5, zorder=4)
            hl.append((r[key][-1], SHORT.get(r['b'], r['name']), c))
    ax.plot(x, floor, color='0.45', lw=2.5, alpha=0.5, solid_capstyle='butt', zorder=2)
    ax.text(x[-2], floor[-2] - 0.015 * yhi, 'noise floor', fontsize=6.5, color='0.4', va='top', ha='center')
    direct_labels(ax, hl, 0.045 * yhi)
    ax.set_xticks(x)
    ax.set_xticklabels([gwl_tex(g) for g in gwls], fontsize=7.5)
    ax.set_xlim(x[0] - 0.2, x[-1] + 0.2)
    ax.set_ylim(0, yhi)
    ax.set_title(title, loc='left', fontsize=8.5)
    ax.set_xlabel('Global warming level' + ('' if use_common else
                  '\n(n = ' + ', '.join(str(n) for n in nmod) + ' models)'), fontsize=8)
axb.set_ylabel('Pair divergence (ICV units)', fontsize=8)
axc.sharey(axb)
axc.tick_params(labelleft=False)

for ax in axes:
    ax.grid(alpha=0.3, lw=0.5)
    ax.tick_params(labelsize=7.5)

# ---- legends along the bottom ----
pa, pb, pc = axa.get_position(), axb.get_position(), axc.get_position()
y_leg = pa.y0 - 0.10
ha = [Line2D([], [], ls='', marker='o', color=color[r['b']], ms=5, label=r['name']) for r in rows]
fig.legend(handles=ha, fontsize=8.0, loc='upper left', bbox_to_anchor=(pa.x0 - 0.04, y_leg),
           ncol=3, frameon=False, handletextpad=0.2, columnspacing=0.8)
hb = [Line2D([], [], color='0.6', lw=1.8, marker='o', ms=3.5, mec='0.35', mew=0.5,
             label='Highlighted biome (colors as in a)'),
      Line2D([], [], color='0.75', lw=0.8, label='Other biomes'),
      Line2D([], [], color='0.45', lw=2.5, alpha=0.5, label='Analytic noise floor')]
fig.legend(handles=hb, fontsize=8.0, loc='upper center', bbox_to_anchor=(0.5 * (pb.x0 + pb.x1+0.1), y_leg),
           ncol=1, frameon=False, handlelength=1.8)
hs = [Line2D([], [], ls='', marker='o', color='0.5', ms=np.sqrt(s), label=gwl_tex(g))
      for s, g in zip(sizes, gwls)]
fig.legend(handles=hs, fontsize=8.0, loc='upper center', bbox_to_anchor=(0.5 * (pc.x0 + pc.x1), y_leg),
           ncol=len(hs), frameon=False, handletextpad=0.1, columnspacing=0.8,
           title='Marker size in (a)', title_fontsize=8.0)

pcm.savefig(fig, 'departure_divergence.png')

print(f'\n--- area-weighted medians per GWL ({", ".join(gwls)}); n models {nmod} ---')
print('    departure mmm | divergence mmm | divergence per-model | comove mmm')
for r in rows:
    f = lambda a: ' '.join(f'{v:5.2f}' for v in a)  # noqa: E731
    print(f'  {r["name"]:<24} {f(r["dep"])} | {f(r["divm"])} | {f(r["div"])} | {f(r["com"])}')
print('  noise floor per-model: ' + ' '.join(f'{v:.2f}' for v in fl)
      + ' | mmm: ' + ' '.join(f'{v:.2f}' for v in flm))
