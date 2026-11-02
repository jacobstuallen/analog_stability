#!/bin/python
#
# Is the overlap of "uncertain" stippling with low forced retention mechanical?
#
# Hypothesis (mechanical): a pair can only be UNCERTAIN (retained by 20-80% of
# models) if its forced (mmm) future distance is near the Dt threshold, so
# uncertainty must pile up where forced retention is low. If that is all that
# is going on:
#   (1) the fraction of uncertain analogs should be a tight function of forced
#       retention (panel a), and
#   (2) uncertainty should peak where the cell's median forced pair distance
#       sits at the threshold and fall off on both sides (panel b).
# Real model disagreement shows up as:
#   - uncertainty in cells with forced retention >= 0.9 (models split even
#     though the forced response keeps the pair), and
#   - pairs in low-retention cells that a MAJORITY of models retain (panel c),
#     i.e. the forced response loses a pair most models keep.
#
# Output: plot/figures/diag_retention_uncertainty.png + console table.
# Dt = DT_STRICT (1 sigma). Area-weighted (cos lat). Coverage mask as in Fig 2.
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi, norm
import plot_common as pcm
from plot_common import wmean, gwl_label, dt_tag

pcm.setup_style()
R = pcm.Retention(mask_low_coverage=True)
ds = R.ds
dt = pcm.DT_STRICT
t = dt_tag(dt)
GWLS = [g for g in ('gwl2', 'gwl3', 'gwl4') if g in R.gwls]
LO_AGREE, HI_AGREE = 0.2, 0.8

# Tenths: kept cells have n = 10 analogs, so retention and uncert are k/10
edges = np.arange(-0.05, 1.06, 0.1)
cent = 0.5 * (edges[:-1] + edges[1:])

# futdist_mmm is the median over analogs of the Mahalanobis distance D
# (ICV units) between the two future climates. Convert to sigma so it is on
# the same scale as Dt. (Assumes m = 2 variables; check cfg if that changes.)
M_VARS = 2


def d_to_sigma(D):
    return norm.ppf(0.5 * (1.0 + chi.cdf(D, M_VARS)))


fig, axes = plt.subplots(3, len(GWLS), figsize=(4.0 * len(GWLS), 10.5),
                         gridspec_kw=dict(hspace=0.45, wspace=0.28, top=0.93))
axes = np.atleast_2d(axes)
if axes.shape[0] != 3:
    axes = axes.T

rows = []
for c, g in enumerate(GWLS):
    ret = R.masked(R.central(g, dt))
    unc = R.masked(ds[f'uncert_{g}_{t}'].values)
    ok = np.isfinite(ret) & np.isfinite(unc)
    w = R.W[ok]
    x, y = ret[ok], unc[ok]

    # ---- (a) conditional distribution of uncert given forced retention ----
    H, _, _ = np.histogram2d(x, y, bins=[edges, edges], weights=w)
    colsum = H.sum(axis=1, keepdims=True)
    Hc = np.where(colsum > 0, H / np.where(colsum > 0, colsum, 1), np.nan)
    ax = axes[0, c]
    pm = ax.pcolormesh(edges, edges, Hc.T, cmap='Greys', vmin=0, vmax=0.6)
    mean_unc = np.array([wmean(y[(x > lo) & (x <= hi)], w[(x > lo) & (x <= hi)])
                         for lo, hi in zip(edges[:-1], edges[1:])])
    ax.plot(cent, mean_unc, 'o-', color='C3', ms=3, lw=1.2, label='mean')
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.15)
    ax.set_xlabel('Forced retention (fraction)', fontsize=8)
    ax.set_ylabel('Fraction of analogs uncertain', fontsize=8)
    ax.set_title(f'a{c + 1}) {gwl_label(g)}: uncertain | forced retention', loc='left', fontsize=8.5)
    ax.tick_params(labelsize=7)
    # share of each retention class in area (top axis annotation)
    area = colsum.ravel() / colsum.sum()
    for xc, a in zip(cent, area):
        if a > 0:
            ax.text(xc, 1.13, f'{100 * a:.0f}', ha='center', va='top', fontsize=5.5,
                    color='0.4')

    # ---- (b) uncert vs median forced pair distance relative to threshold ----
    fd = R.masked(ds[f'futdist_mmm_{g}'].values)
    okb = ok & np.isfinite(fd)
    s_med = d_to_sigma(fd[okb])
    rel = s_med - dt
    ub = unc[okb]; wb = R.W[okb]
    be = np.arange(-1.5, 2.01, 0.1)
    bc = 0.5 * (be[:-1] + be[1:])
    mb = np.array([wmean(ub[(rel > lo) & (rel <= hi)], wb[(rel > lo) & (rel <= hi)])
                   for lo, hi in zip(be[:-1], be[1:])])
    nb = np.array([wb[(rel > lo) & (rel <= hi)].sum() for lo, hi in zip(be[:-1], be[1:])])
    mb[nb < 1e-3 * wb.sum()] = np.nan
    ax = axes[1, c]
    ax.plot(bc, mb, '-', color='k', lw=1.3)
    ax.axvline(0, color='C3', lw=0.8, ls='--')
    ax.set_xlim(be[0], be[-1]); ax.set_ylim(0, 1)
    ax.set_xlabel(r'Median forced pair $\sigma$ minus D$_t$', fontsize=8)
    ax.set_ylabel('Mean fraction uncertain', fontsize=8)
    ax.set_title(f'b{c + 1}) {gwl_label(g)}: uncertain vs distance to threshold', loc='left',
                 fontsize=8.5)
    ax.grid(alpha=0.3, lw=0.5); ax.tick_params(labelsize=7)

    # ---- (c) per-pair model agreement, by forced-retention class of the cell ----
    pc = ds[f'pairconf_{g}_{t}'].values.astype(float)        # (lat, lon, rank), percent
    pc[pc == 255] = np.nan
    pc /= 100.0
    classes = {'forced ret. ≤ 0.3': (ret <= 0.35),
               '0.4-0.8': (ret > 0.35) & (ret <= 0.85),
               '≥ 0.9': (ret > 0.85)}
    ax = axes[2, c]
    pe = np.linspace(0, 1, 21)
    for (lab, cm), col in zip(classes.items(), ('#792b05', '#f07919', '#c9a227')):
        m = cm & ok
        p = pc[m]                                             # (ncell, rank)
        wp = np.broadcast_to(R.W[m][:, None], p.shape)
        fin = np.isfinite(p)
        hist, _ = np.histogram(p[fin], bins=pe, weights=wp[fin])
        hist = hist / hist.sum() if hist.sum() > 0 else hist
        ax.stairs(hist, pe, color=col, lw=1.3, label=lab)
        pf, wf = p[fin], wp[fin]
        rows.append(dict(
            g=g, cls=lab, area=R.W[m].sum() / R.W[ok].sum(),
            unc=np.sum(wf * ((pf > LO_AGREE) & (pf < HI_AGREE))) / wf.sum(),
            maj_keep=np.sum(wf * (pf >= 0.5)) / wf.sum(),
            rob_lost=np.sum(wf * (pf <= LO_AGREE)) / wf.sum()))
    for v in (LO_AGREE, HI_AGREE):
        ax.axvline(v, color='0.5', lw=0.6, ls=':')
    ax.set_xlim(0, 1)
    ax.set_xlabel('Fraction of models retaining the pair', fontsize=8)
    ax.set_ylabel('Area-weighted share of pairs', fontsize=8)
    ax.set_title(f'c{c + 1}) {gwl_label(g)}: per-pair agreement', loc='left', fontsize=8.5)
    ax.legend(fontsize=6.5, frameon=False, loc='upper center')
    ax.grid(alpha=0.3, lw=0.5); ax.tick_params(labelsize=7)

    # cell-level summaries
    unc_w = w * y
    rows_cell = dict(
        r=np.corrcoef(x, y)[0, 1],
        unc_in_high=unc_w[x > 0.85].sum() / unc_w.sum(),
        unc_in_low=unc_w[x <= 0.55].sum() / unc_w.sum())
    rows.append(dict(g=g, cls='CELL', **rows_cell))

cax = fig.add_axes([0.92, 0.70, 0.012, 0.18])
fig.colorbar(pm, cax=cax, label='share of cells in column')
fig.suptitle(f'Is uncertainty mechanical? D$_t$ = {dt:g}$\\sigma$, '
             f'{pcm.HEADLINE} central estimate. Grey numbers: % of area per column.',
             fontsize=9, y=0.995)
pcm.savefig(fig, 'diag_retention_uncertainty.png', dpi=200)

# ==============================
# console
# ==============================
print(f'\n--- Dt={dt:g}: cell level (area-weighted) ---')
print('  corr(forced ret, uncert) | share of uncertain-analog area in cells with ret>=0.9 | ret<=0.5')
for r in rows:
    if r['cls'] == 'CELL':
        print(f'  {r["g"]:<6} r={r["r"]:+.2f}   high={r["unc_in_high"]:.2f}   low={r["unc_in_low"]:.2f}')
print('\n--- pair level, by forced-retention class of the cell ---')
print('  class            area   uncertain   majority-of-models-keep   robustly lost')
for r in rows:
    if r['cls'] != 'CELL':
        print(f'  {r["g"]:<6} {r["cls"]:<18} {r["area"]:.2f}   {r["unc"]:.2f}        '
              f'{r["maj_keep"]:.2f}                     {r["rob_lost"]:.2f}')
print('\nReading: mechanical-only => uncertainty ~0 in ret>=0.9 cells, peaks at 0 in (b), and few'
      '\npairs in low-retention cells kept by a majority of models. Large "majority keep" in'
      '\nlow-retention cells => forced (mmm) loses pairs most models keep (mean != median).')
