#!/bin/python
#
# SI FIGURE: example focal cells for the means-vs-extremes comparison.
# Each panel: one focal cell (star), its 10 extremes-selected analogs (blue)
# and 10 means-selected analogs (orange). Orange filled = means analog that is
# also within Dt=1 in extremes space today (cross run pairconf_gwl0_dt1p0).
# Cells picked objectively: full sets in both runs, overlap closest to the
# biome's area-weighted median, then means-selected share closest to the biome
# mean. Moved out of the main-text figure 2026-11-02.
#
# Writes: means_extremes_examples.png
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import plot_common as pcm
from plot_common import cfg
import means_extremes_common as mec
from means_extremes_common import C_MEANS, C_EXT4, SIG

DT = pcm.DT_STRICT
EXAMPLES = [1, 14, 5]            # boreal, tropical moist, Mediterranean

pcm.setup_style()
M = mec.MeansExtremes()
rows = M.biome_rows(DT)
crossg = {'gwl0': M.ret(M.rc, 'gwl0', DT)}
fig = plt.figure(figsize=(7.2, 2.0 * len(EXAMPLES) + 0.6))
import cartopy.crs as ccrs
import cartopy.feature as cfeature
pc = ccrs.PlateCarree()
N = cfg.N_ANALOGS
pconf = M.rc.get(f'pairconf_gwl0_{pcm.dt_tag(DT)}')
pconf = pconf.values if pconf is not None else None
if pconf is None:
    print('  cross pairconf_gwl0 not found: means analogs drawn without pass/fail')
if M.ov is None:
    print('  compare_varsets.nc not found: examples picked on means-selected share only')

full = (np.nan_to_num(M.se.n_analogs.values) >= N) & (np.nan_to_num(M.sm.n_analogs.values) >= N)
ovl = M.ov['overlap_means_in_ext'].values if M.ov is not None else None
m0 = crossg['gwl0']
ex_list = [b for b in EXAMPLES if any(r['b'] == b for r in rows)]
from matplotlib.gridspec import GridSpec
bot = GridSpec(max(len(ex_list), 1), 1, figure=fig, hspace=0.45)
print('\n--- (c) example focal cells ---')
for k, b in enumerate(ex_list):
    r = next(rr for rr in rows if rr['b'] == b)
    cand = r['bm'] & full & np.isfinite(m0)
    score = np.abs(m0 - M.bmean(m0, r['bm']))
    if ovl is not None:
        cand &= np.isfinite(ovl)
        dov = np.abs(ovl - M.bmedian(ovl, r['bm']))
        cand &= dov <= np.nanmin(np.where(cand, dov, np.nan)) + 1e-6
    ii, jj = np.where(cand)
    if len(ii) == 0:
        print(f'  {r["name"]}: no candidate cell')
        continue
    s = score[ii, jj]
    pick = np.flatnonzero(s <= s.min() + 1e-6)[0]          # deterministic: first in grid order
    i, j = ii[pick], jj[pick]
    flat, flon = M.lat[i], M.lon[j]
    ela, elo = M.se.analog_lat.values[i, j], M.se.analog_lon.values[i, j]
    mla, mlo = M.sm.analog_lat.values[i, j], M.sm.analog_lon.values[i, j]
    ok = pconf[i, j] if pconf is not None else None
    passed = (ok != 255) & (ok >= 50) if ok is not None else np.zeros(N, bool)

    axc = fig.add_subplot(bot[k], projection=ccrs.Robinson(central_longitude=float(flon)))
    lats = np.r_[flat, ela, mla]; lons = np.r_[flon, elo, mlo]
    good = np.isfinite(lats) & np.isfinite(lons)
    # extent around all points (lon relative to the focal cell to handle the dateline)
    dl = (lons[good] - flon + 180) % 360 - 180
    x0, x1 = max(dl.min() - 15, -180), min(dl.max() + 15, 180)
    y0, y1 = max(lats[good].min() - 10, -60), min(lats[good].max() + 10, 85)
    axc.set_extent([x0, x1, y0, y1], crs=ccrs.PlateCarree(central_longitude=float(flon)))
    bmask = np.where(M.biome[::4, ::4] == b, 1.0, np.nan)
    axc.pcolormesh(M.lon[::4], M.lat[::4], bmask, cmap=plt.matplotlib.colors.ListedColormap(['0.86']),
                   vmin=0, vmax=1, transform=pc, zorder=1, rasterized=True)
    axc.add_feature(cfeature.COASTLINE.with_scale('110m'), linewidth=0.3, color='0.4', zorder=2)
    axc.scatter(elo, ela, s=22, marker='o', color=C_EXT4, edgecolor='white', lw=0.5,
                transform=pc, zorder=4)
    axc.scatter(mlo[passed], mla[passed], s=24, marker='s', color=C_MEANS, edgecolor='white',
                lw=0.5, transform=pc, zorder=5)
    axc.scatter(mlo[~passed], mla[~passed], s=24, marker='s', facecolor='white', edgecolor=C_MEANS,
                lw=1.0, transform=pc, zorder=5)
    axc.scatter([flon], [flat], s=90, marker='*', color='k', edgecolor='white', lw=0.5,
                transform=pc, zorder=6)
    n_ok = int(passed.sum()) if pconf is not None else -1
    ov_txt = f', overlap {ovl[i, j]:.0%}' if ovl is not None else ''
    ok_txt = f'\n{n_ok}/{N} means analogs valid for extremes' if n_ok >= 0 else ''
    lab = f'({chr(97 + k)}) '
    axc.set_title(f'{lab}{r["name"]}{ov_txt}{ok_txt}', loc='left', fontsize=6.8)
    print(f'  {r["name"]:<24} focal ({flat:.2f}, {flon:.2f}) overlap '
          f'{ovl[i, j] if ovl is not None else np.nan:.2f} means-valid {n_ok}/{N}')
if ex_list:
    hc = [Line2D([], [], ls='', marker='*', color='k', ms=8, label='Focal site'),
          Line2D([], [], ls='', marker='o', color=C_EXT4, ms=5, label='Extremes-selected analog'),
          Line2D([], [], ls='', marker='s', color=C_MEANS, ms=5, label=f'Means-selected, within {DT:g}{SIG} for extremes'),
          Line2D([], [], ls='', marker='s', mfc='white', mec=C_MEANS, ms=5, label='Means-selected, not within'),
          Line2D([], [], ls='', marker='s', color='0.86', ms=7, label='Biome')]
    fig.legend(handles=hc, fontsize=6.5, loc='lower center', bbox_to_anchor=(0.5, -0.04),
               ncol=3, frameon=False, handletextpad=0.3, columnspacing=0.9)

pcm.savefig(fig, 'means_extremes_examples.png')
print('Done.')
