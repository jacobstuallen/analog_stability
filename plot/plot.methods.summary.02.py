#!/bin/python
# jsallen@ucar.edu
#
# Methods demonstration figure for the sigma-dissimilarity analog method.
#
# Goal: show, for ONE example focal cell, how Mahony (2017) sigma
# dissimilarity is constructed, and make sure she works!
#
# Layout (2 x 2):
#   a) Raw climate space: within-biome candidate cells in TXx-Rx5day,
#       focal cell marked, focal-cell ICV scatter (annual values) +
#       1-sigma ICV ellipse. Shows the variability the metric is scaled to.
#   b) Standardized + PC-rotated space (focal cell's ICV PCs): same
#       candidates, now as standardized anomalies of focal ICV. Concentric
#       sigma rings (chi-distribution 1/2/3 sigma) show how raw distance
#       maps to dissimilarity. Best analog highlighted.
#   c) Map: sigma dissimilarity of every cell in the biome to the FOCAL
#       cell (not to its own best analog) - i.e. "where are this cell's
#       analogs?"
#   d) Histogram: within-biome pairwise distance distribution (sigma),
#       with candidate Dt percentiles (10/15/20) marked for context.
#
# Single representative biome + cell, configurable below.
#
# NOTE: ICV construction (detrend, ddof=2, log1p for rx5day, QC max) and the
# sigma convention (half-normal, clipped [0, sigma_cap]) now match
# analog_historical_sigma.2var.py exactly via icv_utils. This script
# previously used undetrended nanstd/nanmean and a negative-allowed sigma
# convention, which did NOT match the main script despite the header saying
# so -- fixed here.

import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib.patches import Ellipse
from scipy.stats import chi, norm
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from icv_utils import detrend_icv, icv_correlation

# Font/plot settings
font_dir = '/glade/work/jsallen/conda-envs/earth/fonts/Avenir-Medium.otf'
font_manager.fontManager.addfont(font_dir)
prop = font_manager.FontProperties(fname=font_dir)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = prop.get_name()
plt.rcParams['axes.linewidth'] = 0.6
plt.rcParams['xtick.major.width'] = 0.6
plt.rcParams['ytick.major.width'] = 0.6

# User vars
# -------------
data_root = '/glade/campaign/ral/risc/jsallen/TNC'
txx_hist_file = f'{data_root}/CCKP/ERA5/timeseries-txx-annual-mean_era5-x0.25_era5-x0.25-historical_timeseries_mean_1950-2024.nc'
rx5_hist_file = f'{data_root}/CCKP/ERA5/timeseries-rx5day-annual-mean_era5-x0.25_era5-x0.25-historical_timeseries_mean_1950-2024.nc'
biome_file    = '/glade/u/home/jsallen/projects/tnc_2025/analogs/biomes/biomes.analog.gridded.nc'

clim_start = 1991
clim_end   = 2020
pc_var_min = 0.01  # get rid of noise
sigma_cap  = 12.0   # matches main script (was 12.0 here -- reverted to match)

# Per-variable config, mirrored from analog_historical_sigma.2var.py's
# VARIABLES dict so the ICV built here is identical.
TXX_LOG, TXX_QC_MAX = False, 70.0
RX5_LOG, RX5_QC_MAX = True, 3000.0

# Example focal cell: Edwards Plateau, TX (Temperate Grasslands biome).
focal_biome = 10   # grasssslands
focal_lat   = 30.5 # where she at
focal_lon   = -98.5 # where she at
focal_label = 'Edwards Plateau, TX'
biome_name  = 'Temperate Grasslands, Savannas & Shrublands'

dt_levels = [10, 15, 20]   # candidate Dt percentiles to seeeee

outfile = './figures/figure_methods_sigma_demo.png'
# ---------------
# End of user vars

# Load + build ICV exactly as in the main script
# ----------------------------------------------
txx_ds = xr.open_dataset(txx_hist_file).sortby('lat')
rx5_ds = xr.open_dataset(rx5_hist_file).sortby('lat')
biome_ds = xr.open_dataset(biome_file).sortby('lat')

bmin = float(biome_ds.lat.min()); bmax = float(biome_ds.lat.max())
txx_ds = txx_ds.sel(lat=slice(bmin, bmax))
rx5_ds = rx5_ds.sel(lat=slice(bmin, bmax))

txx_ts = txx_ds['timeseries-txx-annual-mean'].sel(time=slice(str(clim_start), str(clim_end)))
rx5_ts = rx5_ds['timeseries-rx5day-annual-mean'].sel(time=slice(str(clim_start), str(clim_end)))
biome_mask = biome_ds['BIOME_ID']

lat_arr = txx_ts.lat.values
lon_arr = txx_ts.lon.values

# Raw (untransformed) arrays -- kept for panel (a) display, which should
# show actual annual values, not log-space ones.
txx_raw = txx_ts.values.astype(np.float64)   # (T, lat, lon)
rx5_raw = rx5_ts.values.astype(np.float64)
txx_raw[txx_raw > TXX_QC_MAX] = np.nan
rx5_raw[rx5_raw > RX5_QC_MAX] = np.nan

# Arrays used for the actual metric, QC'd then log1p'd where main script
# does so (rx5day only).
txx_arr = txx_raw.copy()
rx5_arr = np.log1p(rx5_raw) if RX5_LOG else rx5_raw.copy()

# Detrended ICV (mean, sd ddof=2, anomaly series) -- identical construction
# to analog_historical_sigma.2var.py, via icv_utils.
txx_mean, txx_sd, txx_anom = detrend_icv(txx_arr)
rx5_mean, rx5_sd, rx5_anom = detrend_icv(rx5_arr)
corr = icv_correlation(txx_anom, rx5_anom, txx_sd, rx5_sd)

T = txx_ts.sizes['time']

# Focal cell indices
ilat = int(np.argmin(np.abs(lat_arr - focal_lat)))
ilon = int(np.argmin(np.abs(lon_arr - focal_lon)))

# Within-biome candidate cells
bm = (biome_mask.values == focal_biome)
clat, clon = np.where(bm)
mu_t = txx_mean[clat, clon]
mu_r = rx5_mean[clat, clon]
valid = np.isfinite(mu_t) & np.isfinite(mu_r)
clat, clon = clat[valid], clon[valid]
mu_t, mu_r = mu_t[valid], mu_r[valid]

# Focal ICV (detrended sd/corr; mean unaffected by detrending)
f_mu_t, f_mu_r = txx_mean[ilat, ilon], rx5_mean[ilat, ilon]
f_sd_t, f_sd_r = txx_sd[ilat, ilon], rx5_sd[ilat, ilon]
f_rho = corr[ilat, ilon]
# Raw annual values for the panel (a) scatter (log-space for rx5 if RX5_LOG,
# to stay in the same space as f_mu_r/f_sd_r/the ellipse).
f_txx_years = txx_arr[:, ilat, ilon]
f_rx5_years = rx5_arr[:, ilat, ilon]

# Standardize candidates by focal ICV, rotate into focal PCs (K=2 closed
# form -- equivalent to np.linalg.eigh on the 2x2 correlation matrix used
# in the main script: eigenvalues 1+rho, 1-rho with eigenvectors
# (1,1)/sqrt2, (1,-1)/sqrt2).
zt = (mu_t - f_mu_t) / f_sd_t
zr = (mu_r - f_mu_r) / f_sd_r
inv_sqrt2 = 1 / np.sqrt(2)
pc1 = (zt + zr) * inv_sqrt2
pc2 = (zt - zr) * inv_sqrt2
ev1, ev2 = 1 + f_rho, 1 - f_rho
keep1, keep2 = ev1 >= pc_var_min, ev2 >= pc_var_min
d2 = np.zeros_like(pc1)
if keep1: d2 += pc1**2 / ev1
if keep2: d2 += pc2**2 / ev2
d = np.sqrt(d2)
m = int(keep1) + int(keep2)

# distance to self ~ 0; exclude focal cell itself for best-analog
self_sel = (clat == ilat) & (clon == ilon)
d_excl = d.copy()
d_excl[self_sel] = np.inf
best = int(np.argmin(d_excl))

# Sigma: half-normal convention, matches sigma_from_distance() in the main
# script exactly (0 = identical, monotone increasing, never negative).
def to_sigma(dist, m):
    p = np.clip(chi.cdf(dist, df=m), 0.0, 1.0 - 1e-25)
    return np.clip(norm.ppf(0.5 + p / 2), 0.0, sigma_cap)

sigma_all = to_sigma(d, m)

print(f'Focal: {focal_label}  TXx={f_mu_t:.1f} '
      f'Rx5({"log1p " if RX5_LOG else ""}mm)={f_mu_r:.2f} rho={f_rho:.2f} M={m}')
print(f'Best analog dmaha={d_excl[best]:.4f} sigma={sigma_all[best]:.3f} '
      f'at lat={lat_arr[clat[best]]:.2f} lon={lon_arr[clon[best]]:.2f}')

# Within-biome pairwise distance sample (focal-to-all already gives one row
# for the histogram use focal-to-all sigma as a stand-in demonstration).
sig_pool = sigma_all[~self_sel]

# show me whatchu got
# -------------------
fig = plt.figure(figsize=(10.5, 8.0))
gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25,
                      left=0.08, right=0.95, top=0.90, bottom=0.08)

# (a) Raw climate space + focal ICV
axa = fig.add_subplot(gs[0, 0])
axa.scatter(mu_t, mu_r, s=4, c='#bdbdbd', alpha=0.5, edgecolors='none',
            label='Within-biome cells', zorder=1)
axa.scatter(f_txx_years, f_rx5_years, s=14, c='#2166ac', alpha=0.7,
            edgecolors='none', label='Focal ICV (annual)', zorder=3)
cov = np.array([[f_sd_t**2, f_rho*f_sd_t*f_sd_r],
                [f_rho*f_sd_t*f_sd_r, f_sd_r**2]])
evals, evecs = np.linalg.eigh(cov)
order = np.argsort(evals)[::-1]
evals, evecs = evals[order], evecs[:, order]
angle = np.degrees(np.arctan2(evecs[1, 0], evecs[0, 0]))
for nsig in [1, 2]:
    w, h = 2 * nsig * np.sqrt(evals)
    axa.add_patch(Ellipse((f_mu_t, f_mu_r), w, h, angle=angle,
                          fill=False, edgecolor='#2166ac', lw=1.0,
                          alpha=0.6, ls='--', zorder=4))
axa.scatter(f_mu_t, f_mu_r, s=120, marker='*', c='k', zorder=5,
            label='Focal cell')
axa.scatter(mu_t[best], mu_r[best], s=70, marker='D', facecolor='#d6604d',
            edgecolor='white', lw=0.6, zorder=5, label='Best analog')
axa.set_xlabel('TXx (°C)', fontsize=9)
axa.set_ylabel('Rx5day (log1p mm)' if RX5_LOG else 'Rx5day (mm)', fontsize=9)
axa.set_title('a) Raw climate space + focal ICV: Edwards Plateau, TX', loc='left', fontsize=10)
axa.legend(fontsize=7, loc='best', framealpha=0.9)
axa.tick_params(labelsize=8)
axa.grid(alpha=0.3)

# Standardized + rotated space with sigma rings
axb = fig.add_subplot(gs[0, 1])
u1 = pc1 / np.sqrt(ev1) if keep1 else np.zeros_like(pc1)
u2 = pc2 / np.sqrt(ev2) if keep2 else np.zeros_like(pc2)
sc = axb.scatter(u1, u2, s=5, c=sigma_all, cmap='viridis', vmin=0, vmax=4,
                 alpha=0.7, edgecolors='none', zorder=2)
theta = np.linspace(0, 2*np.pi, 200)
for nsig in [1, 2, 3]:
    D_ring = chi.ppf(norm.cdf(nsig), df=m)
    axb.plot(D_ring*np.cos(theta), D_ring*np.sin(theta), color='k',
             lw=0.7, alpha=0.5, ls='--', zorder=3)
    axb.text(D_ring*0.71, D_ring*0.71, f'{nsig}\u03c3', fontsize=7,
             alpha=0.7, ha='center', va='center')
axb.scatter(0, 0, s=120, marker='*', c='k', zorder=5)
axb.scatter(u1[best], u2[best], s=70, marker='D', facecolor='#d6604d',
            edgecolor='white', lw=0.6, zorder=5)
axb.set_xlabel('Whitened PC1 (ICV units)', fontsize=9)
axb.set_ylabel('Whitened PC2 (ICV units)', fontsize=9)
axb.set_title('b) Standardized + rotated to focal ICV', loc='left', fontsize=10)
axb.set_aspect('equal', adjustable='datalim')
axb.tick_params(labelsize=8)
axb.grid(alpha=0.3)
cb = fig.colorbar(sc, ax=axb, fraction=0.046, pad=0.04)
cb.set_label(r'$\sigma$ dissimilarity', fontsize=8)
cb.ax.tick_params(labelsize=7)

# (c) Map of sigma to focal cell across the biome
axc = fig.add_subplot(gs[1, 0], projection=ccrs.Miller())
axc.add_feature(cfeature.LAND, facecolor='#f0f0f0', zorder=0)
axc.add_feature(cfeature.COASTLINE, lw=0.3, zorder=1)
sc2 = axc.scatter(lon_arr[clon], lat_arr[clat], c=sigma_all, s=2,
                  cmap='viridis', vmin=0, vmax=4, transform=ccrs.PlateCarree(),
                  zorder=2)
axc.scatter(focal_lon, focal_lat, s=150, marker='*', c='red',
            edgecolor='white', lw=0.5, transform=ccrs.PlateCarree(), zorder=3)
axc.set_global()
axc.set_title(r'c) $\sigma$ dissimilarity to focal cell (this biome)', loc='left', fontsize=10)
cb2 = fig.colorbar(sc2, ax=axc, fraction=0.025, pad=0.04)
cb2.set_label(r' $\sigma$ dissimilarity', fontsize=8)
cb2.ax.tick_params(labelsize=7)

# (d) Distance distribution + Dt candidates
axd = fig.add_subplot(gs[1, 1])
axd.hist(sig_pool[np.isfinite(sig_pool)], bins=100, color='#bdbdbd',
         edgecolor='white', lw=0.3)
for lev, col in zip(dt_levels, ['#4393c3', '#f4a582', '#d6604d']):
    v = np.nanpercentile(sig_pool, lev)
    axd.axvline(v, color=col, lw=1.2, ls='--',
                label=f'D$_t$ = {lev}th pctile ({v:.2f}\u03c3)')
axd.set_xlabel(r' $\sigma$ dissimilarity to focal cell', fontsize=9)
axd.set_ylabel('Cell count', fontsize=9)
axd.set_title('d) Within-biome dissimilarity distribution', loc='left', fontsize=10)
axd.legend(fontsize=7, framealpha=0.9)
axd.tick_params(labelsize=8)
axd.grid(alpha=0.3, axis='y')
axd.set_xlim(0, 9)

os.makedirs('./figures', exist_ok=True)
plt.savefig(outfile, dpi=400, bbox_inches='tight')
plt.show()
print(f'Saved: {outfile}')
plt.close()
