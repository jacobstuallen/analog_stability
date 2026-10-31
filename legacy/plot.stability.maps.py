#!/bin/python
# Figure 1: global maps of analog dissimilarity at GWL2 and GWL3.

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import cartopy.crs as ccrs
import cartopy.feature as cfeature

data_root = '/glade/campaign/ral/risc/jsallen/TNC'
infile = f'{data_root}/analog_stability/analog_stability_gwl.nc'
outfile = './figures/figure_stability_maps.png'
vmax = 4.0

try:
    fdir = '/glade/work/jsallen/conda-envs/earth/fonts/Avenir-Medium.otf'
    font_manager.fontManager.addfont(fdir)
    prop = font_manager.FontProperties(fname=fdir)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = prop.get_name()
except Exception:
    pass
plt.rcParams['axes.linewidth'] = 0.6

ds = xr.open_dataset(infile)
lat = ds['lat'].values; lon = ds['lon'].values
s2 = ds['sigma_gwl2_mean'].values
s3 = ds['sigma_gwl3_mean'].values
LON, LAT = np.meshgrid(lon, lat)
n_models = ds.attrs.get('n_models', '?')
varlist = ds.attrs.get('variables', '')

fig = plt.figure(figsize=(6, 6.5))
gs = fig.add_gridspec(2, 1, wspace=0.06, left=0.04, right=0.96,
                      top=0.92, bottom=0.16)
proj = ccrs.Robinson()
cmap = 'magma'

def draw(ax, field, title):
    ax.add_feature(cfeature.LAND, facecolor='#e8e8e8', zorder=0)
    fin = np.isfinite(field)
    sc = ax.scatter(LON[fin], LAT[fin], c=field[fin], s=0.8, cmap=cmap,
                    vmin=0, vmax=vmax, transform=ccrs.PlateCarree(),
                    zorder=1, rasterized=True)
    ax.add_feature(cfeature.COASTLINE, lw=0.25, zorder=2)
    ax.set_global()
    ax.set_title(title, loc='left', fontsize=11)
    return sc

axA = fig.add_subplot(gs[0, 0], projection=proj)
draw(axA, s2, r'a) GWL2 (+2$\sigma$) dissimilarity to historical analog')
axB = fig.add_subplot(gs[1, 0], projection=proj)
scB = draw(axB, s3, r'b) GWL3 (+3$\sigma$')

cax = fig.add_axes([0.20, 0.10, 0.60, 0.025])
cb = fig.colorbar(scB, cax=cax, orientation='horizontal', extend='max')
cb.set_label(r'$\sigma$ dissimilarity', fontsize=9)
cb.ax.tick_params(labelsize=8)

#fig.suptitle(f'Analog dissimilarity under global warming {varlist} '
#             f'({n_models} CMIP6 models, SSP5-8.5)', fontsize=12, y=0.97)

import os
os.makedirs('./figures', exist_ok=True)
plt.savefig(outfile, dpi=400)
print(f'Saved: {outfile}')
plt.close()
