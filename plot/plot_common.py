# plot_common.py
#
# Shared loading, masking, labels and map helpers for the plot/ scripts.
# Every figure reads its data and choices from here, so the headline estimate,
# coverage mask and biome order are the same in all figures.
#
# HEADLINE (env ANALOG_HEADLINE):
#   'mmm'    (default): central estimate = retention from the multi-model-mean
#            future (forced response). Per-model spread / tiers = uncertainty.
#   'median': central estimate = ensemble median of per-model retention.
import os
import sys
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib.colors import ListedColormap

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import analog_config as cfg  # noqa: E402
from analog_common import area_weights, wmean, wmedian  # noqa: E402,F401

FIG_DIR = os.environ.get('ANALOG_FIG_DIR', os.path.join(cfg.OUT_DIR, 'figures'))
HEADLINE = os.environ.get('ANALOG_HEADLINE', 'mmm')
if HEADLINE not in ('mmm', 'median'):
    raise ValueError(f'unknown ANALOG_HEADLINE {HEADLINE!r}')

# Biomes shown, in display order (tropics -> high latitudes). Biomes with
# fewer than MIN_PLOT_CELLS cells with analogs are skipped and reported.
BIOME_SHORT = {
    14: 'Trop. moist broadleaf', 12: 'Trop. dry broadleaf', 11: 'Trop. coniferous',
    13: 'Trop. grass/savanna', 3: 'Flooded grass/savanna', 4: 'Mangroves',
    2: 'Deserts/xeric', 5: 'Mediterranean', 6: 'Montane grass/shrub',
    10: 'Temperate grass/savanna', 8: 'Temperate broadleaf', 9: 'Temperate conifer',
    1: 'Boreal', 15: 'Tundra', 7: 'Rock and ice',
}
MIN_PLOT_CELLS = 50

DT_STRICT, DT_LOOSE = cfg.DT_SLICES


def setup_style():
    """Avenir if available (Casper conda env), else matplotlib default."""
    fdir = '/glade/work/jsallen/conda-envs/earth/fonts/Avenir-Medium.otf'
    if os.path.exists(fdir):
        font_manager.fontManager.addfont(fdir)
        prop = font_manager.FontProperties(fname=fdir)
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = prop.get_name()
    plt.rcParams['axes.linewidth'] = 0.6
    plt.rcParams['xtick.major.width'] = 0.6
    plt.rcParams['ytick.major.width'] = 0.6


def gwl_label(g):
    """'gwl1p5' -> '1.5°C'."""
    return f"{cfg.GWL_KEYS[g]:g}°C" if g in cfg.GWL_KEYS else g


def dt_tag(dt):
    """1.0 -> 'dt1p0' (variable-name suffix used by analog.retention.py)."""
    return 'dt' + f'{dt:.1f}'.replace('.', 'p')


def savefig(fig, name, dpi=400):
    os.makedirs(FIG_DIR, exist_ok=True)
    f = os.path.join(FIG_DIR, name)
    fig.savefig(f, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {f}')
    return f


class Retention:
    """analog_retention.nc + coverage mask + area weights."""

    def __init__(self, mask_low_coverage=True):
        print(f'Loading {cfg.RETENTION_FILE} ...')
        ds = xr.open_dataset(cfg.RETENTION_FILE)
        sets = xr.open_dataset(cfg.SETS_FILE)
        self.ds = ds
        self.lat = ds['lat'].values
        self.lon = ds['lon'].values
        self.Dt = ds['Dt'].values
        self.biome_id = ds['biome_id'].values
        self.n_analogs = ds['n_analogs'].values
        self.s_used = sets['s_used_km'].values
        self.land = np.isfinite(self.biome_id) & (self.biome_id > 0)
        has = np.isfinite(self.n_analogs) & (self.n_analogs > 0)
        # full set at full spacing: selection was not compromised
        self.full = has & (self.n_analogs >= cfg.N_ANALOGS) & (self.s_used >= cfg.S_TRY[0])
        self.has = has
        self.keep = self.full if mask_low_coverage else has
        self.W = area_weights(self.lat, len(self.lon))
        self.gwls = [g for g in cfg.GWLS if g != 'gwl0' and f'retention_{g}' in ds]

    def di(self, dt):
        return int(np.argmin(np.abs(self.Dt - dt)))

    def n_models(self, g):
        return int(self.ds.attrs.get(f'n_models_{g}', -1))

    def central(self, g, dt=None):
        """Headline retention (nlat, nlon[, nDt])."""
        v = f'retention_mmm_{g}' if HEADLINE == 'mmm' else f'retention_{g}'
        a = self.ds[v].values
        return a if dt is None else a[:, :, self.di(dt)]

    def central_label(self):
        return ('forced response (multi-model-mean future)' if HEADLINE == 'mmm'
                else 'ensemble median of per-model retention')

    def biomes(self, mask=None, verbose=True):
        """[(id, short name, cell mask)] for plotted biomes, display order."""
        mask = self.has if mask is None else mask
        out, skipped = [], []
        for b, name in BIOME_SHORT.items():
            bm = (self.biome_id == b) & mask
            if bm.sum() < MIN_PLOT_CELLS:
                skipped.append(name)
                continue
            out.append((b, name, bm))
        if verbose and skipped:
            print(f'  skipped (< {MIN_PLOT_CELLS} cells with analogs): {", ".join(skipped)}')
        return out

    def masked(self, arr2d):
        """NaN outside the coverage mask (draws as grey over the land backdrop)."""
        out = np.full(arr2d.shape, np.nan)
        sel = self.keep & np.isfinite(arr2d)
        out[sel] = arr2d[sel]
        return out


# ------------------------------
# Maps
# ------------------------------
def map_axes(fig, spec):
    import cartopy.crs as ccrs
    return fig.add_subplot(spec, projection=ccrs.Robinson())


def add_map(ax, R, field, cmap, vmin, vmax, title, norm=None):
    """Field on a grey land backdrop; masked land stays grey, ocean light."""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    pc = ccrs.PlateCarree()
    grey = np.where(R.land, 1.0, np.nan)
    ax.pcolormesh(R.lon, R.lat, grey, cmap=ListedColormap(['0.82']), vmin=0, vmax=1,
                  transform=pc, zorder=1, rasterized=True)
    kw = dict(norm=norm) if norm is not None else dict(vmin=vmin, vmax=vmax)
    pm = ax.pcolormesh(R.lon, R.lat, field, cmap=cmap, transform=pc, zorder=2,
                       rasterized=True, **kw)
    ax.set_extent([-180, 180, -60, 90], crs=pc)
    ax.add_feature(cfeature.OCEAN, facecolor='whitesmoke', zorder=2)
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.3, zorder=3)
    ax.set_title(title, loc='left', fontsize=9)
    return pm


def coarse_stipple(ax, R, flag2d, block=10, frac=0.5, **kw):
    """Dots on a coarse grid (block x block cells) where >= frac of the kept
    land cells in the block have flag2d True. Readable at global scale."""
    import cartopy.crs as ccrs
    nlat, nlon = flag2d.shape
    ys, xs = [], []
    for i in range(0, nlat - block + 1, block):
        for j in range(0, nlon - block + 1, block):
            k = R.keep[i:i + block, j:j + block]
            if k.sum() < 0.25 * block * block:
                continue
            if np.mean(flag2d[i:i + block, j:j + block][k]) >= frac:
                ys.append(R.lat[i + block // 2]); xs.append(R.lon[j + block // 2])
    style = dict(s=0.6, c='k', marker='.', linewidths=0, zorder=4)
    style.update(kw)
    ax.scatter(xs, ys, transform=ccrs.PlateCarree(), **style)
