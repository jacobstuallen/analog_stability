# means_extremes_common.py
#
# Loading and biome statistics shared by plot.means.extremes.py (main text)
# and plot.means.extremes.si.py (SI). All shares are in EXTREMES space
# (TXx, Rx5day) unless stated, cos(lat) area-weighted.
#
# Inputs (env overrides as in compare.varsets.py):
#   ANALOG_EXT_DIR    extremes run       analog_sets.nc, analog_retention.nc,
#                                        random_baseline_extremes.nc
#   ANALOG_MEANS_DIR  means run          analog_sets.nc, analog_retention.nc,
#                                        compare_varsets.nc, random_baseline_means.nc
#   ANALOG_CROSS_DIR  cross run          analog_retention.nc (means-selected
#                                        analogs scored on extremes; must include gwl0)
#
# Central estimate follows ANALOG_HEADLINE (plot_common): 'mmm' (default) or
# 'median'. At GWL0 both equal the ERA5 value (delta mode).
import os
import numpy as np
import xarray as xr
import plot_common as pcm
from plot_common import cfg, wmean, wmedian, area_weights, BIOME_SHORT, MIN_PLOT_CELLS

EXT = os.environ.get('ANALOG_EXT_DIR', f'{cfg.DATA_ROOT}/analog_stability')
MEA = os.environ.get('ANALOG_MEANS_DIR', f'{cfg.DATA_ROOT}/analog_stability_means')
CRO = os.environ.get('ANALOG_CROSS_DIR', f'{cfg.DATA_ROOT}/analog_stability_cross')
RAND_EXT = os.environ.get('ANALOG_RANDOM_EXT', f'{EXT}/random_baseline_extremes.nc')
RAND_MEA = os.environ.get('ANALOG_RANDOM_MEANS', f'{MEA}/random_baseline_means.nc')

# Colors (validated: CVD dE >= 13 between chromatic slots; grey is the neutral
# baseline and always carries an open marker as secondary encoding)
C_RAND = '#8c8c8c'
C_MEANS = '#d95f02'
C_EXT2 = '#4292c6'
C_EXT4 = '#08519c'

# Figure text uses matplotlib mathtext, not unicode symbols
SIG = r'$\sigma$'
DEG = r'$^{\circ}$C'
TIMES = r'$\times$'


def gwl_tex(g):
    """'gwl1p5' -> '1.5$^{\\circ}$C' (mathtext twin of plot_common.gwl_label)."""
    return f'{cfg.GWL_KEYS[g]:g}{DEG}' if g in cfg.GWL_KEYS else g


def _open(path, required=True):
    if not os.path.exists(path):
        if required:
            raise SystemExit(f'missing input: {path}')
        print(f'  not found (panel skipped): {path}')
        return None
    return xr.open_dataset(path).sortby('lat')


def _rv(g):
    return f'retention_mmm_{g}' if pcm.HEADLINE == 'mmm' else f'retention_{g}'


class MeansExtremes:
    def __init__(self, need_means=False):
        self.se = _open(f'{EXT}/analog_sets.nc')
        self.sm = _open(f'{MEA}/analog_sets.nc')
        self.re = _open(f'{EXT}/analog_retention.nc')
        self.rc = _open(f'{CRO}/analog_retention.nc')
        self.rb = _open(RAND_EXT)
        self.rm = _open(f'{MEA}/analog_retention.nc', required=need_means)
        self.rbm = _open(RAND_MEA, required=False)
        self.ov = _open(f'{MEA}/compare_varsets.nc', required=False)
        self.lat, self.lon = self.se.lat.values, self.se.lon.values
        for name in ['sm', 're', 'rc', 'rb', 'rm', 'rbm', 'ov']:
            d = getattr(self, name)
            if d is None:
                continue
            if (len(d.lat) != len(self.lat) or not np.allclose(d.lat.values, self.lat)
                    or len(d.lon) != len(self.lon) or not np.allclose(d.lon.values, self.lon)):
                raise SystemExit(f'grid mismatch: {name}')
        if self.rc.attrs.get('variable_set') not in (None, 'extremes'):
            raise SystemExit('cross run is not scored in extremes space')
        if not any(v in self.rc for v in ('retention_mmm_gwl0', 'retention_gwl0')):
            raise SystemExit('cross run has no gwl0 (rerun with ANALOG_GWLS=gwl0,...)')
        self.W = area_weights(self.lat, len(self.lon))
        self.biome = self.se.biome_id.values
        self.has_e = np.nan_to_num(self.se.n_analogs.values) > 0
        self.has_m = np.nan_to_num(self.sm.n_analogs.values) > 0
        self.both = self.has_e & self.has_m
        self.gwls = [g for g in cfg.GWL_KEYS if _rv(g) in self.re and _rv(g) in self.rc]
        print(f'Headline: {pcm.HEADLINE} | GWLs: {", ".join(self.gwls)}')

    # ---- per-cell fields (nlat, nlon) ----
    def ret(self, ds, g, dt):
        """Central retention; GWL0 falls back to whichever variable exists."""
        v = _rv(g)
        if g == 'gwl0' and v not in ds:
            v = 'retention_mmm_gwl0' if 'retention_mmm_gwl0' in ds else 'retention_gwl0'
        return ds[v].sel(Dt=dt, method='nearest').values

    def ret_pct(self, ds, g, dt, p):
        v = f'retention_{g}_p{p}'
        return ds[v].sel(Dt=dt, method='nearest').values if v in ds else None

    def ext0(self, dt):
        """Extremes-selected analogs within Dt today (~1 at Dt=1 by the sigma ceiling)."""
        if any(v in self.re for v in ('retention_mmm_gwl0', 'retention_gwl0')):
            return self.ret(self.re, 'gwl0', dt)
        sig = self.se.analog_sigma.values
        n = np.isfinite(sig).sum(axis=-1)
        return np.where(n > 0, np.sum(sig < dt, axis=-1) / np.maximum(n, 1), np.nan)

    def rand(self, dt, means_space=False):
        d = self.rbm if means_space else self.rb
        return d[f'random_share_{pcm.dt_tag(dt)}'].values

    def n_models(self, g):
        return int(self.re.attrs.get(f'n_models_{g}', -1))

    # ---- biome statistics ----
    def bmean(self, arr, bm):
        return wmean(arr[bm], self.W[bm])

    def bmedian(self, arr, bm):
        return wmedian(arr[bm], self.W[bm])

    def biome_rows(self, dt=1.0):
        """Plotted biomes sorted by lift (means-selected / random share at dt),
        lowest first. bm = cells with both analog sets; rmask = sampled
        random-baseline focal cells."""
        r1, m0 = self.rand(dt), self.ret(self.rc, 'gwl0', dt)
        rows, skipped = [], []
        for b, name in BIOME_SHORT.items():
            bm = (self.biome == b) & self.both
            rmask = (self.biome == b) & np.isfinite(r1)
            if bm.sum() < MIN_PLOT_CELLS or rmask.sum() == 0:
                skipped.append(name)
                continue
            rnd, mns = self.bmean(r1, rmask), self.bmean(m0, bm)
            rows.append(dict(b=b, name=name, bm=bm, rmask=rmask, rand=rnd, m0=mns,
                             lift=mns / max(rnd, 1e-6)))
        if skipped:
            print(f'  skipped biomes: {", ".join(skipped)}')
        return sorted(rows, key=lambda r: r['lift'])
