# analog_config.py
#
# Single source of truth for paths, variables, models, GWL windows, biomes
# and method constants. Every analysis, diag and plot script imports from
# here, so a change made once applies everywhere.
#
# Paths can be overridden with environment variables (useful for testing on
# a subset or a different machine) without editing this file.

import os

# ==============================
# Paths
# ==============================
DATA_ROOT = os.environ.get('ANALOG_DATA_ROOT', '/glade/campaign/ral/risc/jsallen/TNC')
CCKP_ERA5 = f'{DATA_ROOT}/CCKP/ERA5'
CCKP_CMIP6 = f'{DATA_ROOT}/CCKP/CMIP6'
# Variable set (env ANALOG_VARSET): 'extremes' (TXx, Rx5day; the paper's main
# set) or 'means' (annual-mean tas, annual-total pr; the conventional analog
# variables, for the means-vs-extremes comparison). Each set writes to its
# own output folder so runs never overwrite each other.
VARSET = os.environ.get('ANALOG_VARSET', 'extremes')
if VARSET not in ('extremes', 'means'):
    raise ValueError(f'unknown ANALOG_VARSET {VARSET!r}')
_OUT_DEFAULT = f'{DATA_ROOT}/analog_stability' + ('' if VARSET == 'extremes' else '_means')
OUT_DIR = os.environ.get('ANALOG_OUT_DIR', _OUT_DEFAULT)
BIOME_FILE = os.environ.get(
    'ANALOG_BIOME_FILE',
    '/glade/u/home/jsallen/projects/tnc_2025/analogs/biomes/biomes.analog.gridded.nc')
ECO_FILE = os.environ.get(
    'ANALOG_ECO_FILE',
    '/glade/work/jsallen/projects/tnc_2025/draft_03/ecoregion/ecoregions.analog.gridded.nc')

# Output files
HIST_SIGMA_FILE = os.path.join(OUT_DIR, 'analog_historical_sigma.nc')
HIST_DIST_FILE = os.path.join(OUT_DIR, 'analog_historical_dist_summary.npz')
# ANALOG_SETS_FILE: evaluate retention on ANOTHER variable set's analogs
# (cross-evaluation, e.g. means-selected analogs scored on extremes). GWL0
# retention is then < 1 by design: those analogs were not gated on these
# variables.
SETS_FILE = os.environ.get('ANALOG_SETS_FILE', os.path.join(OUT_DIR, 'analog_sets.nc'))
# RETENTION_FILE, STABILITY_FILE, STABILITY_NPZ: set below (depend on MODEL_SET)

# ==============================
# Method constants
# ==============================
CLIM_START, CLIM_END = 1991, 2020   # ERA5 reference period and ICV ruler
PC_VAR_MIN = 0.01                   # drop ICV PCs with eigenvalue below this
SIGMA_CAP = 8.0                     # sigma saturates here (use D_maha beyond)
MIN_BIOME_CELLS = 500               # skip biomes with fewer valid cells
BUFFER_KM = 250.0                   # trivial-neighbor exclusion
SCENARIO = 'ssp585'

# How future climatologies are built (see analog_common.load_future_gwl):
#   'delta': ERA5 1991-2020 mean + (model GWL window - model 1991-2020 window).
#            Removes each model's bias, so a pair can only diverge through
#            differing CHANGES. Default for all results.
#   'raw'  : model GWL window as-is (pre-2026-11 behavior). Kept only to show
#            how much of the old signal was model bias.
FUTURE_MODE = os.environ.get('ANALOG_FUTURE_MODE', 'delta')
MODEL_HIST_END = 2014   # last year in the CCKP historical files; later years
                        # of the baseline come from the scenario run

# Analog set selection
N_ANALOGS = 10
SIGMA_CEILING = 1.0
S_TRY = [500.0, 400.0, 300.0, 250.0]
DECORR_KM = 400.0

# Retention
DT_GRID_START, DT_GRID_STOP, DT_GRID_STEP = 0.5, 4.0, 0.25
DT_SLICES = [1.0, 2.0]
# Model agreement (reliability tiers): a pair is ROBUSTLY RETAINED if at least
# ROBUST_AGREE of models keep it within Dt, ROBUSTLY LOST if at most
# 1 - ROBUST_AGREE do, otherwise UNCERTAIN (IPCC-style 80% agreement).
ROBUST_AGREE = 0.8

# ==============================
# Variables
# ==============================
# key -> ERA5 annual timeseries file + varname, CCKP token, log1p flag, QC max.
# To add a variable, add an entry here; all scripts pick it up.
def _era5_ts(token):
    return (f'{CCKP_ERA5}/timeseries-{token}-annual-mean_era5-x0.25_'
            f'era5-x0.25-historical_timeseries_mean_1950-2024.nc')

_VARIABLE_SETS = {'extremes': {
    'txx': {
        'era5_ts': _era5_ts('txx'),
        'era5_var': 'timeseries-txx-annual-mean',
        'cckp_token': 'txx',
        'log': False,
        'qc_max': 70.0,
    },
    'rx5day': {
        'era5_ts': _era5_ts('rx5day'),
        'era5_var': 'timeseries-rx5day-annual-mean',
        'cckp_token': 'rx5day',
        'log': True,
        'qc_max': 3000.0,
    },
}, 'means': {
    'tas': {                       # annual mean near-surface temperature (degC)
        'era5_ts': _era5_ts('tas'),
        'era5_var': 'timeseries-tas-annual-mean',
        'cckp_token': 'tas',
        'log': False,
        'qc_max': 60.0,
    },
    'pr': {                        # annual TOTAL precipitation (mm), CCKP 'sum'
        'era5_ts': _era5_ts('pr'),
        'era5_var': 'timeseries-pr-annual-mean',
        'cckp_token': 'pr',
        'log': True,               # same multiplicative treatment as Rx5day
        'qc_max': 15000.0,         # world record ~12,000 mm/yr
    },
}}
VARIABLES = _VARIABLE_SETS[VARSET]
VKEYS = list(VARIABLES.keys())
K = len(VKEYS)

# ==============================
# CMIP6 ensemble (model -> variant)
# ==============================
MODELS = {
    'access-cm2': 'r1i1p1f1', 'access-esm1-5': 'r1i1p1f1', 'bcc-csm2-mr': 'r1i1p1f1',
    'canesm5': 'r1i1p1f1', 'cmcc-esm2': 'r1i1p1f1', 'cnrm-cm6-1': 'r1i1p1f2',
    'cnrm-esm2-1': 'r1i1p1f2', 'ec-earth3': 'r1i1p1f1', 'ec-earth3-veg-lr': 'r1i1p1f1',
    'fgoals-g3': 'r3i1p1f1', 'gfdl-cm4': 'r1i1p1f1', 'gfdl-esm4': 'r1i1p1f1',
    'giss-e2-1-g': 'r1i1p1f2', 'hadgem3-gc31-ll': 'r1i1p1f3', 'hadgem3-gc31-mm': 'r1i1p1f3',
    'inm-cm4-8': 'r1i1p1f1', 'inm-cm5-0': 'r1i1p1f1', 'ipsl-cm6a-lr': 'r1i1p1f1',
    'kace-1-0-g': 'r1i1p1f1', 'kiost-esm': 'r1i1p1f1', 'miroc-es2l': 'r1i1p1f2',
    'miroc6': 'r1i1p1f1', 'mpi-esm1-2-hr': 'r1i1p1f1', 'mpi-esm1-2-lr': 'r1i1p1f1',
    'mri-esm2-0': 'r1i1p1f1', 'nesm3': 'r1i1p1f1', 'noresm2-lm': 'r1i1p1f1',
    'noresm2-mm': 'r1i1p1f1', 'ukesm1-0-ll': 'r1i1p1f2',
    # taiesm1 removed 2026-10-31: no CCKP txx/rx5day files exist for it
}

# ==============================
# GWL windows (20-yr, inclusive, relative to 1850-1900), SSP5-8.5.
# GWL_WINDOWS[gwl][model] = (start, end). A model is used at a GWL only if it
# has a window there, so ensembles can differ by GWL (GWL4: 18 of 30 models).
#
# GWL_SOURCE (env ANALOG_GWL_SOURCE):
#   'ar6' (default): IPCC AR6 Atlas per-member windows (Hauser et al.,
#         zenodo 3591806), for the exact member CCKP downscaled. One method
#         for all levels. Table in data/gwl_windows_ar6_ssp585.csv.
#   'tye': Tye et al. 2025 Appendix B (ensemble-mean periods), GWL2/3 only.
#         The windows behind the draft results; kept for comparison.
# ==============================
GWL_KEYS = {'gwl1p5': 1.5, 'gwl2': 2.0, 'gwl3': 3.0, 'gwl4': 4.0}   # name -> deg C
GWL_TABLE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'data', 'gwl_windows_ar6_ssp585.csv')
GWL_SOURCE = os.environ.get('ANALOG_GWL_SOURCE', 'ar6')

_TYE_GWL23 = {
    'access-cm2': (2024, 2043, 2043, 2062), 'access-esm1-5': (2027, 2046, 2048, 2067),
    'bcc-csm2-mr': (2030, 2049, 2052, 2071), 'canesm5': (2019, 2038, 2036, 2055),
    'cmcc-esm2': (2033, 2052, 2049, 2068), 'cnrm-cm6-1': (2029, 2048, 2048, 2067),
    'cnrm-esm2-1': (2029, 2048, 2048, 2067), 'ec-earth3': (2025, 2044, 2046, 2065),
    'ec-earth3-veg-lr': (2029, 2048, 2047, 2066), 'gfdl-cm4': (2030, 2049, 2048, 2067),
    'gfdl-esm4': (2038, 2057, 2061, 2080), 'giss-e2-1-g': (2028, 2047, 2052, 2071),
    'hadgem3-gc31-ll': (2020, 2039, 2039, 2058), 'hadgem3-gc31-mm': (2023, 2042, 2037, 2056),
    'inm-cm4-8': (2037, 2056, 2060, 2079), 'inm-cm5-0': (2035, 2054, 2063, 2082),
    'ipsl-cm6a-lr': (2027, 2046, 2045, 2064), 'kace-1-0-g': (2020, 2039, 2041, 2060),
    'kiost-esm': (2037, 2056, 2062, 2081), 'miroc-es2l': (2033, 2052, 2056, 2075),
    'miroc6': (2037, 2056, 2061, 2080), 'mpi-esm1-2-hr': (2039, 2058, 2064, 2083),
    'mpi-esm1-2-lr': (2040, 2059, 2063, 2082), 'mri-esm2-0': (2029, 2048, 2053, 2072),
    'nesm3': (2026, 2045, 2046, 2065), 'noresm2-lm': (2041, 2060, 2063, 2082),
    'taiesm1': (2021, 2040, 2037, 2056), 'ukesm1-0-ll': (2019, 2038, 2034, 2053),
}


def _load_ar6_windows(path):
    import csv
    by_level = {}
    with open(path) as f:
        rows = [l for l in f if not l.startswith('#')]
    for r in csv.DictReader(rows):
        if MODELS.get(r['model']) != r['ensemble']:
            continue
        by_level.setdefault(float(r['warming_level']), {})[r['model']] = (
            int(r['start_year']), int(r['end_year']))
    return {k: by_level.get(v, {}) for k, v in GWL_KEYS.items()}


if GWL_SOURCE == 'ar6':
    GWL_WINDOWS = _load_ar6_windows(GWL_TABLE)
elif GWL_SOURCE == 'tye':
    GWL_WINDOWS = {'gwl2': {m: (w[0], w[1]) for m, w in _TYE_GWL23.items()},
                   'gwl3': {m: (w[2], w[3]) for m, w in _TYE_GWL23.items()}}
else:
    raise ValueError(f'unknown ANALOG_GWL_SOURCE {GWL_SOURCE!r}')

# 'gwl0' is the baseline window itself (sanity check: in delta mode it must
# reproduce ERA5 exactly, so retention(Dt > 1) = 1). Not a real warming level.
GWL_WINDOWS['gwl0'] = {m: (CLIM_START, CLIM_END) for m in MODELS}

_default_gwls = 'gwl1p5,gwl2,gwl3,gwl4' if GWL_SOURCE == 'ar6' else 'gwl2,gwl3'
GWLS = os.environ.get('ANALOG_GWLS', _default_gwls).split(',')

# MODEL_SET (env ANALOG_MODEL_SET):
#   'per_gwl' (default): each GWL uses every model that reaches it.
#   'common': every GWL uses only models that reach ALL requested GWLs, so
#             levels are directly comparable (scaling test). Output files
#             get a '_common' suffix so both can coexist.
MODEL_SET = os.environ.get('ANALOG_MODEL_SET', 'per_gwl')
if MODEL_SET not in ('per_gwl', 'common'):
    raise ValueError(f'unknown ANALOG_MODEL_SET {MODEL_SET!r}')
_SFX = '_common' if MODEL_SET == 'common' else ''
RETENTION_FILE = os.path.join(OUT_DIR, f'analog_retention{_SFX}.nc')
STABILITY_FILE = os.path.join(OUT_DIR, f'analog_stability_gwl{_SFX}.nc')
STABILITY_NPZ = os.path.join(OUT_DIR, f'per_model_biome_sigma{_SFX}.npz')
SCALING_FILE = os.path.join(OUT_DIR, 'analog_scaling.nc')   # always common models
MODEL_ICV_FILE = os.path.join(OUT_DIR, 'model_icv_ratio.nc')  # diag.model.icv.py

# ==============================
# Biomes (WWF major habitat types; 98/99 lake/rock-ice codes not analyzed)
# ==============================
BIOMES = {
    1: 'Boreal Forests/Taiga', 2: 'Deserts & Xeric Shrublands',
    3: 'Flooded Grasslands & Savannas', 4: 'Mangroves',
    5: 'Mediterranean Forests, Woodlands & Scrub', 6: 'Montane Grasslands & Shrublands',
    7: 'Rock and Ice', 8: 'Temperate Broadleaf & Mixed Forests',
    9: 'Temperate Conifer Forests', 10: 'Temperate Grasslands, Savannas & Shrublands',
    11: 'Tropical & Subtropical Coniferous Forests', 12: 'Tropical & Subtropical Dry Broadleaf Forests',
    13: 'Tropical & Subtropical Grasslands, Savannas & Shrublands',
    14: 'Tropical & Subtropical Moist Broadleaf Forests', 15: 'Tundra',
}

ICV_DEFINITION = (
    'ERA5 1991-2020 annual index series, per grid cell, LINEARLY DETRENDED before '
    'taking the SD (ddof=2). Fixed observational ruler: the same ICV is applied to '
    'historical and to every CMIP6 model/GWL; ICV is never recomputed per model. '
    'Cross-variable ICV correlations are computed from the same detrended anomalies. '
    'Focal-cell convention (Mahony 2017): D_ji is scored in focal cell j ICV PC space '
    'and is asymmetric.')
