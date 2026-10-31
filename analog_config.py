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
OUT_DIR = os.environ.get('ANALOG_OUT_DIR', f'{DATA_ROOT}/analog_stability')
BIOME_FILE = os.environ.get(
    'ANALOG_BIOME_FILE',
    '/glade/u/home/jsallen/projects/tnc_2025/analogs/biomes/biomes.analog.gridded.nc')
ECO_FILE = os.environ.get(
    'ANALOG_ECO_FILE',
    '/glade/work/jsallen/projects/tnc_2025/draft_03/ecoregion/ecoregions.analog.gridded.nc')

# Output files
HIST_SIGMA_FILE = os.path.join(OUT_DIR, 'analog_historical_sigma.nc')
HIST_DIST_FILE = os.path.join(OUT_DIR, 'analog_historical_dist_summary.npz')
SETS_FILE = os.path.join(OUT_DIR, 'analog_sets.nc')
RETENTION_FILE = os.path.join(OUT_DIR, 'analog_retention.nc')
STABILITY_FILE = os.path.join(OUT_DIR, 'analog_stability_gwl.nc')
STABILITY_NPZ = os.path.join(OUT_DIR, 'per_model_biome_sigma.npz')

# ==============================
# Method constants
# ==============================
CLIM_START, CLIM_END = 1991, 2020   # ERA5 reference period and ICV ruler
PC_VAR_MIN = 0.01                   # drop ICV PCs with eigenvalue below this
SIGMA_CAP = 8.0                     # sigma saturates here (use D_maha beyond)
MIN_BIOME_CELLS = 500               # skip biomes with fewer valid cells
BUFFER_KM = 250.0                   # trivial-neighbor exclusion
SCENARIO = 'ssp585'

# Analog set selection
N_ANALOGS = 10
SIGMA_CEILING = 1.0
S_TRY = [500.0, 400.0, 300.0, 250.0]
DECORR_KM = 400.0

# Retention
DT_GRID_START, DT_GRID_STOP, DT_GRID_STEP = 0.5, 4.0, 0.25
DT_SLICES = [1.0, 2.0]

# ==============================
# Variables
# ==============================
# key -> ERA5 annual timeseries file + varname, CCKP token, log1p flag, QC max.
# To add a variable, add an entry here; all scripts pick it up.
def _era5_ts(token):
    return (f'{CCKP_ERA5}/timeseries-{token}-annual-mean_era5-x0.25_'
            f'era5-x0.25-historical_timeseries_mean_1950-2024.nc')

VARIABLES = {
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
}
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
    'noresm2-mm': 'r1i1p1f1', 'taiesm1': 'r1i1p1f1', 'ukesm1-0-ll': 'r1i1p1f2',
}

# ==============================
# GWL windows, SSP5-8.5 (Tye et al. 2025, Appendix B), 20-yr, inclusive.
# GWL_WINDOWS[gwl][model] = (start, end). Models absent from a GWL are
# skipped for that GWL (fgoals-g3 and noresm2-mm have no published window).
# To add GWL1.5 / GWL4, add a key here and list it in GWLS.
# ==============================
_GWL23 = {
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
GWL_WINDOWS = {
    'gwl2': {m: (w[0], w[1]) for m, w in _GWL23.items()},
    'gwl3': {m: (w[2], w[3]) for m, w in _GWL23.items()},
}
GWLS = ['gwl2', 'gwl3']

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
