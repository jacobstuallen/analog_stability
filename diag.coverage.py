# Why are cells low-coverage?
#
# There are FOUR distinct failure paths in analog_set_selection.py, with
# different causes and different fixes. This script attributes each gray cell
# to one of them, globally and per biome.
#
#   A. BIOME SKIPPED       biome has < min_biome_cells -> every cell gray.
#   B. BUFFER KILLS POOL   every same-biome cell is within BUFFER_KM -> no
#                          candidates at all. Small/fragmented biomes.
#   C. QUALITY GATE EMPTY  candidates exist, but NONE pass sigma <= CEILING.
#                          The cell is a climatic OUTLIER within its own biome.
#   D. SHORT SET           some analogs found, but fewer than N_ANALOGS.
#
# C is the interesting one: it means "this place has no climatic peer inside
# its own biome," which is a RESULT, not a bug. B and A are structural.
#
# Run after analog_set_selection.py.

import numpy as np
import xarray as xr

import analog_config as cfg

biome_file = cfg.BIOME_FILE
sets_file = cfg.SETS_FILE

# shared with analog.set.selection.py via analog_config
N_ANALOGS = cfg.N_ANALOGS   # was 30 here (stale); selection uses 10
BUFFER_KM = cfg.BUFFER_KM
SIGMA_CEILING = cfg.SIGMA_CEILING
min_biome_cells = cfg.MIN_BIOME_CELLS
COVERAGE_MIN = 10          # what the plotting script calls "low coverage"

biomes = cfg.BIOMES

# named regions you flagged, to attribute individually
REGIONS = {
    'Sahara core':        (18.0, 28.0,   0.0, 25.0),
    'India (Gangetic)':   (22.0, 30.0,  75.0, 88.0),
    'Tibetan Plateau':    (30.0, 36.0,  82.0, 95.0),
    'Mexico (central)':   (18.0, 26.0, -106.0, -98.0),
    'Chile (central)':   (-38.0, -30.0, -73.0, -70.0),
    'California':         (34.0, 40.0, -123.0, -119.0),
    'Spain':              (37.0, 43.0,  -8.0,   0.0),
    'SE Australia':      (-38.0, -30.0, 143.0, 152.0),
    'SE United States':   (30.0, 36.0, -92.0, -80.0),
    'Amazon (contrast)':  (-8.0,  -2.0, -70.0, -55.0),
}

bds = xr.open_dataset(biome_file).sortby('lat')
ds = xr.open_dataset(sets_file)
lat = ds.lat.values
lon = ds.lon.values

bid = bds['BIOME_ID'].sel(lat=slice(lat.min(), lat.max())).values
nfound = ds['n_analogs'].values
s_used = ds['s_used_km'].values
effn = ds['eff_n'].values
sig = ds['analog_sig'].values if 'analog_sig' in ds else None

land = np.isfinite(bid) & (bid > 0)

# ---------------------------------------------------------------
# Classify every land cell
# ---------------------------------------------------------------
# Which biomes were skipped wholesale?
skipped = set()
for b in biomes:
    nb = (bid == b).sum()
    if 0 < nb < min_biome_cells:      # present, but too small to process
        skipped.add(b)

nf = np.where(np.isfinite(nfound), nfound, 0)

cls = np.full(bid.shape, '', dtype=object)
cls[land & np.isin(bid, list(skipped))] = 'A_biome_skipped'
rest = land & ~np.isin(bid, list(skipped))
cls[rest & (nf == 0)] = 'BC_zero_analogs'
cls[rest & (nf > 0) & (nf < COVERAGE_MIN)] = 'D_short_set'
cls[rest & (nf >= COVERAGE_MIN) & (nf < N_ANALOGS)] = 'D_partial_set'
cls[rest & (nf >= N_ANALOGS)] = 'OK_full_set'

print('=' * 76)
print('GLOBAL ATTRIBUTION OF LOW-COVERAGE CELLS')
print('=' * 76)
print(f'  N_ANALOGS={N_ANALOGS}  BUFFER_KM={BUFFER_KM}  '
      f'SIGMA_CEILING={SIGMA_CEILING}')
print(f'  "low coverage" assumed to mean n_analogs < {COVERAGE_MIN}\n')
nland = land.sum()
for k in ['OK_full_set', 'D_partial_set', 'D_short_set', 'BC_zero_analogs',
          'A_biome_skipped']:
    m = (cls == k) & land
    print(f'  {k:<18s} {m.sum():>9,}  ({m.sum()/nland:6.1%} of land)')
print(f'  {"TOTAL LAND":<18s} {nland:>9,}')

if skipped:
    print(f'\n  Biomes skipped entirely (< {min_biome_cells} cells):')
    for b in sorted(skipped):
        print(f'    {b:2d}  {biomes[b]:<50s} {(bid==b).sum():>7,} cells')

# ---------------------------------------------------------------
# Per biome
# ---------------------------------------------------------------
print()
print('=' * 76)
print('PER BIOME')
print('=' * 76)
print(f'  {"biome":<48s} {"cells":>8s} {"%zero":>7s} {"%<N":>7s} {"med n":>6s}')
print('  ' + '-' * 72)
for b, name in biomes.items():
    m = (bid == b)
    if m.sum() == 0:
        continue
    if b in skipped:
        print(f'  {name:<48s} {m.sum():>8,} {"SKIPPED (too few cells)":>22s}')
        continue
    nfb = nf[m]
    pz = np.mean(nfb == 0)
    pl = np.mean(nfb < N_ANALOGS)
    print(f'  {name:<48s} {m.sum():>8,} {pz:6.1%} {pl:6.1%} '
          f'{np.median(nfb):6.0f}')

# ---------------------------------------------------------------
# The regions you flagged
# ---------------------------------------------------------------
print()
print('=' * 76)
print('THE REGIONS YOU FLAGGED')
print('=' * 76)
LO, LA = np.meshgrid(lon, lat)
print(f'  {"region":<20s} {"cells":>7s} {"%zero":>7s} {"%<N":>7s} '
      f'{"med n":>6s} {"med s_km":>9s}  dominant biome')
print('  ' + '-' * 74)
for rname, (la0, la1, lo0, lo1) in REGIONS.items():
    m = land & (LA >= la0) & (LA <= la1) & (LO >= lo0) & (LO <= lo1)
    if m.sum() == 0:
        print(f'  {rname:<20s}  (no land cells in box)')
        continue
    nfb = nf[m]
    sb = s_used[m]
    bb = bid[m].astype(int)
    vals, cnts = np.unique(bb[bb > 0], return_counts=True)
    dom = biomes.get(int(vals[np.argmax(cnts)]), '?') if len(vals) else '?'
    domfrac = cnts.max() / len(bb) if len(vals) else 0
    meds = np.nanmedian(sb) if np.isfinite(sb).any() else np.nan
    print(f'  {rname:<20s} {m.sum():>7,} {np.mean(nfb==0):6.1%} '
          f'{np.mean(nfb<N_ANALOGS):6.1%} {np.median(nfb):6.0f} '
          f'{meds:9.0f}  {dom[:28]} ({domfrac:.0%})')

# ---------------------------------------------------------------
# Is it path B (buffer) or path C (quality gate)?
# Distinguish by asking: for zero-analog cells, how BIG is their biome?
# A tiny biome -> buffer killed the pool (B).
# A huge biome -> candidates existed, the GATE rejected them all (C).
# ---------------------------------------------------------------
print()
print('=' * 76)
print('DISAMBIGUATING PATH B (buffer) vs PATH C (quality gate)')
print('=' * 76)
print('  For cells with ZERO analogs, how large is their biome?')
print('  Large biome + zero analogs => candidates EXISTED but ALL failed the')
print('  sigma<=%.1f gate. That cell is a climatic OUTLIER in its own biome.'
      % SIGMA_CEILING)
print()
zero = land & (nf == 0) & ~np.isin(bid, list(skipped))
if zero.sum() == 0:
    print('  No zero-analog cells outside skipped biomes.')
else:
    zb = bid[zero].astype(int)
    print(f'  {"biome of zero-analog cell":<44s} {"n_zero":>8s} {"biome sz":>9s} '
          f'{"%zero":>6s}  verdict')
    print('  ' + '-' * 74)
    for b in np.unique(zb):
        if b <= 0:
            continue
        cnt = (zb == b).sum()
        size = (bid == b).sum()
        pz = cnt / size
        # A biome big enough that the 250 km buffer cannot possibly exhaust it
        # still has candidates; if they all fail, it is the GATE.
        if size > 5000:
            verdict = 'PATH C: gate rejected all (outlier cell)'
        elif size > 1000:
            verdict = 'PATH C likely (biome large enough for a pool)'
        else:
            verdict = 'PATH B: biome too small/fragmented for buffer'
        print(f'  {biomes.get(int(b), "?")[:44]:<44s} {cnt:>8,} {size:>9,} '
              f'{pz:5.0%}  {verdict}')

print()
print('=' * 76)
print('WHAT TO DO')
print('=' * 76)
print("""  PATH C cells are NOT a bug. They are the finding: these places have no
  climatic peer within their own biome, even today. That is worth a figure
  in its own right ("present-day analog scarcity"), and it explains WHY the
  retention map is grey there: you cannot lose analogs you never had.

  If you want coverage there, the knobs are:
    - raise SIGMA_CEILING (admits worse analogs; weakens the gate)
    - search ACROSS biomes (changes the scientific question)
    - lower N_ANALOGS (only helps PATH D, not B or C)

  PATH A/B cells are structural. Lower min_biome_cells to rescue A; nothing
  rescues B except shrinking BUFFER_KM.""")
