# Diagnostic: does the sigma_cap actually bind on the FUTURE analog distances?
#
# sigma = norm.ppf(0.5 + chi.cdf(D, df=m)/2) compresses violently in the tail.
# For df=2 it pins at sigma_cap=8 once D_maha ~ 8.28, so every pair beyond that
# reports the same capped value regardless of how far apart they really are.
#
# This script answers three questions:
#   Q1. Where does sigma saturate?  (map + per-biome fraction at cap)
#   Q2. How much dynamic range is lost? (D_maha spread among capped cells)
#   Q3. Does D_maha still discriminate where sigma does not?
#
# Run AFTER analog_future_stability.py (the patched version that writes
# dmaha_* and capfrac_*).

import numpy as np
import xarray as xr
from scipy.stats import chi, norm

data_root = '/glade/campaign/ral/risc/jsallen/TNC'
stab_file = f'{data_root}/analog_stability/analog_stability_gwl.nc'
sigma_cap = 8.0
GWLS = ['gwl2', 'gwl3']

biomes = {
    1: 'Boreal Forests/Taiga', 2: 'Deserts & Xeric Shrublands',
    5: 'Mediterranean Forests, Woodlands & Scrub',
    6: 'Montane Grasslands & Shrublands', 7: 'Rock and Ice',
    8: 'Temperate Broadleaf & Mixed Forests', 9: 'Temperate Conifer Forests',
    10: 'Temperate Grasslands, Savannas & Shrublands',
    12: 'Tropical & Subtropical Dry Broadleaf Forests',
    13: 'Tropical & Subtropical Grasslands, Savannas & Shrublands',
    14: 'Tropical & Subtropical Moist Broadleaf Forests', 15: 'Tundra',
}

ds = xr.open_dataset(stab_file)
bid = ds['biome_id'].values

# ---------------------------------------------------------------
# Where does the sigma transform stop discriminating? (df=2)
# ---------------------------------------------------------------
print('=' * 78)
print('THE TRANSFORM ITSELF (df=2)')
print('=' * 78)
print('  D_maha ->  sigma      (note how sigma stops moving)')
for d in [2, 3, 4, 5, 6, 6.5, 7, 8, 10, 15, 20, 30]:
    p = np.clip(chi.cdf(d, df=2), 0, 1 - 1e-15)
    sg = np.clip(norm.ppf(0.5 + p / 2), 0, sigma_cap)
    flag = '  <-- PINNED' if sg >= sigma_cap - 1e-6 else ''
    print(f'   {d:5.1f}   -> {sg:6.3f}{flag}')
d_at_cap = float(chi.ppf(2 * norm.cdf(sigma_cap) - 1, df=2))
print(f'\n  sigma hits the cap at D_maha = {d_at_cap:.2f}')
print('  Every pair beyond that distance is reported identically.')

# ---------------------------------------------------------------
# Q1 / Q2 / Q3, per biome
# ---------------------------------------------------------------
for g in GWLS:
    sg = ds[f'sigma_{g}_mean'].values
    dm = ds[f'dmaha_{g}_mean'].values
    cf = ds[f'capfrac_{g}'].values

    print()
    print('=' * 78)
    print(f'{g.upper()}: does the cap bind?')
    print('=' * 78)
    print(f'  {"biome":<50s} {"%cells":>7s} {"sigma":>7s} {"D_maha":>8s} {"D p90":>7s}')
    print(f'  {"":<50s} {"at cap":>7s} {"med":>7s} {"med":>8s} {"":>7s}')
    print('  ' + '-' * 74)

    rows = []
    for b, name in biomes.items():
        m = (bid == b) & np.isfinite(sg) & np.isfinite(dm)
        if m.sum() == 0:
            continue
        s_b = sg[m]; d_b = dm[m]
        # fraction of CELLS whose ensemble-mean sigma is pinned
        frac_cap = float(np.mean(s_b >= sigma_cap - 1e-6))
        rows.append((frac_cap, name, float(np.median(s_b)),
                     float(np.median(d_b)), float(np.percentile(d_b, 90))))

    for frac_cap, name, smed, dmed, dp90 in sorted(rows, reverse=True):
        warn = ' !!' if frac_cap > 0.10 else ''
        print(f'  {name:<50s} {frac_cap:6.1%} {smed:7.2f} {dmed:8.2f} '
              f'{dp90:7.2f}{warn}')

    # ---- Q2: how much range is hidden behind the cap? ----
    capped = np.isfinite(sg) & (sg >= sigma_cap - 1e-6) & np.isfinite(dm)
    if capped.sum() > 0:
        dc = dm[capped]
        print(f'\n  Among the {capped.sum():,} cells where sigma = {sigma_cap} exactly,')
        print(f'  D_maha actually ranges {dc.min():.2f} to {dc.max():.2f} '
              f'(median {np.median(dc):.2f}).')
        print(f'  sigma reports ALL of these as identical. That spread is the')
        print(f'  information the cap is destroying.')
    else:
        print(f'\n  No cells pinned at the cap. sigma is safe to use for {g}.')

    # ---- Q3: correlation between sigma and D_maha, capped vs not ----
    ok = np.isfinite(sg) & np.isfinite(dm)
    if ok.sum() > 10:
        r_all = np.corrcoef(sg[ok], dm[ok])[0, 1]
        unc = ok & (sg < sigma_cap - 1e-6)
        r_unc = np.corrcoef(sg[unc], dm[unc])[0, 1] if unc.sum() > 10 else np.nan
        print(f'\n  corr(sigma, D_maha) all cells      : {r_all:+.3f}')
        print(f'  corr(sigma, D_maha) uncapped only  : {r_unc:+.3f}')
        print('  (a much weaker "all cells" corr means the cap is flattening'
              ' the signal)')

# ---------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------
print()
print('=' * 78)
print('VERDICT')
print('=' * 78)
worst = 0.0
for g in GWLS:
    sgv = ds[f'sigma_{g}_mean'].values
    ok = np.isfinite(sgv)
    worst = max(worst, float(np.mean(sgv[ok] >= sigma_cap - 1e-6)))
print(f'  Max fraction of land cells pinned at sigma_cap: {worst:.1%}')
if worst < 0.01:
    print('  -> Cap is not binding. sigma is fine; D_maha is a supplement.')
elif worst < 0.10:
    print('  -> Cap binds in a minority of cells. Report sigma, but show the')
    print('     D_maha supplemental and state where sigma saturates.')
else:
    print('  -> Cap binds substantially. Do the quantitative analysis in D_maha.')
    print('     sigma should appear only for illustrative numbers in the text.')
