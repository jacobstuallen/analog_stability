import numpy as np

f = '/glade/campaign/ral/risc/jsallen/TNC/analog_stability/analog_historical_dist_summary.npz'
z = np.load(f)

biomes = {
    1:'Boreal Forests/Taiga', 2:'Deserts & Xeric Shrublands',
    5:'Mediterranean Forests, Woodlands & Scrub', 6:'Montane Grasslands & Shrublands',
    7:'Rock and Ice', 8:'Temperate Broadleaf & Mixed Forests',
    9:'Temperate Conifer Forests', 10:'Temperate Grasslands, Savannas & Shrublands',
    12:'Tropical & Subtropical Dry Broadleaf Forests',
    13:'Tropical & Subtropical Grasslands, Savannas & Shrublands',
    14:'Tropical & Subtropical Moist Broadleaf Forests', 15:'Tundra',
}

print('keys in file:', sorted(z.files)[:6], '...')

# recover the percentile levels from any biome present
lev = None
for b in biomes:
    k = f'biome_{b}_pctile_levels'
    if k in z.files:
        lev = z[k]
        break
print('\npercentile levels:', lev.astype(int))

for tag, unit in [('dmaha_pctiles', 'D_maha'), ('sigma_pctiles', 'sigma')]:
    print(f'\n===== within-biome PAIRWISE {unit} distribution =====')
    hdr = '  ' + f'{"biome":<52s}' + ''.join(f'{int(p):>8d}' for p in lev)
    print(hdr)
    print('  ' + '-'*(52 + 8*len(lev)))
    for b, name in biomes.items():
        k = f'biome_{b}_{tag}'
        if k not in z.files:
            continue
        v = z[k]
        print('  ' + f'{name:<52s}' + ''.join(f'{x:>8.3f}' for x in v))
