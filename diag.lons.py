import numpy as np, xarray as xr

ds  = xr.open_dataset(ret_file)
eco = xr.open_dataset(eco_file, mask_and_scale=False)

for name, d in [('retention', ds), ('ecoregion', eco)]:
    la, lo = d['lat'].values, d['lon'].values
    print(f'{name:10s} lat[0]={la[0]:8.3f} lat[-1]={la[-1]:8.3f} '
          f'n={len(la):4d} {"ASC" if la[0] < la[-1] else "DESC"}')
    print(f'{" ":10s} lon[0]={lo[0]:8.3f} lon[-1]={lo[-1]:8.3f} '
          f'n={len(lo):4d} {"0-360" if lo.max() > 180 else "-180/180"}')

print('\nlat identical:', np.allclose(ds.lat.values, eco.lat.values))
print('lon identical:', np.allclose(ds.lon.values, eco.lon.values))
