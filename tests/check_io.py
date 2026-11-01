# Tests for analog_common I/O helpers:
#  - _read_years on a file with an UNSORTED and DUPLICATED time axis returns
#    each requested year once, in order, with the first record of a repeat
#  - ensure_out_dir copies the parent's permission bits (incl. setgid)
# Usage: python tests/check_io.py
import sys
import os
import stat
import tempfile
import numpy as np
import xarray as xr
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from analog_common import _read_years, ensure_out_dir

tmp = tempfile.mkdtemp()
years = [2015, 2016, 2018, 2017, 2019, 2019, 2020]          # unsorted + repeated 2019
vals = np.array([15, 16, 18, 17, 19, 99, 20], float)        # second 2019 = 99 (must be dropped)
t = [np.datetime64(f'{y}-07-01') for y in years]
da = np.broadcast_to(vals[:, None, None], (len(years), 2, 2)).copy()
p = os.path.join(tmp, 'x.nc')
xr.Dataset({'v': (['time', 'lat', 'lon'], da)},
           coords={'time': t, 'lat': [0.0, 1.0], 'lon': [0.0, 1.0]}).to_netcdf(p)
out = _read_years(p, 'v', 2016, 2019, -10, 10)[:, 0, 0]
print('read 2016-2019 ->', out)
assert np.array_equal(out, [16, 17, 18, 19])

par = os.path.join(tmp, 'parent'); os.makedirs(par)
os.chmod(par, 0o2770)
child = ensure_out_dir(os.path.join(par, 'child'))
mode = stat.S_IMODE(os.stat(child).st_mode)
print(f'parent {oct(stat.S_IMODE(os.stat(par).st_mode))} child {oct(mode)}')
assert mode == 0o2770
print('PASS')
