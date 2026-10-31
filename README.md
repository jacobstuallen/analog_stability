# analog_stability

Do historical within-biome climate analogs (TXx, Rx5day; Mahony et al. 2017
sigma dissimilarity) stay analogs under global warming levels?

## Layout
- `analog_config.py`: all paths, variables, models, GWL windows, biomes and
  method constants. Edit here, not in the scripts.
- `analog_common.py`, `icv_utils.py`: shared loading and distance code.
- Pipeline, in order:
  1. `analog.historical.sigma.2var.py`: single best historical analog per cell.
  2. `analog.set.selection.py`: diverse top-N analog sets -> `analog_sets.nc`.
  3. `analog.retention.py`: retention of each set under GWLs.
  4. `analog.future.stability.py`: future focal vs historical best analog.
- `diag.*.py`: diagnostics. `plot/`: figures. `ecoregion/`: ecoregion gridding.
- `tests/`: synthetic-data regression check (see below).

## Running on Casper
    qsub -v SCRIPT=analog.set.selection.py pbs.submit.htc

Paths can be overridden with `ANALOG_DATA_ROOT`, `ANALOG_OUT_DIR`,
`ANALOG_BIOME_FILE` and `ANALOG_ECO_FILE`.

## Future climatologies
`FUTURE_MODE` (env `ANALOG_FUTURE_MODE`) is `delta` by default: ERA5
1991-2020 + (model GWL window - model 1991-2020 window, historical + SSP
years). `raw` uses model output as-is and is kept only for comparison.
`ANALOG_GWLS` picks warming levels (default `gwl2,gwl3`; `gwl0` is the
baseline sanity check).

## Regression check
    python tests/make_synth.py /tmp/synth
    export ANALOG_DATA_ROOT=/tmp/synth ANALOG_OUT_DIR=/tmp/out \
           ANALOG_BIOME_FILE=/tmp/synth/biomes/biomes.analog.gridded.nc
    for s in analog.historical.sigma.2var.py analog.set.selection.py \
             analog.retention.py analog.future.stability.py; do python $s; done
    python tests/compare_outputs.py <reference_out_dir> /tmp/out
    # delta-mode sanity check (run retention + stability with ANALOG_GWLS=gwl0,gwl2,gwl3)
    python tests/check_gwl0.py /tmp/out

## Data not in the repo
TNC terrestrial ecoregions shapefile (`ecoregion/shp/`), all NetCDF inputs and
outputs (CCKP ERA5 and CMIP6 on glade).
