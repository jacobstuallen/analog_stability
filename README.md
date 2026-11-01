# analog_stability

Do historical within-biome climate analogs (TXx, Rx5day; Mahony et al. 2017
sigma dissimilarity) stay analogs under global warming levels?

## Layout
- `analog_config.py`: all paths, variables, models, GWL windows, biomes and
  method constants. Edit here, not in the scripts.
- `analog_common.py`, `icv_utils.py`: shared loading and distance code.
- Pipeline, in order:
  1. `analog.historical.sigma.2var.py`: single best historical analog per cell
     (methods context and Dt calibration distributions only).
  2. `analog.set.selection.py`: diverse top-N analog sets -> `analog_sets.nc`.
  3. `analog.retention.py`: retention of each set under GWLs, plus local
     departure, pair divergence and co-movement -> `analog_retention.nc`.
     Per-model ensemble medians, plus `*_mmm_*` from the multi-model-mean
     future (forced response, sampling noise reduced); analytic noise floors
     in the file attributes.
  4. `analog.scaling.py`: linear-scaling test of pair divergence with GWL
     (GWL1.5-4, common models, delta mode) -> `analog_scaling.nc`.
- `legacy/`: retired single-best-analog stability pipeline (first draft).
- `diag.*.py`: diagnostics. `diag.model.icv.py` (model vs ERA5 ICV) feeds the
  noise model of `analog.scaling.py`; run it before scaling. `plot/`: figures. `ecoregion/`: ecoregion gridding.
- `tests/`: synthetic-data regression check (see below).

## Running on Casper
    qsub -v SCRIPT=analog.set.selection.py pbs.submit.htc

Paths can be overridden with `ANALOG_DATA_ROOT`, `ANALOG_OUT_DIR`,
`ANALOG_BIOME_FILE` and `ANALOG_ECO_FILE`.

## Future climatologies
`FUTURE_MODE` (env `ANALOG_FUTURE_MODE`) is `delta` by default: ERA5
1991-2020 + (model GWL window - model 1991-2020 window, historical + SSP
years). `raw` uses model output as-is and is kept only for comparison.
`ANALOG_GWLS` picks warming levels (default `gwl1p5,gwl2,gwl3,gwl4`;
`gwl0` is the baseline sanity check).

## Warming levels
Windows come from the IPCC AR6 Atlas per-member table (Hauser et al.,
zenodo 3591806), stored in `data/gwl_windows_ar6_ssp585.csv` and rebuilt
with `tools/make_gwl_table.py`. 30 models reach GWL1.5-3; 18 reach GWL4.
`ANALOG_GWL_SOURCE=tye` restores the Tye et al. 2025 GWL2/3 windows used in
the first draft. `ANALOG_MODEL_SET=common` restricts every GWL to models
that reach all requested GWLs (outputs get a `_common` suffix).

## Regression check
    python tests/make_synth.py /tmp/synth
    export ANALOG_DATA_ROOT=/tmp/synth ANALOG_OUT_DIR=/tmp/out \
           ANALOG_BIOME_FILE=/tmp/synth/biomes/biomes.analog.gridded.nc
    for s in analog.historical.sigma.2var.py analog.set.selection.py \
             analog.retention.py analog.scaling.py; do python $s; done
    python tests/compare_outputs.py <reference_out_dir> /tmp/out
    # delta-mode sanity check (run retention with ANALOG_GWLS=gwl0,gwl2,gwl3)
    python tests/check_gwl0.py /tmp/out
    # statistics behind analog.scaling.py
    python tests/check_scaling_stats.py
    # per-variable Mahalanobis shares (attribution)
    python tests/check_var_shares.py

## Data not in the repo
TNC terrestrial ecoregions shapefile (`ecoregion/shp/`), all NetCDF inputs and
outputs (CCKP ERA5 and CMIP6 on glade).
