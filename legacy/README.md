# legacy

Retired 2026-10-31 (review plan step 3, issue H). Kept for reference and for
reproducing the first-draft figures; not maintained.

`analog.future.stability.py` scored each cell's future against its single
best historical analog (no sigma gate, no spatial spread). That is a
different analog definition from the diverse analog sets used for retention,
and the comparison mixed the focal cell's own departure with pair
divergence. Both are now separated in `analog.retention.py`
(`departure_*`, `divergence_*`, `comove_*`, `futdist_*`).

Plot and diag scripts here read `analog_stability_gwl.nc` /
`per_model_biome_sigma.npz` written by that script. Run from the repo root:
`python legacy/analog.future.stability.py`.

## Retired 2026-11-01
- `figures/`: figures from the single-best-analog pipeline (first draft).
- `plot.methods.summary.py` (older ICV construction) and
  `plot.methods.summary.global.py`. The current methods figure is
  `plot/plot.methods.summary.py` (formerly `plot.methods.summary.02.py`).
