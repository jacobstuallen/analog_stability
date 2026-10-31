# icv_utils.py
#
# Shared construction of the ERA5 interannual climatic variability (ICV)
# ruler used by all analog scripts.
#
# DESIGN DECISIONS (see methods):
#   * The ICV ruler is ERA5-based and FIXED. It is computed once from the
#     1991-2020 ERA5 annual index series and applied unchanged to both the
#     historical and future analysis. It is never recomputed per CMIP6 model.
#     Rationale: across-model spread in Rx5day ICV is large (CV up to ~1.8 in
#     tropical biomes) and would otherwise leak methodological scatter into
#     the precipitation dimension of the sigma metric.
#   * The ICV is DETRENDED. A per-cell linear trend is removed from each
#     annual index series before taking the standard deviation, so that the
#     1991-2020 forced warming signal is not absorbed into the denominator
#     as if it were interannual variability.
#   * The ICV is PER FOCAL CELL (Mahony et al. 2017 convention). The distance
#     from focal cell j to candidate i is scored in j's own ICV principal
#     component space, so the metric is asymmetric: D_ji != D_ij.
#
# NOTE: detrending does NOT move the climatological mean. With a mean-centered
# time axis the intercept of the linear fit equals the period mean, so
# clim_mean is identical to the undetrended nanmean. Only the SD and the
# cross-variable correlations change.

import numpy as np

MIN_YEARS = 5   # cells with fewer finite years than this get a NaN ICV


def detrend_icv(arr):
    """Per-cell linear detrend of an annual index series.

    Parameters
    ----------
    arr : (T, nlat, nlon) float64
        Annual index series (already QC-masked and, where applicable,
        log1p-transformed). May contain NaNs.

    Returns
    -------
    clim : (nlat, nlon)
        Period mean (equal to the linear-fit intercept on a centered time
        axis; unchanged by detrending).
    sd : (nlat, nlon)
        Standard deviation of the residual about the fitted linear trend,
        with ddof=2 (slope and intercept both fitted).
    anom : (T, nlat, nlon)
        Residual anomaly series about the fitted trend. Used downstream for
        the cross-variable ICV correlation matrix and for any inter-cell
        temporal correlation.
    """
    arr = np.asarray(arr, dtype=np.float64)
    T = arr.shape[0]

    t = np.arange(T, dtype=np.float64)
    t = t - t.mean()                       # center -> intercept == mean

    finite = np.isfinite(arr)
    a = np.where(finite, arr, 0.0)
    w = finite.astype(np.float64)

    n = w.sum(axis=0)                      # finite years per cell
    ok = n > 0

    tw = (w * t[:, None, None]).sum(axis=0)
    tbar = np.divide(tw, n, out=np.full(n.shape, np.nan), where=ok)
    ybar = np.divide(a.sum(axis=0), n, out=np.full(n.shape, np.nan), where=ok)

    tc = (t[:, None, None] - tbar[None]) * w
    yc = (a - ybar[None]) * w

    sxy = (tc * yc).sum(axis=0)
    sxx = (tc * tc).sum(axis=0)
    with np.errstate(invalid='ignore', divide='ignore'):
        slope = np.where(sxx > 0, sxy / sxx, 0.0)

    fitted = ybar[None] + slope[None] * (t[:, None, None] - tbar[None])
    anom = np.where(finite, arr - fitted, np.nan)

    clim = ybar
    with np.errstate(invalid='ignore', divide='ignore'):
        sd = np.sqrt(np.nansum(anom ** 2, axis=0)
                     / np.maximum(n - 2.0, 1.0))     # ddof=2
    sd = np.where(n >= MIN_YEARS, sd, np.nan)
    sd = np.where(sd > 0, sd, np.nan)

    return clim, sd, anom


def icv_correlation(anom_i, anom_j, sd_i, sd_j):
    """Per-cell Pearson correlation between two detrended ICV anomaly series.

    Uses the pairwise-finite year count with ddof=2 (matching detrend_icv),
    rather than a hardcoded T/(T-1) factor, so cells with missing years are
    handled correctly and the normalization is consistent with sd_i / sd_j.
    """
    prod = anom_i * anom_j
    npair = np.sum(np.isfinite(prod), axis=0).astype(np.float64)
    with np.errstate(invalid='ignore', divide='ignore'):
        cov = np.nansum(prod, axis=0) / np.maximum(npair - 2.0, 1.0)
        r = cov / (sd_i * sd_j)
    r = np.where(npair >= MIN_YEARS, r, np.nan)
    return np.clip(r, -0.999, 0.999)


def build_icv_corr(VKEYS, icv_sd, icv_series):
    """Assemble the {(i,j): corr_grid} dict of ICV cross-correlations."""
    K = len(VKEYS)
    icv_corr = {}
    for i in range(K):
        for j in range(i + 1, K):
            icv_corr[(i, j)] = icv_correlation(
                icv_series[VKEYS[i]], icv_series[VKEYS[j]],
                icv_sd[VKEYS[i]], icv_sd[VKEYS[j]],
            )
    return icv_corr
