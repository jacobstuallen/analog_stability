#!/bin/python
# Author: Mostly Claude from analog.historical.sigma2var.py (jsallen@ucar.edu)
#
# ANALOG SET SELECTION (historical-only, one-time). Writes analog_sets.nc,
# which the retention script then reads (analogous to how method 1 reads
# analog_historical_sigma.nc).
#
# For each focal land cell we build a DIVERSE set of up to N within-biome
# historical analogs, designed for the adaptation-transfer question: which
# present-day regions share this cell's extreme-climate regime, drawn from
# spatially distinct locations so the set is not redundant.
#
# Selection algorithm, per focal cell (all HISTORICAL ERA5):
#   1. Candidate pool: same-biome cells OUTSIDE the 250 km buffer (the
#      diagnostic's decorrelation result: closer than this and an "analog"
#      is just the same weather system), with finite climatology.
#   2. Score historical sigma dissimilarity (Mahony 2017, in the FOCAL
#      cell's ICV PC space) to every candidate.
#   3. QUALITY GATE: drop candidates with sigma > SIGMA_CEILING. Anchored to
#      the literature (Parks et al. 2022 used <0.5 sigma for "good"; we use
#      the looser ~1 sigma so we are not over-restrictive). This stops
#      relaxation from ever trading match quality for hitting N.
#   4. GREEDY SPREAD at spacing s = S_TRY[0] (500 km): walk the gated pool in
#      ascending sigma, accept a candidate only if it is >= s from EVERY
#      already-accepted analog. Stop at N. The buffer stops trivial neighbors
#      of the focal cell; s stops the analogs from clustering among
#      themselves (the diagnostic showed top-N analogs are interchangeable on
#      quality in homogeneous biomes, so WHICH N you keep is a spread choice).
#   5. STEPWISE RELAXATION: if fewer than N accepted, retry at the next
#      smaller s in S_TRY (down to the 250 km floor), re-walking from the
#      same gated pool. Take the first s that yields N. The quality gate is
#      NEVER relaxed, so relaxation only ever trades spacing, not quality.
#   6. If still < N at the floor, accept what we have (honest "few diverse
#      good analogs" outcome; recorded, not hidden).
#
# Per-cell outputs recorded (so retention can be read honestly later):
#   analog_lat/lon  (nlat,nlon,N)  padded NaN
#   analog_sigma    (nlat,nlon,N)  historical sigma of each accepted analog
#   n_analogs       (nlat,nlon)    count actually found (<= N)
#   s_used_km       (nlat,nlon)    final spacing that produced the set
#   eff_n           (nlat,nlon)    effective sample size of the set
#                                  (Kish-style, decorrelation-weighted;
#                                   ADJUSTABLE -- see compute_eff_n)
#
# Variable set is the same CONFIG list as method 1 (TXx, Rx5day now).

import numpy as np
import xarray as xr
import time

import analog_config as cfg
from analog_common import (load_biomes, load_era5_icv, haversine_km,
                           sigma_from_distance, ensure_out_dir)

# ==============================
# Configuration (edit in analog_config.py; aliased here)
# ==============================
out_dir = cfg.OUT_DIR
outfile = cfg.SETS_FILE
clim_start, clim_end = cfg.CLIM_START, cfg.CLIM_END
pc_var_min = cfg.PC_VAR_MIN
min_biome_cells = cfg.MIN_BIOME_CELLS
sigma_cap = cfg.SIGMA_CAP

# ---- selection knobs ----
N_ANALOGS = cfg.N_ANALOGS          # target set size (practicality; no quality knee)
BUFFER_KM = cfg.BUFFER_KM          # trivial-neighbor exclusion (decorrelation result)
SIGMA_CEILING = cfg.SIGMA_CEILING  # quality gate (Parks 2022 <0.5 good; ~1 loose)
S_TRY = cfg.S_TRY                  # spacing relaxation ladder, floor = buffer
DECORR_KM = cfg.DECORR_KM          # spatial scale for effective-N downweighting
focal_chunk = 2000                 # focal cells processed per progress print

VKEYS, K = cfg.VKEYS, cfg.K
biomes = cfg.BIOMES


# ==============================
# Helpers
# ==============================
def load_inputs():
    """Load biome grid + per-variable ERA5 ICV (clim mean, sd, correlations).
    Same construction as the historical and retention scripts."""
    print("Loading biome mask...")
    biome_grid, bmin, bmax = load_biomes()
    clim_mean, icv_sd, _, icv_corr, lat_arr, lon_arr = load_era5_icv(bmin, bmax)
    return (biome_grid, clim_mean, icv_sd, icv_corr,
            lat_arr, lon_arr, len(lat_arr), len(lon_arr))


def sigma_focal_to_candidates(fi, fj, ci, cj, clim_mean, icv_sd, icv_corr):
    """Historical sigma from focal (fi,fj) to candidate cells (ci,cj),
    scored in the FOCAL cell's ICV PC space. Vectorized over candidates.
    Returns sigma array (len = n candidates), or None if focal invalid."""
    mu_f = np.array([clim_mean[vk][fi, fj] for vk in VKEYS])
    sd_f = np.array([icv_sd[vk][fi, fj] for vk in VKEYS])
    if not (np.all(np.isfinite(mu_f)) and np.all(np.isfinite(sd_f)) and np.all(sd_f > 0)):
        return None
    R = np.eye(K)
    for (i, j), grid in icv_corr.items():
        rij = grid[fi, fj]; R[i, j] = rij; R[j, i] = rij
    evals, evecs = np.linalg.eigh(R)
    cand = np.column_stack([clim_mean[vk][ci, cj] for vk in VKEYS])  # (m,K)
    diff = (cand - mu_f) / sd_f
    proj = diff @ evecs
    keep = evals >= pc_var_min
    with np.errstate(invalid='ignore', divide='ignore'):
        contrib = np.where(keep[None, :], proj**2 / evals[None, :], 0.0)
    d = np.sqrt(np.nansum(contrib, axis=1))
    mdims = int(keep.sum())
    if mdims < 1:
        return None
    return sigma_from_distance(d, mdims)


def greedy_spread(order_idx, cand_lat, cand_lon, s_km, n_target):
    """Greedy spatial-spread selection. order_idx is candidate indices sorted
    by ascending sigma (best first). Accept a candidate only if it is >= s_km
    from every already-accepted analog. Returns list of accepted candidate
    indices (into the candidate arrays), up to n_target.
    Serial by nature (each acceptance depends on prior ones), but cheap:
    at most n_target accepted, so the inner distance check is tiny."""
    accepted = []
    acc_lat = []; acc_lon = []
    for idx in order_idx:
        if not accepted:
            accepted.append(idx); acc_lat.append(cand_lat[idx]); acc_lon.append(cand_lon[idx])
        else:
            dmin = haversine_km(cand_lat[idx], cand_lon[idx],
                                np.array(acc_lat), np.array(acc_lon)).min()
            if dmin >= s_km:
                accepted.append(idx); acc_lat.append(cand_lat[idx]); acc_lon.append(cand_lon[idx])
        if len(accepted) >= n_target:
            break
    return accepted


def compute_eff_n(sel_lat, sel_lon, decorr_km=DECORR_KM):
    """Effective sample size of an analog set, downweighting spatially
    clustered members. Kish-style: eff_n = (sum w)^2 / sum(w^2) is the
    classic form for unequal weights; here we instead use the redundancy
    implied by pairwise proximity. We build a similarity matrix C where
    C_ab = exp(-d_ab / decorr_km) (1 on diagonal, ->0 for distant pairs),
    and define eff_n = n^2 / sum(C). Two coincident analogs -> counts as 1;
    n fully-separated analogs -> counts as n.
    ADJUSTABLE: swap this formula freely; retention does not depend on it."""
    n = len(sel_lat)
    if n <= 1:
        return float(n)
    la = np.array(sel_lat); lo = np.array(sel_lon)
    D = haversine_km(la[:, None], lo[:, None], la[None, :], lo[None, :])
    C = np.exp(-D / decorr_km)
    return float(n*n / C.sum())


def select_analog_sets():
    (biome_grid, clim_mean, icv_sd, icv_corr,
     lat_arr, lon_arr, nlat, nlon) = load_inputs()

    # output arrays (padded fixed-width)
    analog_lat = np.full((nlat, nlon, N_ANALOGS), np.nan, np.float32)
    analog_lon = np.full((nlat, nlon, N_ANALOGS), np.nan, np.float32)
    analog_sig = np.full((nlat, nlon, N_ANALOGS), np.nan, np.float32)
    n_found = np.zeros((nlat, nlon), np.int16)
    s_used = np.full((nlat, nlon), np.nan, np.float32)
    eff_n = np.full((nlat, nlon), np.nan, np.float32)

    for b, bname in biomes.items():
        bm = (biome_grid == b)
        ci, cj = np.where(bm)
        if len(ci) < min_biome_cells:
            print(f'Biome {b:2d} ({bname}): {len(ci)} cells, SKIPPING')
            continue
        # valid candidates (finite climatology in all vars)
        valid = np.ones(len(ci), bool)
        for vk in VKEYS:
            valid &= np.isfinite(clim_mean[vk][ci, cj])
        ci, cj = ci[valid], cj[valid]
        if len(ci) < min_biome_cells:
            print(f'Biome {b:2d} ({bname}): {len(ci)} valid, SKIPPING')
            continue
        cand_lat = lat_arr[ci]; cand_lon = lon_arr[cj]
        nb = len(ci)
        t0 = time.time()
        relaxed_cells = 0; short_cells = 0

        for k in range(nb):
            fi, fj = ci[k], cj[k]
            # distance focal -> all candidates; buffer exclusion + self
            dfoc = haversine_km(lat_arr[fi], lon_arr[fj], cand_lat, cand_lon)
            pool = dfoc > BUFFER_KM
            pool[k] = False
            if pool.sum() < 1:
                continue
            pidx = np.where(pool)[0]
            sig = sigma_focal_to_candidates(fi, fj, ci[pidx], cj[pidx],
                                            clim_mean, icv_sd, icv_corr)
            if sig is None:
                continue
            # quality gate
            gate = sig <= SIGMA_CEILING
            if gate.sum() < 1:
                continue
            gidx = pidx[gate]; gsig = sig[gate]
            order = np.argsort(gsig)              # best first, indices into gidx
            ordered_cand = gidx[order]            # candidate indices (into ci/cj)
            ordered_sig = gsig[order]

            # stepwise relaxation over spacing
            chosen = None; s_final = None
            for s_km in S_TRY:
                acc_local = greedy_spread(np.arange(len(ordered_cand)),
                                          cand_lat[ordered_cand], cand_lon[ordered_cand],
                                          s_km, N_ANALOGS)
                if len(acc_local) >= N_ANALOGS:
                    chosen = acc_local; s_final = s_km
                    break
                # keep the best (largest) set seen so far as fallback
                if chosen is None or len(acc_local) > len(chosen):
                    chosen = acc_local; s_final = s_km
            if s_final != S_TRY[0]:
                relaxed_cells += 1
            if len(chosen) < N_ANALOGS:
                short_cells += 1

            sel = ordered_cand[chosen]            # candidate indices
            sel_sig = ordered_sig[chosen]
            m = len(sel)
            analog_lat[fi, fj, :m] = cand_lat[sel]
            analog_lon[fi, fj, :m] = cand_lon[sel]
            analog_sig[fi, fj, :m] = sel_sig
            n_found[fi, fj] = m
            s_used[fi, fj] = s_final
            eff_n[fi, fj] = compute_eff_n(cand_lat[sel], cand_lon[sel])

            if (k+1) % focal_chunk == 0:
                print(f'  biome {b:2d} {bname[:24]:<24} {k+1}/{nb} cells '
                      f'| {time.time()-t0:.0f}s', flush=True)

        print(f'Biome {b:2d} ({bname}): {nb} cells done, '
              f'{relaxed_cells} relaxed, {short_cells} short of {N_ANALOGS} '
              f'| {time.time()-t0:.0f}s')

    return (analog_lat, analog_lon, analog_sig, n_found, s_used, eff_n,
            biome_grid, lat_arr, lon_arr)


def write_output(analog_lat, analog_lon, analog_sig, n_found, s_used, eff_n,
                 biome_grid, lat_arr, lon_arr):
    ensure_out_dir(out_dir)
    nrank = np.arange(N_ANALOGS)
    ds = xr.Dataset(
        coords={'lat': lat_arr, 'lon': lon_arr, 'rank': nrank},
        data_vars={
            'analog_lat':  (['lat', 'lon', 'rank'], analog_lat),
            'analog_lon':  (['lat', 'lon', 'rank'], analog_lon),
            'analog_sigma':(['lat', 'lon', 'rank'], analog_sig),
            'n_analogs':   (['lat', 'lon'], n_found),
            's_used_km':   (['lat', 'lon'], s_used),
            'eff_n':       (['lat', 'lon'], eff_n),
            'biome_id':    (['lat', 'lon'], biome_grid.astype(np.float32)),
        })
    ds.attrs['method'] = (
        'Diverse within-biome historical analog sets (Mahony 2017 sigma, '
        'half-normal convention). Per focal cell: same-biome candidates '
        f'outside {BUFFER_KM:.0f} km buffer, quality-gated at sigma<= '
        f'{SIGMA_CEILING}, greedy spatial-spread top-{N_ANALOGS} with '
        'stepwise spacing relaxation. Feeds the analog-retention method.')
    ds.attrs['icv_definition'] = cfg.ICV_DEFINITION
    ds.attrs['buffer_km'] = BUFFER_KM
    ds.attrs['sigma_ceiling'] = SIGMA_CEILING
    ds.attrs['n_target'] = N_ANALOGS
    ds.attrs['spacing_ladder_km'] = ', '.join(f'{s:.0f}' for s in S_TRY)
    ds.attrs['variables'] = ', '.join(VKEYS)
    ds.attrs['variable_set'] = cfg.VARSET
    ds.to_netcdf(outfile)
    print(f'\nSaved: {outfile}')

    # quick coverage summary
    print('\n--- coverage per biome ---')
    print(f'{"biome":<48} {"med n":>6} {"% short":>8} {"med effN":>9}')
    for b, bname in biomes.items():
        bm = (biome_grid == b)
        nn = n_found[bm]; nn = nn[nn > 0]
        if nn.size == 0:
            continue
        en = eff_n[bm]; en = en[np.isfinite(en)]
        pct_short = 100.0 * np.mean(nn < N_ANALOGS)
        print(f'{bname:<48} {np.median(nn):>6.0f} {pct_short:>7.1f}% '
              f'{np.median(en):>9.1f}')


# ==============================
# Run
# ==============================
results = select_analog_sets()
write_output(*results)
print('\nDone.')
