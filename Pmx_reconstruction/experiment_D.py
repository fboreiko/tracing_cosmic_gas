#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""==============================================================================
EXPERIMENT D  --  is the residual intra-halo? Spherise the halos and see
==============================================================================
A halo's own profile is a function of the WAVEVECTOR: clumpiness shows up
as variation with k-hat at fixed |k|, and the stacked u_bar, being the
k-hat average, throws precisely that away. Spectra are binned in |k|, so
R*_i is the shell average of a product while R_i = f_i u_bar P_halo_gas is
a product of separately shell-averaged factors, and the gap between them
splits into two covariances:

    intra-halo    within each halo, over the directions of the k shell,
                between its own profile and its own cross with the gas --
                its clumps and the gas in them pointing the same way. No
                radial profile can carry it, and no rescaling of the U
                template can either, because the material belongs to a
                RESOLVED halo.
    halo-to-halo  across the bin's haloes, between the shell-averaged
                quantities. It survives even if haloes are perfectly
                smooth spheres, and it is the k -> 0 limit of the whole
                thing.

Randomising each halo's interior replaces its profile by its own angular
average, so it removes the first and keeps the second. Hence

    R*(true) - R*(randomised)   =   the intra-halo term, MEASURED.

If the substructure reading is right, that difference accounts for
essentially all of R - R* at k >~ 2 and R*(randomised) lands on the model.
If it does not, the reading is wrong and the residual is halo-to-halo
covariance -- which is a bin-width problem, not a physics one.

WHAT THIS DOES AND DOES NOT CONTROL FOR

    Randomising the matter destroys the halo's alignment with its environment
    at the same time as its internal clump-gas alignment, so a surviving
    difference is "intra-halo" in the broad sense: clumps, triaxiality and
    filament alignment together. At k >~ 3 the scales in question are well inside r200m, where an alignment
    with the large-scale field is not a plausible carrier.

USAGE
    # measure the randomised cache (one particle pass per species) and
    # score it; the production cache and the u_bar stack must exist
    python -m Pmx_reconstruction.experiment_D --measure

    # score caches that are already on disk, refusing to measure anything
    python -m Pmx_reconstruction.experiment_D

    # the aperture the aperture sweep leaves the dip untouched at
    python -m Pmx_reconstruction.experiment_D --aperture 2 --measure
------------------------------------------------------------------------------
"""
from dataclasses import replace

import numpy as np
import matplotlib
matplotlib.use('Agg')

from utils.pipeline_paths import ensure_parents, plot_path
from utils.plot_data import save_plot_data

from Pmx_reconstruction.pmxlib.bundle import load_or_measure
from Pmx_reconstruction.pmxlib.config import (MassRange, PmxConfig,
                                              base_parser, stem_D)
from Pmx_reconstruction.pmxlib.reconstruction import Profile
from Pmx_reconstruction.pmxlib import plotting as pl
from Pmx_reconstruction.pmxlib import rstar_ustar as ru
from Pmx_reconstruction.pmxlib import spherise

K_TABLE = (0.1, 0.3, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0)
K_DIP = (2.0, 8.0)
EXACT = 1e-12


def model_R(data, tot, aperture, profile):
    """R = sum_i f_in,i u_m(k|M_i) P_halo_gas(k|M_i), over the occupied bins.

    Deliberately NOT compute_R: this experiment always weights by the assigned
    mass, whatever cfg.profile_source says, because that is the pairing R* is
    measured in. Passing the catalogue's n_i M_i here would put the
    0.255-against-0.249 mass gap into the residual and it would read as
    intra-halo structure.
    """
    k = np.asarray(data['k_center'], float)
    M_i = np.asarray(data['M_mean'], float)
    occ = np.flatnonzero(np.asarray(data['counts'], float) > 0)
    u = np.zeros((M_i.size, k.size))
    u[occ] = profile.u(k, M_i[occ], trunc=aperture)
    return np.sum(np.asarray(tot['f_in'], float)[:, None] * u
                  * np.asarray(data['P_halo_gas'], float), axis=0)


def run_experiment_D(cfg, data, randomise='shuffle', aperture=1.0,
                     allow_compute=False, recompute=False, seed=None):
    print("\n" + "=" * 70)
    print(f"EXPERIMENT D -- R* with the halo interiors randomised "
          f"('{randomise}'), x = {aperture:g}")
    print("=" * 70)
    randomise = spherise.check_mode(randomise)
    if randomise is None:
        raise SystemExit("experiment D needs a randomisation mode")

    k = np.asarray(data['k_center'], float)
    z = float(data['redshift'])
    P_true = np.asarray(data['P_matter_gas'], float)
    logM_cen = np.asarray(data['logM_cen'], float)
    mr = MassRange.from_config(cfg, data)
    ks = [kk for kk in K_TABLE if kk <= k[-1]]
    ik = [int(np.argmin(np.abs(k - kk))) for kk in ks]     # table columns

    # --- the two measurements -------------------------------------------------
    prod = ru.load_or_measure_rstar_ustar(cfg, data, aperture=aperture,
                                          allow_compute=allow_compute)
    if prod is None:
        raise SystemExit(
            f"No production R*/U* cache at aperture {aperture:g}:\n"
            f"  {ru.cache_path(cfg, aperture=aperture)}\nBuild it with\n"
            f"      python -m Pmx_reconstruction.pmxlib.rstar_ustar "
            f"--aperture {aperture:g}")
    rand = ru.load_or_measure_rstar_ustar(cfg, data, aperture=aperture,
                                          allow_compute=allow_compute,
                                          recompute=recompute,
                                          randomise=randomise, seed=seed)
    if rand is None:
        seed_arg = '' if seed is None else f' --seed {seed}'
        raise SystemExit(
            f"No '{randomise}' R*/U* cache at aperture {aperture:g}:\n"
            f"  {ru.cache_path(cfg, aperture=aperture, randomise=randomise, seed=seed)}\n"
            f"Build it with\n"
            f"      python -m Pmx_reconstruction.pmxlib.rstar_ustar "
            f"--aperture {aperture:g} --randomise {randomise}{seed_arg}\n"
            f"or re-run this with --measure.")

    # The labels do not change under the randomisation, so everything built
    # from them must be bit-identical; if not, the two caches are not the same
    # measurement and nothing below means what it says.
    for key in ('f_in', 'f_i', 'counts', 'M_mean', 'logM_edges',
                'm2_bin_gas', 'mass_bin_dm', 'mass_bin_gas'):
        if key in prod and key in rand and not np.allclose(
                np.asarray(prod[key], float), np.asarray(rand[key], float),
                rtol=EXACT, atol=0.0):
            raise SystemExit(
                f"[D] the production and randomised caches disagree on "
                f"'{key}', which the randomisation does not touch. Re-measure "
                f"the randomised one with the same --nbins / --logm-min / "
                f"--logm-max / --aperture.")

    # U* is the same unmoved matter by two routes: a residual in production,
    # a direct painting in the randomised run.
    with np.errstate(divide='ignore', invalid='ignore'):
        dev = np.abs(np.asarray(rand['U_star_gas'], float)
                     / np.asarray(prod['U_star_gas'], float) - 1.0)
    print(f"[D] U* residual vs painted: max |ratio - 1| = {np.nanmax(dev):.2e}")

    tot = ru.measured_partition(cfg, prod, data, mr=mr)
    tot_r = ru.measured_partition(cfg, rand, data, mr=mr)
    R_star, R_star_rand = tot['R_star'], tot_r['R_star']
    R_star_i = np.asarray(tot['R_star_i'], float)
    C_i = R_star_i - np.asarray(tot_r['R_star_i'], float)

    # u_j -> 1 whatever the angles, so the two must agree as k -> 0.
    low = k < 0.15
    with np.errstate(divide='ignore', invalid='ignore'):
        dev = R_star_rand[low] / R_star[low] - 1.0
    print(f"[D] R*(randomised)/R*(true) - 1 over k < 0.15: "
          f"max |.| = {np.nanmax(np.abs(dev)):.2e} (must be ~0)")

    # The smeared self-pair, as a fraction of the raw randomised R*.
    term = np.asarray(rand['R_star_gas_selfpair'], float).sum(axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        frac = term / (R_star_rand + term)
    print("\n  k =              " + "".join(f"{kk:>9.2f}" for kk in ks))
    print("  selfpair / raw  " + "".join(f"{frac[j]:9.4f}" for j in ik))

    # --- the model the identity is written against ----------------------------
    # The split is clean only against the MEASURED radial profile; NFW is the
    # fallback, and then the residual also carries the profile error of
    # Experiment C.
    try:
        prof = Profile.from_config(replace(cfg, profile_source='measured'), z,
                                   aperture=aperture,
                                   allow_compute=allow_compute)
        R_meas = model_R(data, tot, aperture, prof)
    except SystemExit as exc:
        print(f"\n[D] no stacked-profile cache at aperture {aperture:g}; the "
              f"split below is against NFW. Build the stack with\n"
              f"      python -m Pmx_reconstruction.pmxlib.u_bar "
              f"--aperture {aperture:g}\n  ({exc})")
        R_meas = model_R(data, tot, aperture, Profile(cfg, z, aperture=aperture))

    # --- the split: total = halo-to-halo - intra-halo, exactly -----------------
    total = R_meas - R_star
    halo = R_meas - R_star_rand
    intra = R_star - R_star_rand
    with np.errstate(divide='ignore', invalid='ignore'):
        share = np.where(total != 0.0, -intra / total, np.nan)
        rows = [('R/R* - 1', R_meas / R_star - 1.0),
                ('R*rand/R* - 1', R_star_rand / R_star - 1.0),
                ('total /P', total / P_true),
                ('halo-to-halo /P', halo / P_true),
                ('-intra-halo /P', -intra / P_true),
                ('intra / total', share)]
    print()
    for name, y in rows:
        if y is not None:
            print(f"  {name:16s}" + "".join(f"{y[j]:9.4f}" for j in ik))

    dip = (k >= K_DIP[0]) & (k <= K_DIP[1])
    with np.errstate(divide='ignore', invalid='ignore'):
        print(f"\n  Over k = {K_DIP[0]:g}--{K_DIP[1]:g} h/cMpc, median:"
              f"\n      R - R*             = "
              f"{np.nanmedian((total / P_true)[dip]):+.4f} of P^me"
              f"\n      of which intra-halo  {np.nanmedian(share[dip]):.1%}"
              f"\n      left over          = "
              f"{np.nanmedian((halo / P_true)[dip]):+.4f} of P^me")

    # Where in halo mass the intra-halo term sits.
    jb = [int(np.argmin(np.abs(k - kk))) for kk in (1.0, 3.0, 5.0) if kk <= k[-1]]
    print("\n  per bin, intra-halo / R*_i:")
    print("    logM " + "".join(f"{k[j]:>12.2f}" for j in jb))
    for i in np.flatnonzero(np.any(R_star_i != 0.0, axis=1) & mr.resolved):
        with np.errstate(divide='ignore', invalid='ignore'):
            print(f"   {logM_cen[i]:5.2f} "
                  + "".join(f"{C_i[i, j] / R_star_i[i, j]:12.4f}" for j in jb))

    # --- figure and plot data ---------------------------------------------------
    d = pl.SpheriseData(
        k=k, k_Ny=float(data['k_Nyquist']),
        k_split=float(data['k_split']) if 'k_split' in data else None,
        P_true=P_true, R_star=R_star, R_star_rand=R_star_rand, R_meas=R_meas,
        logM=logM_cen, C_i=C_i, R_star_i=R_star_i,
        mode=randomise, aperture=float(aperture))
    path = plot_path('pme_reconstruction', cfg.feedback,
                     stem=stem_D(cfg, mr, randomise, aperture, seed))
    ensure_parents(path)
    pl.save_figure(pl.spherise_figure(d), path)
    print(f"\n[D][plot] {path}")
    save_plot_data(path, dict(
        k_center=k, k_Ny=d.k_Ny, P_true=P_true,
        R_star=R_star, R_star_rand=R_star_rand, R_meas=R_meas,
        R_star_i=R_star_i, C_i=C_i,
        logM_cen=logM_cen, M_mean=np.asarray(data['M_mean'], float),
        counts=np.asarray(data['counts'], float),
        f_in=np.asarray(prod['f_in'], float),
        U_star=np.asarray(prod['U_star_gas'], float),
        U_star_rand=np.asarray(rand['U_star_gas'], float),
        randomise=randomise, aperture=float(aperture),
        seed=(-1 if seed is None else int(seed)), redshift=z,
        **{key: np.asarray(data[key]) for key in
           ('k_split', 'ngrid_3d', 'is_3d') if key in data},
        **mr.payload(),
    ), description=('Experiment D: R* with each halo interior randomised at '
                    'fixed radius, against the production R* and the bin-sum '
                    'model; their difference is the intra-halo matter-gas term'))
    return d


def main():
    ap = base_parser("Experiment D: split the profile residual into an "
                     "intra-halo term and a halo-to-halo one by randomising "
                     "the halo interiors.", validate=False)
    ap.add_argument('--randomise', default='shuffle',
                    choices=list(spherise.MODES),
                    help="how to randomise a halo's interior: 'shuffle' "
                         "spherises it")
    ap.add_argument('--aperture', type=float, default=1.0,
                    help="membership radius in units of r200m, matching the "
                         "caches to compare")
    ap.add_argument('--measure', action='store_true',
                    help="measure whatever cache is missing (one particle "
                         "pass per species) instead of failing")
    ap.add_argument('--recompute-randomised', dest='recompute_randomised',
                    action='store_true',
                    help="re-measure the randomised cache even if it exists. "
                         "It redraws the SAME seed; to compare two draws, run "
                         "again with a different --seed")
    ap.add_argument('--seed', default=None,
                    help="which draw of the randomised cache: a non-negative "
                         "int, or 'random' (drawn and printed; needs "
                         "--measure). Omitted, the legacy draw")
    args = ap.parse_args()
    cfg = PmxConfig.from_args(args)
    data = load_or_measure(cfg, cfg.nbins, cfg.logm_min, cfg.logm_max,
                           recompute=args.recompute,
                           recompute_3d=args.recompute_3d)
    run_experiment_D(cfg, data, randomise=args.randomise,
                     aperture=args.aperture, allow_compute=args.measure,
                     recompute=args.recompute_randomised,
                     seed=spherise.resolve_seed(args.seed))


if __name__ == '__main__':
    main()
