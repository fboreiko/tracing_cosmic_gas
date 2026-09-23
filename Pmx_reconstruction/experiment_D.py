#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""==============================================================================
EXPERIMENT D  --  is the residual intra-halo? Spherise the halos and see
==============================================================================
Experiment C showed that measuring the radial profile does not close R: with
u_bar in place of NFW the bin sum still falls short of the truth, and by MORE
than it did with NFW, because NFW's over-concentration had been propping it up.
So what is missing is not radial. This experiment asks what it is instead.

THE IDENTITY BEING TESTED
    With the measured profile and the measured weights, the shortfall is not an
    approximation but exactly a covariance across the halos of a bin,

        R*_i - R_i = (N_i / M_tot) Cov_{j in i}( M_j^a u_j(k), c_j(k) ),

    between where halo j's mass sits (u_j, its own clumpy non-spherical
    profile) and how halo j correlates with the gas (c_j). It is exact at each
    k vector; on the binned spectra the average runs over the directions in
    the k shell as well, which is where halo shape aligned with the
    surrounding field enters. Two very different things live in there:

      intra-halo    a halo's clumps hold gas, so the matter in a clump and the
                    gas in the same clump correlate on the clump's own scale.
                    No radial profile can carry this, and no rescaling of the
                    U template can either, because the material belongs to a
                    RESOLVED halo.
      halo-to-halo  within a bin, more massive or more strongly biased halos
                    have both a larger M u and a larger c. This survives even
                    if halos are perfectly smooth spheres, and it is the
                    k -> 0 limit of the same covariance.

    Randomising each halo's interior independently of everything else replaces
    u_j by its own spherised version and leaves c_j alone, so it removes the
    first and keeps the second. Hence

        R*(true) - R*(randomised)   =   the intra-halo term, MEASURED.

    If the substructure reading is right, that difference accounts for
    essentially all of R - R* at k >~ 2 and R*(randomised) lands on the model.
    If it does not, the reading is wrong and the residual is halo-to-halo
    covariance -- which is a bin-width problem, not a physics one.

WHAT THIS DOES AND DOES NOT CONTROL FOR
    Randomising the matter destroys the halo's alignment with its environment
    at the same time as its internal clump-gas alignment, so a surviving
    difference is "intra-halo" in the broad sense: clumps, triaxiality and
    filament alignment together. Separating THOSE needs a run in which each
    halo's gas is rotated with its matter and P_halo_gas is re-measured against
    the rotated gas field, which is a second measurement and not a mode here.
    At k >~ 3 the scales in question are well inside r200m, where an alignment
    with the large-scale field is not a plausible carrier.

    'shuffle' and 'rotate' have the SAME expectation (see pmxlib.spherise);
    'rotate' is the noisier of the two and is a cross-check, not a control.

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
import numpy as np
import matplotlib
matplotlib.use('Agg')

from utils.pipeline_paths import ensure_parents, plot_path
from utils.plot_data import save_plot_data

from Pmx_reconstruction.pmxlib.bundle import load_or_measure
from Pmx_reconstruction.pmxlib.config import MassRange, PmxConfig, base_parser
from Pmx_reconstruction.pmxlib.reconstruction import Profile
from Pmx_reconstruction.pmxlib import plotting as pl
from Pmx_reconstruction.pmxlib import rstar_ustar as ru
from Pmx_reconstruction.pmxlib import spherise

K_TABLE = (0.1, 0.3, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0)
# Where "the residual" is summarised. Below this the covariance is the
# halo-to-halo one by construction (u -> 1), and the interesting claim is
# entirely about the small scales.
K_DIP = (2.0, 8.0)
# How closely the two caches must agree on things the randomisation cannot
# touch. Not a tolerance: the labels are identical, so these are the same
# floating-point sums in the same order.
EXACT = 1e-12


def _at(k, y, kk):
    """y at the k bin nearest kk."""
    return float(np.asarray(y, float)[int(np.argmin(np.abs(k - kk)))])


def check_caches_agree(prod, rand):
    """Everything the randomisation leaves alone must be bit-identical.

    The labels do not change, so f_part, the counts, the mean masses and the
    mass moments are the same sums over the same particles. If any of them
    moves, the two caches are not the same measurement and nothing downstream
    means what it says -- so this fails loudly rather than warning.
    """
    for key in ('f_part', 'f_i', 'counts', 'M_mean', 'logM_edges',
                'm2_bin_gas', 'mass_bin_dm', 'mass_bin_gas'):
        if key not in prod or key not in rand:
            continue
        a = np.asarray(prod[key], float)
        b = np.asarray(rand[key], float)
        if a.shape != b.shape or not np.allclose(a, b, rtol=EXACT, atol=0.0):
            raise SystemExit(
                f"[D] the production and randomised caches disagree on "
                f"'{key}', which the randomisation does not touch. They were "
                f"not measured from the same catalogue, aperture or mass "
                f"binning; re-measure the randomised one with the same "
                f"--nbins / --logm-min / --logm-max / --aperture.")
    print("[D] the two caches agree exactly on f_part, counts, M_mean and "
          "the mass moments")


def check_ustar(prod, rand):
    """U* is the same measurement by two routes: a residual and a painting.

    The production cache takes U* as delta_m minus the summed per-bin fields;
    the randomised one paints the unassigned particles directly. Those
    particles did not move, so the two must agree -- to painting precision
    rather than exactly, since they are different sums over different grids.
    """
    a = np.asarray(prod['U_star_gas'], float)
    b = np.asarray(rand['U_star_gas'], float)
    with np.errstate(divide='ignore', invalid='ignore'):
        dev = np.abs(b / a - 1.0)
    worst = float(np.nanmax(dev[np.isfinite(dev)])) if np.any(np.isfinite(dev)) else np.nan
    print(f"[D] U* by the two routes (residual vs painted): max |ratio - 1| = "
          f"{worst:.2e}")
    if np.isfinite(worst) and worst > 1e-3:
        print("[D] WARNING: U* should be the SAME matter painted twice. A gap "
              "this size means the membership differed between the two runs "
              "-- check that both used the same --aperture and catalogue.")
    return worst


def check_low_k(R_star, R_star_rand, k, kmax=0.15):
    """R* must be untouched by the randomisation as k -> 0.

    u_j -> 1 whatever the angles, so on scales far larger than a halo the
    spherised field and the true one are the same field. This is the physical
    check that the randomisation moved particles WITHIN their hosts and not
    somewhere else; it needs no reference measurement.
    """
    sel = k < kmax
    if np.count_nonzero(sel) < 3:
        sel = np.zeros_like(k, dtype=bool)
        sel[:5] = True
    with np.errstate(divide='ignore', invalid='ignore'):
        dev = R_star_rand[sel] / R_star[sel] - 1.0
    worst = float(np.nanmax(np.abs(dev)))
    print(f"[D] R*(randomised)/R*(true) - 1 over k < {kmax:g}: "
          f"mean {np.nanmean(dev):+.2e}, max |.| {worst:.2e} "
          f"(must be ~0: u_j -> 1 there)")
    if worst > 0.02:
        print("[D] WARNING: the two disagree on the largest scales, where "
              "they cannot. Particles left their hosts, or the halo centres "
              "used for the randomisation are not the catalogue's.")
    return worst


def randomisation_noise(cache, data):
    """sigma(k) on R*(randomised) from the finite number of member particles.

    The random directions are uncorrelated with the gas, so they bias nothing
    and add variance only: the moved matter contributes a white field of power
    P_n = L^2 sum_p m_p^2 / M_tot^2, and crossing a white field with the gas
    gives sqrt(P_n P_gg / N_modes) per k bin.

    P_n is built out of the MEASURED P^shot_gg rather than from an analytic
    L^2 sum m^2, so the TSC window and the k binning are carried along with it
    exactly as they are in the self-pair subtraction.

    Returns None on a cache written before m2_bin_mat existed.
    """
    if 'm2_bin_mat' not in cache:
        return None
    f_g = float(cache['f_g'])
    m2_in = float(np.sum(np.asarray(cache['m2_bin_mat'], float)))
    P_shot = np.asarray(cache['P_shot_gg'], float)
    P_n = f_g ** 2 * (m2_in / float(cache['m2_tot_gas'])) * P_shot

    P_gg = np.asarray(data.get('P_gas_gas_raw', data['P_gas_gas']), float)
    nm = _mode_count(cache)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.sqrt(np.abs(P_n * P_gg) / np.maximum(nm, 1.0))


def report_selfpair(rand, R_star_rand, k):
    """How much of R*(randomised) was the smeared self-pair, and where.

    The intra-halo term is a difference of two spectra whose self-pair
    corrections differ, so its credibility is bounded by the smaller of the
    two. Printed next to the answer for the same reason self_pairs prints its
    own fractions: past the point where the correction is most of the raw
    number, what is left is a difference of two large numbers.
    """
    if 'R_star_gas_selfpair' not in rand:
        print("[D] WARNING: the randomised cache carries no smeared self-pair "
              "correction, so R*(randomised) is biased HIGH and the "
              "intra-halo term below is biased LOW -- badly so wherever the "
              "gas spectrum is shot-dominated. Re-measure it with "
              "--recompute-randomised.")
        return None
    term = np.asarray(rand['R_star_gas_selfpair'], float).sum(axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        frac = term / (R_star_rand + term)
    ks = [kk for kk in K_TABLE if kk <= k[-1]]
    print("\n  [shot] the smeared self-pair removed from R*(randomised), as a "
          "fraction of the raw value:")
    print("  k =              " + "".join(f"{kk:>9.2f}" for kk in ks))
    print("  m^2 j0(kr)^2    " + "".join(f"{_at(k, frac, kk):9.4f}"
                                         for kk in ks))
    worst = float(np.nanmax(np.abs(frac[np.isfinite(frac)]))) if np.any(
        np.isfinite(frac)) else np.nan
    if np.isfinite(worst) and worst > 0.3:
        print(f"  NOTE: up to {worst:.0%}. Read the intra-halo term below "
              f"only where this is small; past that it is a difference of two "
              f"large numbers.")
    return frac


def _mode_count(cache):
    """Modes per k bin, taking the 3-D count wherever the splice did."""
    nm = np.asarray(cache.get('nmodes', np.ones(1)), float)
    if 'is_3d' not in cache:
        return nm
    take = np.asarray(cache['is_3d'], bool)
    for key in ('nmodes_eff_3d', 'nmodes_3d'):
        if key in cache:
            return np.where(take, np.asarray(cache[key], float), nm)
    return nm


def model_R(data, tot, aperture, profile):
    """R = sum_i f_part_i u_m(k|M_i) P_halo_gas(k|M_i), over the occupied bins.

    Deliberately NOT compute_R: this experiment always weights by the assigned
    mass, whatever cfg.profile_source says, because that is the pairing the
    covariance identity is written in. Passing the catalogue's n_i M_i here
    would put the 0.255-against-0.249 mass gap into the residual and it would
    read as intra-halo structure.
    """
    k = np.asarray(data['k_center'], float)
    M_i = np.asarray(data['M_mean'], float)
    occ = np.asarray(data['counts'], float) > 0
    w = np.asarray(tot['f_part'], float)
    u = np.zeros((M_i.size, k.size))
    idx = np.flatnonzero(occ)
    u[idx] = profile.u(k, M_i[idx], trunc=aperture)
    return np.sum(w[:, None] * u * np.asarray(data['P_halo_gas'], float),
                  axis=0)


def run_experiment_D(cfg, data, randomise='shuffle', aperture=1.0,
                     allow_compute=False, recompute=False):
    print("\n" + "=" * 70)
    print(f"EXPERIMENT D -- R* with the halo interiors randomised "
          f"('{randomise}'), x = {aperture:g}")
    print("=" * 70)

    randomise = spherise.check_mode(randomise)
    if randomise is None:
        raise SystemExit("experiment D needs a randomisation mode")

    k = np.asarray(data['k_center'], float)
    k_Ny = float(data['k_Nyquist'])
    k_split = float(data['k_split']) if 'k_split' in data else None
    z = float(data['redshift'])
    P_true = np.asarray(data['P_matter_gas'], float)
    logM_cen = np.asarray(data['logM_cen'], float)
    mr = MassRange.from_config(cfg, data)

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
                                          randomise=randomise)
    if rand is None:
        raise SystemExit(
            f"No '{randomise}' R*/U* cache at aperture {aperture:g}:\n"
            f"  {ru.cache_path(cfg, aperture=aperture, randomise=randomise)}\n"
            f"Build it with\n"
            f"      python -m Pmx_reconstruction.pmxlib.rstar_ustar "
            f"--aperture {aperture:g} --randomise {randomise}\n"
            f"or re-run this with --measure.")

    check_caches_agree(prod, rand)
    check_ustar(prod, rand)

    tot = ru.measured_partition(cfg, prod, data, mr=mr)
    tot_r = ru.measured_partition(cfg, rand, data, mr=mr)
    R_star, R_star_rand = tot['R_star'], tot_r['R_star']
    check_low_k(R_star, R_star_rand, k)
    report_selfpair(rand, R_star_rand, k)

    # --- the model the identity is written against ----------------------------
    # The identity R*_i - R_i = Cov(M u, c) holds with the MEASURED radial
    # profile; with NFW the residual also carries the profile error of
    # Experiment C and the split below is not clean. R_nfw is drawn anyway,
    # because it is what the production reconstruction uses.
    R_meas = measured_profile_R(cfg, data, tot, z, aperture, allow_compute)
    R_nfw = model_R(data, tot, aperture, Profile(cfg, z, aperture=aperture))
    if R_meas is None:
        R_meas, R_nfw = R_nfw, None

    sigma = randomisation_noise(rand, data)

    # --- the split ------------------------------------------------------------
    total = R_meas - R_star            # what experiment B calls profile error
    halo = R_meas - R_star_rand        # what survives spherisation
    intra = R_star - R_star_rand       # the term with no representation
    C_i = (np.asarray(tot['R_star_i'], float)
           - np.asarray(tot_r['R_star_i'], float))

    _report(k, P_true, R_star, R_star_rand, R_meas, R_nfw, total, halo, intra,
            sigma, randomise)
    _report_bins(k, logM_cen, C_i, np.asarray(tot['R_star_i'], float), mr)

    d = pl.SpheriseData(
        k=k, k_Ny=k_Ny, k_split=k_split, P_true=P_true, R_star=R_star,
        R_star_rand=R_star_rand, R_meas=R_meas, R_nfw=R_nfw, sigma=sigma,
        logM=logM_cen, C_i=C_i, R_star_i=np.asarray(tot['R_star_i'], float),
        mode=randomise, aperture=float(aperture))
    return _finish(cfg, data, d, mr, prod, rand, randomise, aperture)


def measured_profile_R(cfg, data, tot, z, aperture, allow_compute):
    """R built on the stacked u_bar, or None if that stack is not available.

    Returning None rather than failing: the headline of this experiment is
    R*(true) - R*(randomised), which needs no model at all. Losing the
    measured profile costs the clean split, not the measurement, so it is
    worth saying so and carrying on with NFW rather than refusing to run.
    """
    from dataclasses import replace
    try:
        prof = Profile.from_config(replace(cfg, profile_source='measured'), z,
                                   aperture=aperture,
                                   allow_compute=allow_compute)
    except SystemExit as exc:
        print(f"\n[D] no stacked-profile cache at aperture {aperture:g}, so "
              f"the split below is against NFW and carries Experiment C's "
              f"profile error on top of the covariance it is meant to "
              f"isolate. Build the stack with\n"
              f"      python -m Pmx_reconstruction.pmxlib.u_bar "
              f"--aperture {aperture:g}\n  ({exc})")
        return None
    return model_R(data, tot, aperture, prof)


def _report(k, P, R_star, R_rand, R_meas, R_nfw, total, halo, intra, sigma,
            mode):
    ks = [kk for kk in K_TABLE if kk <= k[-1]]
    head = "".join(f"{kk:>9.2f}" for kk in ks)
    print(f"\n  k =              {head}")

    def row(name, y):
        print(f"  {name:16s}" + "".join(f"{_at(k, y, kk):9.4f}" for kk in ks))

    with np.errstate(divide='ignore', invalid='ignore'):
        row('R/R* - 1        ', R_meas / R_star - 1.0)
        if R_nfw is not None:
            row('R/R* - 1  (NFW) ', R_nfw / R_star - 1.0)
        row('R*rand/R* - 1   ', R_rand / R_star - 1.0)
        print()
        # total = halo + (-intra), exactly. The three are printed with that
        # sign convention so the columns add up on the page.
        row('total /P        ', total / P)
        row('halo-to-halo /P ', halo / P)
        row('-intra-halo /P  ', -intra / P)
        if sigma is not None:
            row('sigma /P        ', sigma / P)
        print()
        share_k = np.where(total != 0.0, -intra / total, np.nan)
        row('intra / total   ', share_k)

    lo, hi = K_DIP
    sel = (k >= lo) & (k <= hi)
    if not np.any(sel):
        return
    with np.errstate(divide='ignore', invalid='ignore'):
        share = float(np.nanmedian(share_k[sel]))
        left = float(np.nanmedian((halo / P)[sel]))
        gap = float(np.nanmedian((total / P)[sel]))
    print(f"\n  Over k = {lo:g}--{hi:g} h/cMpc, median:")
    print(f"      R - R*             = {gap:+.4f} of P^me   (the residual)")
    print(f"      of which intra-halo  {share:.1%}")
    print(f"      left over          = {left:+.4f} of P^me   (halo-to-halo)")
    _verdict(share, left, gap)


def _verdict(share, left, gap):
    """Say which way the experiment came out, in the terms it was set up in."""
    if not np.isfinite(share):
        print("  -> nothing to read: the residual is zero over this range.")
    elif share > 0.7 and abs(left) < 0.3 * abs(gap):
        print("  -> the residual IS intra-halo. Spherising the halos removes "
              "most of it and what is left is small, so no radial profile and "
              "no rescaling of the U template can close R: the model needs a "
              "term of its own for the correlation between a halo's clumps "
              "and the gas in them.")
    elif share < 0.3:
        print("  -> the residual is NOT intra-halo. It survives spherising "
              "the halos, so it is the halo-to-halo covariance of M u with "
              "the gas cross INSIDE a bin -- a bin-width and mass-scatter "
              "problem. Narrow the bins (--nbins) and see it shrink before "
              "modelling anything.")
    else:
        print("  -> mixed: neither term dominates. The bin-width null test "
              "(--nbins 60 against 30) is the one that separates them, since "
              "the halo-to-halo part shrinks with the bin width and the "
              "intra-halo part does not.")


def _report_bins(k, logM, C_i, R_star_i, mr):
    """Where in halo mass the intra-halo term sits, at three scales."""
    ks = [kk for kk in (1.0, 3.0, 5.0) if kk <= k[-1]]
    idx = [int(np.argmin(np.abs(k - kk))) for kk in ks]
    occ = np.any(R_star_i != 0.0, axis=1) & mr.resolved
    print("\n  per bin, intra-halo / R*_i:")
    print("    logM " + "".join(f"{k[j]:>12.2f}" for j in idx))
    for i in np.flatnonzero(occ):
        with np.errstate(divide='ignore', invalid='ignore'):
            vals = [C_i[i, j] / R_star_i[i, j] if R_star_i[i, j] != 0 else np.nan
                    for j in idx]
        print(f"   {logM[i]:5.2f} " + "".join(f"{v:12.4f}" for v in vals))
    print("    A substructure term grows with host mass and with k. A "
          "bin-width artefact does neither.")


def _finish(cfg, data, d, mr, prod, rand, randomise, aperture):
    ap = '' if float(aperture) == 1.0 else f'_ap{float(aperture):g}'
    models = ''
    if cfg.colossus_conc_model != PmxConfig.colossus_conc_model:
        models += f'_cm-{cfg.colossus_conc_model}'
    models += cfg.clump_tag
    if cfg.ngrid_3d:
        models += f'_3d{cfg.ngrid_3d}k{cfg.k_split:g}'
    stem = (f'expD_spherise_gas_{cfg.mass_def}_nb{cfg.nbins}'
            f'_logMmin{cfg.logm_min:.2f}_logMmax{cfg.logm_max:.2f}'
            f'{"" if mr.is_default else mr.tag()}'
            f'_{randomise}{ap}{models}')
    path = plot_path('pme_reconstruction', cfg.feedback, stem=stem)
    ensure_parents(path)
    pl.save_figure(pl.spherise_figure(d), path)
    print(f"\n[D][plot] {path}")

    save_plot_data(path, dict(
        k_center=d.k, k_Ny=d.k_Ny, P_true=d.P_true,
        R_star=d.R_star, R_star_rand=d.R_star_rand,
        R_meas=d.R_meas, R_nfw=d.R_nfw,
        R_star_i=d.R_star_i, C_i=d.C_i,
        sigma=(d.sigma if d.sigma is not None else np.array([])),
        logM_cen=d.logM, M_mean=np.asarray(data['M_mean'], float),
        counts=np.asarray(data['counts'], float),
        f_part=np.asarray(prod['f_part'], float),
        U_star=np.asarray(prod['U_star_gas'], float),
        U_star_rand=np.asarray(rand['U_star_gas'], float),
        randomise=randomise, aperture=float(aperture),
        redshift=float(data['redshift']),
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
                    help="how to randomise a halo's interior. 'shuffle' "
                         "spherises it, 'rotate' turns it rigidly; both have "
                         "the same expectation and 'shuffle' has much the "
                         "smaller variance, so 'rotate' is a cross-check")
    ap.add_argument('--aperture', type=float, default=1.0,
                    help="membership radius in units of r200m, matching the "
                         "caches to compare")
    ap.add_argument('--measure', action='store_true',
                    help="measure whatever cache is missing (one particle "
                         "pass per species) instead of failing")
    ap.add_argument('--recompute-randomised', dest='recompute_randomised',
                    action='store_true',
                    help="re-measure the randomised cache even if it exists, "
                         "e.g. to see how far two draws differ")
    args = ap.parse_args()
    cfg = PmxConfig.from_args(args)
    data = load_or_measure(cfg, cfg.nbins, cfg.logm_min, cfg.logm_max,
                           recompute=args.recompute,
                           recompute_3d=args.recompute_3d)
    run_experiment_D(cfg, data, randomise=args.randomise,
                     aperture=args.aperture, allow_compute=args.measure,
                     recompute=args.recompute_randomised)


if __name__ == '__main__':
    main()
