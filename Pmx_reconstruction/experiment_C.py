#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""==============================================================================
EXPERIMENT C  --  the measured halo profile against the model it replaces
==============================================================================
R carries exactly one modelled ingredient, u_m(k|M): every P_halo_gas in it is
measured, so the gas profile is whatever the simulation says. This script asks
what that one ingredient costs, by putting NFW + c(M,z) next to the stacked
profile pmxlib.u_bar measures off the particles:

    u_bar(k|M_i) = sum_{p in i} m_p j0(k r_p) / sum_{p in i} m_p

The quantity to read is the LOWER panel, u_bar/u_NFW - 1. It is not a
diagnostic of the profile in the abstract: it is exactly the factor by which
bin i's term of R is wrong, so a 20% gap at k = 3 for a bin carrying 5% of the
reconstruction is a 1% error on R there, and can be weighed directly against
the 5% band the other figures draw.

THREE CURVES PER MASS BIN
    solid    u_bar, measured
    dashed   u_nfw at c(M,z) from --concentration-model, which is what a
             --profile nfw run actually uses
    dotted   u_nfw at a concentration FITTED to u_bar

The third is what separates the two ways the model can be wrong. If the dotted
curve lands on the solid one, NFW is the right shape and only c(M,z) was off --
a one-parameter fix, and the fitted c(M) is a relation you could extrapolate
below the catalogue floor to improve U. If it does not, the profile is not NFW
in this simulation at these masses, and no concentration relation will save it.
Baryons are the reason to expect the second: c(M,z) here is calibrated on
dark-matter-only runs, while u_bar carries whatever feedback did to the gas.

WHY THE k RANGE EXTENDS PAST NYQUIST
    u_bar comes from a radial histogram, not a grid, so it is defined at any k
    and the box's Nyquist frequency does not bound it. It is drawn past k_Ny
    (marked) because the comparison is still meaningful there even though no
    spectrum in this pipeline is. It stops at 2 k_Ny rather than going on
    forever because the radial shells ARE finite: past there the shell width
    rings at the per-cent level and would read as structure in the profile.

    python -m Pmx_reconstruction.experiment_C --aperture 1
------------------------------------------------------------------------------
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')

from scipy.optimize import minimize_scalar

from utils.pipeline_paths import ensure_parents, plot_path
from utils.plot_data import save_plot_data

from Pmx_reconstruction.pmxlib.config import PmxConfig, base_parser
from Pmx_reconstruction.pmxlib import plotting as pl
from Pmx_reconstruction.pmxlib import u_bar as ub
from Pmx_reconstruction.pmxlib.nfw import conc_of, r200m_of_M, u_nfw

# Where the fit stops caring. Below this the profile has already fallen out of
# R and matching it would trade away the k range that still contributes.
U_FIT_MIN = 0.05
# Where u_m stops being ~1 and the profile starts mattering. The verdict below
# is summarised over this region only.
UM_MATTERS = 0.9
C_BOUNDS = (0.5, 40.0)
N_K = 240
# How far past Nyquist to draw. u_bar is grid-free so it is defined anywhere,
# but the radial shells are not infinitely fine and the binning error grows as
# k^2: measured against an exact NFW at the largest halo in the box, NX = 128
# shells give 4.6e-4 out to k_Ny, 3.5e-3 out to 2 k_Ny and 8.4e-3 out to
# 3 k_Ny. Two is where that is still well under anything being looked at here.
K_MAX_OVER_NY = 2.0


def fit_concentration(k, u_meas, M, rhobar_m, trunc=1.0, kmax=None):
    """The concentration whose NFW profile best matches u_meas.

    Least squares in log u over the k where u_meas is still above U_FIT_MIN,
    which weights every decade of k alike instead of letting the large scales
    -- where every profile gives 1 and there is nothing to fit -- dominate.
    Returns (c, n_points_used); c is NaN if there is nothing to fit.
    """
    k = np.asarray(k, dtype=float)
    u_meas = np.asarray(u_meas, dtype=float)
    sel = np.isfinite(u_meas) & (u_meas > U_FIT_MIN)
    if kmax is not None:
        sel &= k <= kmax
    if np.count_nonzero(sel) < 4:
        return np.nan, int(np.count_nonzero(sel))

    kk, target = k[sel], np.log(u_meas[sel])

    def cost(logc):
        u = u_nfw(kk, M, float(np.exp(logc)), rhobar_m, trunc=trunc)
        u = np.where(u > 1e-12, u, 1e-12)
        return float(np.sum((np.log(u) - target) ** 2))

    out = minimize_scalar(cost, bounds=np.log(C_BOUNDS), method='bounded')
    return float(np.exp(out.x)), int(kk.size)


def pick_bins(cache, n_show):
    """Indices of the bins to draw: n_show of them, evenly spaced in log M
    over the range u_bar is actually measured on."""
    # resolved_bins, not the stored bin_ok: the gate is re-derived on load so
    # it can be moved without re-measuring, and this has to agree with what
    # measured_range and u_bar_interp will do with the same cache.
    ok, _ = ub.resolved_bins(cache)
    idx = np.flatnonzero(ok)
    if idx.size == 0:
        raise SystemExit(
            "No bin in the u_bar cache is both occupied and past the "
            "resolution floor; there is nothing to compare. Check the "
            "per-bin table from\n"
            "      python -m Pmx_reconstruction.pmxlib.u_bar")
    if idx.size <= n_show:
        return idx
    return idx[np.unique(np.linspace(0, idx.size - 1, n_show).astype(int))]


def run_experiment_C(cfg, aperture=1.0, n_show=6, allow_compute=False):
    print("\n" + "=" * 70)
    print("EXPERIMENT C -- measured u_bar(k|M) against NFW + c(M,z)")
    print("=" * 70)

    cache = ub.load_or_measure_u_bar(cfg, aperture=aperture,
                                     allow_compute=allow_compute)
    if cache is None:
        raise SystemExit(
            f"No stacked-profile cache at aperture {aperture:g}:\n"
            f"  {ub.cache_path(cfg, aperture)}\nBuild it with\n"
            f"      python -m Pmx_reconstruction.pmxlib.u_bar "
            f"--aperture {aperture:g}")

    z = float(cache['redshift'])
    lo, hi = ub.measured_range(cache)
    print(f"[C] cache measured over logM {lo:.2f}..{hi:.2f}, "
          f"floor logM = {float(cache['logm_floor']):.2f}, "
          f"aperture = {float(cache['aperture']):g} r200m")

    # k is ours to choose: u_bar has no grid in it. Run past Nyquist, marked,
    # because the comparison stays meaningful where the spectra do not.
    k_f = 2.0 * np.pi / cfg.box
    k_Ny = np.pi * cfg.grid / cfg.box
    k = np.logspace(np.log10(k_f), np.log10(K_MAX_OVER_NY * k_Ny), N_K)

    idx = pick_bins(cache, n_show)
    M = np.asarray(cache['M_mean'], float)[idx]
    logM = np.log10(M)
    u_meas = ub.u_bar_of_k(cache, k)[idx]

    c_model = np.array([conc_of(cfg, m, z) for m in M])
    u_model = np.array([u_nfw(k, M[n], c_model[n], cfg.rhobar_m, trunc=aperture)
                        for n in range(M.size)])
    c_fit = np.full(M.size, np.nan)
    u_fit = np.zeros_like(u_model)
    for n in range(M.size):
        c_fit[n], npts = fit_concentration(k, u_meas[n], M[n], cfg.rhobar_m,
                                           trunc=aperture)
        u_fit[n] = (u_nfw(k, M[n], c_fit[n], cfg.rhobar_m, trunc=aperture)
                    if np.isfinite(c_fit[n]) else np.nan)

    # --- where the model stops being good enough --------------------------------
    def _k_at(ratio, level):
        """Lowest k where |ratio| first exceeds `level`, NaN if it never does."""
        bad = np.flatnonzero(np.isfinite(ratio) & (np.abs(ratio) > level))
        return float(k[bad[0]]) if bad.size else np.nan

    with np.errstate(divide='ignore', invalid='ignore'):
        ratio_model = u_meas / u_model - 1.0
        ratio_fit = u_meas / u_fit - 1.0

    r200 = np.array([r200m_of_M(m, cfg.rhobar_m) for m in M])
    print(f"\n  u_bar(k_f) must be 1: max departure "
          f"{np.nanmax(np.abs(u_meas[:, 0] - 1.0)):.2e}")
    print("\n  logM    n_halo   c(M,z)   c_fit  c_fit/c   k(5%)   k(20%)   "
          "at k=1    at k=3   at k_Ny")
    for n, j in enumerate(idx):
        def _at(kk):
            return float(np.interp(kk, k, ratio_model[n]))
        print(f"  {logM[n]:5.2f}  {int(cache['counts'][j]):7d}  "
              f"{c_model[n]:6.2f}  {c_fit[n]:6.2f}  {c_fit[n] / c_model[n]:6.2f}  "
              f"{_k_at(ratio_model[n], 0.05):6.2f}  "
              f"{_k_at(ratio_model[n], 0.20):6.2f}  "
              f"{_at(1.0):+7.3f}  {_at(3.0):+7.3f}  {_at(k_Ny):+7.3f}")
    print(f"\n  k(5%) / k(20%): where |u_bar/u_NFW - 1| first passes that level. "
          f"k_Ny = {k_Ny:.2f} h/cMpc.")
    print("  The last three columns ARE the fractional error the profile puts "
          "into that bin's term of R.")

    # Summarised only where the profile is DOING something. Averaged over the
    # whole k range these numbers are both ~0, because u -> 1 on large scales
    # whatever the profile is and that region outnumbers the rest; the verdict
    # below would then be decided by the part of the axis with no information
    # in it.
    where = (u_meas < UM_MATTERS) & (k[None, :] <= k_Ny)
    if not where.any():
        print(f"\n  no bin drops below u_m = {UM_MATTERS:g} within k_Ny = "
              f"{k_Ny:.2f}: the profile is ~1 everywhere the box resolves, so "
              f"the model costs nothing here whatever it says.")
    else:
        med_mod = float(np.nanmedian(np.abs(ratio_model[where])))
        med_fit = float(np.nanmedian(np.abs(ratio_fit[where])))
        n_pt = int(np.count_nonzero(where))
        print(f"\n  median |residual| over the {n_pt} (bin, k) points with "
              f"u_m < {UM_MATTERS:g} and k < k_Ny:"
              f"\n      {med_mod:.4f} at c(M,z),  {med_fit:.4f} at fitted c")
        _verdict(med_mod, med_fit)
    return _finish(cfg, cache, dict(
        k=k, k_Ny=k_Ny, logM=logM, u_meas=u_meas, u_model=u_model,
        u_fit=u_fit, c_model=c_model, c_fit=c_fit, r200m=r200,
        aperture=float(aperture), idx=idx, M=M, z=z))


def _verdict(med_mod, med_fit):
    """Which of the two ways the model can be wrong this run is looking at."""
    if med_fit < 0.5 * med_mod:
        print("  -> most of the gap is the CONCENTRATION, not the NFW shape: a "
              "c(M,z) fitted here would recover it, and could be extrapolated "
              "below the catalogue floor to improve U as well.")
    else:
        print("  -> refitting c does NOT close the gap: the profile is not NFW "
              "at these masses in this simulation, and no concentration "
              "relation will fix R. Use --profile measured.")


def _finish(cfg, cache, a):
    """Draw, save and return; split out only to keep run_experiment_C readable."""
    k, k_Ny, idx, M = a['k'], a['k_Ny'], a['idx'], a['M']
    aperture = a['aperture']
    d = pl.ProfileData(k=k, k_Ny=k_Ny, logM=a['logM'], u_meas=a['u_meas'],
                       u_model=a['u_model'], u_fit=a['u_fit'],
                       c_model=a['c_model'], c_fit=a['c_fit'],
                       r200m=a['r200m'], aperture=aperture)
    models = ''
    if cfg.concentration_source != PmxConfig.concentration_source:
        models += f'_conc-{cfg.concentration_source}'
    if cfg.colossus_conc_model != PmxConfig.colossus_conc_model:
        models += f'_cm-{cfg.colossus_conc_model}'
    ap = '' if float(aperture) == 1.0 else f'_ap{float(aperture):g}'
    stem = (f'expC_profile_{cfg.mass_def}_nbu{int(cache["nbins"])}'
            f'_logMu{float(cache["logm_min"]):g}-{float(cache["logm_max"]):g}'
            f'_nshow{idx.size}{ap}{models}')
    path = plot_path('pme_reconstruction', cfg.feedback, stem=stem)
    ensure_parents(path)
    pl.save_figure(pl.profile_figure(d), path)
    print(f"\n[C][plot] {path}")

    save_plot_data(path, dict(
        k=k, k_Ny=k_Ny, logM=a['logM'], M_mean=M, bin_index=idx,
        u_bar=a['u_meas'], u_nfw=a['u_model'], u_nfw_fit=a['u_fit'],
        c_model=a['c_model'], c_fit=a['c_fit'], r200m=a['r200m'],
        counts=np.asarray(cache['counts'], float)[idx],
        npart_mean=np.asarray(cache['npart_mean'], float)[idx],
        aperture=float(aperture), redshift=a['z'],
        logm_floor=float(cache['logm_floor']),
        concentration_source=cfg.concentration_source,
        colossus_conc_model=cfg.colossus_conc_model,
    ), description=('Experiment C: the stacked matter profile u_bar(k|M) '
                    'against the NFW + c(M,z) model R uses in its place'))
    return d


def main():
    ap = base_parser("Experiment C: compare the measured stacked profile "
                     "u_bar(k|M) with NFW + c(M,z).", validate=False)
    ap.add_argument('--aperture', type=float, default=1.0,
                    help="which u_bar cache to read, in units of r200m")
    ap.add_argument('--n-show', dest='n_show', type=int, default=6,
                    help="how many mass bins to draw, spread evenly in log M "
                         "over the measured range")
    ap.add_argument('--measure', action='store_true',
                    help="measure the u_bar cache if it is missing (one "
                         "particle pass per species) instead of failing")
    args = ap.parse_args()
    cfg = PmxConfig.from_args(args)
    run_experiment_C(cfg, aperture=args.aperture, n_show=args.n_show,
                     allow_compute=args.measure)


if __name__ == '__main__':
    main()
