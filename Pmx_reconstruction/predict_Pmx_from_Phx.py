#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Reconstruct a matter-tracer spectrum P_matter_x(k) from halo-tracer cross-spectra
P_halo_x(k; M_i) measured in halo-mass bins, using only the assumption that all
matter lives in the resolved halos.

The tracer x is a run-time switch: x = gas gives P(m x gas), x = dm gives
P(m x dm), x = matter gives the matter AUTO spectrum P(m x m).

    The construction
    ----------------
    If every particle of matter belongs to some halo, the matter density field
    is the halo field, mass-weighted and smeared by each halo's internal matter
    profile u_m(k|M):

        delta_m(k) = (1/rhobar_m) * sum_j M_j u_m(k|M_j) e^{i k.x_j}

    Grouping halos into mass bins i (centres M_i, comoving number density n_i)
    and cross-correlating with ANY second field x -- the derivation never
    touches x, which is why one script covers all three tracers:

     ┌──────────────────────────────────────────────────────────────────────┐
     │ P_matter_x(k) = (1/rhobar_m) sum_i n_i M_i u_m(k|M_i) P_halo_x(k;M_i)│
     └──────────────────────────────────────────────────────────────────────┘

    Everything on the right is either measured (P_halo_x, the halo counts) or
    known (rhobar_m from cosmology, u_m from NFW + a c(M,z) relation). No
    *normalized* halo mass function is needed: the weight n_i M_i =
    (count_i / L^3) * M_i comes straight from the raw histogram of halo counts
    and the box volume.

    P_halo_x(k; M_i) and the truths are measured directly from FLAMINGO, reusing
    the field/FFT/binning machinery of HATF_reconstruction.py via utils.*.

NAMING CONVENTION
    Every spectrum in this file is named for the pair of fields it correlates,

        P_<first field>_<second field>,

    with the fields drawn from {halo, matter, dm, gas}, and 'x' standing for
    whichever tracer --target has selected. So P_halo_gas and P_halo_dm are the
    per-bin measured halo cross spectra of shape (nbins, nk); P_matter_gas,
    P_matter_dm, P_matter_matter, P_dm_gas and P_dm_dm are the measured truths
    of shape (nk,); and P_halo_x / P_matter_x_true / P_dm_x are the aliases the
    current target resolves to. 

ONE SWITCH
    TARGET_MODE ('matter_gas' | 'matter_dm' | 'matter_matter')   --target
        Which spectrum is being reconstructed, i.e. what plays the role of the
        second field x. This selects which set of halo cross-spectra is fed to
        the reconstruction and which truth it is compared against, both through
        the TRACER_INFO table; there is no per-tracer branching anywhere else.

    The FIRST field is not a choice. The reconstruction's weights are built from
    TOTAL halo mass and TOTAL matter density, so P(m x x) with
    delta_m = f_c delta_dm + f_g delta_gas is the only thing the right-hand side
    targets, and it is always the truth. P(dm x x) is still measured and drawn
    on both panels, because the gap between the two is the DM-vs-matter
    definitional mismatch and it sets the floor on achievable agreement -- but
    it is a reference curve, never a denominator.

    All halo cross-spectra and all measured truths go into a single pass and one
    bundle, so flipping --target afterwards costs nothing and needs no
    re-measurement.

WHAT IT DOES
    1. Loads (or computes and caches) the FLAMINGO 2-D projected gas and DM
       delta fields, using exactly the same paths as HATF, so any field HATF
       has already written is reused rather than recomputed.
        1b. Builds the matter field delta_m = f_c delta_dm + f_g delta_gas, with
            the fractions taken from the actual particle mass sums. The
            reconstruction's weights use TOTAL halo mass and TOTAL matter
            density, so P(m x x) is what it actually targets; P(dm x x) alone
            differs at high k, where feedback makes gas smoother than DM.
    2. Splits the halo catalogue into NBINS log-spaced bins in halo mass over
       [10^LOGM_MIN, 10^LOGM_MAX] Msun/h, and measures BOTH P_halo_gas(k; M_i)
       and P_halo_dm(k; M_i) for each bin, plus the truths. The halo field and
       its FFT are built once per bin and crossed against both tracers, so
       covering the second tracer costs one extra binning pass, not a second
       run; the third tracer costs nothing at all, since P_halo_matter is a
       linear combination of the other two.
    3. Caches all of that in ONE npz bundle. On every later run the bundle is
       loaded and steps 1-2 are skipped entirely.
    3b. Removes the SELF-PAIR (shot-noise) part of every measured spectrum that
       has one. delta_m = f_c delta_dm + f_g delta_gas is built out of the very
       particles that also make up the tracer field, so P(m x gas) contains
       f_g P^shot_gg, P(m x dm) contains f_c P^shot_dd, and the matter auto
       contains f_c^2 P^shot_dd + f_g^2 P^shot_gg. None of that is clustering,
       and none of it is anything the reconstruction can produce, so leaving it
       in charges the halo-model identity for a discreteness floor. The two
       self-pair spectra P^shot_dd and P^shot_gg are measured ONCE, in this
       pipeline's own convention (TSC window, L^2 factor, k binning) by painting
       each species at uniformly random positions with its real masses, and are
       cached in their own npz keyed on the grid rather than the mass binning.
       The bundle then stores all THREE variants of every affected spectrum
       side by side -- raw, self-pair term, corrected -- so the correction is a
       stored quantity, not one re-derived on every run; --no-shot-subtract
       chooses which variant a given run works with. See section 3b. This
       mirrors, and is deliberately numerically identical to, what
       measure_u_tilde.py does to R_star, U_star and its truths; the two
       scripts do not import each other.
    4. Runs the reconstruction and compares it to the measured truth.

A NOTE ON PROJECTED (2-D) SPECTRA
    The repo works with fields projected along one axis. For a full-box
    projection the 2-D field is the column average of the 3-D one, so

        delta_2D(k_perp) = delta_3D(k_perp, k_z = 0),

    i.e. the 2-D modes ARE a subset of the 3-D modes (those with k_z = 0). The
    halo-model identity above holds mode by mode, so it holds unchanged for the
    projected fields with u_m evaluated at |k| = k_perp, and for an auto
    spectrum of those modes just as much as for a cross spectrum of them. The
    only requirement is that P_halo_x and P_matter_x share a normalisation
    convention, which they do here (both carry the same box^2 factor as in
    HATF). No Limber approximation and no extra projection kernel is involved.

DEPENDENCIES
    numpy, scipy, matplotlib, plus:
        camb      linear P(k), computed once and cached to DATA_ROOT as a
                  two-column table (the run takes ~20 s, every later run reads
                  the file)
        colossus  Tinker+08 dn/dM, Tinker+10 b(M), and optionally c(M,z),
                  evaluated ON THE CAMB SPECTRUM rather than on colossus's own
                  Eisenstein & Hu default -- see _ColossusBackend
        astropy   background densities, i.e. rho_crit(0)
    All are hard requirements: they are imported at module scope and there is no
    fallback path. Cosmological parameters are read from utils.sim_params;
    nothing is hard-coded here.

        pip install camb colossus astropy

USAGE
    Run from the repo root (it imports `utils.*` the same way HATF does):

        python -m Pmx_reconstruction.predict_Pmx_from_Phx                            # gas, from cache
        python -m Pmx_reconstruction.predict_Pmx_from_Phx --target matter_dm         # dm, same bundle
        python -m Pmx_reconstruction.predict_Pmx_from_Phx --target matter_matter     # the auto spectrum
        python -m Pmx_reconstruction.predict_Pmx_from_Phx --recompute                # re-measure
        python -m Pmx_reconstruction.predict_Pmx_from_Phx --nbins 40 --logm-min 11.5
        python -m Pmx_reconstruction.predict_Pmx_from_Phx --no-shot-subtract         # keep self-pairs
        python -m Pmx_reconstruction.predict_Pmx_from_Phx --recompute-shot           # re-measure them

    EXPERIMENT A -- extrapolating P_halo_x(k|M) below M_min (section 6):

        # the validation test: hide everything below 10^12, then score every
        # extrapolation against the exact contribution of the hidden bins
        python -m Pmx_reconstruction.experiment_A --split-logm 12.0 \
               --extrap flat simhc bias halomodel --hmf catalog

        # same, but with the mass weight from Tinker+08 instead of the
        # catalogue: now the HMF is on trial as well as the extrapolation
        python -m Pmx_reconstruction.experiment_A --split-logm 12.0 --hmf tinker

        # production: correct below the catalogue limit, no exact target exists
        python -m Pmx_reconstruction.experiment_A

        # published c(M,z) instead of the built-in power law, and the
        # EH98 fitting function instead of the CAMB spectrum, for comparison
        python -m Pmx_reconstruction.experiment_A --concentration colossus
        python -m Pmx_reconstruction.experiment_A --power-spectrum eisenstein98

    Any change to the mass binning, mass definition or grid produces a new
    bundle filename, so caches never silently go stale. Note that --target does
    NOT appear in the bundle name, on purpose: all three tracers live in one
    bundle.
--------------------------------------------------------------------------------
"""

import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

from utils.pipeline_paths import ensure_parents, plot_path
from utils.plot_data import save_plot_data

from Pmx_reconstruction.pmxlib.bundle import load_or_measure
from Pmx_reconstruction.pmxlib.config import (PmxConfig,
                                              TRACER_INFO, add_binning_args,
                                              add_concentration_args,
                                              add_experiment_a_args,
                                              add_selfpair_args, add_target_args)
from Pmx_reconstruction.pmxlib.nfw import concentration, u_nfw
from Pmx_reconstruction.pmxlib.self_pairs import use_self_pair_corrected

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Computer Modern']
rcParams['text.usetex'] = False


# ==============================================================================
# 4.  THE RECONSTRUCTION  (this is the part that matters)
# ==============================================================================
#   P_matter_x(k) = (1/cfg.rhobar_m) sum_i n_i M_i u_m(k|M_i) P_halo_x(k; M_i)
# ------------------------------------------------------------------------------
def reconstruct_P_matter_x(cfg, k, P_halo_x, M_i, n_i, z):
    """Reconstruct P_matter_x(k) from measured P_halo_x(k; M_i) and the halo histogram.

    Nothing in here knows or cares which tracer e is: that is the whole content
    of the identity, and the reason one function covers both modes.

    Parameters
    ----------
    k    : (nk,)        wavenumbers [h/cMpc]
    P_halo_x : (nbins, nk)  measured halo-tracer cross spectra, one row per mass bin
    M_i  : (nbins,)     representative halo mass per bin [Msun/h]
    n_i  : (nbins,)     comoving number density per bin [(cMpc/h)^-3]
    z    : scalar       redshift (for the concentration relation)

    Returns
    -------
    P_matter_x_rec : (nk,)    reconstructed matter-tracer cross spectrum
    """
    c_i = concentration(M_i, z, source=cfg.concentration_source,
                        colossus_model=cfg.colossus_conc_model,
                        sim_params=cfg.sim_params, sim_name=cfg.sim_name)
    u_im = np.array([u_nfw(k, M_i[i], c_i[i], cfg.rhobar_m)
                     for i in range(M_i.size)])
    weight = (n_i * M_i)[:, None] / cfg.rhobar_m          # (nbins, 1)
    return np.sum(weight * u_im * P_halo_x, axis=0)       # sum over mass bins



def main():
    ap = argparse.ArgumentParser(
        description='Reconstruct P(matter x x)(k) from binned FLAMINGO '
                    'halo-tracer cross spectra, for x = gas, dm or matter.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--recompute', action='store_true',
                    help='re-measure the spectra even if a cached bundle exists')
    ap.add_argument('--recompute-shot', action='store_true',
                    help='re-measure the self-pair spectra even if a cache '
                         'exists, and rebuild the bundle block from them')
    add_target_args(ap)
    add_binning_args(ap)
    add_concentration_args(ap)
    add_selfpair_args(ap)

    g = ap.add_argument_group(
        'Experiment A',
        'Replace the flat missing-mass template f_u * P_halo_x(k|M_min) by the '
        'integral over the unresolved masses, with P_halo_x(k|M) extrapolated '
        'downwards off a reference bin. The implementation now lives in '
        'experiment_A.py, which can also be run on its own against the same '
        'cached bundle.')
    g.add_argument('--experiment', choices=('none', 'A'), default='none',
                   help="run Experiment A after the standard reconstruction")
    add_experiment_a_args(ap)

    args = ap.parse_args()

    # One frozen config, built once and passed down. The mass_def guard that
    # used to sit here is now in PmxConfig.__post_init__, so it fires for every
    # consumer rather than only for the script that has a main().
    cfg = PmxConfig.from_args(args)

    tracer = cfg.tracer                         # 'gas', 'dm' or 'matter'
    info = TRACER_INFO[tracer]
    tag = info['tag']                           # for filenames
    x_sym = info['sym']                         # for latex labels

    print(f"[cosmo] sim={cfg.sim_name}, feedback={cfg.feedback}, z={cfg.z}")
    print(f"[cosmo] h={cfg.h}, Omega_m={cfg.omega_m}, L_box={cfg.box} cMpc/h, ngrid={cfg.grid}")
    print(f"[cosmo] cfg.rhobar_m = {cfg.rhobar_m:.4e} (Msun/h)/(Mpc/h)^3")
    print(f"[mode]  target = {cfg.target_mode}  ->  x = {tracer}, reconstructing "
          f"P(matter x {tracer}) from {info['halo_key']}(k; M_i)")
    if tracer == 'matter':
        print("[mode]  one-sided route: the second delta_m is left as a "
              "MEASURED field, so\n        u_m enters once and the "
              "missing-mass deficit stays linear in f_u.")
    print()

    data = load_or_measure(cfg, cfg.nbins, cfg.logm_min, cfg.logm_max,
                           cfg.nkbins, recompute=args.recompute,
                           self_pairs=cfg.subtract_self_pairs,
                           recompute_shot=args.recompute_shot,
                           shot_nreal=cfg.shot_nreal,
                           shot_seed=cfg.shot_seed,
                           shot_chunk=cfg.shot_chunk)

    # --- pick the variant to work with, section 3b -----------------------------
    # The bundle carries the raw spectra, the self-pair terms and the corrected
    # spectra side by side; this is where one of them is put in force for the
    # rest of the run. Doing it in a single place is what makes the correction
    # apply "everywhere" downstream without a flag threaded through the call
    # graph, and doing it AFTER the bundle has been written is what keeps the
    # file on disk independent of the command line.
    shot_removed = use_self_pair_corrected(cfg, data,
                                           subtract=cfg.subtract_self_pairs)

    k = data['k_center']

    # --- pick out the tracer-dependent pieces ----------------------------------
    # Three arrays, all selected through the same table, so there is no longer a
    # per-tracer branch here:
    #   P_halo_x        (nbins, nk)  drives the reconstruction
    #   P_matter_x_true (nk,)        the truth it is scored against
    #   P_dm_x          (nk,)        P(dm x x), the definitional-mismatch curve
    P_halo_x = data[info['halo_key']]
    P_matter_x_true = data[info['truth_key']]
    P_dm_x = data[info['dm_cross_key']]

    P_dm_gas_ref = data['P_dm_gas']              # the fixed dm-gas reference curve

    counts = data['counts']
    M_i = data['M_mean']                  # <M> in the bin, not the bin centre
    logM_cen = data['logM_cen']
    k_Nyquist = float(data['k_Nyquist'])
    box = float(data['box'])
    z = float(data['redshift'])

    n_i = counts / box ** 3               # comoving number density [(cMpc/h)^-3]

    print("\n[sim] halo histogram (FLAMINGO):")
    for i in range(counts.size):
        print(f"    logM={logM_cen[i]:5.2f}  count={counts[i]:10.0f}  "
              f"n_i={n_i[i]:.3e}  n_i M_i={n_i[i] * M_i[i]:.3e}")

    mass_frac_resolved = float(np.sum(n_i * M_i) / cfg.rhobar_m)
    print(f"[sim] resolved mass fraction sum(n_i M_i)/cfg.rhobar_m = "
          f"{mass_frac_resolved:.3f}")

    print(f"\n[truth] P(m x {tracer}), delta_m = f_c delta_dm + f_g delta_gas "
          f"(f_c={float(data['f_c']):.4f}, f_g={float(data['f_g']):.4f})")
    P_selfpair_truth = data.get(f"{info['truth_key']}_selfpair")
    print(f"[truth] self-pair terms {'REMOVED' if shot_removed else 'PRESENT'}"
          + ("" if shot_removed else "  <-- the reconstruction cannot produce them"))

    # --- reconstruction --------------------------------------------------------
    P_matter_x_rec = reconstruct_P_matter_x(cfg, k, P_halo_x, M_i, n_i, z)

    # --- optional missing-mass correction --------------------------------------
    f_u = 1.0 - mass_frac_resolved
    Delta_P_approx = 0.0
    occupied = np.flatnonzero(counts > 0)
    if occupied.size == 0:
        raise SystemExit("No occupied mass bins: check --logm-min/--logm-max "
                         "against the catalogue.")

    if cfg.apply_missing_mass and f_u > 1e-3:
        template = P_halo_x[occupied[0]]      # lowest occupied mass bin
        Delta_P_approx = f_u * template
        P_matter_x_corr = P_matter_x_rec + Delta_P_approx
        print(f"[corr] unresolved mass fraction f_u = {f_u:.3f}; template taken "
              f"from the logM={logM_cen[occupied[0]]:.2f} bin")
    else:
        P_matter_x_corr = P_matter_x_rec
        f_u = 0.0

    # --- comparison plot -------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True, dpi=300,
                                   gridspec_kw=dict(height_ratios=[3, 1], hspace=0.05))

    truth_label = (rf'truth $\delta_m\times\delta_{{{x_sym}}}$, '
                   r'$\delta_m=f_c\delta_{\rm dm}+f_g\delta_{\rm gas}$')
    ax1.loglog(k, np.abs(P_matter_x_true), 'k-', lw=2.5, label=truth_label)

    # P(dm x e), for scale: the gap to the black curve is the DM-vs-matter
    # definitional mismatch, and sets the floor on achievable agreement.
    mismatch_label = rf'$\delta_{{\rm dm}}\times\delta_{{{x_sym}}}$'
    # In 'matter_gas' mode this curve and the dm-gas reference are the same
    # array, so draw it once under a combined label rather than twice.
    ref_is_mismatch = cfg.show_dmgas_reference and tracer == 'gas'
    if ref_is_mismatch:
        mismatch_label += ' (= reference)'
    ax1.loglog(k, np.abs(P_dm_x), color='0.55', ls='-', lw=1.0,
               label=mismatch_label)

    # the fixed dm-gas reference, on both panels in every mode
    show_ref = cfg.show_dmgas_reference and not ref_is_mismatch
    if show_ref:
        ax1.loglog(k, np.abs(P_dm_gas_ref), color='C0', ls='-.', lw=1.2,
                   label=r'$\delta_{\rm dm}\times\delta_{\rm gas}$ (reference)')

    ax1.loglog(k, np.abs(P_matter_x_rec), 'C1--', lw=2.0,
               label=rf'$P_{{\rm m,{tag}}}$ reconstructed')
    if f_u > 1e-3:
        ax1.loglog(k, np.abs(P_matter_x_corr), 'C2:', lw=2.0,
                   label=r'reconstructed + missing-mass')

    # The self-pair term, drawn whether or not it was removed: on a log axis it
    # is immediately obvious at which k it stops being a footnote.
    if P_selfpair_truth is not None and np.any(P_selfpair_truth > 0):
        ax1.loglog(k, np.abs(P_selfpair_truth), color='C3', ls=':', lw=1.2,
                   label=('self-pair term (removed)' if shot_removed
                          else 'self-pair term (NOT removed)'))

    ax1.axvline(k_Nyquist, c='grey', ls=':', lw=1.2, label='$k_{Nyquist}$')
    ax1.set_ylabel(rf'$P_{{\rm m,{tag}}}(k)\,L_{{\rm box}}^2$')
    ax1.legend(frameon=False, fontsize=10)
    ax1.set_title(rf'$P_{{\rm m,{tag}}}$ reconstruction from $P_{{\rm h,{tag}}}(k;M_i)$, '
                  rf'FLAMINGO {cfg.feedback}, $z={z}$'
                  + ('' if shot_removed else '  [self-pairs NOT removed]'))

    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = P_matter_x_rec / P_matter_x_true
        ratio_corr = P_matter_x_corr / P_matter_x_true

    ax2.semilogx(k, ratio, 'C1--', lw=2)
    if f_u > 1e-3:
        ax2.semilogx(k, ratio_corr, 'C2:', lw=2)
    with np.errstate(divide='ignore', invalid='ignore'):
        ax2.semilogx(k, P_dm_x / P_matter_x_true, color='0.55', ls='-', lw=1.0)
        if show_ref:
            ax2.semilogx(k, P_dm_gas_ref / P_matter_x_true, color='C0', ls='-.', lw=1.2)
    ax2.axhline(1.0, color='k', lw=0.8)
    ax2.axvline(k_Nyquist, c='grey', ls=':', lw=1.2)
    ax2.fill_between(k, 0.95, 1.05, color='0.85', zorder=0)
    ax2.set_ylim(0.0, 2.0)
    ax2.set_ylabel('rec / truth')
    ax2.set_xlabel(r'$k$ [h/cMpc]')

    p_out = plot_path(
        'pme_reconstruction', cfg.feedback,
        stem=(
            f'P_matter_{tag}_from_P_halo_{tag}_{cfg.mass_def}_nb{cfg.nbins}_'
            f'logMmin{args.logm_min:.2f}_logMmax{args.logm_max:.2f}'
            f'{"" if shot_removed else "_noshot"}'
        )
    )
    ensure_parents(p_out)
    fig.savefig(p_out, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"\n[plot] saved {p_out}")

    save_plot_data(p_out, {
        'k_center': k,
        'target_mode': cfg.target_mode,
        'tracer': tracer,
        'P_matter_x_true': P_matter_x_true,
        'P_dm_x': P_dm_x,
        'P_dmgas_reference': P_dm_gas_ref,
        'f_c': float(data['f_c']),
        'f_g': float(data['f_g']),
        'P_matter_x_rec': P_matter_x_rec,
        'P_matter_x_corr': P_matter_x_corr,
        'P_halo_x': P_halo_x,
        'counts': counts,
        'M_mean': M_i,
        'logM_cen': logM_cen,
        'n_i': n_i,
        'mass_frac_resolved': mass_frac_resolved,
        'k_Nyquist': k_Nyquist,
        'box': box,
        'redshift': z,
        # Self-pair bookkeeping: the flag, what was removed from the truth, and
        # the raw truth, so a saved figure can always be un-corrected again.
        'self_pairs_removed': shot_removed,
        'P_selfpair_truth': (np.zeros_like(k) if P_selfpair_truth is None
                             else P_selfpair_truth),
        'P_matter_x_true_raw': data.get(f"{info['truth_key']}_raw",
                                        P_matter_x_true),
    }, description=(f'P_matter_{tag} reconstructed from binned FLAMINGO '
                    f'P_halo_{tag}(k;M_i), target={cfg.target_mode}, '
                    f'self-pair terms '
                    f'{"removed" if shot_removed else "left in"}'))

    # --- numerical accuracy checks ------------------------------------------------
    if cfg.apply_missing_mass and f_u > 1e-3:
        Delta_P_meas = P_matter_x_true - P_matter_x_rec
        Delta_P_frac = Delta_P_meas / Delta_P_approx

        # plot it against k
        fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
        ax.semilogx(k, Delta_P_frac, 'C5:', lw=1.5, label=r'$\Delta P_{\rm meas}/\Delta P_{\rm approx}$')
        ax.axhline(1.0, color='k', lw=0.8)
        ax.set_xlabel(r'$k$ [h/cMpc]')
        ax.set_ylabel(r'$\Delta P_{\rm meas}/\Delta P_{\rm approx}$')
        ax.set_title(f'Missing-mass correction accuracy check, f_u={f_u:.3f}')
        p_check = plot_path(
            'pme_reconstruction', cfg.feedback,
            stem=(
                f'P_matter_{tag}_missing_mass_check_{cfg.mass_def}_nb{cfg.nbins}_'
                f'logMmin{args.logm_min:.2f}_logMmax{args.logm_max:.2f}'
                f'{"" if shot_removed else "_noshot"}'
            )
        )
        ensure_parents(p_check)
        fig.savefig(p_check, bbox_inches='tight', dpi=300)
        plt.close(fig)

    # --- Experiment A -----------------------------------------------------------
    # Imported inside the branch on purpose: a plain run must not pay for CAMB
    # or the colossus mass-function and bias modules.
    if args.experiment == 'A':
        from Pmx_reconstruction.experiment_A import run_experiment_A
        from Pmx_reconstruction.pmxlib.config import ExperimentAOptions
        run_experiment_A(cfg, data, ExperimentAOptions.from_args(args),
                         tracer, tag, x_sym)

    print("\n" + "=" * 70)
    print("Done.")
    print("=" * 70)


if __name__ == '__main__':
    main()