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

        python predict_P_matter_x.py                            # gas, from cache
        python predict_P_matter_x.py --target matter_dm         # dm, same bundle
        python predict_P_matter_x.py --target matter_matter     # the auto spectrum
        python predict_P_matter_x.py --recompute                # re-measure
        python predict_P_matter_x.py --nbins 40 --logm-min 11.5
        python predict_P_matter_x.py --no-shot-subtract         # keep self-pairs
        python predict_P_matter_x.py --recompute-shot           # re-measure them

    EXPERIMENT A -- extrapolating P_halo_x(k|M) below M_min (section 6):

        # the validation test: hide everything below 10^12, then score every
        # extrapolation against the exact contribution of the hidden bins
        python predict_P_matter_x.py --experiment A --split-logm 12.0 \
               --extrap flat simhc bias halomodel --hmf catalog

        # same, but with the mass weight from Tinker+08 instead of the
        # catalogue: now the HMF is on trial as well as the extrapolation
        python predict_P_matter_x.py --experiment A --split-logm 12.0 --hmf tinker

        # production: correct below the catalogue limit, no exact target exists
        python predict_P_matter_x.py --experiment A

        # published c(M,z) instead of the built-in power law, and the
        # EH98 fitting function instead of the CAMB spectrum, for comparison
        python predict_P_matter_x.py --experiment A --concentration colossus
        python predict_P_matter_x.py --experiment A --power-spectrum eisenstein98

    Any change to the mass binning, mass definition or grid produces a new
    bundle filename, so caches never silently go stale. Note that --target does
    NOT appear in the bundle name, on purpose: all three tracers live in one
    bundle.
--------------------------------------------------------------------------------
"""

import argparse
import gc
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.special import sici

import camb   

from colossus.cosmology import cosmology as colossus_cosmology
from colossus.lss import (peaks as colossus_peaks,
                            mass_function as colossus_mf,
                            bias as colossus_bias)
from colossus.halo import concentration as colossus_conc

import astropy.units as u
from astropy.cosmology import FlatLambdaCDM

from abacusnbody.analysis.tsc import tsc_parallel

from utils.catalog_loaders import load_halo_properties, load_particle_properties
from utils.delta_fields import compute_delta_2d, compute_delta_field_and_mass
from utils.power_spectrum_utils import (
    compute_2d_fft,
    compute_k_grid_2d,
    bin_power_spectrum_2d,
)
from utils.pipeline_paths import (
    DATA_ROOT,
    delta_2d_path as _delta_2d_path,
    ensure_parents,
    get_halo_file_path,
    get_particle_file_path,
    plot_path,
)
from utils.plot_data import save_plot_data
from utils.sim_params import get_sim_params

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Computer Modern']
rcParams['text.usetex'] = False


# ==============================================================================
# 0.  FIXED CONFIGURATION  --  FLAMINGO strongest AGN, and nothing else
# ==============================================================================
SIM_NAME = 'flamingo'
FEEDBACK = 'strongest_AGN'

SIM_PARAMS = get_sim_params(SIM_NAME)
BOX     = SIM_PARAMS['box_size_cMpc_h']     # 681.0 cMpc/h
NGRID   = SIM_PARAMS['ngrid_default']       # 2048
NTHREAD = SIM_PARAMS['nthread_default']
Z_EVAL  = SIM_PARAMS['redshift']            # 0.74
H_LITTLE = SIM_PARAMS['h']
OMEGA_M  = SIM_PARAMS['omega_m']

# Critical density today in (Msun/h)/(Mpc/h)^3. The h's cancel, so this is the
# same number as 2.775e11 h^2 Msun/Mpc^3. Comoving mean matter density does not
# evolve, so rhobar_m is redshift-independent in these units.
# Taken from astropy, which reproduces the familiar literal 2.77536627e11 to
# nine digits. See critical_density_h_units() in section 6.
RHO_CRIT_0 = None            # filled in below, once the helper is defined
rhobar_m = None

# --- halo binning -------------------------------------------------------------
# Halos are binned by their virial mass. In the FLAMINGO catalogue the closest
# available definition is m200b (M200 w.r.t. the mean background density), which
# is also what the r200m <-> M conversion in u_nfw assumes. Switching this to
# 'm200c' would make r200m_of_M inconsistent, hence the guard in main().
MASS_DEF  = 'm200b'
LOGM_MIN  = 11.0          # log10(M / [Msun/h])
LOGM_MAX  = 15.0
NBINS     = 30            # log-spaced bins between LOGM_MIN and LOGM_MAX
CENTRALS_ONLY = True      # satellites repeat their host's m200b; counting them
                          # would double-count halo mass in n_i M_i

# Unit of the halo-mass array in the catalogue, expressed in Msun/h. Particle
# masses in this repo are stored in 1e10 Msun/h; halo masses are read straight
# out of /galaxies/m200b. If the sanity check below fires, set this to 1e10.
HALO_MASS_UNIT_MSUN_H = 1.0

NKBINS = 600             # None -> bin_power_spectrum_2d's default, max(32, ngrid//10)

# --- self-pair (shot-noise) subtraction ---------------------------------------
# See section 3b. The measurement is a Monte-Carlo one: each species is painted
# at uniformly random positions with its true masses and the auto spectrum of
# the resulting field is, in expectation, exactly the self-pair term that the
# real field carries. The seed is fixed so that a cache is reproducible, and it
# is the same seed measure_u_tilde.py uses.
SUBTRACT_SELF_PAIRS = True    # switched off by --no-shot-subtract
SHOT_SEED = 12345
SHOT_NREAL = 1                # random realisations averaged; more = less noise
SHOT_CHUNK = 5e7              # particles per painting chunk (caps peak memory)

# --- which spectrum is being reconstructed ------------------------------------
# This is the second field x in P_matter_x = (1/rhobar_m) sum_i n_i M_i u_m P_halo_x.
# The identity is blind to x, so this is purely a choice of what to point it at:
#   'matter_gas'    : x = gas,    reconstruct P(matter x gas)    from P_halo_gas
#   'matter_dm'     : x = dm,     reconstruct P(matter x dm)     from P_halo_dm
#   'matter_matter' : x = matter, reconstruct P(matter x matter) from P_halo_matter
# All three halo cross-spectra live in the same bundle, so switching is free.
TARGET_MODE = 'matter_gas'

# NAMING CONVENTION, used everywhere below
#     P_<first field>_<second field>
# with the fields drawn from {halo, matter, dm, gas} and 'x' standing for
# whichever tracer the current --target has selected. So:
#     P_halo_gas, P_halo_dm, P_halo_matter   per-bin, shape (nbins, nk)
#     P_matter_gas, P_matter_dm, P_matter_matter, P_dm_gas, P_dm_dm   truths, (nk,)
#     P_halo_x, P_matter_x_true, P_dm_x      the target-dependent aliases
# Nothing is called P_hg, P_me or P_dme any more: with three modes in play the
# one-letter abbreviations stopped being readable.

TRACER_OF_TARGET = {'matter_gas': 'gas',
                    'matter_dm': 'dm',
                    'matter_matter': 'matter'}

# Everything downstream keys off this table rather than off string surgery on
# the target name, so adding a fourth tracer is a single entry here plus a
# measurement in section 3.
#
#   halo_key     the per-bin halo cross spectrum that drives the reconstruction
#   truth_key    P(matter x x), what the reconstruction is scored against
#   dm_cross_key P(dm x x), the definitional-mismatch curve: the same
#                reconstruction run against a DM-only first field would target
#                this instead, so the gap between the two is the floor on
#                achievable agreement, not an error
#   sym          latex symbol for the tracer in plot labels
TRACER_INFO = {
    'gas': dict(tag='gas', sym=r'\rm gas',
                halo_key='P_halo_gas',
                truth_key='P_matter_gas',
                dm_cross_key='P_dm_gas'),
    'dm': dict(tag='dm', sym=r'\rm dm',
               halo_key='P_halo_dm',
               truth_key='P_matter_dm',
               dm_cross_key='P_dm_dm'),
    # P(dm x matter) is P(matter x dm) by symmetry, hence the shared key.
    'matter': dict(tag='matter', sym=r'\rm m',
                   halo_key='P_halo_matter',
                   truth_key='P_matter_matter',
                   dm_cross_key='P_matter_dm'),
}

# --- what plays the role of "matter" in the truth: not a choice ---------------
# The reconstruction's weights use M200b (TOTAL halo mass) and
# rhobar_m = Omega_m rho_crit (TOTAL matter density), so the right-hand side of
# the halo-model identity targets P(total matter x e) and nothing else:
#
#     delta_m = f_c delta_dm + f_g delta_gas (+ f_star delta_star)
#
# P(dm x e) is measured too and plotted as the definitional-mismatch curve, but
# it is not a competing definition of the truth -- comparing the reconstruction
# against it would just be scoring the identity for a mismatch that is built
# into the question. The mismatch grows at high k, where feedback makes gas
# smoother than DM.
#
# NOTE: star particles are in neither field, so delta_m as built here is still
# missing f_star ~ 1% of the matter, and those are the MOST clustered baryons.
# The residual inconsistency with rhobar_m (which does include stars) is at that
# level. Note also that P(m x x) contains an explicit f_x P_xx term, so unlike a
# pure cross-spectrum it is not entirely free of the tracer's shot noise -- with
# weight f_g ~ 0.16 for gas, f_c ~ 0.84 for DM, and 1 for matter, where the
# target is a genuine auto spectrum and nothing cancels at all. That self-pair
# term is measured and removed in section 3b; what is left here is the genuine
# f_x P_xx CLUSTERING, which the identity does have to reproduce.

# --- the always-on reference curve --------------------------------------------
# P(delta_dm x delta_gas) is drawn on both panels in every run, whatever the
# target is. It is the cleanest single picture of baryon-vs-DM clustering in the
# box, so having it fixed on the axes makes the gas-target and dm-target plots
# directly comparable to each other rather than each being self-referential.
# In 'matter_gas' mode it IS the definitional-mismatch curve P(dm x e), so it is
# drawn once and labelled as both rather than plotted twice.
SHOW_DMGAS_REFERENCE = True

# --- missing-mass correction --------------------------------------------------
# sum_i n_i M_i < rhobar_m always: mass in halos below 10^LOGM_MIN and in the
# diffuse IGM is not represented. The correction below adds f_u * P_halo_x of the
# LOWEST mass bin, i.e. it assumes the unresolved mass cross-correlates with the
# tracer like the smallest resolved halos do. That is an approximation and
# mildly over-corrects (unresolved mass is less biased than 10^11 Msun/h halos),
# but it needs no linear theory and no bias model. It is reported separately
# from the raw reconstruction so you can always see how much it moved things.
APPLY_MISSING_MASS = True


# ==============================================================================
# 1.  NFW PROFILE  u_m(k|M)   (normalized: u_m -> 1 as k -> 0)
# ==============================================================================
def delta_vir_200m():
    """Halo mass is M200m: mean interior density = 200 * rhobar_m."""
    return 200.0


def r200m_of_M(M):
    """Comoving radius r200m [Mpc/h] for M200m [Msun/h].

    M = (4/3) pi r^3 * 200 * rhobar_m  =>  r = (3M / (4 pi 200 rhobar_m))^(1/3).
    Using the comoving rhobar_m gives a comoving radius, consistent with a
    comoving-k profile transform.
    """
    return (3.0 * M / (4.0 * np.pi * delta_vir_200m() * rhobar_m)) ** (1.0 / 3.0)


def u_nfw(k, M, c):
    """Normalized Fourier transform of an NFW halo of mass M, concentration c.

    k [h/Mpc] (array), M [Msun/h] (scalar), c [-] (scalar). Returns u(k).
    """
    r200 = r200m_of_M(M)
    rs = r200 / c
    mc = np.log(1.0 + c) - c / (1.0 + c)     # mass normalisation
    kr = np.asarray(k, dtype=float) * rs
    kr = np.where(kr > 0, kr, 1e-12)         # k=0 mode is never used, keep sici finite
    si_1c, ci_1c = sici((1.0 + c) * kr)
    si_0, ci_0 = sici(kr)
    term = (np.sin(kr) * (si_1c - si_0)
            - np.sin(c * kr) / ((1.0 + c) * kr)
            + np.cos(kr) * (ci_1c - ci_0))
    return term / mc


# ==============================================================================
# 2.  CONCENTRATION-MASS RELATION  c(M, z)
# ==============================================================================
# A simple power law in M200m, with parameters chosen to sit on the Diemer+19 /
# FLAMINGO-DMO median at these redshifts (c ~ 5-6 at cluster scales, ~9-10 for
# small halos). Swap in an exact colossus call if you want; with x = gas the
# reconstruction is only mildly sensitive to c, since P_halo_gas already carries the
# gas profile and u_m only redistributes the *matter* weight.
#
# That last sentence does NOT survive the switch to x = dm or x = matter. There
# the measured P_halo_dm (or P_halo_matter) carries the true matter profile and
# the truth goes as u_true^2, so the reconstruction/truth ratio at high k is
# essentially u_model/u_true and this function is being tested directly. Tune
# c0/alpha before concluding anything about the halo-model identity from a
# high-k residual in either mode.
# Which c(M,z) to use. 'powerlaw' is the built-in fit below and is the DEFAULT,
# so that upgrading this file does not silently move every existing plot: u_m
# enters the baseline reconstruction, not just Experiment A. 'colossus' swaps in
# a published relation and is the better choice for new work -- especially in
# 'matter_dm' and 'matter_matter' modes, where the reconstruction/truth ratio at
# high k is close to
# u_model/u_true and this function is what is really being tested.
CONCENTRATION_SOURCE = 'powerlaw'      # set from --concentration
COLOSSUS_CONC_MODEL = 'diemer19'       # set from --concentration-model
_conc_warned = False


def _ensure_colossus_cosmology():
    """Set the colossus cosmology from the parameter file, once.

    concentration() can be called without a HaloModel ever being built (the
    baseline reconstruction uses u_m too), so the cosmology has to be
    establishable on its own.
    """
    if colossus_cosmology.current_cosmo is not None:
        return
    colossus_cosmology.setCosmology(
        f'{SIM_NAME}_hatf',
        **dict(flat=True, H0=100.0 * H_LITTLE, Om0=OMEGA_M,
               Ob0=SIM_PARAMS.get('omega_b'), sigma8=SIM_PARAMS.get('sigma8'),
               ns=SIM_PARAMS.get('n_s'), Tcmb0=SIM_PARAMS.get('tcmb0', 2.725),
               relspecies=False))


def _concentration_colossus(M, z):
    global _conc_warned
    _ensure_colossus_cosmology()
    scalar = np.isscalar(M)
    M = np.atleast_1d(np.asarray(M, dtype=float))
    # The unresolved-mass integral reaches down to 10^8 Msun/h, below the
    # calibration range of most c(M,z) fits. colossus warns and extrapolates;
    # that is acceptable here because u_m ~ 1 for those halos at every k we use,
    # so the extrapolated c barely affects the answer. Surface it once rather
    # than letting it either spam the log or vanish.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        c = colossus_conc.concentration(M, f'{int(DELTA_HALO)}m', z,
                                        model=COLOSSUS_CONC_MODEL)
    if caught and not _conc_warned:
        print(f"  [conc] {COLOSSUS_CONC_MODEL}: some masses lie outside the "
              "model's calibration range and are extrapolated. Harmless here "
              "(u_m -> 1 for those halos), but noted.")
        _conc_warned = True
    return float(c[0]) if scalar else c


def concentration(M, z):
    """Concentration-mass relation, c(M200m, z)."""
    if CONCENTRATION_SOURCE == 'colossus':
        return _concentration_colossus(M, z)
    c0, Mpiv, alpha, beta = 7.5, 1e12, 0.090, 0.65
    return c0 * (M / Mpiv) ** (-alpha) * (1.0 + z) ** (-beta)


# ==============================================================================
# 3.  MEASUREMENT FROM FLAMINGO  (this replaces the synthetic universe)
# ==============================================================================
def bundle_path(nbins, logm_min, logm_max, ngrid, nkbins):
    """One npz per distinct measurement configuration.

    Everything that changes the numbers is in the filename, so a stale bundle
    can never be picked up for a different binning or grid. TARGET_MODE is
    deliberately absent: one bundle serves all three tracers.
    """
    nk = 'default' if nkbins is None else str(nkbins)
    stem = (f'spectra_v4_{FEEDBACK}_{MASS_DEF}'
            f'_logM{logm_min:g}-{logm_max:g}_nb{nbins}'
            f'_ngrid{ngrid}_nk{nk}'
            f'_{"cen" if CENTRALS_ONLY else "all"}.npz')
    return DATA_ROOT / SIM_NAME / 'pme_inputs' / stem


def mass_fractions_path():
    """Tiny cache for the component mass fractions (a few bytes, but the
    particle read that produces them is not cheap)."""
    return DATA_ROOT / SIM_NAME / 'pme_inputs' / f'mass_fractions_{FEEDBACK}.npz'


def component_mass_fractions(source='particles'):
    """Return (f_c, f_g): DM and gas fractions of the DM+gas mass budget.

    These are the weights in delta_m = f_c delta_dm + f_g delta_gas.

    source='particles' sums the actual particle masses in the snapshot (exact
    for the two fields we have, cached so it happens at most once).
    source='cosmology' uses f_g = Omega_b/Omega_m with no file access, which
    effectively assigns the stellar mass to the gas field and so slightly
    over-weights the smoother component.
    """
    if source == 'cosmology':
        f_g = SIM_PARAMS['omega_b'] / SIM_PARAMS['omega_m']
        print(f"  mass fractions from cosmology: f_c={1 - f_g:.5f}, f_g={f_g:.5f}")
        return 1.0 - f_g, f_g

    path = mass_fractions_path()
    if path.exists():
        with np.load(path) as f:
            f_c, f_g = float(f['f_c']), float(f['f_g'])
        print(f"  mass fractions from cache: f_c={f_c:.5f}, f_g={f_g:.5f}")
        return f_c, f_g

    print("  Summing particle masses to get component fractions (one-off) ...")
    particles_file = get_particle_file_path(FEEDBACK, sim_name=SIM_NAME)

    gas_p = load_particle_properties(particles_file, 'gas', requested=('mass',),
                                     sim_name=SIM_NAME, Lbox=BOX)
    M_gas = float(np.sum(gas_p['mass']))
    del gas_p
    gc.collect()

    dm_p = load_particle_properties(particles_file, 'dm', requested=('mass',),
                                    sim_name=SIM_NAME, Lbox=BOX)
    M_dm = float(np.sum(dm_p['mass']))
    del dm_p
    gc.collect()

    total = M_dm + M_gas
    f_c, f_g = M_dm / total, M_gas / total

    # Cross-check against the cosmological baryon fraction. They should not be
    # equal -- the difference is roughly the stellar mass, which is in neither
    # field -- but a large discrepancy means something is wrong.
    f_b_cosmo = SIM_PARAMS['omega_b'] / SIM_PARAMS['omega_m']
    print(f"    M_dm={M_dm:.4e}, M_gas={M_gas:.4e} (1e10 Msun/h)")
    print(f"    f_c={f_c:.5f}, f_g={f_g:.5f}  "
          f"(cosmological Omega_b/Omega_m = {f_b_cosmo:.5f}; "
          f"the deficit {f_b_cosmo - f_g:+.5f} is mostly stars)")
    if not (0.5 * f_b_cosmo < f_g < 1.5 * f_b_cosmo):
        print("    WARNING: gas fraction is far from the cosmological baryon "
              "fraction. Check the particle file and mass units.")

    ensure_parents(path)
    np.savez(path, f_c=f_c, f_g=f_g, M_dm=M_dm, M_gas=M_gas)
    return f_c, f_g


def _load_or_compute_delta(label):
    """Load a FLAMINGO 2-D projected delta field, computing it only if absent.

    Uses the same paths as HATF_reconstruction, so a field HATF has already
    written is picked up here for free (and vice versa).
    """
    cfg = {'sim_name': SIM_NAME, 'feedback': FEEDBACK}
    path = _delta_2d_path(cfg, label)

    if path.exists():
        print(f"  Loading {label} field from {path} ...")
        field = np.load(path)
        print(f"    shape {field.shape}")
        return field

    print(f"  {label} field not found. Computing (this is the slow part) ...")
    particles_file = get_particle_file_path(FEEDBACK, sim_name=SIM_NAME)
    field, _ = compute_delta_field_and_mass(
        tracer=label,
        sim_name=SIM_NAME,
        dm_particles_file=particles_file if label == 'dm' else None,
        gas_particles_file=particles_file,   # 'dm' needs it too, for the rescale
        box=BOX,
        ngrid=NGRID,
        nthread=NTHREAD,
    )
    ensure_parents(path)
    np.save(path, field)
    print(f"    computed and saved to {path}")
    gc.collect()
    return field


def _check_mass_units(masses_msun_h):
    """Loud failure if the catalogue mass unit is not what we assumed."""
    finite = masses_msun_h[np.isfinite(masses_msun_h) & (masses_msun_h > 0)]
    if finite.size == 0:
        raise RuntimeError("No positive halo masses found in the catalogue.")
    logm_hi = np.log10(np.max(finite))
    if not (12.5 < logm_hi < 16.5):
        raise RuntimeError(
            f"Halo masses look wrong: max log10(M) = {logm_hi:.2f}, expected the "
            f"most massive FLAMINGO halo near 10^15 Msun/h.\n"
            f"HALO_MASS_UNIT_MSUN_H is currently {HALO_MASS_UNIT_MSUN_H:g}; if the "
            f"catalogue stores masses in 1e10 Msun/h, set it to 1e10."
        )
    print(f"    mass sanity check ok: max log10(M) = {logm_hi:.2f}")


def _looks_like_shot_noise_floor(P):
    """True if the last few k bins are flat to within 5%, the tell-tale shape of
    a discreteness floor rather than physics. Only ever applied to the auto
    spectrum, which is the one quantity here that has such a floor."""
    if P.size < 5:
        return False
    tail = P[-4:]
    return bool(np.all(np.isfinite(tail))
                and np.ptp(tail) < 0.05 * np.abs(np.mean(tail)))


def binned_spectrum(prod, k_grid, nkbins):
    """The one and only k-binning convention in this file.

    Takes the real part of a product of two transforms (or |transform|^2),
    azimuthally averages it over the k bins and applies the L_box^2 factor that
    every spectrum in this repo carries.

    It is a module-level function rather than a closure inside measure_spectra
    because the self-pair spectra of section 3b are measured in a completely
    separate pass, possibly years apart in wall-clock time, and MUST come out in
    the identical convention or the subtraction is meaningless. Sharing the
    function is the cheapest way to guarantee that.
    """
    kb, kc, P = bin_power_spectrum_2d(prod, k_grid, NGRID, BOX,
                                      nkbins=nkbins, k_min=2.0 * np.pi / BOX)
    return np.asarray(kb), np.asarray(kc), np.asarray(P) * BOX ** 2


def measure_spectra(nbins, logm_min, logm_max, nkbins):
    """Measure P_halo_x(k; M_i) per mass bin for BOTH tracers, plus the truths.

    Returns a dict ready to be written to the npz bundle.
    """
    print("=" * 70)
    print("Measuring FLAMINGO spectra (no cached bundle found)")
    print("=" * 70)

    # --- 3a. gas and DM projected fields --------------------------------------
    # Both survive the whole mass-bin loop now: each halo field is crossed
    # against both. Two complex 2048^2 transforms is ~134 MB, which is nothing
    # next to the particle read that produced them.
    print("\nProjected fields:")
    delta_gas = _load_or_compute_delta('gas')
    delta_dm = _load_or_compute_delta('dm')

    gas_fft = compute_2d_fft(delta_gas, NGRID)
    dm_fft = compute_2d_fft(delta_dm, NGRID)
    del delta_gas, delta_dm
    gc.collect()

    k_grid = compute_k_grid_2d(NGRID, BOX)
    k_Nyquist = np.pi * NGRID / BOX
    # k_min is not set here any more: the 2 pi / L lower edge now lives in
    # binned_spectrum(), which is shared with the self-pair pass of section 3b
    # so that the two can never end up on different k bins.

    # --- 3b. the matter field --------------------------------------------------
    # delta_m = f_c delta_dm + f_g delta_gas. The FFT is linear, so this is done
    # on the transforms directly: no third real-space array is ever allocated,
    # and the result is exact rather than an approximation.
    print("\nComponent mass fractions:")
    f_c, f_g = component_mass_fractions()
    m_fft = f_c * dm_fft + f_g * gas_fft

    # --- 3c. the truths --------------------------------------------------------
    # Four spectra, two per tracer. P(m x e) is what the reconstruction actually
    # targets, since the weights n_i M_i / rhobar_m are built from TOTAL halo
    # mass and TOTAL matter density. P(dm x e) is kept for comparison: the
    # difference between them is the definitional mismatch, negligible at low k
    # and growing once feedback-driven gas displacement becomes comparable to
    # 1/k. P(dm x gas) doubles as the fixed reference curve on both panels.
    def _binned(prod):
        return binned_spectrum(prod, k_grid, nkbins)

    k_bins, k_center, P_matter_gas = _binned((m_fft * np.conj(gas_fft)).real)
    _, _, P_dm_gas = _binned((dm_fft * np.conj(gas_fft)).real)
    _, _, P_matter_dm = _binned((m_fft * np.conj(dm_fft)).real)
    _, _, P_dm_dm = _binned(np.abs(dm_fft) ** 2)
    # The two autos that 'matter_matter' mode needs. P_matter_matter is the
    # truth for that target; P_gas_gas is not used by the reconstruction at all
    # and is kept purely so that the closure check below has something to close.
    _, _, P_matter_matter = _binned(np.abs(m_fft) ** 2)
    _, _, P_gas_gas = _binned(np.abs(gas_fft) ** 2)

    print(f"\nTruths measured on {k_center.size} k bins, "
          f"k = [{k_center[0]:.4f}, {k_center[-1]:.2f}] h/cMpc")
    with np.errstate(divide='ignore', invalid='ignore'):
        dev_g = np.abs(P_matter_gas / P_dm_gas - 1.0)
        dev_d = np.abs(P_matter_dm / P_dm_dm - 1.0)
        dev_m = np.abs(P_matter_matter / P_dm_dm - 1.0)
    for name, dev in (('gas', dev_g), ('dm', dev_d), ('matter', dev_m)):
        finite = np.isfinite(dev)
        if np.any(finite):
            print(f"  x={name:6s}  |P_matter_x/P_dm_x - 1|: "
                  f"{np.nanmax(dev[finite]):.4f} max, "
                  f"{dev[finite][0]:.4f} at k_min")

    # Closure: delta_m = f_c delta_dm + f_g delta_gas is linear, so the matter
    # auto MUST be the quadratic combination of the three measured spectra.
    # Any deviation beyond round-off means f_c/f_g or the binning is wrong, and
    # it is much easier to catch here than downstream in a reconstruction ratio.
    P_mm_closure = (f_c ** 2 * P_dm_dm
                    + 2.0 * f_c * f_g * P_dm_gas
                    + f_g ** 2 * P_gas_gas)
    with np.errstate(divide='ignore', invalid='ignore'):
        closure = np.abs(P_matter_matter / P_mm_closure - 1.0)
    closure_max = float(np.nanmax(closure[np.isfinite(closure)]))
    print(f"  closure |P_mm / (f_c^2 P_dd + 2 f_c f_g P_dg + f_g^2 P_gg) - 1| "
          f"= {closure_max:.2e} max")
    if closure_max > 1e-6:
        print("  WARNING: the matter auto spectrum does not close on its "
              "components. Check f_c, f_g and the k binning before using "
              "'matter_matter' mode.")

    # Both autos have a genuine shot-noise floor. P_dm_dm propagates into
    # P(m x dm) with weight f_c; in 'matter_matter' mode the target IS an auto
    # spectrum, so nothing cancels and the floor enters at full weight.
    for name, P_auto in (('P_dm_dm', P_dm_dm),
                         ('P_matter_matter', P_matter_matter),
                         ('P_gas_gas', P_gas_gas)):
        if _looks_like_shot_noise_floor(P_auto):
            print(f"  WARNING: {name} is flat to within 5% across the last four "
                  f"k bins. That is the shape of a discreteness floor, not "
                  f"physics; check the particle number before trusting the "
                  f"high-k end of any target that leans on it.")

    del m_fft
    gc.collect()

    # --- 3d. halo catalogue ----------------------------------------------------
    print("\nHalo catalogue:")
    halo_file = get_halo_file_path(FEEDBACK, sim_name=SIM_NAME)
    print(f"  {halo_file}")
    requested = ['pos', MASS_DEF]
    if CENTRALS_ONLY:
        requested.append('centrals')
    halo_props = load_halo_properties(halo_file, requested, sim_name=SIM_NAME)
    if halo_props is None:
        raise RuntimeError('Failed to load halo properties')

    all_pos = np.asarray(halo_props['pos'])
    all_mass = np.asarray(halo_props[MASS_DEF], dtype=np.float64) * HALO_MASS_UNIT_MSUN_H

    if CENTRALS_ONLY:
        cen = np.asarray(halo_props['centrals']).astype(bool)
        all_pos = all_pos[cen]
        all_mass = all_mass[cen]
        print(f"  centrals only: {all_mass.size} objects")
    else:
        print(f"  all objects: {all_mass.size}")
    del halo_props
    gc.collect()

    _check_mass_units(all_mass)

    # --- 3e. log-spaced mass bins ---------------------------------------------
    logM_edges = np.linspace(logm_min, logm_max, nbins + 1)
    logM_cen = 0.5 * (logM_edges[:-1] + logM_edges[1:])
    M_cen = 10.0 ** logM_cen

    with np.errstate(divide='ignore', invalid='ignore'):
        logm_all = np.log10(all_mass)
    in_range = np.isfinite(logm_all) & (logm_all >= logm_min) & (logm_all < logm_max)
    print(f"  in [{logm_min}, {logm_max}): {np.count_nonzero(in_range)} halos "
          f"({np.count_nonzero(in_range) / all_mass.size:.1%} of the sample)")

    bin_index = np.full(all_mass.size, -1, dtype=np.int64)
    bin_index[in_range] = np.digitize(logm_all[in_range], logM_edges) - 1
    bin_index[bin_index == nbins] = nbins - 1     # right edge into the last bin

    counts = np.zeros(nbins, dtype=np.float64)
    M_mean = np.zeros(nbins, dtype=np.float64)    # <M> in the bin, more faithful
    P_halo_gas = np.zeros((nbins, k_center.size), dtype=np.float64)
    P_halo_dm = np.zeros((nbins, k_center.size), dtype=np.float64)

    print("\nPer-bin halo cross spectra (gas and dm from the same halo field):")
    for i in range(nbins):
        sel = bin_index == i
        n_in_bin = int(np.count_nonzero(sel))
        counts[i] = n_in_bin
        if n_in_bin == 0:
            print(f"  bin {i:2d}  logM = {logM_cen[i]:5.2f}  EMPTY, skipped")
            continue

        M_mean[i] = float(np.mean(all_mass[sel]))
        pos_i = np.ascontiguousarray(all_pos[sel], dtype=np.float32)

        delta_h = compute_delta_2d(pos_i, BOX, NGRID, None, nthread=NTHREAD)
        halo_fft = compute_2d_fft(delta_h, NGRID)
        del delta_h, pos_i

        _, _, P_halo_gas[i] = _binned((gas_fft * np.conj(halo_fft)).real)
        _, _, P_halo_dm[i] = _binned((dm_fft * np.conj(halo_fft)).real)

        del halo_fft
        gc.collect()

        print(f"  bin {i:2d}  logM = {logM_cen[i]:5.2f}  N = {n_in_bin:8d}  "
              f"log10<M> = {np.log10(M_mean[i]):5.2f}  "
              f"P_halo_gas(k_min) = {P_halo_gas[i, 0]:.4e}  P_halo_dm(k_min) = {P_halo_dm[i, 0]:.4e}")

    del gas_fft, dm_fft, all_pos, all_mass
    gc.collect()

    # Where a bin is empty, fall back to the geometric bin centre so downstream
    # arithmetic stays finite; its weight is zero anyway because counts = 0.
    M_mean = np.where(counts > 0, M_mean, M_cen)

    return dict(
        k_bins=np.asarray(k_bins),
        k_center=np.asarray(k_center),
        k_Nyquist=k_Nyquist,
        logM_edges=logM_edges,
        logM_cen=logM_cen,
        M_cen=M_cen,
        M_mean=M_mean,
        counts=counts,
        P_halo_gas=P_halo_gas,
        P_halo_dm=P_halo_dm,
        # P_halo_matter is NOT stored: it is f_c P_halo_dm + f_g P_halo_gas
        # exactly, and deriving it on load keeps the bundle from carrying a
        # redundant (nbins, nk) array that could drift out of step with f_c/f_g.
        P_matter_gas=P_matter_gas,
        P_dm_gas=P_dm_gas,
        P_matter_dm=P_matter_dm,
        P_dm_dm=P_dm_dm,
        P_matter_matter=P_matter_matter,
        P_gas_gas=P_gas_gas,
        f_c=f_c,
        f_g=f_g,
        box=BOX,
        ngrid=NGRID,
        redshift=Z_EVAL,
        mass_def=MASS_DEF,
        centrals_only=CENTRALS_ONLY,
        feedback=FEEDBACK,
    )


def derive_P_halo_matter(data):
    """Add P_halo_matter to a loaded bundle, in place.

    delta_m = f_c delta_dm + f_g delta_gas and the FFT is linear, so the
    halo-matter cross spectrum is an exact linear combination of the two
    measured halo cross spectra:

        P_halo_matter(k; M_i) = f_c P_halo_dm(k; M_i) + f_g P_halo_gas(k; M_i)

    No new measurement, no approximation. This is the entire cost of the
    right-hand side of 'matter_matter' mode.
    """
    if 'P_halo_matter' in data:
        return data
    f_c, f_g = float(data['f_c']), float(data['f_g'])
    data['P_halo_matter'] = f_c * data['P_halo_dm'] + f_g * data['P_halo_gas']
    return data


# Keys every bundle must carry. P_halo_matter is absent on purpose: it is
# derived on load, not stored.
_REQUIRED_KEYS = ('k_center', 'counts', 'M_mean', 'f_c', 'f_g',
                  'P_halo_gas', 'P_halo_dm', 'P_matter_gas', 'P_dm_gas',
                  'P_matter_dm', 'P_dm_dm', 'P_matter_matter', 'P_gas_gas')


def load_or_measure(nbins, logm_min, logm_max, nkbins, recompute=False,
                    self_pairs=True, recompute_shot=False,
                    shot_nreal=SHOT_NREAL, shot_seed=SHOT_SEED,
                    shot_chunk=SHOT_CHUNK):
    """Load the cached bundle if it exists, otherwise measure and write it.

    The bundle carries all THREE variants of every spectrum that has a self-pair
    term -- the uncorrected measurement, the self-pair term itself, and the
    corrected difference -- so that the correction is a stored quantity rather
    than something re-derived on every run. See section 3b for the block's
    layout and ensure_self_pair_block() for how it is built.

    A bundle written before section 3b existed simply lacks that block; it is
    added and the bundle rewritten in place, so no existing bundle ever has to
    have its (expensive) per-bin halo spectra re-measured. That is why the
    filename version is still v4.

    The dict returned always has the RAW spectrum under the canonical name.
    Choosing which variant to work with is use_self_pair_corrected()'s job, and
    it is a separate step on purpose: what gets written to disk should not
    depend on a command-line switch.
    """
    path = bundle_path(nbins, logm_min, logm_max, NGRID, nkbins)

    data = None
    if path.exists() and not recompute:
        print(f"Loading pre-computed spectra bundle:\n  {path}")
        with np.load(path, allow_pickle=False) as f:
            data = {key: f[key] for key in f.files}
        missing = [key for key in _REQUIRED_KEYS if key not in data]
        if missing:
            print(f"  Bundle is missing {missing}; re-measuring.")
            data = None
        else:
            print(f"  {data['P_halo_gas'].shape[0]} mass bins, "
                  f"{data['k_center'].size} k bins, all three tracers, "
                  f"f_c={float(data['f_c']):.4f}, f_g={float(data['f_g']):.4f}. "
                  f"Skipping measurement.")

    fresh = data is None
    if fresh:
        if recompute and path.exists():
            print("--recompute given: ignoring the existing bundle and "
                  "re-measuring.")
        data = measure_spectra(nbins, logm_min, logm_max, nkbins)

    # The self-pair block, added to the dict and then persisted with it. It is
    # rewritten whenever the spectra themselves were re-measured, whenever the
    # block was absent, and whenever --recompute-shot asks for it.
    added = ensure_self_pair_block(data, nkbins, allow_measure=self_pairs,
                                   recompute=recompute_shot, nreal=shot_nreal,
                                   seed=shot_seed, chunk=shot_chunk)

    if fresh or added:
        ensure_parents(path)
        # P_halo_matter is derived, not measured; keep it out of the file so it
        # can never drift out of step with f_c/f_g.
        to_save = {key: value for key, value in data.items()
                   if key != 'P_halo_matter'}
        np.savez_compressed(path, **to_save)
        print(f"\nSpectra bundle {'saved' if fresh else 'updated'}:\n  {path}")

    return derive_P_halo_matter(data)


# ==============================================================================
# 3b.  SELF-PAIR (SHOT-NOISE) SPECTRA AND THEIR SUBTRACTION
# ==============================================================================
# WHY THERE IS ANYTHING TO SUBTRACT
#     The tracer fields and the matter field are not built from independent
#     samples: delta_m = f_c delta_dm + f_g delta_gas is made of the SAME
#     particles as delta_dm and delta_gas. Crossing two fields that share
#     particles leaves a self-pair term -- every particle correlating with
#     itself -- which is a discreteness artefact, not clustering:
#
#         <delta_m delta_gas*>   = f_c P_dm_gas + f_g P_gas_gas
#                                  ... and P_gas_gas carries P^shot_gg
#         <delta_m delta_dm*>    = f_c P_dm_dm + f_g P_dm_gas
#                                  ... and P_dm_dm carries P^shot_dd
#         <delta_m delta_m*>     = f_c^2 P_dm_dm + 2 f_c f_g P_dm_gas
#                                                + f_g^2 P_gas_gas
#
#     so the self-pair content of the three truths is
#
#         P_matter_gas    : f_g   P^shot_gg
#         P_matter_dm     : f_c   P^shot_dd
#         P_matter_matter : f_c^2 P^shot_dd + f_g^2 P^shot_gg
#         P_dm_dm         :       P^shot_dd
#         P_gas_gas       :       P^shot_gg
#         P_dm_gas        :       0        (disjoint particle sets)
#         P_halo_gas/dm   :       0        (halo centres are not particles)
#
#     The last two are zero IDENTICALLY, not approximately: a self-pair term is
#     an object correlating with itself, and no dm particle is a gas particle,
#     nor is any halo centre a particle. They get no entry in the stored block
#     at all -- an array of zeros and a '_corrected' copy of the raw would just
#     be two ways of saying nothing happened.
#
#     The right-hand side of the halo-model identity contains no self-pairs at
#     all -- it is built from P_halo_x, which has none -- so scoring it against
#     an un-subtracted truth charges the identity for a floor it never claimed
#     to produce. At the high-k end of a 2048^2 grid that floor is not a
#     rounding error, and in 'matter_matter' mode it enters at full weight.
#
# HOW P^shot IS MEASURED
#     Not from a formula. The painted fields carry a TSC assignment window and
#     its aliases, the masses are not equal, and the L^2 / binning conventions
#     are this repo's own; writing down sum m^2 / (sum m)^2 * L^2 and hoping
#     would be a guess. Instead each species is painted at UNIFORMLY RANDOM
#     positions with its real per-particle masses and the auto spectrum of the
#     resulting field is taken through binned_spectrum(). A random catalogue has
#     no clustering, so in expectation that auto spectrum IS the self-pair term
#     of the real field, measured through exactly the same window, normalisation
#     and k bins. This is the same estimator measure_u_tilde.py uses for its
#     R_star / U_star correction, with the same seed.
#
# WHAT IS STORED, AND WHERE
#     Two files, with different lifetimes.
#
#     P^shot_dd and P^shot_gg depend only on (feedback, ngrid, box, k binning,
#     seed, realisations) -- NOT on the mass binning -- so they live in their own
#     small npz, shotnoise_v1_*.npz, and are shared by every --nbins / --logm-*
#     / --target run. The measurement is the expensive part (one particle-mass
#     read plus one paint + FFT per species) and it happens exactly once.
#
#     The spectra bundle then carries the whole correction, resolved against its
#     own spectra. For every spectrum with a self-pair term it holds three
#     arrays rather than one:
#
#         <key>              the raw measurement
#         <key>_selfpair     the self-pair term computed for it
#         <key>_corrected    the difference
#
#     together with P_shot_dm, P_shot_gas and the seed / realisation count that
#     produced them. So the bundle is self-describing: the correction is a
#     stored, auditable quantity rather than something re-derived from two files
#     on every run, and a bundle can be read either way after the fact -- by
#     this script, by measure_u_tilde.py, or by hand -- without re-running
#     anything. Nothing is overwritten: the canonical name is always the RAW
#     spectrum, which is also what keeps a bundle readable by code written
#     before any of this existed.
#
#     Which variant a given RUN works with is a separate decision, taken once by
#     use_self_pair_corrected() and controlled by --no-shot-subtract. What is on
#     disk does not depend on the command line.
# ------------------------------------------------------------------------------

# Number of z cells used by _paint_uniform_random(). See the note there; 3 is
# the smallest value that is safe against the _rightwrap index range.
_NZ_PAINT = 3

SHOT_SPECIES = ('dm', 'gas')


def shot_path(ngrid, nkbins, nreal=SHOT_NREAL, seed=SHOT_SEED):
    """Cache for the self-pair spectra, beside the spectra bundle.

    The mass binning is deliberately absent from the name: P^shot depends on the
    particles and the grid, not on how halos were binned, so one file serves
    every bundle built on the same grid and k binning. Everything that DOES
    change the numbers is in the name, so a stale file can never be picked up.
    """
    nk = 'default' if nkbins is None else str(nkbins)
    stem = (f'shotnoise_v1_{FEEDBACK}'
            f'_ngrid{ngrid}_nk{nk}_nreal{nreal}_seed{seed}.npz')
    return DATA_ROOT / SIM_NAME / 'pme_inputs' / stem


def _paint_uniform_random(mass, ngrid, nthread, rng, chunk=SHOT_CHUNK):
    """Mass of a uniformly random catalogue on the 2-D grid (column sum).

    NOT a call to tsc_parallel with a 2-D grid. In abacusutils 2.1.2 the 2-D
    branch of _tsc_scatter is broken: it sets izw = int16(0) but still writes
    density[ix, iy, izw], i.e. three indices into a two-dimensional array, so
    numba fails to type it. We therefore paint into a genuine 3-D grid with
    _NZ_PAINT cells along z and sum over z. The TSC weights in z sum to unity
    for every particle, so the column sum is EXACTLY the 2-D result and is
    independent of _NZ_PAINT -- not an approximation, and not a different
    smoothing. _NZ_PAINT = 1 would be the obvious choice but is unsafe:
    iz = round(z/box) can be 1, and then izp1 = _rightwrap(2, 1) = 1 indexes
    past the end of a size-1 axis, which numba does not bounds-check.

    The random positions are generated in chunks and painted straight into the
    shared accumulator, so peak memory is chunk * 12 bytes rather than one
    float32 (N, 3) array for the whole species. Weights are divided by the
    GLOBAL mean particle mass (not the chunk's) and the mean restored at the
    end, so the float32 accumulation happens on numbers of order unity and the
    chunking cannot change the answer.
    """
    # Left in its native dtype on purpose: an .astype(float64) of a species'
    # whole mass array is one of the largest allocations this pipeline could
    # make, and nothing here needs it. Only the reductions are done in float64.
    mass = np.asarray(mass)
    m_mean = float(mass.mean(dtype=np.float64)) if mass.size else 1.0
    if not np.isfinite(m_mean) or m_mean == 0.0:
        m_mean = 1.0

    grid = np.zeros((ngrid, ngrid, _NZ_PAINT), dtype=np.float32)
    n_p = mass.size
    step = n_p if chunk is None else max(1, int(chunk))
    for a in range(0, n_p, step):
        b = min(a + step, n_p)
        pos = rng.uniform(0.0, BOX, size=(b - a, 3)).astype(np.float32)
        # uniform() is half-open in float64, but the float32 cast can round up
        # to exactly BOX; tsc wants [0, L). It also wraps pos IN PLACE, which is
        # harmless here because pos is a temporary.
        pos[pos >= BOX] -= np.float32(BOX)
        tsc_parallel(pos, grid, BOX,
                     weights=np.ascontiguousarray(mass[a:b] / m_mean,
                                                  dtype=np.float32),
                     nthread=(nthread or 1))
        del pos
    return grid.sum(axis=2, dtype=np.float64) * m_mean


def _mass_moments(mass, chunk=SHOT_CHUNK):
    """(sum m, sum m^2) in float64, accumulated in chunks.

    np.sum(mass ** 2) would materialise a second array the size of the whole
    species; at FLAMINGO particle counts that is worth avoiding for a two-line
    loop. Chunking also keeps the float32 case from losing the sum to
    accumulation error, since each partial sum is taken in float64.
    """
    mass = np.asarray(mass)
    n_p = mass.size
    step = n_p if chunk is None else max(1, int(chunk))
    s1 = s2 = 0.0
    for a in range(0, n_p, step):
        m = np.asarray(mass[a:min(a + step, n_p)], dtype=np.float64)
        s1 += float(m.sum())
        s2 += float(np.dot(m, m))
    return s1, s2


def measure_shot_spectra(nkbins, nreal=SHOT_NREAL, seed=SHOT_SEED,
                         chunk=SHOT_CHUNK):
    """Measure P^shot_dd and P^shot_gg in this pipeline's own convention.

    One particle-mass read per species (positions are never loaded: they are
    replaced by randoms), then nreal paint + FFT + bin passes, averaged.
    """
    nreal = max(1, int(nreal))
    print("=" * 70)
    print("Measuring self-pair (shot-noise) spectra (no cached file found)")
    print("=" * 70)

    k_grid = compute_k_grid_2d(NGRID, BOX)
    particles_file = get_particle_file_path(FEEDBACK, sim_name=SIM_NAME)
    rng = np.random.default_rng(seed)

    out = dict(ngrid=NGRID, box=BOX, feedback=FEEDBACK, redshift=Z_EVAL,
               nreal=nreal, seed=seed)
    for species in SHOT_SPECIES:
        print(f"\n[{species}] loading particle masses ...")
        part = load_particle_properties(particles_file, species,
                                        requested=('mass',),
                                        sim_name=SIM_NAME, Lbox=BOX)
        mass = np.asarray(part['mass'])
        del part
        gc.collect()

        n_p = mass.size
        M_tot, M2 = _mass_moments(mass, chunk=chunk)
        print(f"    {n_p:.4e} particles, sum m = {M_tot:.6e}, "
              f"sum m^2 / (sum m)^2 = {M2 / M_tot ** 2:.6e}")

        acc = None
        for r in range(nreal):
            print(f"[{species}] painting random realisation {r + 1}/{nreal} ...")
            dens = _paint_uniform_random(mass, NGRID, NTHREAD, rng, chunk=chunk)
            # Exactly the normalisation the real fields use: rho / rhobar - 1,
            # with rhobar the GLOBAL mean of this species per cell.
            delta = dens / (M_tot / NGRID ** 2) - 1.0
            del dens
            fft_r = compute_2d_fft(delta, NGRID)
            del delta
            k_bins, k_center, Pk = binned_spectrum(np.abs(fft_r) ** 2,
                                                   k_grid, nkbins)
            del fft_r
            gc.collect()
            acc = Pk if acc is None else acc + Pk

        P_shot = acc / float(nreal)
        out[f'P_shot_{species}'] = P_shot
        out[f'n_particles_{species}'] = float(n_p)
        out[f'M_tot_{species}'] = M_tot
        out[f'M2_{species}'] = M2
        out['k_bins'] = np.asarray(k_bins)
        out['k_center'] = np.asarray(k_center)

        # Convention check, informational only. IF compute_2d_fft normalises so
        # that a Poisson field has <|delta_k|^2> = sum m^2 / (sum m)^2, then the
        # low-k plateau should sit at L^2 sum m^2 / (sum m)^2. A ratio far from
        # unity does not invalidate anything below -- the MEASURED curve is what
        # gets subtracted, precisely so that no such assumption is needed -- but
        # it does say the convention is not the naive one, which is worth
        # knowing before quoting P^shot in a paper.
        expect = BOX ** 2 * M2 / M_tot ** 2
        lo = np.isfinite(P_shot) & (k_center < 0.3)
        if np.any(lo):
            print(f"    P^shot_{species}{species}: {np.nanmedian(P_shot):.6e} "
                  f"(median over k), {np.nanmedian(P_shot[lo]):.6e} at k < 0.3")
            print(f"    convention check, plateau / (L^2 sum m^2/(sum m)^2) = "
                  f"{np.nanmedian(P_shot[lo]) / expect:.4f} "
                  f"(1 = the naive normalisation)")
        del mass
        gc.collect()

    return out


_SHOT_REQUIRED_KEYS = ('k_center', 'P_shot_dm', 'P_shot_gas')


def load_or_measure_shot(nkbins, recompute=False, nreal=SHOT_NREAL,
                         seed=SHOT_SEED, chunk=SHOT_CHUNK, allow_measure=True):
    """Load the cached self-pair spectra, otherwise measure and write them.

    With allow_measure=False a missing cache returns None instead of triggering
    the (expensive) measurement -- used when the subtraction is switched off and
    the spectra are wanted only for the diagnostic printout.
    """
    path = shot_path(NGRID, nkbins, nreal=nreal, seed=seed)

    if path.exists() and not recompute:
        print(f"Loading cached self-pair spectra:\n  {path}")
        with np.load(path, allow_pickle=False) as f:
            shot = {key: f[key] for key in f.files}
        missing = [key for key in _SHOT_REQUIRED_KEYS if key not in shot]
        if missing:
            print(f"  File is missing {missing}; re-measuring.")
        else:
            return shot

    if recompute and path.exists():
        print("--recompute-shot given: ignoring the cached self-pair spectra.")
    if not allow_measure:
        print(f"No cached self-pair spectra at\n  {path}\n  (not measuring them: "
              f"the subtraction is switched off)")
        return None

    shot = measure_shot_spectra(nkbins, nreal=nreal, seed=seed, chunk=chunk)
    ensure_parents(path)
    np.savez_compressed(path, **shot)
    print(f"\nSelf-pair spectra saved:\n  {path}")
    return shot


def self_pair_terms(data, shot):
    """The self-pair content of every spectrum that HAS one, keyed by its name.

    This is the whole physics of section 3b in one table. f_c and f_g are the
    bundle's own particle-mass fractions, so the weights here are consistent
    with the delta_m that was actually built.

    P_dm_gas is deliberately absent, and so are P_halo_gas / P_halo_dm. Those
    correlate DISJOINT point sets -- no dm particle is also a gas particle, and
    no halo centre is a particle -- so no object ever pairs with itself and
    there is no self-pair term to write down. Not "a term that happens to be
    small", and not "a term we are choosing to neglect": zero identically. They
    are left out of the block entirely rather than stored as arrays of zeros,
    so that the presence of a '<key>_corrected' array always means a correction
    was actually made.

    (They still carry shot VARIANCE, of course -- a finite sample is noisy. But
    variance is not bias, and there is nothing to subtract from the mean.)
    """
    f_c, f_g = float(data['f_c']), float(data['f_g'])
    P_dd = np.asarray(shot['P_shot_dm'], dtype=float)
    P_gg = np.asarray(shot['P_shot_gas'], dtype=float)
    return {
        'P_matter_gas': f_g * P_gg,
        'P_matter_dm': f_c * P_dd,
        'P_matter_matter': f_c ** 2 * P_dd + f_g ** 2 * P_gg,
        'P_dm_dm': P_dd,
        'P_gas_gas': P_gg,
    }


def has_self_pair_block(data):
    """True if the bundle already carries the stored triplets."""
    return all(f'{key}_selfpair' in data and f'{key}_corrected' in data
               for key in self_pair_keys(data))


def self_pair_keys(data):
    """Which of the bundle's spectra have a self-pair term at all.

    See self_pair_terms(): P_dm_gas and the halo cross spectra are not on this
    list because they cross disjoint point sets and their term is identically
    zero.
    """
    return [key for key in ('P_matter_gas', 'P_matter_dm', 'P_matter_matter',
                            'P_dm_dm', 'P_gas_gas')
            if key in data]


# Written by an earlier version of this file, which stored an all-zeros block
# for P_dm_gas. Purged on sight so that a bundle never carries a '_corrected'
# array that corrects nothing.
_LEGACY_BLOCK_KEYS = ('P_dm_gas_selfpair', 'P_dm_gas_corrected')


def ensure_self_pair_block(data, nkbins, allow_measure=True, recompute=False,
                           nreal=SHOT_NREAL, seed=SHOT_SEED, chunk=SHOT_CHUNK):
    """Add the stored self-pair block to a bundle dict, in place.

    For every spectrum that has a self-pair term the bundle ends up carrying
    three arrays rather than one:

        <key>              the raw measurement, exactly as before
        <key>_selfpair     the self-pair term that was computed for it
        <key>_corrected    the difference, <key> - <key>_selfpair

    plus the two underlying spectra P_shot_dm and P_shot_gas and the metadata
    (seed, realisations) that produced them. Nothing is overwritten and nothing
    is thrown away, so a bundle can be read either way after the fact and the
    correction is auditable from the file alone rather than only reproducible
    by re-running this script.

    Returns True if the block was written or rewritten (so the caller knows the
    bundle on disk is now stale), False if it was already there and current.
    """
    fresh = not has_self_pair_block(data)
    legacy = [key for key in _LEGACY_BLOCK_KEYS if key in data]
    if not (fresh or recompute):
        if legacy:
            for key in legacy:
                data.pop(key)
            print(f"  dropped {', '.join(legacy)} from the block: P_dm_gas "
                  f"crosses disjoint point sets and has no self-pair term")
            return True
        n_r = int(data.get('selfpair_nreal', 0))
        s_d = int(data.get('selfpair_seed', 0))
        print(f"  self-pair block present in the bundle "
              f"(seed {s_d}, {n_r} realisation{'s' if n_r != 1 else ''})")
        return False

    if not allow_measure and fresh:
        print("  bundle carries no self-pair block, and --no-shot-subtract "
              "means it is not worth measuring one now.")
        return False

    shot = load_or_measure_shot(nkbins, recompute=recompute, nreal=nreal,
                               seed=seed, chunk=chunk, allow_measure=True)

    # The self-pair spectra live in their own file, keyed on the grid and the k
    # binning but NOT on the mass binning, so re-binning the halos costs nothing
    # here. That does mean the two files can in principle disagree, hence:
    k_c = np.asarray(data['k_center'], dtype=float)
    k_s = np.asarray(shot['k_center'], dtype=float)
    if k_s.shape != k_c.shape or not np.allclose(k_s, k_c, rtol=1e-10, atol=0.0):
        raise SystemExit(
            "The cached self-pair spectra are on a different k binning from the "
            "spectra bundle:\n"
            f"  bundle: {k_c.size} bins, k = [{k_c[0]:.5f}, {k_c[-1]:.4f}]\n"
            f"  shot:   {k_s.size} bins, k = [{k_s[0]:.5f}, {k_s[-1]:.4f}]\n"
            "Re-run with --recompute-shot (they must share --nkbins and the grid)."
        )

    terms = self_pair_terms(data, shot)
    for key in self_pair_keys(data):
        raw = np.asarray(data[key], dtype=float)
        data[key] = raw                       # canonical name stays RAW
        data[f'{key}_selfpair'] = terms[key]
        data[f'{key}_corrected'] = raw - terms[key]
    for key in _LEGACY_BLOCK_KEYS:
        data.pop(key, None)

    data['P_shot_dm'] = np.asarray(shot['P_shot_dm'], dtype=float)
    data['P_shot_gas'] = np.asarray(shot['P_shot_gas'], dtype=float)
    data['selfpair_seed'] = int(shot.get('seed', seed))
    data['selfpair_nreal'] = int(shot.get('nreal', nreal))

    print(f"  self-pair block {'built' if fresh else 'rebuilt'} for "
          f"{', '.join(self_pair_keys(data))}")
    print("  no block for P_dm_gas, P_halo_gas, P_halo_dm: those cross disjoint "
          "point sets, so their self-pair term is identically zero")
    _check_closure_of_block(data)
    return True


def use_self_pair_corrected(data, subtract=True):
    """Choose which stored variant the rest of the run works with, in place.

    With subtract=True the canonical names are pointed at the '_corrected'
    arrays and the raw ones preserved under '<key>_raw'; with subtract=False the
    bundle is left as loaded. Either way every downstream consumer -- the
    reconstruction ratio, the missing-mass check, measured_bias_per_bin,
    Experiment A, the saved payloads -- reads the canonical names and needs no
    correction flag threaded through it.

    P_halo_matter is rebuilt afterwards for symmetry, though it needs no
    correction: it is a combination of the two halo cross spectra, neither of
    which has a self-pair term.

    Returns True if the corrected variant is in force.
    """
    available = has_self_pair_block(data)
    data['self_pairs_removed'] = bool(subtract and available)

    if not available:
        print("[shot] no self-pair block in this bundle; nothing removed.")
        return False

    _report_self_pair_fractions(data, removed=bool(subtract))

    if not subtract:
        print("[shot] self-pair block present but NOT applied "
              "(--no-shot-subtract): the raw spectra are in force.")
        return False

    for key in self_pair_keys(data):
        data[f'{key}_raw'] = np.asarray(data[key], dtype=float)
        data[key] = np.asarray(data[f'{key}_corrected'], dtype=float)

    data.pop('P_halo_matter', None)
    derive_P_halo_matter(data)

    print("[shot] corrected spectra in force for "
          f"{', '.join(self_pair_keys(data))}.")
    return True


def _report_self_pair_fractions(data, removed=True):
    """Print what fraction of each raw spectrum the self-pair term is.

    Always quoted against the RAW measurement, so the number means "this much of
    what came out of the box was discreteness", whichever variant is in force.
    """
    k = np.asarray(data['k_center'], dtype=float)
    k_Ny = float(data['k_Nyquist'])
    keys = [key for key in ('P_matter_gas', 'P_matter_dm', 'P_matter_matter',
                            'P_dm_dm') if f'{key}_selfpair' in data]
    ks = [kk for kk in (0.1, 0.5, 1.0, 2.0, 3.0, 5.0) if kk <= k[-1]]

    print("\n[shot] self-pair fraction of the raw measured spectrum:")
    print("        k =  " + "".join(f"{kk:>9.2f}" for kk in ks))
    worst = {}
    for key in keys:
        raw = np.asarray(data.get(f'{key}_raw', data[key]), dtype=float)
        term = np.asarray(data[f'{key}_selfpair'], dtype=float)
        with np.errstate(divide='ignore', invalid='ignore'):
            frac = term / raw
        print(f"  {key:16s}"
              + "".join(f"{frac[int(np.argmin(np.abs(k - kk)))]:9.3f}" for kk in ks))
        below = np.isfinite(frac) & (k < k_Ny)
        if np.any(below):
            worst[key] = float(np.nanmax(np.abs(frac[below])))

    for key, w in worst.items():
        if w > 0.2:
            print(f"  NOTE: {key} is up to {w:.0%} self-pairs below k_Nyquist. "
                  f"{'Removed below' if removed else 'NOT being removed'}; either "
                  f"way, do not read the high-k end of that curve as clustering.")


def _check_closure_of_block(data):
    """The quadratic closure must survive the subtraction, and it does exactly.

        P_mm - (f_c^2 P^s_dd + f_g^2 P^s_gg)
            = f_c^2 (P_dd - P^s_dd) + 2 f_c f_g P_dg + f_g^2 (P_gg - P^s_gg)

    Note what the cross term does here: NOTHING. P_dm_gas appears on the right
    with no correction of its own, because dm and gas particles are disjoint
    sets and it has no self-pair term. That is precisely why the identity
    survives -- the subtracted pieces on the two sides are f_c^2 P^s_dd +
    f_g^2 P^s_gg either way. So this is the same check measure_spectra runs on
    the raw spectra, re-run on the corrected ones, and if it now fails the
    self-pair WEIGHTS are wrong rather than the fields. Run at build time, so a
    bad block never reaches disk unremarked.
    """
    needed = ('P_matter_matter_corrected', 'P_dm_dm_corrected',
              'P_gas_gas_corrected', 'P_dm_gas')
    if not all(key in data for key in needed):
        return
    f_c, f_g = float(data['f_c']), float(data['f_g'])
    rhs = (f_c ** 2 * data['P_dm_dm_corrected']
           + 2.0 * f_c * f_g * data['P_dm_gas']      # uncorrected, by construction
           + f_g ** 2 * data['P_gas_gas_corrected'])
    with np.errstate(divide='ignore', invalid='ignore'):
        dev = np.abs(np.asarray(data['P_matter_matter_corrected']) / rhs - 1.0)
    dev_max = float(np.nanmax(dev[np.isfinite(dev)]))
    print(f"  closure of the corrected spectra "
          f"|P_mm / (f_c^2 P_dd + 2 f_c f_g P_dg + f_g^2 P_gg) - 1| = "
          f"{dev_max:.2e} max")
    if dev_max > 1e-6:
        print("  WARNING: the corrected matter auto does not close on its "
              "corrected components. Check f_c, f_g and the self-pair weights.")


# ==============================================================================
# 4.  THE RECONSTRUCTION  (this is the part that matters)
# ==============================================================================
#   P_matter_x(k) = (1/rhobar_m) sum_i n_i M_i u_m(k|M_i) P_halo_x(k; M_i)
# ------------------------------------------------------------------------------
def reconstruct_P_matter_x(k, P_halo_x, M_i, n_i, z):
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
    c_i = concentration(M_i, z)
    u_im = np.array([u_nfw(k, M_i[i], c_i[i]) for i in range(M_i.size)])
    weight = (n_i * M_i)[:, None] / rhobar_m          # (nbins, 1)
    return np.sum(weight * u_im * P_halo_x, axis=0)       # sum over mass bins



# ==============================================================================
# 6.  EXPERIMENT A  --  extrapolating P_halo_x(k|M) below M_min
# ==============================================================================
# The baseline missing-mass correction applied in main() is the flat template
#
#     Delta_P(k) = f_u * P_halo_x(k | M_min),                                    (*)
#
# i.e. "the unresolved mass cross-correlates with e exactly like the lowest
# resolved bin does". Experiment A replaces (*) by the integral it is standing
# in for,
#
#     Delta_P(k) = (1/rhobar_m) int_0^Mmin dM M n(M) u_m(k|M) P_halo_x(k|M),
#
# with the unmeasurable integrand P_halo_x(k|M) supplied by an extrapolation off a
# reference bin h_r -- the lowest bin we do have measurements for:
#
#     P_halo_x(k|M)  ~  [P_h_r_e(k) / P_h_r_c(k)] * P_hc(k|M).
#
# c is cold dark matter, which we trust; the bracket freezes the baryonic
# response in at the reference scale.
#
# WHAT ACTUALLY GETS COMPUTED
#     Define the mass transfer function
#         T(k,M) = P_hc(k|M) / P_hc(k|M_r)                          [dimensionless]
#     so that P_halo_x(k|M) = T(k,M) * P_halo_x(k|M_r) and the measured P_h_r_c cancels
#     out of the final expression entirely. Then
#
#         Delta_P(k) = f_u * <u_m(k|M) T(k,M)>_w * P_halo_x(k|M_r)
#                    = f_u * S(k) * P_halo_x(k|M_r),
#
#     which is (*) multiplied by a single shape factor S(k). The old formula is
#     the S = 1 special case, and in the limit u_m -> 1 with both 1-halo terms
#     dropped S -> <b>_w / b(M_r), which is exactly Equation (eq:plateau) of the
#     notes. Experiment A is thus a strict generalisation, not a competitor.
#
# THE FOUR WAYS OF SUPPLYING T
#     'flat'      T = 1 and u_m = 1. Reproduces (*) bit for bit. The control.
#     'simhc'     T = P_halo_dm(k;M_i) / P_halo_dm(k;M_r), straight from the measured
#                 halo-DM cross spectra. This is the literal reading of the
#                 ansatz with "P_hc taken from simulations", and it needs no
#                 cosmology, no mass function and no bias model. Only available
#                 in validation mode, where the "unresolved" bins are really
#                 measured -- which is the point: it tests the ansatz ALONE.
#     'bias'      T = b(M)/b(M_r) from Tinker+10, u_m kept. The 2-halo limit.
#     'halomodel' T from the full 1-halo + 2-halo model.
#
# THE MASS INTEGRAL, AND WHY ITS AMPLITUDE IS NOT TAKEN FROM THE MODEL
#     f_u from an analytic HMF is badly determined: integrating Tinker+08 from
#     10^10 rather than 10^6 Msun/h changes it by a factor of two, because the
#     low-mass end contributes mass logarithmically and never converges. The
#     SHAPE S(k), by contrast, moves by only a few per cent over the same range,
#     since it is a ratio of two integrals against the same weight. So the
#     default (--fu catalog) takes the amplitude from the measured catalogue
#     deficit f_u^cat = 1 - sum_i f_i and the shape from the model. --fu model
#     uses the HMF for both and is there to expose exactly this instability.
#
#     Note that f_u^cat is NOT the f_u of the integral: it also collects mass
#     outside r200b of resolved halos, halos above the top bin, and the
#     dm+gas-vs-total bookkeeping residue. Experiment A improves the treatment
#     of component (i) only, and carries the rest along with the same template.
# ------------------------------------------------------------------------------

DELTA_C = 1.686               # spherical-collapse threshold
DELTA_HALO = 200.0            # w.r.t. MEAN background: matches m200b and r200m_of_M

# Sum of neutrino masses [eV] handed to CAMB. The parameter file does not carry
# it, and Omega_m there is the TOTAL matter density, so setting a non-zero mnu
# here without also adjusting omch2 would change Omega_m. FLAMINGO's fiducial
# (DES Y3 3x2pt+all) uses 0.06 eV; set it here if you want that, and note that
# it suppresses P_lin by ~ a few per cent at the k of interest. It cancels to
# first order in T(k,M), which is a ratio, so the default of 0 is safe for
# Experiment A even though it is not the simulation's cosmology.
NEUTRINO_MASS_EV = 0.0

# Integration range for the unresolved-mass integral, log10(M / [Msun/h]).
INT_LOGM_LO = 8.0             # see the f_u instability note above
INT_NODES = 256

# CAMB table extent. It has to bracket everything colossus will ask for when it
# builds its sigma(R) interpolator, which reaches well beyond the k of the box.
CAMB_KMIN, CAMB_KMAX, CAMB_NK = 1e-6, 3e2, 1000


def _astropy_cosmology():
    """FlatLambdaCDM built from the simulation's own parameter file."""
    return FlatLambdaCDM(H0=100.0 * H_LITTLE, Om0=OMEGA_M,
                         Ob0=SIM_PARAMS.get('omega_b'),
                         Tcmb0=SIM_PARAMS.get('tcmb0', 2.725))


def critical_density_h_units():
    """rho_crit(z=0) in (Msun/h)/(Mpc/h)^3, from astropy.

    The h's cancel, so this is numerically the same as rho_crit in
    h^2 Msun/Mpc^3, i.e. the familiar 2.77536627e11.
    """
    ac = _astropy_cosmology()
    return float((ac.critical_density0 / ac.h ** 2).to(u.Msun / u.Mpc ** 3).value)


# Now that the helper exists, fix the densities the whole module works in.
RHO_CRIT_0 = critical_density_h_units()
rhobar_m = OMEGA_M * RHO_CRIT_0


def camb_linear_power_table():
    """P_lin(k, z=0) from CAMB, cached on disk as a colossus-readable table.

    Returns the path to a two-column file of log10(k [h/Mpc]), log10(P
    [(Mpc/h)^3]), normalised so that sigma_8 equals the value in the parameter
    file. CAMB is run with an arbitrary A_s and the spectrum rescaled by
    (sigma8_target / sigma8_camb)^2, which is exact in linear theory and avoids
    having to solve for A_s.

    The table is what makes the whole stack self-consistent: colossus computes
    sigma(M), the Tinker+08 mass function and the Tinker+10 bias from this same
    spectrum rather than from its built-in Eisenstein & Hu approximation.
    """

    ob, ns = SIM_PARAMS.get('omega_b'), SIM_PARAMS.get('n_s')
    s8, tcmb = SIM_PARAMS.get('sigma8'), SIM_PARAMS.get('tcmb0', 2.725)
    tag = (f"{OMEGA_M:.6f}_{ob:.6f}_{H_LITTLE:.6f}_{s8:.6f}_{ns:.6f}"
           f"_{tcmb:.4f}_{NEUTRINO_MASS_EV:.4f}_{CAMB_KMAX:.0f}")
    path = DATA_ROOT / SIM_NAME / 'pme_inputs' / f'plin_camb_{tag}.txt'
    if path.exists():
        print(f"  [camb] reusing cached linear P(k): {path.name}")
        return path

    print("  [camb] running CAMB for the linear power spectrum (one-off, ~20 s) ...")
    pars = camb.set_params(
        H0=100.0 * H_LITTLE,
        ombh2=ob * H_LITTLE ** 2,
        omch2=(OMEGA_M - ob) * H_LITTLE ** 2,
        ns=ns, TCMB=tcmb, mnu=NEUTRINO_MASS_EV, omk=0.0, As=2.0e-9,
    )
    pars.set_matter_power(redshifts=[0.0], kmax=CAMB_KMAX * 1.7)
    pars.NonLinear = camb.model.NonLinear_none
    results = camb.get_results(pars)

    s8_camb = float(results.get_sigma8_0())
    kh, _, pk = results.get_matter_power_spectrum(
        minkh=CAMB_KMIN, maxkh=CAMB_KMAX, npoints=CAMB_NK)
    pk0 = pk[0] * (s8 / s8_camb) ** 2
    print(f"  [camb] sigma8 = {s8_camb:.5f} from A_s = 2e-9, rescaled to "
          f"{s8:.5f} (factor {(s8 / s8_camb) ** 2:.5f} in P)")

    ensure_parents(path)
    np.savetxt(path, np.column_stack([np.log10(kh), np.log10(pk0)]),
               header='log10(k [h/Mpc])  log10(P [(Mpc/h)^3])')
    print(f"  [camb] table written to {path}")
    return path


# ------------------------------------------------------------------------------
# 6a.  The cosmology backend
# ------------------------------------------------------------------------------
class _ColossusBackend:
    """Tinker+08 n(M) and Tinker+10 b(M) via colossus, on a CAMB spectrum."""

    name = 'colossus'

    def __init__(self, z, power_spectrum='camb'):
        self.z = float(z)
        self.mdef = f'{int(DELTA_HALO)}m'

        params = dict(flat=True, H0=100.0 * H_LITTLE, Om0=OMEGA_M,
                      Ob0=SIM_PARAMS.get('omega_b'),
                      sigma8=SIM_PARAMS.get('sigma8'), ns=SIM_PARAMS.get('n_s'),
                      Tcmb0=SIM_PARAMS.get('tcmb0', 2.725), relspecies=False)
        self.cosmo = colossus_cosmology.setCosmology(f'{SIM_NAME}_hatf', **params)

        if power_spectrum == 'camb':
            self.ps_args = {'model': 'table', 'path': str(camb_linear_power_table())}
            self.ps_label = 'CAMB'
        elif power_spectrum == 'eisenstein98':
            self.ps_args = {'model': 'eisenstein98'}
            self.ps_label = 'Eisenstein & Hu 1998 (colossus built-in)'
        else:
            raise ValueError(f"unknown power spectrum {power_spectrum!r}")

    def Plin(self, k):
        return self.cosmo.matterPowerSpectrum(np.atleast_1d(np.asarray(k, float)),
                                              self.z, **self.ps_args)

    def dndM(self, M):
        M = np.atleast_1d(np.asarray(M, float))
        dndlnM = colossus_mf.massFunction(
            M, self.z, mdef=self.mdef, model='tinker08',
            q_in='M', q_out='dndlnM', ps_args=self.ps_args)
        return dndlnM / M

    def bias(self, M):
        # haloBias() does not forward ps_args to the fitting function, so the
        # peak height is computed explicitly on the CAMB spectrum and fed to
        # haloBiasFromNu. Going through haloBias() directly would silently use
        # colossus's built-in Eisenstein & Hu sigma(M) instead.
        nu = colossus_peaks.peakHeight(np.atleast_1d(np.asarray(M, float)),
                                       self.z, ps_args=self.ps_args)
        return colossus_bias.haloBiasFromNu(nu, z=self.z, mdef=self.mdef,
                                            model='tinker10')

    def describe(self):
        return (f"colossus: P_lin from {self.ps_label}, Tinker+08 dn/dM and "
                f"Tinker+10 b(M) at Delta = {self.mdef}")


class HaloModel:
    """Halo-model P_hc(k|M), on top of an interchangeable cosmology backend.

    Only the dimensionless ratio T(k,M) = P_hc(k|M)/P_hc(k|M_r) is ever used
    downstream, so the absolute normalisation of anything here cancels. That is
    why the neutrino treatment does not matter either, and why the choice of
    linear spectrum moves S(k) far less than it moves P_lin itself.
    """

    def __init__(self, z, power_spectrum='camb'):
        self.z = float(z)
        self.be = _ColossusBackend(z, power_spectrum)
        self._I2h_cache = {}

        print(f"  [halo model] {self.be.describe()}")
        print(f"  [halo model] Omega_m={OMEGA_M:.4f} Omega_b="
              f"{SIM_PARAMS.get('omega_b'):.4f} h={H_LITTLE:.4f} "
              f"sigma8={SIM_PARAMS.get('sigma8'):.4f} n_s={SIM_PARAMS.get('n_s'):.4f}")
        print(f"  [halo model] at z={self.z:g}: b(1e12)="
              f"{float(self.bias(1e12)[0]):.4f}, "
              f"dn/dlnM(1e12)={float(self.dndM(1e12)[0]) * 1e12:.4e}")

    # thin pass-throughs, so callers never touch the backend directly
    def Plin(self, k):
        return self.be.Plin(k)

    def dndM(self, M):
        return self.be.dndM(M)

    def bias(self, M):
        return self.be.bias(M)

    def I2h(self, k):
        """int dM n(M) b(M) W_m(k,M), normalised so that I(k -> 0) = 1.

        The normalisation IS the bias consistency relation of the notes;
        imposing it by hand is what keeps a mass function that does not
        integrate to rhobar_m from leaking into the 2-halo amplitude.
        """
        key = (k.shape, float(k[0]), float(k[-1]))
        if key in self._I2h_cache:
            return self._I2h_cache[key]
        M = np.logspace(6.0, 16.0, 300)
        w = M * self.dndM(M) * self.bias(M) * M / rhobar_m
        u = np.array([u_nfw(k, m, concentration(m, self.z)) for m in M])
        num = np.trapezoid(w[:, None] * u, np.log(M), axis=0)
        out = num / np.trapezoid(w, np.log(M))
        self._I2h_cache[key] = out
        return out

    def P_hc(self, k, M):
        """Halo-CDM cross spectrum, (nM, nk). Units are arbitrary but internally
        consistent: only ratios of this quantity are ever used."""
        M = np.atleast_1d(np.asarray(M, dtype=float))
        u = np.array([u_nfw(k, m, concentration(m, self.z)) for m in M])
        one_halo = (M / rhobar_m)[:, None] * u
        two_halo = self.bias(M)[:, None] * (self.Plin(k) * self.I2h(k))[None, :]
        return one_halo + two_halo


def _trapz_weights(x):
    """Weights w such that sum(w * f) == trapezoid(f, x)."""
    w = np.empty_like(x)
    w[1:-1] = 0.5 * (x[2:] - x[:-2])
    w[0] = 0.5 * (x[1] - x[0])
    w[-1] = 0.5 * (x[-1] - x[-2])
    return w


def transfer_T(k, M_nodes, M_ref, mode, hm=None, T_sim=None):
    """Mass transfer function T(k,M) = P_hc(k|M) / P_hc(k|M_r), shape (nM, nk).

    'flat'      -> 1
    'bias'      -> b(M)/b(M_r), the 2-halo limit
    'halomodel' -> full 1-halo + 2-halo ratio
    'simhc'     -> T_sim, supplied by the caller from measured P_halo_dm
    """
    nM, nk = M_nodes.size, k.size
    if mode == 'flat':
        return np.ones((nM, nk))
    if mode == 'simhc':
        if T_sim is None:
            raise ValueError("mode 'simhc' needs measured P_halo_dm ratios (validation "
                             "mode only)")
        return T_sim
    if hm is None:
        raise ValueError(f"mode {mode!r} needs a HaloModel instance")
    if mode == 'bias':
        return (hm.bias(M_nodes) / float(hm.bias(np.array([M_ref]))[0]))[:, None] \
            * np.ones((1, nk))
    if mode == 'halomodel':
        return hm.P_hc(k, M_nodes) / hm.P_hc(k, np.array([M_ref]))[0][None, :]
    raise ValueError(f"unknown extrapolation mode {mode!r}")


def delta_P_experimentA(k, P_halo_x_ref, M_ref, M_min, mode, hm=None, z=None,
                        cat_M=None, cat_w=None, T_sim=None,
                        int_logm_lo=INT_LOGM_LO, n_nodes=INT_NODES,
                        f_u=None, use_um=True):
    """The Experiment A correction.

        Delta_P(k) = f_u * S(k) * P_halo_x(k|M_r),   S(k) = <u_m(k|M) T(k,M)>_w

    Two ways of supplying the mass weight w = M n(M):

    cat_M / cat_w given
        The nodes and weights come from the halo CATALOGUE (w_i = n_i M_i).
        Used in validation mode, where the pseudo-unresolved bins are really
        measured, so that n(M) is exact and only T is under test.

    cat_M / cat_w absent
        Nodes are log-spaced on [10^int_logm_lo, M_min] and w = M n(M) comes
        from Tinker+08.

    f_u given
        Overrides the model's own mass integral, i.e. the amplitude is taken
        from the catalogue deficit and only the SHAPE S(k) from the model.

    Returns a dict with S, Delta_P, f_u used, f_u from the integral, and the
    per-node T for inspection.
    """
    if cat_M is not None:
        M_nodes = np.asarray(cat_M, dtype=float)
        w = np.asarray(cat_w, dtype=float)
    else:
        M_nodes = np.logspace(int_logm_lo, np.log10(M_min), n_nodes)
        lnM = np.log(M_nodes)
        w = M_nodes * hm.dndM(M_nodes) * M_nodes * _trapz_weights(lnM)

    keep = w > 0
    M_nodes, w = M_nodes[keep], w[keep]
    if T_sim is not None:
        T_sim = np.asarray(T_sim)[keep]

    T = transfer_T(k, M_nodes, M_ref, mode, hm=hm, T_sim=T_sim)

    if use_um and mode != 'flat':
        u = np.array([u_nfw(k, m, concentration(m, z)) for m in M_nodes])
    else:
        u = np.ones((M_nodes.size, k.size))

    S = np.sum(w[:, None] * u * T, axis=0) / np.sum(w)
    f_u_int = float(np.sum(w) / rhobar_m)
    f_u_used = f_u_int if f_u is None else float(f_u)

    return dict(S=S, Delta_P=f_u_used * S * P_halo_x_ref, f_u_used=f_u_used,
                f_u_integral=f_u_int, M_nodes=M_nodes, w=w, T=T, mode=mode)


def measured_bias_per_bin(data, kmax_fit=0.08):
    """Large-scale halo bias per mass bin, b_i = <P_halo_dm / P_dm,dm> over k < kmax.

    Purely a diagnostic: it is what the Tinker+10 curve is checked against
    before b(M) is trusted below M_min. Averaging over the lowest k bins keeps
    it in the linear regime where the ratio is genuinely constant.

    The denominator P_dm_dm is an AUTO spectrum and so carries a self-pair term;
    it arrives here already corrected if section 3b ran, which is the right
    thing -- an uncorrected denominator biases b_i low, mildly at these k and
    badly if kmax_fit is ever pushed up.
    """
    k = data['k_center']
    sel = k < kmax_fit
    if np.count_nonzero(sel) < 3:
        sel = np.zeros_like(k, dtype=bool)
        sel[:5] = True
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = data['P_halo_dm'][:, sel] / data['P_dm_dm'][None, sel]
    b = np.nanmean(np.where(np.isfinite(ratio), ratio, np.nan), axis=1)
    return b, float(k[sel][-1])


# ------------------------------------------------------------------------------
# 6b.  The driver
# ------------------------------------------------------------------------------
def run_experiment_A(data, args, tracer, tag, x_sym):
    """Run and plot Experiment A.

    Two configurations, chosen by --split-logm:

    VALIDATION (--split-logm given)
        Bins above the split are declared resolved; bins below are hidden and
        play the part of the unresolved population. Their exact contribution
            Delta_P_exact(k) = sum_hidden f_i u_m(k|M_i) P_halo_x(k;M_i)
        is known, so every extrapolation can be scored against a truth rather
        than against the final reconstruction, where a dozen other errors are
        superposed. This is the test the notes ask for.

    PRODUCTION (no split)
        Everything is resolved; the correction is applied below the catalogue
        limit and there is no exact target. Only the effect on the final
        reconstruction can be judged.
    """
    info = TRACER_INFO[tracer]
    k = data['k_center']
    P_halo_x = data[info['halo_key']]
    P_matter_x_true = data[info['truth_key']]
    # P_halo_dm is the extrapolation's CDM proxy (the P_hc of the ansatz) and is
    # the same array whatever the target is -- it is not the tracer here.
    P_halo_dm = data['P_halo_dm']
    counts = data['counts']
    M_i = data['M_mean']
    logM_cen = data['logM_cen']
    z = float(data['redshift'])
    k_Ny = float(data['k_Nyquist'])
    n_i = counts / float(data['box']) ** 3

    print("\n" + "=" * 70)
    print("EXPERIMENT A -- extrapolating P_halo_x(k|M) below M_min")
    print("=" * 70)

    occupied = counts > 0

    # --- split the catalogue ---------------------------------------------------
    if args.split_logm is not None:
        hidden = occupied & (logM_cen < args.split_logm)
        resolved = occupied & (logM_cen >= args.split_logm)
        if not np.any(hidden):
            raise SystemExit(f"--split-logm {args.split_logm} hides no occupied "
                             f"bin; nothing to validate against.")
        mode_label = 'validation'
    else:
        hidden = np.zeros_like(occupied)
        resolved = occupied
        mode_label = 'production'

    # The analytic integral must cover the SAME mass range as the exact target,
    # or the two are simply not comparable. In validation mode that range is
    # bounded below by the catalogue, not by 10^INT_LOGM_LO.
    if args.int_logm_min is not None: # practically never used?
        int_logm_lo = float(args.int_logm_min)
    elif args.split_logm is not None: # validation mode
        int_logm_lo = float(data['logM_edges'][0])
        print(f"[A] integral lower limit defaulted to the catalogue edge "
              f"logM = {int_logm_lo:.2f}, to match the exact target's range")
    else: # production mode
        int_logm_lo = INT_LOGM_LO # 8.0 or something like that

    ref = int(np.flatnonzero(resolved)[0]) if args.ref_logm is None else \
        int(np.argmin(np.abs(logM_cen - args.ref_logm)))
    if not resolved[ref]:
        raise SystemExit(f"--ref-logm {args.ref_logm} selects bin {ref}, which is "
                         f"not in the resolved set.")

    M_ref = float(M_i[ref])
    P_halo_x_ref = P_halo_x[ref]
    boundary = int(np.flatnonzero(resolved)[0])
    M_min = 10.0 ** float(data['logM_edges'][boundary])

    f_i = n_i * M_i / rhobar_m
    f_resolved = float(np.sum(f_i[resolved]))
    f_u_cat = 1.0 - f_resolved

    print(f"[A] {mode_label} mode; reference bin {ref} at logM = "
          f"{logM_cen[ref]:.2f} (<M> = {M_ref:.3e} Msun/h)")
    print(f"[A] resolved mass fraction {f_resolved:.4f}, catalogue deficit "
          f"f_u^cat = {f_u_cat:.4f}")
    if np.any(hidden):
        print(f"[A] hiding {np.count_nonzero(hidden)} bins below logM = "
              f"{args.split_logm}, carrying f = {float(np.sum(f_i[hidden])):.4f} "
              f"of the mass")

    # --- the exact target, when we have one ------------------------------------
    Delta_P_exact = None
    f_u_hidden = None
    if np.any(hidden):
        idx = np.flatnonzero(hidden)
        u_hidden = np.array([u_nfw(k, M_i[j], concentration(M_i[j], z)) for j in idx])
        Delta_P_exact = np.sum(f_i[idx][:, None] * u_hidden * P_halo_x[idx], axis=0)
        f_u_hidden = float(np.sum(f_i[idx]))
        print(f"[A] exact hidden contribution built from the measured bins; "
              f"f_u(hidden) = {f_u_hidden:.4f}")

    # --- halo model, only if a mode needs it -----------------------------------
    modes = list(args.extrap)
    if 'simhc' in modes and not np.any(hidden):
        print("[A] dropping mode 'simhc': it needs measured P_hc in the "
              "unresolved range, which only exists in validation mode.")
        modes.remove('simhc')
    # 'flat' needs it too whenever the mass integral is analytic, since that is
    # where its f_u comes from.
    needs_hm = (any(m in ('bias', 'halomodel') for m in modes)
                or (args.hmf == 'tinker' and any(m != 'simhc' for m in modes)))
    hm = None
    if needs_hm:
        print("[A] building the halo model ...")
        hm = HaloModel(z, power_spectrum=args.power_spectrum)

    # --- bias sanity check ------------------------------------------------------
    b_meas, k_fit = measured_bias_per_bin(data)
    if hm is not None:
        print(f"[A] measured vs Tinker+10 bias (from P_halo_dm/P_dm,dm, k < {k_fit:.3f}):")
        for j in np.flatnonzero(occupied)[::max(1, np.count_nonzero(occupied) // 8)]:
            print(f"      logM={logM_cen[j]:5.2f}  b_meas={b_meas[j]:6.3f}  "
                  f"b_T10={float(hm.bias(np.array([M_i[j]]))[0]):6.3f}")

    # --- the amplitude ----------------------------------------------------------
    if args.fu == 'catalog':
        f_u_amp = f_u_hidden if f_u_hidden is not None else f_u_cat
    else:
        f_u_amp = None      # let each mode use its own mass integral
    print(f"[A] amplitude source: --fu {args.fu}"
          + (f" -> f_u = {f_u_amp:.4f}" if f_u_amp is not None else
             " -> from the Tinker+08 integral, per mode"))

    # --- catalogue weights for the integral, in validation mode -----------------
    # The per-mode nodes and weights are picked inside the loop below; all that
    # is needed up here is the guard and the measured T_sim, which 'simhc' reads.
    # Note that 'simhc' only knows T at the measured bin masses, so it is always
    # evaluated on the catalogue nodes even when the other modes integrate over a
    # continuous mass function.
    T_sim = None
    if args.hmf == 'catalog' and not np.any(hidden):
        raise SystemExit("--hmf catalog needs --split-logm: outside validation "
                         "mode there is no catalogue below M_min.")
    if np.any(hidden):
        idx = np.flatnonzero(hidden)
        with np.errstate(divide='ignore', invalid='ignore'):
            T_sim_full = P_halo_dm[idx] / P_halo_dm[ref][None, :]
        T_sim = np.where(np.isfinite(T_sim_full), T_sim_full, 0.0)

    # --- run every requested mode ----------------------------------------------
    results = {}
    for mode in modes:
        use_cat = (args.hmf == 'catalog') or (mode == 'simhc')
        if use_cat and not np.any(hidden):
            continue
        if use_cat:
            idx = np.flatnonzero(hidden)
            cm, cw = M_i[idx], n_i[idx] * M_i[idx]
        else:
            cm = cw = None
        res = delta_P_experimentA(
            k, P_halo_x_ref, M_ref, M_min, mode, hm=hm, z=z,
            cat_M=cm, cat_w=cw,
            T_sim=T_sim if mode == 'simhc' else None,
            int_logm_lo=int_logm_lo, n_nodes=INT_NODES,
            f_u=f_u_amp,
        )
        results[mode] = res
        S = res['S']
        print(f"[A] mode {mode:10s}: f_u(int) = {res['f_u_integral']:.4f}, "
              f"f_u(used) = {res['f_u_used']:.4f}, "
              f"S(k_min) = {S[0]:.3f}, S(k=1) = {np.interp(1.0, k, S):.3f}, "
              f"S(k_Ny) = {S[-1]:.3g}")
        if Delta_P_exact is not None:
            with np.errstate(divide='ignore', invalid='ignore'):
                r = res['Delta_P'] / Delta_P_exact
            good = np.isfinite(r) & (k < k_Ny)
            print(f"                 Delta_P/exact: {r[good][0]:.3f} at k_min, "
                  f"{np.interp(1.0, k, r):.3f} at k=1, "
                  f"median |1-r| = {np.nanmedian(np.abs(r[good] - 1.0)):.3f}")

    # --- the bias/halomodel spread, as an error bar on the discarded term -------
    if 'bias' in results and 'halomodel' in results:
        with np.errstate(divide='ignore', invalid='ignore'):
            spread = results['halomodel']['S'] / results['bias']['S'] - 1.0
        print("[A] spread halomodel/bias - 1 (this is the size of the 1-halo term "
              "the\n    factorisation argument dropped, NOT a ranking of the two):")
        for kk in (0.1, 0.3, 1.0, 3.0):
            if kk < k[-1]:
                print(f"      k = {kk:4.1f}: {np.interp(kk, k, spread):+.3f}")
        big = k[np.abs(spread) > 0.5]
        if big.size:
            print(f"      |spread| exceeds 50% above k = {big[0]:.2f}; past that "
                  f"point neither\n      mode is controlled and only the "
                  f"--split-logm validation can decide.")

    # --- figure 1: the shape factor S(k) ---------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True, dpi=300,
                             gridspec_kw=dict(height_ratios=[1, 1], hspace=0.08))
    ax = axes[0]
    # 'flat' is the control, so it is drawn in grey-dashed rather than black:
    # black is reserved for the truth on the reconstruction panel.
    colours = {'flat': '0.35', 'simhc': 'C3', 'bias': 'C0', 'halomodel': 'C2'}
    styles = {'flat': '--', 'simhc': '-', 'bias': '--', 'halomodel': '-.'}
    for mode, res in results.items():
        ax.loglog(k, np.abs(res['S']), color=colours.get(mode, 'C4'),
                  ls=styles.get(mode, ':'), lw=1.8, label=f"$S(k)$, {mode}")
    if Delta_P_exact is not None and f_u_hidden:
        with np.errstate(divide='ignore', invalid='ignore'):
            S_exact = Delta_P_exact / (f_u_hidden * P_halo_x_ref)
        ax.loglog(k, np.abs(S_exact), 'C1:', lw=2.5, label=r'$S(k)$ exact (measured)')
    ax.axvline(k_Ny, c='grey', ls=':', lw=1.2)
    ax.axhline(1.0, c='grey', lw=0.6)
    ax.set_ylabel(r'$S(k)=\langle u_m T\rangle_w$')
    ax.legend(frameon=False, fontsize=9)
    ax.set_title(rf'Experiment A shape factor, $M_{{\rm min}}={M_min:.2e}$, '
                 rf'ref logM$={logM_cen[ref]:.2f}$, $z={z}$')
 
    ax = axes[1]
    if Delta_P_exact is not None:
        for mode, res in results.items():
            with np.errstate(divide='ignore', invalid='ignore'):
                ax.semilogx(k, res['Delta_P'] / Delta_P_exact,
                            color=colours.get(mode, 'C4'),
                            ls=styles.get(mode, ':'), lw=1.8, label=mode)
        ax.axhline(1.0, c='k', lw=0.8)
        ax.fill_between(k, 0.9, 1.1, color='0.85', zorder=0)
        ax.set_ylim(0.0, 2.0)
        ax.set_ylabel(r'$\Delta P_{\rm pred}/\Delta P_{\rm exact}$')
        ax.legend(frameon=False, fontsize=9, ncol=2)
    else:
        for mode, res in results.items():
            ax.loglog(k, np.abs(res['Delta_P']), color=colours.get(mode, 'C4'),
                      ls=styles.get(mode, ':'), lw=1.8, label=mode)
        ax.set_ylabel(r'$\Delta P(k)\,L_{\rm box}^2$')
        ax.legend(frameon=False, fontsize=9)
    ax.axvline(k_Ny, c='grey', ls=':', lw=1.2)
    ax.set_xlabel(r'$k$ [h/cMpc]')
 
    stem = (f'expA_shape_{tag}_{MASS_DEF}_nb{args.nbins}'
            f'_split{"none" if args.split_logm is None else f"{args.split_logm:.2f}"}'
            f'_ref{logM_cen[ref]:.2f}_hmf{args.hmf}_fu{args.fu}'
            f'{"" if bool(data.get("self_pairs_removed", False)) else "_noshot"}')
    p1 = plot_path('pme_reconstruction', FEEDBACK, stem=stem)
    ensure_parents(p1)
    fig.savefig(p1, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"\n[A][plot] {p1}")
 
    # --- figure 2: the ansatz itself, bin by bin -------------------------------
    p2 = None
    if np.any(hidden):
        idx = np.flatnonzero(hidden)
        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
        show = idx[:: max(1, idx.size // 6)]
        for n, j in enumerate(show):
            with np.errstate(divide='ignore', invalid='ignore'):
                T_e = P_halo_x[j] / P_halo_x_ref          # what we want
                T_c = P_halo_dm[j] / P_halo_dm[ref]         # what we use as a proxy
            ax.semilogx(k, T_e, color=f'C{n}', ls='-', lw=1.6,
                        label=rf'logM$={logM_cen[j]:.2f}$')
            ax.semilogx(k, T_c, color=f'C{n}', ls='--', lw=1.2)
        ax.axvline(k_Ny, c='grey', ls=':', lw=1.2)
        ax.set_xlabel(r'$k$ [h/cMpc]')
        ax.set_ylabel(r'$P^{he}(k|M)/P^{h_re}$ (solid) vs $P^{hc}(k|M)/P^{h_rc}$ (dashed)')
        ax.set_title('Experiment A ansatz: does the tracer cancel in the ratio?')
        ax.legend(frameon=False, fontsize=9, ncol=2)
        ax.set_ylim(0, 2)
        stem2 = stem.replace('expA_shape', 'expA_ansatz')
        p2 = plot_path('pme_reconstruction', FEEDBACK, stem=stem2)
        ensure_parents(p2)
        fig.savefig(p2, bbox_inches='tight', dpi=300)
        plt.close(fig)
        print(f"[A][plot] {p2}")
 
    # --- figure 3: what it does to the reconstruction --------------------------
    n_use = np.where(resolved, n_i, 0.0)
    P_rec_resolved = reconstruct_P_matter_x(k, P_halo_x, M_i, n_use, z)
 
    if Delta_P_exact is not None:
        P_target = P_rec_resolved + Delta_P_exact
        ref_label = 'full-catalogue rec.'
        ratio_label = 'rec / full-catalogue rec.'
    else:
        P_target = P_matter_x_true
        ref_label = 'truth'
        ratio_label = 'rec / truth'
 
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True, dpi=300,
                                   gridspec_kw=dict(height_ratios=[3, 1], hspace=0.05))
    if Delta_P_exact is not None:
        ax1.loglog(k, np.abs(P_matter_x_true), color='0.7', ls='-', lw=1.5,
                   label=rf'full truth $\delta_m\times\delta_{{{x_sym}}}$ '
                         rf'(incl. mass below the catalogue)')
        ax1.loglog(k, np.abs(P_target), 'k-', lw=2.5,
                   label='target: all occupied bins')
        ax1.loglog(k, np.abs(P_rec_resolved), color='0.5', ls='-', lw=1.5,
                   label=rf'bins above the split only')
    else:
        ax1.loglog(k, np.abs(P_matter_x_true), 'k-', lw=2.5,
                   label=rf'truth $\delta_m\times\delta_{{{x_sym}}}$')
        ax1.loglog(k, np.abs(P_rec_resolved), color='0.5', ls='-', lw=1.5,
                   label='resolved bins only')
    for mode, res in results.items():
        ax1.loglog(k, np.abs(P_rec_resolved + res['Delta_P']), color=colours.get(mode, 'C4'),
                   ls=styles.get(mode, ':'), lw=1.8, label=f'+ {mode}')
    ax1.axvline(k_Ny, c='grey', ls=':', lw=1.2)
    ax1.set_ylabel(rf'$P_{{\rm m,{tag}}}(k)\,L_{{\rm box}}^2$')
    ax1.legend(frameon=False, fontsize=9)
    ax1.set_title(rf'Experiment A, {mode_label} mode, FLAMINGO {FEEDBACK}, $z={z}$')
 
    with np.errstate(divide='ignore', invalid='ignore'):
        if Delta_P_exact is not None:
            ax2.semilogx(k, P_matter_x_true / P_target, color='0.7', lw=1.5)
        ax2.semilogx(k, P_rec_resolved / P_target, color='0.5', lw=1.5)
        for mode, res in results.items():
            ax2.semilogx(k, (P_rec_resolved + res['Delta_P']) / P_target,
                         color=colours.get(mode, 'C4'), ls=styles.get(mode, ':'),
                         lw=1.8)
    ax2.axhline(1.0, color='k', lw=0.8)
    ax2.fill_between(k, 0.95, 1.05, color='0.85', zorder=0)
    ax2.axvline(k_Ny, c='grey', ls=':', lw=1.2)
    # Validation is a near-1 comparison, so a tighter range; production has to
    # accommodate a correction that may be large.
    ax2.set_ylim(0.6, 1.5) if Delta_P_exact is not None else ax2.set_ylim(0.0, 2.0)
    ax2.set_ylabel(ratio_label)
    ax2.set_xlabel(r'$k$ [h/cMpc]')
 
    stem3 = stem.replace('expA_shape', 'expA_reconstruction')
    p3 = plot_path('pme_reconstruction', FEEDBACK, stem=stem3)
    ensure_parents(p3)
    fig.savefig(p3, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"[A][plot] {p3}")

    # --- save everything --------------------------------------------------------
    payload = {
        'k_center': k, 'tracer': tracer, 'mode': mode_label,
        'ref_bin': ref, 'ref_logM': float(logM_cen[ref]), 'M_ref': M_ref,
        'M_min': M_min, 'split_logm': (np.nan if args.split_logm is None
                                       else float(args.split_logm)),
        'hmf_source': args.hmf, 'fu_source': args.fu,
        'int_logm_min': float(int_logm_lo),
        'f_u_cat': f_u_cat, 'f_resolved': f_resolved,
        'self_pairs_removed': bool(data.get('self_pairs_removed', False)),
        'b_measured': b_meas, 'logM_cen': logM_cen, 'M_mean': M_i, 'n_i': n_i,
        'P_halo_x_ref': P_halo_x_ref, 'P_rec_resolved': P_rec_resolved, 'P_matter_x_true': P_matter_x_true,
    }
    if Delta_P_exact is not None:
        payload['Delta_P_exact'] = Delta_P_exact
        payload['f_u_hidden'] = f_u_hidden
    for mode, res in results.items():
        payload[f'S_{mode}'] = res['S']
        payload[f'Delta_P_{mode}'] = res['Delta_P']
        payload[f'f_u_integral_{mode}'] = res['f_u_integral']
    if hm is not None:
        payload['b_tinker10'] = hm.bias(M_i)
        payload['dndM_tinker08'] = hm.dndM(M_i)
    save_plot_data(p1, payload,
                   description=('Experiment A: P_halo_x extrapolated below M_min by '
                                'freezing the baryon response at a reference bin'))
    return results


# ==============================================================================
# 7.  MAIN
# ==============================================================================
def main():
    ap = argparse.ArgumentParser(
        description='Reconstruct P(matter x x)(k) from binned FLAMINGO '
                    'halo-tracer cross spectra, for x = gas, dm or matter.')
    ap.add_argument('--recompute', action='store_true',
                    help='re-measure the spectra even if a cached bundle exists')
    ap.add_argument('--nbins', type=int, default=NBINS,
                    help=f'number of log-spaced halo mass bins (default {NBINS})')
    ap.add_argument('--logm-min', type=float, default=LOGM_MIN,
                    help=f'lower log10(M/[Msun/h]) edge (default {LOGM_MIN})')
    ap.add_argument('--logm-max', type=float, default=LOGM_MAX,
                    help=f'upper log10(M/[Msun/h]) edge (default {LOGM_MAX})')
    ap.add_argument('--nkbins', type=int, default=NKBINS,
                    help='number of k bins (default: bin_power_spectrum_2d default)')
    ap.add_argument('--target', choices=sorted(TRACER_OF_TARGET), default=TARGET_MODE,
                    help="which spectrum to reconstruct: 'matter_gas' "
                         "(x = gas, from P_halo_gas), 'matter_dm' (x = dm, from "
                         "P_halo_dm) or 'matter_matter' (x = matter, from "
                         "P_halo_matter = f_c P_halo_dm + f_g P_halo_gas). All "
                         "three live in the same bundle, so switching needs no "
                         "re-measurement.")

    s = ap.add_argument_group(
        'Self-pair (shot-noise) subtraction',
        'delta_m is built from the same particles as the tracer fields, so '
        'P(m x x) carries a self-pair term that no halo-model reconstruction '
        'can produce. It is measured once from random catalogues, and the '
        'bundle stores the raw spectra, the self-pair terms and the corrected '
        'spectra side by side. See section 3b.')
    s.add_argument('--no-shot-subtract', action='store_true',
                   help='work with the RAW spectra rather than the corrected '
                        'ones (for comparison; the default uses the corrected). '
                        'Both variants are in the bundle either way, so this '
                        'changes only which is in force, never what is on disk. '
                        'If the bundle has no self-pair block yet, this flag '
                        'also stops one being measured.')
    s.add_argument('--recompute-shot', action='store_true',
                   help='re-measure the self-pair spectra even if a cache '
                        'exists, and rebuild the bundle block from them')
    s.add_argument('--shot-nreal', type=int, default=SHOT_NREAL,
                   help=f'random realisations averaged into P^shot (default '
                        f'{SHOT_NREAL}). More costs one paint + FFT per species '
                        f'each and beats down the Monte-Carlo noise as 1/sqrt(n), '
                        f'which only matters at the lowest k, where the term '
                        f'itself is negligible.')
    s.add_argument('--shot-seed', type=int, default=SHOT_SEED,
                   help=f'RNG seed for the random catalogues (default {SHOT_SEED}, '
                        f'the same seed measure_u_tilde.py uses). It is part of '
                        f'the cache filename.')
    s.add_argument('--shot-chunk', type=float, default=SHOT_CHUNK,
                   help='particles per painting chunk; caps peak memory and '
                        'cannot change the result')

    g = ap.add_argument_group(
        'Experiment A',
        'Replace the flat missing-mass template f_u * P_halo_x(k|M_min) by the '
        'integral over the unresolved masses, with P_halo_x(k|M) extrapolated '
        'downwards off a reference bin.')
    g.add_argument('--experiment', choices=('none', 'A'), default='none',
                   help="run Experiment A after the standard reconstruction")
    g.add_argument('--extrap', nargs='+', default=['flat', 'bias', 'halomodel'],
                   choices=('flat', 'simhc', 'bias', 'halomodel'),
                   help="how to supply the mass transfer function T(k,M). "
                        "'flat' is the current f_u*P_halo_x(k|M_min) control; "
                        "'simhc' uses the measured P_halo_dm ratio and needs "
                        "--split-logm; 'bias' is the 2-halo limit b(M)/b(M_r); "
                        "'halomodel' is the full 1h+2h ratio.")
    g.add_argument('--split-logm', type=float, default=None,
                   help="VALIDATION: treat occupied bins below this log10(M) as "
                        "unresolved, hide them from the reconstruction, and "
                        "score every extrapolation against their exact, "
                        "measured contribution. This is the test that isolates "
                        "the extrapolation from every other error in the chain.")
    g.add_argument('--ref-logm', type=float, default=None,
                   help="log10(M) of the reference bin h_r (default: the lowest "
                        "resolved occupied bin)")
    g.add_argument('--hmf', choices=('tinker', 'catalog'), default='tinker',
                   help="where the weight M n(M) in the integral comes from. "
                        "'catalog' uses the measured n_i M_i of the hidden bins "
                        "(validation only) so that ONLY T is under test; "
                        "'tinker' uses Tinker+08.")
    g.add_argument('--fu', choices=('catalog', 'model'), default='catalog',
                   help="amplitude of the correction: 'catalog' takes f_u from "
                        "the measured mass deficit and only the shape S(k) from "
                        "the model; 'model' takes both from the HMF integral, "
                        "which is unstable to the lower integration limit.")
    g.add_argument('--int-logm-min', type=float, default=None,
                   help=f"lower log10(M) limit of the unresolved integral. "
                        f"Default: the catalogue's own lower edge in validation "
                        f"mode, so that the integral covers exactly the mass "
                        f"range the exact target covers; {INT_LOGM_LO} in "
                        f"production mode. f_u depends strongly on this, S(k) "
                        f"barely at all.")

    g.add_argument('--power-spectrum', choices=('camb', 'eisenstein98'),
                   default='camb',
                   help="linear P(k) feeding sigma(M), n(M) and b(M). 'camb' is "
                        "exact and is cached to disk after the first run; "
                        "'eisenstein98' is the ~5%%-accurate fitting function. "
                        "n(M) and b(M) are Tinker+08/+10 via colossus either way.")
    g.add_argument('--concentration', choices=('powerlaw', 'colossus'),
                   default='powerlaw',
                   help="c(M,z) relation. 'powerlaw' is the built-in fit and is "
                        "the default so that results do not shift silently; "
                        "'colossus' uses a published model and is preferable "
                        "for new work.")
    g.add_argument('--concentration-model', default='diemer19',
                   help="colossus c(M,z) model name when --concentration "
                        "colossus (e.g. diemer19, duffy08, bhattacharya13)")

    args = ap.parse_args()

    global CONCENTRATION_SOURCE, COLOSSUS_CONC_MODEL
    CONCENTRATION_SOURCE = args.concentration
    COLOSSUS_CONC_MODEL = args.concentration_model

    if MASS_DEF != 'm200b':
        raise ValueError(
            f"MASS_DEF is {MASS_DEF!r}, but r200m_of_M assumes a mean-background "
            f"(200m) definition. Either use 'm200b' or change r200m_of_M to match."
        )

    tracer = TRACER_OF_TARGET[args.target]      # 'gas', 'dm' or 'matter'
    info = TRACER_INFO[tracer]
    tag = info['tag']                           # for filenames
    x_sym = info['sym']                         # for latex labels

    print(f"[cosmo] sim={SIM_NAME}, feedback={FEEDBACK}, z={Z_EVAL}")
    print(f"[cosmo] h={H_LITTLE}, Omega_m={OMEGA_M}, L_box={BOX} cMpc/h, ngrid={NGRID}")
    print(f"[cosmo] rhobar_m = {rhobar_m:.4e} (Msun/h)/(Mpc/h)^3")
    print(f"[mode]  target = {args.target}  ->  x = {tracer}, reconstructing "
          f"P(matter x {tracer}) from {info['halo_key']}(k; M_i)")
    if tracer == 'matter':
        print("[mode]  one-sided route: the second delta_m is left as a "
              "MEASURED field, so\n        u_m enters once and the "
              "missing-mass deficit stays linear in f_u.")
    print()

    data = load_or_measure(args.nbins, args.logm_min, args.logm_max,
                           args.nkbins, recompute=args.recompute,
                           self_pairs=not args.no_shot_subtract,
                           recompute_shot=args.recompute_shot,
                           shot_nreal=args.shot_nreal,
                           shot_seed=args.shot_seed,
                           shot_chunk=args.shot_chunk)

    # --- pick the variant to work with, section 3b -----------------------------
    # The bundle carries the raw spectra, the self-pair terms and the corrected
    # spectra side by side; this is where one of them is put in force for the
    # rest of the run. Doing it in a single place is what makes the correction
    # apply "everywhere" downstream without a flag threaded through the call
    # graph, and doing it AFTER the bundle has been written is what keeps the
    # file on disk independent of the command line.
    shot_removed = use_self_pair_corrected(
        data, subtract=not args.no_shot_subtract)

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

    mass_frac_resolved = float(np.sum(n_i * M_i) / rhobar_m)
    print(f"[sim] resolved mass fraction sum(n_i M_i)/rhobar_m = "
          f"{mass_frac_resolved:.3f}")

    print(f"\n[truth] P(m x {tracer}), delta_m = f_c delta_dm + f_g delta_gas "
          f"(f_c={float(data['f_c']):.4f}, f_g={float(data['f_g']):.4f})")
    P_selfpair_truth = data.get(f"{info['truth_key']}_selfpair")
    print(f"[truth] self-pair terms {'REMOVED' if shot_removed else 'PRESENT'}"
          + ("" if shot_removed else "  <-- the reconstruction cannot produce them"))

    # --- reconstruction --------------------------------------------------------
    P_matter_x_rec = reconstruct_P_matter_x(k, P_halo_x, M_i, n_i, z)

    # --- optional missing-mass correction --------------------------------------
    f_u = 1.0 - mass_frac_resolved
    Delta_P_approx = 0.0
    occupied = np.flatnonzero(counts > 0)
    if occupied.size == 0:
        raise SystemExit("No occupied mass bins: check --logm-min/--logm-max "
                         "against the catalogue.")

    if APPLY_MISSING_MASS and f_u > 1e-3:
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
    ref_is_mismatch = SHOW_DMGAS_REFERENCE and tracer == 'gas'
    if ref_is_mismatch:
        mismatch_label += ' (= reference)'
    ax1.loglog(k, np.abs(P_dm_x), color='0.55', ls='-', lw=1.0,
               label=mismatch_label)

    # the fixed dm-gas reference, on both panels in every mode
    show_ref = SHOW_DMGAS_REFERENCE and not ref_is_mismatch
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
                  rf'FLAMINGO {FEEDBACK}, $z={z}$'
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
        'pme_reconstruction', FEEDBACK,
        stem=(
            f'P_matter_{tag}_from_P_halo_{tag}_{MASS_DEF}_nb{args.nbins}_'
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
        'target_mode': args.target,
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
                    f'P_halo_{tag}(k;M_i), target={args.target}, '
                    f'self-pair terms '
                    f'{"removed" if shot_removed else "left in"}'))

    # --- numerical accuracy checks ------------------------------------------------
    if APPLY_MISSING_MASS and f_u > 1e-3:
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
            'pme_reconstruction', FEEDBACK,
            stem=(
                f'P_matter_{tag}_missing_mass_check_{MASS_DEF}_nb{args.nbins}_'
                f'logMmin{args.logm_min:.2f}_logMmax{args.logm_max:.2f}'
                f'{"" if shot_removed else "_noshot"}'
            )
        )
        ensure_parents(p_check)
        fig.savefig(p_check, bbox_inches='tight', dpi=300)
        plt.close(fig)

    # --- Experiment A -----------------------------------------------------------
    if args.experiment == 'A':
        run_experiment_A(data, args, tracer, tag, x_sym)

    print("\n" + "=" * 70)
    print("Done.")
    print("=" * 70)


if __name__ == '__main__':
    main()