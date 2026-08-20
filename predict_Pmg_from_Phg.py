#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Reconstruct the matter-gas cross-power spectrum P_mg(k) from halo-gas
cross-spectra P_hg(k; M_i) measured in halo-mass bins, using only the
assumption that all matter lives in the resolved halos.

    The construction
    ----------------
    If every particle of matter belongs to some halo, the matter density field
    is the halo field, mass-weighted and smeared by each halo's internal matter
    profile u_m(k|M):

        delta_m(k) = (1/rhobar_m) * sum_j M_j u_m(k|M_j) e^{i k.x_j}

    Grouping halos into mass bins i (centres M_i, comoving number density n_i)
    and cross-correlating with the gas field:

        ┌──────────────────────────────────────────────────────────────────┐
        │  P_me(k) = (1/rhobar_m) * sum_i  n_i M_i u_m(k|M_i) P_he(k; M_i) │
        └──────────────────────────────────────────────────────────────────┘

    Everything on the right is either measured (P_hg, the halo counts) or known
    (rhobar_m from cosmology, u_m from NFW + a c(M,z) relation). No *normalized*
    halo mass function is needed: the weight n_i M_i = (count_i / L^3) * M_i
    comes straight from the raw histogram of halo counts and the box volume.

    P_hg(k; M_i) and the truth P_mg(k) are measured directly from FLAMINGO, 
    reusing the field/FFT/binning machinery of HATF_reconstruction.py via utils.*.

WHAT IT DOES
    1. Loads (or computes and caches) the FLAMINGO 2-D projected gas and DM
       delta fields, using exactly the same paths as HATF, so any field HATF
       has already written is reused rather than recomputed.
    1b. Builds the matter field delta_m = f_c delta_dm + f_g delta_gas, with the
       fractions taken from the actual particle mass sums. The reconstruction's
       weights use TOTAL halo mass and TOTAL matter density, so P(m x gas) is
       what it actually targets; P(dm x gas) alone differs at high k, where
       feedback makes gas smoother than DM. Both are measured and stored, and
       --truth picks between them.
    2. Splits the halo catalogue into NBINS log-spaced bins in halo mass over
       [10^LOGM_MIN, 10^LOGM_MAX] Msun/h, and measures P_hg(k; M_i) for each
       bin plus the two truths.
    3. Caches all of that in ONE npz bundle. On every later run the bundle is
       loaded and step 1-2 are skipped entirely.
    4. Runs the reconstruction and compares it to the measured truth.

A NOTE ON PROJECTED (2-D) SPECTRA
    The repo works with fields projected along one axis. For a full-box
    projection the 2-D field is the column average of the 3-D one, so

        delta_2D(k_perp) = delta_3D(k_perp, k_z = 0),

    i.e. the 2-D modes ARE a subset of the 3-D modes (those with k_z = 0). The
    halo-model identity above holds mode by mode, so it holds unchanged for the
    projected fields with u_m evaluated at |k| = k_perp. The only requirement is
    that P_hg and P_mg share a normalisation convention, which they do here
    (both carry the same box^2 factor as in HATF). No Limber approximation and
    no extra projection kernel is involved.

USAGE
    Run from the repo root (it imports `utils.*` the same way HATF does):

        python predict_pme_from_phe.py                 # load bundle if present
        python predict_pme_from_phe.py --recompute     # force re-measurement
        python predict_pme_from_phe.py --nbins 40 --logm-min 11.5

    Any change to the mass binning, mass definition or grid produces a new
    bundle filename, so caches never silently go stale.
--------------------------------------------------------------------------------
"""

import argparse
import gc
from pathlib import Path
 
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.special import sici
 
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
RHO_CRIT_0 = 2.77536627e11
rhobar_m   = OMEGA_M * RHO_CRIT_0
 
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
 
NKBINS = None             # None -> bin_power_spectrum_2d's default, max(32, ngrid//10)
 
# --- which field plays the role of "matter" in the truth ----------------------
# The reconstruction's weights use M200b (TOTAL halo mass) and
# rhobar_m = Omega_m rho_crit (TOTAL matter density), so the right-hand side of
# the halo-model identity targets P(total matter x gas). Comparing it against
# P(dm x gas) therefore carries a definitional mismatch that grows at high k,
# where gas is smoother than DM because of feedback:
#
#     delta_m = f_c delta_dm + f_g delta_gas (+ f_star delta_star)
#
# Both truths are measured and stored in the bundle; this switch only chooses
# which one is compared against and plotted, so flipping it costs nothing.
#   'dm_gas' : delta_m built from the DM and gas fields    <- recommended
#   'dm'     : DM field alone (the previous behaviour, kept for comparison)
#
# NOTE: star particles are in neither field, so 'dm_gas' is still missing
# f_star ~ 1% of the matter, and those are the MOST clustered baryons. The
# residual inconsistency with rhobar_m (which does include stars) is at that
# level. Note also that P(m x gas) contains an explicit f_g P_gas,gas term, so
# unlike a pure cross-spectrum it is not entirely free of gas shot noise.
MATTER_FIELD = 'dm_gas'
 
# --- missing-mass correction --------------------------------------------------
# sum_i n_i M_i < rhobar_m always: mass in halos below 10^LOGM_MIN and in the
# diffuse IGM is not represented. The correction below adds f_u * P_hg of the
# LOWEST mass bin, i.e. it assumes the unresolved mass cross-correlates with gas
# like the smallest resolved halos do. That is an approximation and mildly
# over-corrects (unresolved mass is less biased than 10^11 Msun/h halos), but it
# needs no linear theory and no bias model. It is reported separately from the
# raw reconstruction so you can always see how much it moved things.
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
# small halos). Swap in an exact colossus call if you want; the reconstruction
# is only mildly sensitive to c, since P_hg already carries the gas profile and
# u_m only redistributes the *matter* weight.
def concentration(M, z):
    c0, Mpiv, alpha, beta = 7.5, 1e12, 0.090, 0.65
    return c0 * (M / Mpiv) ** (-alpha) * (1.0 + z) ** (-beta)
 
 
# ==============================================================================
# 3.  MEASUREMENT FROM FLAMINGO  (this replaces the synthetic universe)
# ==============================================================================
def bundle_path(nbins, logm_min, logm_max, ngrid, nkbins):
    """One npz per distinct measurement configuration.
 
    Everything that changes the numbers is in the filename, so a stale bundle
    can never be picked up for a different binning or grid.
    """
    nk = 'default' if nkbins is None else str(nkbins)
    # v2: bundle now also carries the DM-only truth and the mass fractions.
    stem = (f'phg_massbins_v2_{FEEDBACK}_{MASS_DEF}'
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
        print(f"    WARNING: gas fraction is far from the cosmological baryon "
              f"fraction. Check the particle file and mass units.")
 
    ensure_parents(path)
    np.savez(path, f_c=f_c, f_g=f_g, M_dm=M_dm, M_gas=M_gas)
    return f_c, f_g
 
 
def _load_or_compute_delta(label):
    """Load a FLAMINGO 2-D projected delta field, computing it only if absent.
 
    Uses the same paths as HATF_reconstruction, so a field HATF already wrote is
    picked up here for free (and vice versa).
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
 
 
def measure_spectra(nbins, logm_min, logm_max, nkbins):
    """Measure P_hg(k; M_i) per mass bin and the truth P_mg(k) from FLAMINGO.
 
    Returns a dict ready to be written to the npz bundle.
    """
    print("=" * 70)
    print("Measuring FLAMINGO spectra (no cached bundle found)")
    print("=" * 70)
 
    # --- 3a. gas and DM projected fields --------------------------------------
    print("\nProjected fields:")
    delta_gas = _load_or_compute_delta('gas')
    delta_dm = _load_or_compute_delta('dm')
 
    gas_fft = compute_2d_fft(delta_gas, NGRID)
    dm_fft = compute_2d_fft(delta_dm, NGRID)
    del delta_gas, delta_dm
    gc.collect()
 
    k_grid = compute_k_grid_2d(NGRID, BOX)
    k_min = 2.0 * np.pi / BOX
    k_Nyquist = np.pi * NGRID / BOX
 
    # --- 3b. the matter field --------------------------------------------------
    # delta_m = f_c delta_dm + f_g delta_gas. The FFT is linear, so this is done
    # on the transforms directly: no third real-space array is ever allocated,
    # and the result is exact rather than an approximation.
    print("\nComponent mass fractions:")
    f_c, f_g = component_mass_fractions()
    m_fft = f_c * dm_fft + f_g * gas_fft
 
    # --- 3c. the two truths ----------------------------------------------------
    # P(m x gas) is the one the reconstruction actually targets, since the
    # weights n_i M_i / rhobar_m are built from TOTAL halo mass and TOTAL matter
    # density. P(dm x gas) is kept for comparison: the difference between them
    # is the definitional mismatch, which is negligible at low k and grows once
    # feedback-driven gas displacement becomes comparable to 1/k.
    k_bins, k_center, P_mg_true = bin_power_spectrum_2d(
        (m_fft * np.conj(gas_fft)).real, k_grid, NGRID, BOX,
        nkbins=nkbins, k_min=k_min,
    )
    P_mg_true = np.asarray(P_mg_true) * BOX ** 2
 
    _, _, P_dmg_true = bin_power_spectrum_2d(
        (dm_fft * np.conj(gas_fft)).real, k_grid, NGRID, BOX,
        nkbins=nkbins, k_min=k_min,
    )
    P_dmg_true = np.asarray(P_dmg_true) * BOX ** 2
 
    print(f"\nTruths measured on {k_center.size} k bins, "
          f"k = [{k_center[0]:.4f}, {k_center[-1]:.2f}] h/cMpc")
    with np.errstate(divide='ignore', invalid='ignore'):
        dev = np.abs(P_mg_true / P_dmg_true - 1.0)
    finite = np.isfinite(dev)
    if np.any(finite):
        print(f"  |P_mg/P_dmg - 1|: {np.nanmax(dev[finite]):.4f} max, "
              f"{dev[finite][0]:.4f} at k_min")
 
    del dm_fft, m_fft
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
    P_hg = np.zeros((nbins, k_center.size), dtype=np.float64)
 
    print("\nPer-bin halo-gas cross spectra:")
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
 
        _, _, P_i = bin_power_spectrum_2d(
            (gas_fft * np.conj(halo_fft)).real, k_grid, NGRID, BOX,
            nkbins=nkbins, k_min=k_min,
        )
        P_hg[i] = np.asarray(P_i) * BOX ** 2
 
        del halo_fft
        gc.collect()
 
        print(f"  bin {i:2d}  logM = {logM_cen[i]:5.2f}  N = {n_in_bin:8d}  "
              f"log10<M> = {np.log10(M_mean[i]):5.2f}  "
              f"P_hg(k_min) = {P_hg[i, 0]:.4e}")
 
    del gas_fft, all_pos, all_mass
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
        P_hg=P_hg,
        P_mg_true=P_mg_true,
        P_dmg_true=P_dmg_true,
        f_c=f_c,
        f_g=f_g,
        box=BOX,
        ngrid=NGRID,
        redshift=Z_EVAL,
        mass_def=MASS_DEF,
        centrals_only=CENTRALS_ONLY,
        feedback=FEEDBACK,
    )
 
 
def load_or_measure(nbins, logm_min, logm_max, nkbins, recompute=False):
    """Load the cached bundle if it exists, otherwise measure and write it."""
    path = bundle_path(nbins, logm_min, logm_max, NGRID, nkbins)
 
    required = ('k_center', 'P_hg', 'P_mg_true', 'P_dmg_true', 'counts',
                'M_mean', 'f_c', 'f_g')
 
    if path.exists() and not recompute:
        print(f"Loading pre-computed spectra bundle:\n  {path}")
        with np.load(path, allow_pickle=False) as f:
            data = {key: f[key] for key in f.files}
        missing = [key for key in required if key not in data]
        if missing:
            print(f"  Bundle is missing {missing}; re-measuring.")
        else:
            print(f"  {data['P_hg'].shape[0]} mass bins, "
                  f"{data['k_center'].size} k bins, "
                  f"f_c={float(data['f_c']):.4f}, f_g={float(data['f_g']):.4f}. "
                  f"Skipping measurement.")
            return data
 
    if recompute and path.exists():
        print("--recompute given: ignoring the existing bundle and re-measuring.")
 
    data = measure_spectra(nbins, logm_min, logm_max, nkbins)
    ensure_parents(path)
    np.savez_compressed(path, **data)
    print(f"\nSpectra bundle saved:\n  {path}")
    return data
 
 
# ==============================================================================
# 4.  THE RECONSTRUCTION  (this is the part that matters)
# ==============================================================================
#   P_mg(k) = (1/rhobar_m) sum_i n_i M_i u_m(k|M_i) P_hg(k; M_i)
# ------------------------------------------------------------------------------
def reconstruct_Pmg(k, P_hg, M_i, n_i, z):
    """Reconstruct P_mg(k) from measured P_hg(k; M_i) and the halo histogram.
 
    Parameters
    ----------
    k    : (nk,)        wavenumbers [h/cMpc]
    P_hg : (nbins, nk)  measured halo-gas cross spectra, one row per mass bin
    M_i  : (nbins,)     representative halo mass per bin [Msun/h]
    n_i  : (nbins,)     comoving number density per bin [(cMpc/h)^-3]
    z    : scalar       redshift (for the concentration relation)
 
    Returns
    -------
    P_mg_rec : (nk,)    reconstructed matter-gas cross spectrum
    """
    c_i = concentration(M_i, z)
    u_im = np.array([u_nfw(k, M_i[i], c_i[i]) for i in range(M_i.size)])
    weight = (n_i * M_i)[:, None] / rhobar_m          # (nbins, 1)
    return np.sum(weight * u_im * P_hg, axis=0)       # sum over mass bins
 
 
# ==============================================================================
# 5.  MAIN
# ==============================================================================
def main():
    ap = argparse.ArgumentParser(
        description='Reconstruct P_mg(k) from binned FLAMINGO halo-gas cross spectra.')
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
    ap.add_argument('--truth', choices=['dm_gas', 'dm'], default=MATTER_FIELD,
                    help="which field is 'matter' in the truth: 'dm_gas' (default, "
                         "f_c*delta_dm + f_g*delta_gas) or 'dm' (DM alone). Both are "
                         "stored in the bundle, so switching needs no re-measurement.")
    args = ap.parse_args()
 
    if MASS_DEF != 'm200b':
        raise ValueError(
            f"MASS_DEF is {MASS_DEF!r}, but r200m_of_M assumes a mean-background "
            f"(200m) definition. Either use 'm200b' or change r200m_of_M to match."
        )
 
    print(f"[cosmo] sim={SIM_NAME}, feedback={FEEDBACK}, z={Z_EVAL}")
    print(f"[cosmo] h={H_LITTLE}, Omega_m={OMEGA_M}, L_box={BOX} cMpc/h, ngrid={NGRID}")
    print(f"[cosmo] rhobar_m = {rhobar_m:.4e} (Msun/h)/(Mpc/h)^3\n")
 
    data = load_or_measure(args.nbins, args.logm_min, args.logm_max,
                           args.nkbins, recompute=args.recompute)
 
    k = data['k_center']
    P_hg = data['P_hg']
    P_mg_full = data['P_mg_true']     # delta_m = f_c delta_dm + f_g delta_gas
    P_dmg = data['P_dmg_true']        # DM alone
    P_mg_true = P_mg_full if args.truth == 'dm_gas' else P_dmg
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
 
    print(f"\n[truth] using '{args.truth}' "
          f"(f_c={float(data['f_c']):.4f}, f_g={float(data['f_g']):.4f})")
    with np.errstate(divide='ignore', invalid='ignore'):
        mismatch = P_mg_full / P_dmg - 1.0
    for kk in (0.05, 0.5, 5.0):
        if kk <= k[-1]:
            j = int(np.argmin(np.abs(k - kk)))
            print(f"    k={k[j]:6.3f}  P_mg/P_dmg - 1 = {mismatch[j]:+.4f}")
    print("    (this is the DM-vs-matter definitional mismatch, not a "
          "reconstruction error)")
 
    # --- reconstruction --------------------------------------------------------
    P_mg_rec = reconstruct_Pmg(k, P_hg, M_i, n_i, z)
 
    # --- optional missing-mass correction --------------------------------------
    f_u = 1.0 - mass_frac_resolved
    if APPLY_MISSING_MASS and f_u > 1e-3:
        occupied = np.flatnonzero(counts > 0)
        template = P_hg[occupied[0]]      # lowest occupied mass bin
        P_mg_corr = P_mg_rec + f_u * template
        print(f"[corr] unresolved mass fraction f_u = {f_u:.3f}; template taken "
              f"from the logM={logM_cen[occupied[0]]:.2f} bin")
    else:
        P_mg_corr = P_mg_rec
        f_u = 0.0
 
    # --- comparison plot -------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True, dpi=300,
                                   gridspec_kw=dict(height_ratios=[3, 1], hspace=0.05))
 
    truth_label = (r'truth $\delta_m\times\delta_{gas}$, '
                   r'$\delta_m=f_c\delta_{dm}+f_g\delta_{gas}$'
                   if args.truth == 'dm_gas'
                   else r'truth $\delta_{dm}\times\delta_{gas}$')
    ax1.loglog(k, np.abs(P_mg_true), 'k-', lw=2.5, label=truth_label)
    # the truth not being used, for scale: the gap between the two black curves
    # is the definitional mismatch, and sets the floor on achievable agreement
    other = P_dmg if args.truth == 'dm_gas' else P_mg_full
    other_label = (r'$\delta_{dm}\times\delta_{gas}$ (unused)'
                   if args.truth == 'dm_gas'
                   else r'$\delta_m\times\delta_{gas}$ (unused)')
    ax1.loglog(k, np.abs(other), color='0.55', ls='-', lw=1.0, label=other_label)
    ax1.loglog(k, np.abs(P_mg_rec), 'C1--', lw=2.0, label=r'$P_{mg}$ reconstructed')
    if f_u > 1e-3:
        ax1.loglog(k, np.abs(P_mg_corr), 'C2:', lw=2.0,
                   label=r'reconstructed + missing-mass')
    ax1.axvline(k_Nyquist, c='grey', ls=':', lw=1.2, label='$k_{Nyquist}$')
    ax1.set_ylabel(r'$P_{mg}(k)\,L_{\rm box}^2$')
    ax1.legend(frameon=False, fontsize=10)
    ax1.set_title(rf'$P_{{mg}}$ reconstruction from $P_{{hg}}(k;M_i)$, '
                  rf'FLAMINGO {FEEDBACK}, $z={z}$')
 
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = P_mg_rec / P_mg_true
        ratio_corr = P_mg_corr / P_mg_true
 
    ax2.semilogx(k, ratio, 'C1--', lw=2)
    if f_u > 1e-3:
        ax2.semilogx(k, ratio_corr, 'C2:', lw=2)
    with np.errstate(divide='ignore', invalid='ignore'):
        ax2.semilogx(k, other / P_mg_true, color='0.55', ls='-', lw=1.0)
    ax2.axhline(1.0, color='k', lw=0.8)
    ax2.axvline(k_Nyquist, c='grey', ls=':', lw=1.2)
    ax2.fill_between(k, 0.95, 1.05, color='0.85', zorder=0)
    ax2.set_ylim(0.0, 2.0)
    ax2.set_ylabel('rec / truth')
    ax2.set_xlabel(r'$k$ [h/cMpc]')
 
    p_out = plot_path('pme_reconstruction', FEEDBACK,
                      stem=f'P_mg_from_P_hg_{MASS_DEF}_nb{args.nbins}_{args.truth}')
    ensure_parents(p_out)
    fig.savefig(p_out, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"\n[plot] saved {p_out}")
 
    save_plot_data(p_out, {
        'k_center': k,
        'P_mg_true': P_mg_true,
        'P_mg_dmgas': P_mg_full,
        'P_dmg': P_dmg,
        'truth_choice': args.truth,
        'f_c': float(data['f_c']),
        'f_g': float(data['f_g']),
        'P_mg_rec': P_mg_rec,
        'P_mg_corr': P_mg_corr,
        'P_hg': P_hg,
        'counts': counts,
        'M_mean': M_i,
        'logM_cen': logM_cen,
        'n_i': n_i,
        'mass_frac_resolved': mass_frac_resolved,
        'k_Nyquist': k_Nyquist,
        'box': box,
        'redshift': z,
    }, description='P_mg reconstructed from binned FLAMINGO P_hg(k;M_i)')
 
    # --- numerical summary -----------------------------------------------------
    print("\n[result] reconstruction vs truth:")
    for kk in [0.05, 0.2, 0.5, 1.0, 3.0, 8.0]:
        if kk > k[-1]:
            continue
        j = int(np.argmin(np.abs(k - kk)))
        print(f"    k={k[j]:6.3f}  truth={P_mg_true[j]:.4e}  rec={P_mg_rec[j]:.4e}  "
              f"ratio={ratio[j]:.4f}  corr_ratio={ratio_corr[j]:.4f}")
 
    print("\n" + "=" * 70)
    print("Done.")
    print("=" * 70)
 
 
if __name__ == '__main__':
    main()
