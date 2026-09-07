#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Measurement of the P_halo_x spectra bundle from FLAMINGO, and its cache.

Was section 3 of predict_Pmx_from_Phx.py. The bundle is one npz per distinct
measurement configuration, carrying all three tracers, so --target is free and
experiment_A.py picks up a bundle written by predict_Pmx_from_Phx.py without
re-measuring anything.

The filename format is load-bearing: every cached bundle on the cluster depends
on it. Do not change bundle_path's stem.
"""
import gc

import numpy as np

from utils.catalog_loaders import load_particle_properties
from utils.delta_fields import compute_delta_2d, compute_delta_field_and_mass
from utils.pipeline_paths import (DATA_ROOT, ensure_parents,
                                  get_particle_file_path, delta_2d_path)
from utils.power_spectrum_utils import compute_2d_fft, compute_k_grid_2d

from Pmx_reconstruction.pmxlib.binning import load_binned_halos, log_mass_bin_edges
from Pmx_reconstruction.pmxlib.painting import binned_spectrum
from Pmx_reconstruction.pmxlib.self_pairs import ensure_self_pair_block

# ==============================================================================
# 3.  MEASUREMENT FROM FLAMINGO  (this replaces the synthetic universe)
# ==============================================================================
def bundle_path(cfg, nbins, logm_min, logm_max, ngrid, nkbins):
    """One npz per distinct measurement configuration.

    Everything that changes the numbers is in the filename, so a stale bundle
    can never be picked up for a different binning or grid. TARGET_MODE is
    deliberately absent: one bundle serves all three tracers.
    """
    nk = 'default' if nkbins is None else str(nkbins)
    stem = (f'spectra_v4_{cfg.feedback}_{cfg.mass_def}'
            f'_logM{logm_min:g}-{logm_max:g}_nb{nbins}'
            f'_ngrid{ngrid}_nk{nk}'
            f'_{"cen" if cfg.centrals_only else "all"}.npz')
    return DATA_ROOT / cfg.sim_name / 'pme_inputs' / stem


def mass_fractions_path(cfg):
    """Tiny cache for the component mass fractions (a few bytes, but the
    particle read that produces them is not cheap)."""
    return DATA_ROOT / cfg.sim_name / 'pme_inputs' / f'mass_fractions_{cfg.feedback}.npz'


def component_mass_fractions(cfg, source='particles'):
    """Return (f_c, f_g): DM and gas fractions of the DM+gas mass budget.

    These are the weights in delta_m = f_c delta_dm + f_g delta_gas.

    source='particles' sums the actual particle masses in the snapshot (exact
    for the two fields we have, cached so it happens at most once).
    source='cosmology' uses f_g = Omega_b/Omega_m with no file access, which
    effectively assigns the stellar mass to the gas field and so slightly
    over-weights the smoother component.
    """
    if source == 'cosmology':
        f_g = cfg.sim_params['omega_b'] / cfg.sim_params['omega_m']
        print(f"  mass fractions from cosmology: f_c={1 - f_g:.5f}, f_g={f_g:.5f}")
        return 1.0 - f_g, f_g

    path = mass_fractions_path(cfg)
    if path.exists():
        with np.load(path) as f:
            f_c, f_g = float(f['f_c']), float(f['f_g'])
        print(f"  mass fractions from cache: f_c={f_c:.5f}, f_g={f_g:.5f}")
        return f_c, f_g

    print("  Summing particle masses to get component fractions (one-off) ...")
    particles_file = get_particle_file_path(cfg.feedback, sim_name=cfg.sim_name)

    gas_p = load_particle_properties(particles_file, 'gas', requested=('mass',),
                                     sim_name=cfg.sim_name, Lbox=cfg.box)
    M_gas = float(np.sum(gas_p['mass']))
    del gas_p
    gc.collect()

    dm_p = load_particle_properties(particles_file, 'dm', requested=('mass',),
                                    sim_name=cfg.sim_name, Lbox=cfg.box)
    M_dm = float(np.sum(dm_p['mass']))
    del dm_p
    gc.collect()

    total = M_dm + M_gas
    f_c, f_g = M_dm / total, M_gas / total

    # Cross-check against the cosmological baryon fraction. They should not be
    # equal -- the difference is roughly the stellar mass, which is in neither
    # field -- but a large discrepancy means something is wrong.
    f_b_cosmo = cfg.sim_params['omega_b'] / cfg.sim_params['omega_m']
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


def _load_or_compute_delta(cfg, label):
    """Load a FLAMINGO 2-D projected delta field, computing it only if absent.

    Uses the same paths as HATF_reconstruction, so a field HATF has already
    written is picked up here for free (and vice versa).
    """
    path_cfg = {'sim_name': cfg.sim_name, 'feedback': cfg.feedback}
    path = delta_2d_path(path_cfg, label)

    if path.exists():
        print(f"  Loading {label} field from {path} ...")
        field = np.load(path)
        print(f"    shape {field.shape}")
        return field

    print(f"  {label} field not found. Computing (this is the slow part) ...")
    particles_file = get_particle_file_path(cfg.feedback, sim_name=cfg.sim_name)
    field, _ = compute_delta_field_and_mass(
        tracer=label,
        sim_name=cfg.sim_name,
        dm_particles_file=particles_file if label == 'dm' else None,
        gas_particles_file=particles_file,   # 'dm' needs it too, for the rescale
        box=cfg.box,
        ngrid=cfg.grid,
        nthread=cfg.threads,
    )
    ensure_parents(path)
    np.save(path, field)
    print(f"    computed and saved to {path}")
    gc.collect()
    return field


def measure_spectra(cfg, nbins, logm_min, logm_max, nkbins):
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
    delta_gas = _load_or_compute_delta(cfg, 'gas')
    delta_dm = _load_or_compute_delta(cfg, 'dm')

    gas_fft = compute_2d_fft(delta_gas, cfg.grid)
    dm_fft = compute_2d_fft(delta_dm, cfg.grid)
    del delta_gas, delta_dm
    gc.collect()

    k_grid = compute_k_grid_2d(cfg.grid, cfg.box)
    k_Nyquist = np.pi * cfg.grid / cfg.box
    # k_min is not set here any more: the 2 pi / L lower edge now lives in
    # binned_spectrum(cfg, ), which is shared with the self-pair pass of section 3b
    # so that the two can never end up on different k bins.

    # --- 3b. the matter field --------------------------------------------------
    # delta_m = f_c delta_dm + f_g delta_gas. The FFT is linear, so this is done
    # on the transforms directly: no third real-space array is ever allocated,
    # and the result is exact rather than an approximation.
    print("\nComponent mass fractions:")
    f_c, f_g = component_mass_fractions(cfg)
    m_fft = f_c * dm_fft + f_g * gas_fft

    # --- 3c. the truths --------------------------------------------------------
    # Four spectra, two per tracer. P(m x e) is what the reconstruction actually
    # targets, since the weights n_i M_i / cfg.rhobar_m are built from TOTAL halo
    # mass and TOTAL matter density. P(dm x e) is kept for comparison: the
    # difference between them is the definitional mismatch, negligible at low k
    # and growing once feedback-driven gas displacement becomes comparable to
    # 1/k. P(dm x gas) doubles as the fixed reference curve on both panels.
    def _binned(prod):
        return binned_spectrum(cfg, prod, k_grid, nkbins)

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

    del m_fft
    gc.collect()

    # --- 3d/3e. halo catalogue and log-spaced mass bins ------------------------
    # Binning lives in pmxlib.binning so that measure_u_tilde's cache indexes
    # the same bins as this bundle by construction rather than by comment.
    print("\nHalo catalogue:")
    all_pos, all_mass, bin_index, logM_edges, _ = load_binned_halos(cfg)
    _, logM_cen, M_cen = log_mass_bin_edges(nbins, logm_min, logm_max)

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

        delta_h = compute_delta_2d(pos_i, cfg.box, cfg.grid, None, nthread=cfg.threads)
        halo_fft = compute_2d_fft(delta_h, cfg.grid)
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
        box=cfg.box,
        ngrid=cfg.grid,
        redshift=cfg.z,
        mass_def=cfg.mass_def,
        centrals_only=cfg.centrals_only,
        feedback=cfg.feedback,
    )


def derive_P_halo_matter(cfg, data):
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


def load_or_measure(cfg, nbins, logm_min, logm_max, nkbins, recompute=False,
                    self_pairs=True, recompute_shot=False,
                    shot_nreal=None, shot_seed=None, shot_chunk=None):
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
    shot_nreal = cfg.shot_nreal if shot_nreal is None else shot_nreal
    shot_seed = cfg.shot_seed if shot_seed is None else shot_seed
    shot_chunk = cfg.shot_chunk if shot_chunk is None else shot_chunk

    path = bundle_path(cfg, nbins, logm_min, logm_max, cfg.grid, nkbins)

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
        data = measure_spectra(cfg, nbins, logm_min, logm_max, nkbins)

    # The self-pair block, added to the dict and then persisted with it. It is
    # rewritten whenever the spectra themselves were re-measured, whenever the
    # block was absent, and whenever --recompute-shot asks for it.
    added = ensure_self_pair_block(cfg, data, nkbins, allow_measure=self_pairs,
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

    return derive_P_halo_matter(cfg, data)