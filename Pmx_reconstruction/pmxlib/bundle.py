#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Measurement of the P_halo_gas spectra bundle from FLAMINGO, and its cache.

The bundle is one npz per distinct measurement configuration, so whichever of
experiment_A.py and experiment_B.py runs first builds it and the other picks it
up without re-measuring anything.
"""
import gc

import numpy as np

from utils.catalog_loaders import load_particle_properties
from utils.delta_fields import compute_delta_2d, compute_delta_field_and_mass
from utils.pipeline_paths import (DATA_ROOT, ensure_parents,
                                  get_particle_file_path, delta_2d_path)
from utils.power_spectrum_utils import (bin_mode_stats, compute_2d_fft,
                                        compute_k_grid_2d)

from Pmx_reconstruction.pmxlib.binning import load_binned_halos, log_mass_bin_edges
from Pmx_reconstruction.pmxlib.painting import binned_spectrum
from Pmx_reconstruction.pmxlib.self_pairs import (measure_shot_spectra,
                                                 subtract_self_pairs)

def bundle_path(cfg, nbins, logm_min, logm_max, ngrid):
    """One npz per distinct measurement configuration."""
    stem = (f'spectra_v5_{cfg.feedback}_{cfg.mass_def}'
            f'_logM{logm_min:g}-{logm_max:g}_nb{nbins}'
            f'_ngrid{ngrid}_{cfg.kbin_tag}'
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

    ensure_parents(path)
    np.savez(path, f_c=f_c, f_g=f_g, M_dm=M_dm, M_gas=M_gas)
    return f_c, f_g


def _load_or_compute_delta(cfg, label):
    """Load a FLAMINGO 2-D projected delta field, computing it only if absent."""
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


def measure_spectra(cfg, nbins, logm_min, logm_max):
    """Measure P_halo_gas and P_halo_dm per mass bin, plus the truths.

    Returns a dict ready to be written to the npz bundle.
    """
    print("=" * 70)
    print("Measuring FLAMINGO spectra (no cached bundle found)")
    print("=" * 70)

    # --- gas and DM projected fields --------------------------------------
    print("\nProjected fields:")
    delta_gas = _load_or_compute_delta(cfg, 'gas')
    delta_dm = _load_or_compute_delta(cfg, 'dm')

    gas_fft = compute_2d_fft(delta_gas, cfg.grid)
    dm_fft = compute_2d_fft(delta_dm, cfg.grid)
    del delta_gas, delta_dm
    gc.collect()

    k_grid = compute_k_grid_2d(cfg.grid, cfg.box)
    k_Nyquist = np.pi * cfg.grid / cfg.box
    k_bins = cfg.k_bins
    # k_center is the mode-weighted mean |k| of the bin, not the midpoint of
    # its edges: that is the k the bin's value refers to, and where a bin is
    # wide compared to the k it sits at -- the lowest few -- the two differ by
    # several per cent. Models are evaluated on this k, so they and the
    # measurement refer to the same place.
    nmodes, k_center = bin_mode_stats(k_grid, k_bins)
    print(f"  k binning: {k_center.size} log bins, "
          f"{cfg.kbins_per_decade:g}/decade, width floored at "
          f"{cfg.kbin_wmin_kf:g} k_f; k = {k_center[0]:.4f} .. {k_center[-1]:.3f}")

    # --- the matter field --------------------------------------------------
    # delta_m = f_c delta_dm + f_g delta_gas.
    print("\nComponent mass fractions:")
    f_c, f_g = component_mass_fractions(cfg)
    m_fft = f_c * dm_fft + f_g * gas_fft

    # --- the truths --------------------------------------------------------
    def _binned(prod):
        return binned_spectrum(cfg, prod, k_grid)

    P_matter_gas = _binned((m_fft * np.conj(gas_fft)).real)
    P_dm_gas = _binned((dm_fft * np.conj(gas_fft)).real)
    P_dm_dm = _binned(np.abs(dm_fft) ** 2)
    P_gas_gas = _binned(np.abs(gas_fft) ** 2)

    del m_fft
    gc.collect()

    # --- the self-pair spectra ---------------------------------------------
    shot = measure_shot_spectra(cfg)

    # --- halo catalogue and log-spaced mass bins ---------------------------
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

        P_halo_gas[i] = _binned((gas_fft * np.conj(halo_fft)).real)
        P_halo_dm[i] = _binned((dm_fft * np.conj(halo_fft)).real)

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
        nmodes=np.asarray(nmodes),
        k_Nyquist=k_Nyquist,
        logM_edges=logM_edges,
        logM_cen=logM_cen,
        M_mean=M_mean,
        counts=counts,
        P_halo_gas=P_halo_gas,
        P_halo_dm=P_halo_dm,
        P_matter_gas=P_matter_gas,
        P_dm_gas=P_dm_gas,
        P_dm_dm=P_dm_dm,
        P_gas_gas=P_gas_gas,
        **shot,
        f_c=f_c,
        f_g=f_g,
        box=cfg.box,
        ngrid=cfg.grid,
        redshift=cfg.z,
        mass_def=cfg.mass_def,
        centrals_only=cfg.centrals_only,
        feedback=cfg.feedback,
    )


# Keys every bundle must carry; a bundle missing any of them is re-measured.
_REQUIRED_KEYS = ('k_center', 'k_bins', 'nmodes', 'counts', 'M_mean', 'f_c', 'f_g',
                  'P_halo_gas', 'P_halo_dm', 'P_matter_gas', 'P_dm_gas',
                  'P_dm_dm', 'P_gas_gas', 'P_shot_dm', 'P_shot_gas')


def load_or_measure(cfg, nbins, logm_min, logm_max, recompute=False):
    """Load the cached bundle if it exists, otherwise measure and write it."""
    path = bundle_path(cfg, nbins, logm_min, logm_max, cfg.grid)

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
                  f"{data['k_center'].size} k bins, "
                  f"f_c={float(data['f_c']):.4f}, f_g={float(data['f_g']):.4f}. "
                  f"Skipping measurement.")

    if data is None:
        if recompute and path.exists():
            print("--recompute given: ignoring the existing bundle and "
                  "re-measuring.")
        data = measure_spectra(cfg, nbins, logm_min, logm_max)
        ensure_parents(path)
        np.savez_compressed(path, **data)
        print(f"\nSpectra bundle saved:\n  {path}")

    return subtract_self_pairs(data)