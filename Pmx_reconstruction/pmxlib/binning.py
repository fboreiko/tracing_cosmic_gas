#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Halo catalogue loading and log-spaced mass binning.

This was the invariant maintained by hand and by comment: measure_spectra built
the bins in predict_Pmx_from_Phx.py, and load_binned_centrals in
measure_u_tilde.py reproduced them "bit for bit (same edges, same right-edge
rule, same centrals cut), so bin i here is bin i in the bundle". Now there is
one definition and the invariant holds by construction.
"""
import gc

import numpy as np

from utils.catalog_loaders import load_halo_properties
from utils.pipeline_paths import get_halo_file_path


def log_mass_bin_edges(nbins, logm_min, logm_max):
    """Bin edges, bin centres in log10, and centre masses.

    Returns (logM_edges, logM_cen, M_cen).
    """
    logM_edges = np.linspace(logm_min, logm_max, nbins + 1)
    logM_cen = 0.5 * (logM_edges[:-1] + logM_edges[1:])
    return logM_edges, logM_cen, 10.0 ** logM_cen


def assign_mass_bins(mass, nbins, logm_min, logm_max):
    """bin_index (-1 for out of range) and the in_range mask.

    The right edge goes into the last bin (bin_index == nbins -> nbins - 1).
    That rule, and the half-open [logm_min, logm_max) test, are what make bin i
    in a u_tilde cache the same bin i as in the spectra bundle.
    """
    logM_edges, _, _ = log_mass_bin_edges(nbins, logm_min, logm_max)
    with np.errstate(divide='ignore', invalid='ignore'):
        logm = np.log10(mass)
    in_range = np.isfinite(logm) & (logm >= logm_min) & (logm < logm_max)
    bin_index = np.full(mass.size, -1, dtype=np.int64)
    bin_index[in_range] = np.digitize(logm[in_range], logM_edges) - 1
    bin_index[bin_index == nbins] = nbins - 1     # right edge into the last bin
    return bin_index, in_range, logM_edges


def check_mass_units(masses_msun_h, halo_mass_unit_msun_h):
    """Loud failure if the catalogue mass unit is not what we assumed."""
    finite = masses_msun_h[np.isfinite(masses_msun_h) & (masses_msun_h > 0)]
    if finite.size == 0:
        raise RuntimeError("No positive halo masses found in the catalogue.")
    logm_hi = np.log10(np.max(finite))
    if not (12.5 < logm_hi < 16.5):
        raise RuntimeError(
            f"Halo masses look wrong: max log10(M) = {logm_hi:.2f}, expected the "
            f"most massive FLAMINGO halo near 10^15 Msun/h.\n"
            f"halo_mass_unit_msun_h is currently {halo_mass_unit_msun_h:g}; if the "
            f"catalogue stores masses in 1e10 Msun/h, set it to 1e10."
        )
    print(f"    mass sanity check ok: max log10(M) = {logm_hi:.2f}")


def load_halo_catalogue(cfg, extra_props=()):
    """Positions and masses of the halos, with the centrals cut applied.

    Any names in extra_props are loaded alongside and returned in the extras
    dict, cut the same way. Nothing is binned here.
    """
    halo_file = get_halo_file_path(cfg.feedback, sim_name=cfg.sim_name)
    print(f"  {halo_file}")
    requested = ['pos', cfg.mass_def] + list(extra_props)
    if cfg.centrals_only:
        requested.append('centrals')
    props = load_halo_properties(halo_file, requested, sim_name=cfg.sim_name)

    pos = np.asarray(props['pos'], dtype=np.float64)
    mass = np.asarray(props[cfg.mass_def], dtype=np.float64) * cfg.halo_mass_unit_msun_h
    extras = {name: np.asarray(props[name], dtype=np.float64)
              for name in extra_props}

    if cfg.centrals_only:
        cen = np.asarray(props['centrals']).astype(bool)
        pos, mass = pos[cen], mass[cen]
        extras = {k: v[cen] for k, v in extras.items()}
        print(f"  centrals only: {mass.size} halos")
    else:
        print(f"  all objects: {mass.size}")

    del props
    gc.collect()
    return pos, mass, extras


def load_binned_halos(cfg, extra_props=(), restrict_to_range=False):
    """Load the catalogue, check the mass units, and bin by mass.

    Returns (pos, mass, bin_index, logM_edges, extras). With restrict_to_range
    the arrays are cut down to the in-range halos, which is what the u~_m
    membership search wants; measure_spectra keeps the full arrays and selects
    per bin instead.
    """
    pos, mass, extras = load_halo_catalogue(cfg, extra_props=extra_props)
    check_mass_units(mass, cfg.halo_mass_unit_msun_h)

    bin_index, in_range, logM_edges = assign_mass_bins(
        mass, cfg.nbins, cfg.logm_min, cfg.logm_max)
    print(f"  in [{cfg.logm_min}, {cfg.logm_max}): {np.count_nonzero(in_range)} halos "
          f"({np.count_nonzero(in_range) / mass.size:.1%} of the sample)")

    if restrict_to_range:
        pos, mass, bin_index = pos[in_range], mass[in_range], bin_index[in_range]
        extras = {k: v[in_range] for k, v in extras.items()}

    return pos, mass, bin_index, logM_edges, extras