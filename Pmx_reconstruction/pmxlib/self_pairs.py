#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Self-pair (shot-noise) spectra, and their removal from the measured spectra.

Both the spectra bundle and the R*/U* cache are corrected here, by one applier
off two term-builders, so the raw/corrected convention cannot differ between
the two files that get differenced against each other.

<delta_m delta_gas*> = f_c P_dm_gas + f_g P_gas_gas, and P_gas_gas carries
P^shot_gg, so the self-pair content of the spectra that have one is

P_matter_gas : f_g P^shot_gg
P_gas_gas    :     P^shot_gg
P_dm_dm      :     P^shot_dd

HOW P^shot IS MEASURED
Each species is painted at uniformly random 2-D positions with its real
per-particle masses, and the auto spectrum of the resulting field is taken
through binned_spectrum(). A random catalogue has no clustering, so in
expectation that auto spectrum IS the self-pair term of the real field.
------------------------------------------------------------------------------
"""
import gc

import numpy as np

from utils.catalog_loaders import load_particle_properties
from utils.pipeline_paths import get_particle_file_path
from utils.power_spectrum_utils import compute_2d_fft, compute_k_grid_2d

from Pmx_reconstruction.pmxlib.painting import (binned_spectrum, mass_moments,
                                                paint_uniform_random_2d)

SHOT_SPECIES = ('dm', 'gas')

SHOT_SEED = 12345
SHOT_CHUNK = 5e7

def measure_shot_spectra(cfg, nkbins):
    """P^shot_dd and P^shot_gg, in this pipeline's own k-binning convention."""
    print("\nSelf-pair (shot-noise) spectra, from uniformly random catalogues:")
    k_grid = compute_k_grid_2d(cfg.grid, cfg.box)
    particles_file = get_particle_file_path(cfg.feedback, sim_name=cfg.sim_name)
    rng = np.random.default_rng(SHOT_SEED)

    out = {}
    for species in SHOT_SPECIES:
        part = load_particle_properties(particles_file, species,
                                        requested=('mass',),
                                        sim_name=cfg.sim_name, Lbox=cfg.box)
        mass = np.asarray(part['mass'])
        del part
        gc.collect()

        M_tot, M2 = mass_moments(mass, chunk=SHOT_CHUNK)
        print(f"  [{species}] {mass.size:.4e} particles, "
              f"sum m^2 / (sum m)^2 = {M2 / M_tot ** 2:.6e}")

        dens = paint_uniform_random_2d(mass, cfg.box, cfg.grid, cfg.threads,
                                       rng, chunk=SHOT_CHUNK)
        delta = dens / (M_tot / cfg.grid ** 2) - 1.0
        del dens, mass
        gc.collect()
        fft_r = compute_2d_fft(delta, cfg.grid)
        del delta
        _, _, P_shot = binned_spectrum(cfg, np.abs(fft_r) ** 2, k_grid, nkbins)
        del fft_r
        gc.collect()

        out[f'P_shot_{species}'] = P_shot
        print(f"  [{species}] P^shot = {np.nanmedian(P_shot):.4e} (median over k)")

    return out


def self_pair_terms(data):
    """The self-pair content of each of the BUNDLE's spectra, keyed by name."""
    f_g = float(data['f_g'])
    P_dd = np.asarray(data['P_shot_dm'], dtype=float)
    P_gg = np.asarray(data['P_shot_gas'], dtype=float)
    return {
        'P_matter_gas': f_g * P_gg,
        'P_dm_dm': P_dd,
        'P_gas_gas': P_gg,
    }


def rstar_self_pair_terms(cache):
    """The same, for the R*/U* cache's spectra. Its ingredients, its keys."""
    f_g = float(cache['f_g'])
    P_gg = np.asarray(cache['P_shot_gg'], dtype=float)
    share = (np.asarray(cache['m2_bin_gas'], dtype=float)
             / float(cache['m2_tot_gas']))
    R_i = f_g * share[:, None] * P_gg[None, :]
    return {'R_star_gas': R_i, 'U_star_gas': f_g * P_gg - R_i.sum(axis=0)}


def subtract_self_pairs(data, terms=None, report=True):
    """Apply `terms` in place: canonical becomes corrected, raw kept alongside.

        <key>  = raw - selfpair,  with <key>_raw and <key>_selfpair beside it

    `terms` defaults to the bundle's; rstar_ustar passes rstar_self_pair_terms.
    One applier so both files mean the same thing by the raw/corrected names --
    they get differenced against each other. `report` off for per-bin arrays.
    """
    terms = self_pair_terms(data) if terms is None else terms
    applied = [key for key in terms if key in data]
    for key in applied:
        raw = np.asarray(data[key], dtype=float)
        data[f'{key}_raw'] = raw
        data[f'{key}_selfpair'] = np.asarray(terms[key], dtype=float)
        data[key] = raw - data[f'{key}_selfpair']

    if report:
        _report_self_pair_fractions(data, applied)
    return data


def _report_self_pair_fractions(data, keys):
    """What fraction of each raw spectrum the self-pair term was."""
    k = np.asarray(data['k_center'], dtype=float)
    k_Ny = float(data['k_Nyquist'])
    ks = [kk for kk in (0.1, 0.5, 1.0, 2.0, 3.0, 5.0) if kk <= k[-1]]

    print("\n[shot] self-pair fraction of the raw measured spectrum "
          "(removed from every one of them):")
    print("        k =  " + "".join(f"{kk:>9.2f}" for kk in ks))
    worst = {}
    for key in keys:
        raw = np.asarray(data[f'{key}_raw'], dtype=float)
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
            print(f"  NOTE: {key} was up to {w:.0%} self-pairs below k_Nyquist. "
                  f"It is removed, but do not read the high-k end of that curve "
                  f"as clustering: what is left there is a small difference of "
                  f"two large numbers.")
