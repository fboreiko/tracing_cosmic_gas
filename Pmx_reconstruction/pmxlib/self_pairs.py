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
------------------------------------------------------------------------------
"""
import gc

import numpy as np

from utils.catalog_loaders import load_particle_properties
from utils.pipeline_paths import get_particle_file_path
from Pmx_reconstruction.pmxlib.kspace import bin_spectrum, fft
from Pmx_reconstruction.pmxlib.painting import (mass_moments,
                                                paint_uniform_random_2d,
                                                paint_uniform_random_3d)

SHOT_SPECIES = ('dm', 'gas')

SHOT_SEED = 12345
SHOT_CHUNK = 5e7

def measure_shot_spectra(cfg, binning):
    """P^shot_dd and P^shot_gg on the grid `binning` was built for.

    Same code on the 2-D and 3-D grids. L^2 sum m^2 / (sum m)^2 is the exact
    expectation in the projected convention. The 3-D grid is deconvolved, so
    there the printed ratio tests the painting normalisation and the window
    at once; the 2-D one is not, so there it is ~1 only where W ~ 1.
    """
    dim, ngrid = binning['dim'], binning['ngrid']
    paint = paint_uniform_random_2d if dim == 2 else paint_uniform_random_3d
    print(f"\nSelf-pair (shot-noise) spectra on the {ngrid}^{dim} grid, from "
          f"uniformly random catalogues:")
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
        expected = cfg.box ** 2 * M2 / M_tot ** 2

        dens = paint(mass, cfg.box, ngrid, cfg.threads, rng, chunk=SHOT_CHUNK)
        delta = dens / (M_tot / ngrid ** dim) - 1.0
        del dens, mass
        gc.collect()
        fft_r = fft(delta)
        del delta
        P_shot = bin_spectrum(np.abs(fft_r) ** 2, binning)
        del fft_r
        gc.collect()

        out[f'P_shot_{species}'] = P_shot
        med = float(np.nanmedian(P_shot))
        print(f"  [{species}] P^shot = {med:.4e}, analytic {expected:.4e}, "
              f"ratio {med / expected:.4f} (~1 where the window is ~1 or divided out)")

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


def smeared_share(cache):
    """sum_{q in i} m_q^2 j0(k r_q)^2 / sum_gas m_q^2, shape (nbins, nk).
    The self-pair share of bin i AFTER its members have been randomised.
    """
    W = np.asarray(cache['m2_shell_gas'], dtype=float)
    S = np.asarray(cache['m2r_shell_gas'], dtype=float)
    k = np.asarray(cache['k_center'], dtype=float)
    with np.errstate(invalid='ignore', divide='ignore'):
        r_bar = np.where(W > 0, S / np.where(W > 0, W, 1.0), 0.0)
    out = np.zeros((W.shape[0], k.size))
    for i in range(W.shape[0]):
        occ = W[i] > 0
        if not np.any(occ):
            continue
        # np.sinc(x) = sin(pi x)/(pi x), so j0(k r) = np.sinc(k r / pi).
        j0 = np.sinc(np.outer(r_bar[i][occ], k) / np.pi)
        out[i] = W[i][occ] @ (j0 ** 2)
    return out / float(cache['m2_tot_gas'])


def rstar_self_pair_terms(cache, randomised=False):
    f_g = float(cache['f_g'])
    P_gg = np.asarray(cache['P_shot_gg'], dtype=float)
    share = (np.asarray(cache['m2_bin_gas'], dtype=float)
             / float(cache['m2_tot_gas']))
    R_i = f_g * share[:, None] * P_gg[None, :]
    U = f_g * P_gg - R_i.sum(axis=0)
    if not randomised:
        return {'R_star_gas': R_i, 'U_star_gas': U}
    if 'm2_shell_gas' not in cache:
        print("  [shot] WARNING: this randomised cache predates the smeared "
              "self-pair correction and carries no m2_shell_gas, so R*_i is "
              "left UNCORRECTED and is biased high by up to the full "
              "self-pair term. Re-measure it with --randomise ... "
              "--recompute.")
        return {'U_star_gas': U}
    return {'R_star_gas': f_g * smeared_share(cache) * P_gg[None, :],
            'U_star_gas': U}


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
