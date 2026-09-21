#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The large-scale half of the spectra bundle, measured on a cubic grid.

Projecting to k_z = 0 costs a factor 2k/k_f in modes -- 4.7x in sigma at
k = 0.1 and 10x at k = 0.5 -- and buys nothing below the projection scale. So
the same spectra are re-measured on a small cubic grid, converted to the 2-D
convention by painting.as_projected, and spliced in below --k-split.

Everything here carries the SAME KEY NAMES as the 2-D bundle, on the same k
bin edges, so the splice in bundle.stitch is a straight assignment and
self_pairs.subtract_self_pairs works on either dict unchanged.

The price is one particle read per species: the cached 2-D delta fields cannot
be un-projected.
"""
import gc

import numpy as np

from utils.catalog_loaders import load_particle_properties
from utils.pipeline_paths import (DATA_ROOT, ensure_parents,
                                  get_particle_file_path)
from utils.power_spectrum_utils import Binner3D, compute_3d_fft

from Pmx_reconstruction.pmxlib.binning import load_binned_halos, log_mass_bin_edges
from Pmx_reconstruction.pmxlib.painting import (binned_spectrum_3d, overdensity_3d,
                                                paint_3d)
from Pmx_reconstruction.pmxlib.self_pairs import measure_shot_spectra_3d

BUNDLE3D_VERSION = 'v1'

# What stitch_on_k splices: the bundle's spectra and the R*/U* cache's, plus
# the _raw / _selfpair companions subtract_self_pairs leaves beside them.
_STITCH_PREFIXES = ('P_', 'R_star', 'U_star')


def bundle3d_path(cfg, nbins, logm_min, logm_max, ngrid3):
    stem = (f'spectra3d_{BUNDLE3D_VERSION}_{cfg.feedback}_{cfg.mass_def}'
            f'_logM{logm_min:g}-{logm_max:g}_nb{nbins}'
            f'_n3d{ngrid3}_{cfg.kbin_tag}'
            f'_{"cen" if cfg.centrals_only else "all"}.npz')
    return DATA_ROOT / cfg.sim_name / 'pme_inputs' / stem


def species_delta_3d(cfg, species, ngrid3):
    """Painted 3-D overdensity of one particle species.

    Same field the 2-D path builds, minus the projection: TSC with the
    per-particle masses as weights, divided by the global mean.
    """
    particles_file = get_particle_file_path(cfg.feedback, sim_name=cfg.sim_name)
    part = load_particle_properties(particles_file, species,
                                    requested=('mass', 'pos'),
                                    sim_name=cfg.sim_name, Lbox=cfg.box)
    pos = np.mod(np.asarray(part['pos'], dtype=np.float32), np.float32(cfg.box))
    pos[pos >= cfg.box] -= np.float32(cfg.box)
    mass = np.asarray(part['mass'], dtype=np.float64)
    del part
    gc.collect()
    print(f"    {pos.shape[0]:.3e} {species} particles")

    dens = paint_3d(pos, mass, cfg.box, ngrid3, cfg.threads)
    del pos, mass
    gc.collect()
    return overdensity_3d(dens, ngrid3)


def measure_spectra_3d(cfg, nbins, logm_min, logm_max, k_bins,
                       f_c, f_g, ngrid3):
    """measure_spectra's 3-D twin: same spectra, same k edges, same keys."""
    print("=" * 70)
    print(f"Measuring the large scales on a {ngrid3}^3 grid")
    print("=" * 70)

    binner = Binner3D(ngrid3, cfg.box, k_bins)
    k_center = binner.k_center
    print(f"  k_Nyquist = {binner.k_Nyquist:.3f} h/cMpc, cell = "
          f"{cfg.box / ngrid3:.3f} cMpc/h, TSC window deconvolved")

    def _binned(prod):
        return binned_spectrum_3d(cfg, prod, binner)

    print("\n3-D fields:")
    gas_fft = compute_3d_fft(species_delta_3d(cfg, 'gas', ngrid3), ngrid3)
    dm_fft = compute_3d_fft(species_delta_3d(cfg, 'dm', ngrid3), ngrid3)
    gc.collect()

    print(f"\n  matter field from the bundle's fractions: "
          f"f_c={f_c:.5f}, f_g={f_g:.5f}")
    m_fft = f_c * dm_fft + f_g * gas_fft

    P_matter_gas = _binned((m_fft * np.conj(gas_fft)).real)
    P_dm_gas = _binned((dm_fft * np.conj(gas_fft)).real)
    P_dm_dm = _binned(np.abs(dm_fft) ** 2)
    P_gas_gas = _binned(np.abs(gas_fft) ** 2)
    del m_fft
    gc.collect()

    shot = measure_shot_spectra_3d(cfg, ngrid3, binner)

    print("\nHalo catalogue:")
    all_pos, all_mass, bin_index, logM_edges, _ = load_binned_halos(cfg)
    _, logM_cen, M_cen = log_mass_bin_edges(nbins, logm_min, logm_max)

    counts = np.zeros(nbins, dtype=np.float64)
    M_mean = np.zeros(nbins, dtype=np.float64)
    P_halo_gas = np.zeros((nbins, k_center.size), dtype=np.float64)
    P_halo_dm = np.zeros((nbins, k_center.size), dtype=np.float64)

    print("\nPer-bin halo cross spectra:")
    for i in range(nbins):
        sel = bin_index == i
        n_in_bin = int(np.count_nonzero(sel))
        counts[i] = n_in_bin
        if n_in_bin == 0:
            print(f"  bin {i:2d}  logM = {logM_cen[i]:5.2f}  EMPTY, skipped")
            continue

        M_mean[i] = float(np.mean(all_mass[sel]))
        dens_h = paint_3d(all_pos[sel], None, cfg.box, ngrid3, cfg.threads)
        halo_fft = compute_3d_fft(overdensity_3d(dens_h, ngrid3), ngrid3)
        del dens_h

        P_halo_gas[i] = _binned((gas_fft * np.conj(halo_fft)).real)
        P_halo_dm[i] = _binned((dm_fft * np.conj(halo_fft)).real)
        del halo_fft
        gc.collect()

        print(f"  bin {i:2d}  logM = {logM_cen[i]:5.2f}  N = {n_in_bin:8d}  "
              f"P_halo_gas(k_min) = {P_halo_gas[i, 0]:.4e}")

    del gas_fft, dm_fft, all_pos, all_mass
    gc.collect()

    M_mean = np.where(counts > 0, M_mean, M_cen)

    return dict(
        k_bins=np.asarray(k_bins),
        k_center=k_center,
        k_Nyquist=binner.k_Nyquist,
        k_Nyquist_3d=binner.k_Nyquist,
        k_eff=binner.k_eff,
        nmodes=binner.nmodes,
        nmodes_eff=binner.nmodes_eff,
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
        ngrid_3d=ngrid3,
        redshift=cfg.z,
        mass_def=cfg.mass_def,
        centrals_only=cfg.centrals_only,
        feedback=cfg.feedback,
    )


_REQUIRED_KEYS_3D = ('k_center', 'nmodes', 'nmodes_eff', 'k_eff', 'counts',
                     'M_mean',
                     'P_halo_gas', 'P_halo_dm', 'P_matter_gas', 'P_dm_gas',
                     'P_dm_dm', 'P_gas_gas', 'P_shot_dm', 'P_shot_gas')


def load_or_measure_3d(cfg, nbins, logm_min, logm_max, k_bins,
                       f_c, f_g, recompute=False):
    """Load the cached 3-D bundle if it exists, otherwise measure and write it.

    The k edges are the 2-D run's, so a cache built against a different bundle
    is refused rather than interpolated onto.
    """
    ngrid3 = cfg.grid_3d
    path = bundle3d_path(cfg, nbins, logm_min, logm_max, ngrid3)

    data = None
    if path.exists() and not recompute:
        print(f"Loading pre-computed 3-D bundle:\n  {path}")
        with np.load(path, allow_pickle=False) as f:
            data = {key: f[key] for key in f.files}
        missing = [key for key in _REQUIRED_KEYS_3D if key not in data]
        if missing:
            print(f"  Bundle is missing {missing}; re-measuring.")
            data = None
        elif not np.allclose(np.asarray(data['k_bins'], float), k_bins):
            print("  Cached 3-D bundle sits on different k edges from the "
                  "2-D bundle; re-measuring.")
            data = None
        else:
            print(f"  {data['P_halo_gas'].shape[0]} mass bins, "
                  f"{data['k_center'].size} k bins. Skipping measurement.")

    if data is None:
        if recompute and path.exists():
            print("--recompute-3d given: ignoring the existing 3-D bundle.")
        data = measure_spectra_3d(cfg, nbins, logm_min, logm_max,
                                  k_bins, f_c, f_g, ngrid3)
        ensure_parents(path)
        np.savez_compressed(path, **data)
        print(f"\n3-D bundle saved:\n  {path}")

    return data


# The splice
def stitch_on_k(data, d3, k_split, report=True, label='bundle'):
    """Replace k < k_split in every P_* spectrum with the 3-D measurement.

    Both sides must already be self-pair corrected: each is corrected with its
    OWN shot spectra, then spliced, so no bin mixes a 2-D shot term with a 3-D
    signal. The pre-splice arrays are kept as <key>_2d and <key>_3d, which is
    what lets rstar_ustar hand the right P^shot_gg to each of its two caches.
    """
    k = np.asarray(data['k_center'], dtype=float)
    nk = k.size
    take = (k < float(k_split)) & (np.asarray(d3['nmodes'], dtype=float) > 0)

    stitched = []
    for key in list(data):
        if not key.startswith(_STITCH_PREFIXES) or key.endswith(('_2d', '_3d')):
            continue
        if key not in d3:
            continue
        v = np.asarray(data[key], dtype=float)
        v3 = np.asarray(d3[key], dtype=float)
        if v.ndim == 0 or v.shape[-1] != nk or v3.shape != v.shape:
            continue
        out = v.copy()
        out[..., take] = v3[..., take]
        data[f'{key}_2d'], data[f'{key}_3d'] = v, v3
        data[key] = out
        stitched.append(key)

    data['k_split'] = float(k_split)
    data['is_3d'] = take
    data['nmodes_3d'] = np.asarray(d3['nmodes'], dtype=float)
    if 'ngrid_3d' in d3:
        data['ngrid_3d'] = int(d3['ngrid_3d'])
    # Diagnostics, carried when the 3-D side has them; only report_stitch reads
    # them, and only the bundle goes through that.
    for src, dst in (('nmodes_eff', 'nmodes_eff_3d'), ('k_eff', 'k_eff_3d'),
                     ('k_Nyquist_3d', 'k_Nyquist_3d')):
        if src in d3:
            data[dst] = np.asarray(d3[src], dtype=float)

    if report:
        print(f"\n[3d] {label}: {np.count_nonzero(take)} of {nk} k bins taken "
              f"from the {int(d3.get('ngrid_3d', 0))}^3 grid "
              f"(k < {k_split:g}); {len(stitched)} spectra spliced")
    return data


def report_stitch(data, ks=(0.03, 0.05, 0.1, 0.2, 0.3, 0.5)):
    """The mode count either side of the splice, and the overlap-band ratio.

    k_eff_3d against k_eff_2d is the cheap check that the two halves weight
    the inside of a bin the same way; the ratio is the check that they measure
    the same thing. A flat offset near 0.98 is an undeconvolved window, one
    near L_box is the projection factor, and scatter at the level of sigma_2d
    with no offset is what a correct splice looks like.
    """
    k = np.asarray(data['k_center'], dtype=float)
    n2 = np.asarray(data['nmodes_2d'], dtype=float)
    n3 = np.asarray(data['nmodes_eff_3d'], dtype=float)
    ke2 = np.asarray(data['k_eff_2d'], dtype=float)
    ke3 = np.asarray(data['k_eff_3d'], dtype=float)
    k_split = float(data['k_split'])

    print("\n[3d] modes per k bin and the sigma ~ 1/sqrt(N) they imply:")
    print("        k      N_2d   sigma_2d      N_3d   sigma_3d     gain"
          "   k_eff_3d/k_eff_2d")
    for kk in ks:
        if kk >= k[-1]:
            continue
        i = int(np.argmin(np.abs(k - kk)))
        if n2[i] <= 0 or n3[i] <= 0:
            continue
        s2, s3 = 1 / np.sqrt(n2[i]), 1 / np.sqrt(n3[i])
        print(f"  {k[i]:7.3f}  {n2[i]:8.0f}   {s2:8.2%}  {n3[i]:9.0f}   "
              f"{s3:8.2%}   {s2 / s3:6.2f}x   {ke3[i] / ke2[i]:14.6f}")

    band = (k >= 0.5 * k_split) & (k < k_split)
    if not band.any():
        return
    exp = float(np.nanmean(1 / np.sqrt(n2[band])))
    print(f"\n[3d] overlap band {0.5 * k_split:g} < k < {k_split:g}: "
          f"3-D / 2-D per spectrum (1 = consistent, "
          f"scatter ~{exp:.4f} expected)")
    for key in sorted(nm for nm in data
                      if nm.startswith('P_') and nm.endswith('_3d')):
        base = key[:-3]
        v2 = np.asarray(data[f'{base}_2d'], dtype=float)
        v3 = np.asarray(data[key], dtype=float)
        if v2.ndim != 1:
            continue
        with np.errstate(divide='ignore', invalid='ignore'):
            r = v3[band] / v2[band]
        r = r[np.isfinite(r)]
        if r.size == 0:
            continue
        print(f"  {base:22s} mean {np.mean(r):6.4f}  scatter {np.std(r):6.4f}")
