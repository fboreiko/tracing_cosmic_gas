#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Self-pair (shot-noise) spectra and their subtraction.

<delta_m delta_gas*>   = f_c P_dm_gas + f_g P_gas_gas
... and P_gas_gas carries P^shot_gg
<delta_m delta_dm*>    = f_c P_dm_dm + f_g P_dm_gas
... and P_dm_dm carries P^shot_dd
<delta_m delta_m*>     = f_c^2 P_dm_dm + 2 f_c f_g P_dm_gas
+ f_g^2 P_gas_gas

so the self-pair content of the three truths is

P_matter_gas    : f_g   P^shot_gg
P_matter_dm     : f_c   P^shot_dd
P_matter_matter : f_c^2 P^shot_dd + f_g^2 P^shot_gg
P_dm_dm         :       P^shot_dd
P_gas_gas       :       P^shot_gg

HOW P^shot IS MEASURED
Each species is painted at uniformly random 2D positions with its real
per-particle masses and the auto spectrum of the resulting field is taken
through binned_spectrum(). A random catalogue has no clustering, so in
expectation that auto spectrum IS the self-pair term of the real field.
------------------------------------------------------------------------------
"""
import gc

import numpy as np

from utils.catalog_loaders import load_particle_properties
from utils.pipeline_paths import (DATA_ROOT, ensure_parents,
                                  get_particle_file_path)
from utils.power_spectrum_utils import compute_2d_fft, compute_k_grid_2d

from Pmx_reconstruction.pmxlib.painting import (binned_spectrum, mass_moments,
                                                paint_uniform_random_2d)

SHOT_SPECIES = ('dm', 'gas')

def shot_path(cfg, ngrid, nkbins, nreal=None, seed=None):
    """Cache for the self-pair spectra, beside the spectra bundle."""
    nreal = cfg.shot_nreal if nreal is None else nreal
    seed = cfg.shot_seed if seed is None else seed
    nk = 'default' if nkbins is None else str(nkbins)
    stem = (f'shotnoise_v1_{cfg.feedback}'
            f'_ngrid{ngrid}_nk{nk}_nreal{nreal}_seed{seed}.npz')
    return DATA_ROOT / cfg.sim_name / 'pme_inputs' / stem


def measure_shot_spectra(cfg, nkbins, nreal=None, seed=None,
                         chunk=None):
    """Measure P^shot_dd and P^shot_gg in this pipeline's own convention."""
    nreal = cfg.shot_nreal if nreal is None else nreal
    seed = cfg.shot_seed if seed is None else seed
    chunk = cfg.shot_chunk if chunk is None else chunk
    nreal = max(1, int(nreal))
    print("=" * 70)
    print("Measuring self-pair (shot-noise) spectra (no cached file found)")
    print("=" * 70)

    k_grid = compute_k_grid_2d(cfg.grid, cfg.box)
    particles_file = get_particle_file_path(cfg.feedback, sim_name=cfg.sim_name)
    rng = np.random.default_rng(seed)

    out = dict(ngrid=cfg.grid, box=cfg.box, feedback=cfg.feedback, redshift=cfg.z,
               nreal=nreal, seed=seed)
    for species in SHOT_SPECIES:
        print(f"\n[{species}] loading particle masses ...")
        part = load_particle_properties(particles_file, species,
                                        requested=('mass',),
                                        sim_name=cfg.sim_name, Lbox=cfg.box)
        mass = np.asarray(part['mass'])
        del part
        gc.collect()

        n_p = mass.size
        M_tot, M2 = mass_moments(mass, chunk=chunk)
        print(f"    {n_p:.4e} particles, sum m = {M_tot:.6e}, "
              f"sum m^2 / (sum m)^2 = {M2 / M_tot ** 2:.6e}")

        acc = None
        for r in range(nreal):
            print(f"[{species}] painting random realisation {r + 1}/{nreal} ...")
            dens = paint_uniform_random_2d(
                mass, cfg.box, cfg.grid, cfg.threads, rng, chunk=chunk)
            delta = dens / (M_tot / cfg.grid ** 2) - 1.0
            del dens
            fft_r = compute_2d_fft(delta, cfg.grid)
            del delta
            k_bins, k_center, Pk = binned_spectrum(
                cfg, np.abs(fft_r) ** 2, k_grid, nkbins)
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

        del mass
        gc.collect()

    return out


_SHOT_REQUIRED_KEYS = ('k_center', 'P_shot_dm', 'P_shot_gas')


def load_or_measure_shot(cfg, nkbins, recompute=False, nreal=None,
                         seed=None, chunk=None, allow_measure=True):
    """Load the cached self-pair spectra, otherwise measure and write them."""
    nreal = cfg.shot_nreal if nreal is None else nreal
    seed = cfg.shot_seed if seed is None else seed
    chunk = cfg.shot_chunk if chunk is None else chunk
    path = shot_path(cfg, cfg.grid, nkbins, nreal=nreal, seed=seed)

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

    shot = measure_shot_spectra(cfg, nkbins, nreal=nreal, seed=seed, chunk=chunk)
    ensure_parents(path)
    np.savez_compressed(path, **shot)
    print(f"\nSelf-pair spectra saved:\n  {path}")
    return shot


def self_pair_terms(cfg, data, shot):
    """The self-pair content of every spectrum"""
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
    """Which of the bundle's spectra have a self-pair term at all."""
    return [key for key in ('P_matter_gas', 'P_matter_dm', 'P_matter_matter',
                            'P_dm_dm', 'P_gas_gas')
            if key in data]


_LEGACY_BLOCK_KEYS = ('P_dm_gas_selfpair', 'P_dm_gas_corrected')


def ensure_self_pair_block(cfg, data, nkbins, allow_measure=True, recompute=False,
                           nreal=None, seed=None, chunk=None):
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
    nreal = cfg.shot_nreal if nreal is None else nreal
    seed = cfg.shot_seed if seed is None else seed
    chunk = cfg.shot_chunk if chunk is None else chunk
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

    shot = load_or_measure_shot(cfg, nkbins, recompute=recompute, nreal=nreal,
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

    terms = self_pair_terms(cfg, data, shot)
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


def use_self_pair_corrected(cfg, data, subtract=True):
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
    # local import: bundle imports this module, so this cannot be top-level
    from Pmx_reconstruction.pmxlib.bundle import derive_P_halo_matter
    derive_P_halo_matter(cfg, data)

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