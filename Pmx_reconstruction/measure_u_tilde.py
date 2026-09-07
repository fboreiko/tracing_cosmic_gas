#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Measure the true per-bin matter profile u~_m(k|M_i) from the particle fields,
and with it R_star, U_star, rho_u and the two error terms of Equation (59).

THE PARTITION (Equations 55-58 of the notes)
    Every dm and gas particle is assigned to at most ONE catalogue central: the
    most massive central whose r200b sphere contains it (centrals processed in
    descending mass, first assignment wins). Particles inside no sphere, or
    inside the sphere of a central outside [10^LOGM_MIN, 10^LOGM_MAX), are
    "out". This is a partition of the particles, hence exact and linear:

        delta_m = sum_i delta_m^(i) + delta_m^out,

    where delta_m^(i) = rho_m^(i) / rhobar_m is the density of bin i's
    particles normalised by the GLOBAL mean (not the bin's own), so that the
    sum reproduces 1 + delta_m mode by mode. Crossing with a tracer x,

        P_matter_x = sum_i R_star_i + U_star,
        R_star_i(k) = <delta_m^(i) delta_x*>,   U_star(k) = <delta_m^out delta_x*>.

    R_star_i is the bin's TRUE contribution to the cross spectrum -- it is
    exactly what the reconstruction's f_i u_m^NFW P_halo_x(k;M_i) is standing
    in for. Dividing gives the tracer-cross-weighted mean profile of the bin,

        u~_m(k|M_i) = R_star_i(k) / [ f_i P_halo_x(k;M_i) ],            Eq. (56)

    which is what u_m^NFW is being compared against, and

        rho_u(k) = R / R_star = sum_i omega_i u_m^NFW / u~_m.           Eq. (57)

    u~_m depends on the TRACER (it is weighted by each halo's own cross-
    correlation C_j with x), so it is measured separately for x = gas and dm.
    The first-moment profile <u_m>_i -- the plain mass-weighted mean of the
    bin's matter profiles, no tracer weighting -- is measured too, by stacking
    the same particles on a common centre. The gap u~_m - <u_m> is the tracer
    weighting; the gap <u_m> - u_m^NFW is the c(M) relation and truncation.

    U_star is built as delta_m - sum_i delta_m^(i) in Fourier space, so the
    closure R_star + U_star = P_matter_x holds to round-off by construction;
    the particle bookkeeping is checked independently on the mass sums.

WHAT IS SAVED
    One npz per (binning, grid) in DATA_ROOT/<sim>/pme_inputs/, next to the
    spectra bundle, holding R_star_<x>, U_star_<x>, u_tilde_<x>, u_stack,
    u_nfw, the assigned mass per bin and the closure diagnostics. The plot step
    reads the spectra bundle for P_halo_x, P_matter_x and the reconstruction.

COST
    One pass over the particles per species (load, periodic KD-tree,
    membership), then NBINS TSC paintings and FFTs per species, plus NBINS
    more for the stacks. Memory: the particle positions of one species at a
    time (as the existing pipeline already does), the KD-tree on them, one
    int32 label per particle, and NBINS x ngrid^2 float64 accumulators (~1 GB
    at ngrid = 2048, NBINS = 30). Use --skip-stack to drop the stacks.

USAGE  (repo root, like predict_Pmx_from_Phx.py; the spectra bundle must exist)
        python measure_u_tilde.py                    # measure, then plot
        python measure_u_tilde.py --plot-only        # re-plot from cache
        python measure_u_tilde.py --recompute
        python measure_u_tilde.py --nbins 30 --logm-min 11 --logm-max 15
        python measure_u_tilde.py --concentration colossus   # for the u_nfw curve
        python measure_u_tilde.py --show-bins 12.0 13.0 14.0 14.7
--------------------------------------------------------------------------------
"""
import argparse
import gc
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from abacusnbody.analysis.tsc import tsc_parallel

import tracing_cosmic_gas.Pmx_reconstruction.predict_Pmx_from_Phx as P
from utils.catalog_loaders import load_halo_properties, load_particle_properties
from utils.power_spectrum_utils import (compute_2d_fft, compute_k_grid_2d,
                                        bin_power_spectrum_2d)
from utils.pipeline_paths import (DATA_ROOT, ensure_parents,
                                  get_halo_file_path, get_particle_file_path,
                                  plot_path)
from utils.plot_data import save_plot_data

TRACER_COLOR = {'gas': 'C1', 'dm': 'C0', 'matter': 'C2'}
SPECIES = ('dm', 'gas')


# ==============================================================================
# 1.  Paths
# ==============================================================================
def ustar_path(nbins, logm_min, logm_max, ngrid, nkbins):
    """Companion cache to bundle_path: same stem, different prefix."""
    nk = 'default' if nkbins is None else str(nkbins)
    stem = (f'ustar_v3_{P.FEEDBACK}_{P.MASS_DEF}'
            f'_logM{logm_min:g}-{logm_max:g}_nb{nbins}'
            f'_ngrid{ngrid}_nk{nk}'
            f'_{"cen" if P.CENTRALS_ONLY else "all"}.npz')
    return DATA_ROOT / P.SIM_NAME / 'pme_inputs' / stem


# ==============================================================================
# 2.  Halo catalogue, binned exactly as measure_spectra does it
# ==============================================================================
def load_binned_centrals(nbins, logm_min, logm_max):
    """Positions, masses, radii and bin index of the in-range centrals.

    Reproduces the binning of measure_spectra bit for bit (same edges, same
    right-edge rule, same centrals cut), so bin i here is bin i in the bundle.
    r200b is NOT taken from the catalogue but recomputed from M200b with the
    module's r200m_of_M, i.e. with exactly the truncation radius u_nfw assumes.
    The catalogue value is loaded only for a consistency print.
    """
    halo_file = get_halo_file_path(P.FEEDBACK, sim_name=P.SIM_NAME)
    print(f"  {halo_file}")
    requested = ['pos', P.MASS_DEF, 'r200b']
    if P.CENTRALS_ONLY:
        requested.append('centrals')
    props = load_halo_properties(halo_file, requested, sim_name=P.SIM_NAME)

    pos = np.asarray(props['pos'], dtype=np.float64)
    mass = np.asarray(props[P.MASS_DEF], dtype=np.float64) * P.HALO_MASS_UNIT_MSUN_H
    r_cat = np.asarray(props['r200b'], dtype=np.float64)
    if P.CENTRALS_ONLY:
        cen = np.asarray(props['centrals']).astype(bool)
        pos, mass, r_cat = pos[cen], mass[cen], r_cat[cen]
    del props
    gc.collect()

    logM_edges = np.linspace(logm_min, logm_max, nbins + 1)
    with np.errstate(divide='ignore', invalid='ignore'):
        logm = np.log10(mass)
    in_range = np.isfinite(logm) & (logm >= logm_min) & (logm < logm_max)
    bin_index = np.full(mass.size, -1, dtype=np.int64)
    bin_index[in_range] = np.digitize(logm[in_range], logM_edges) - 1
    bin_index[bin_index == nbins] = nbins - 1

    pos, mass, r_cat, bin_index = (pos[in_range], mass[in_range],
                                   r_cat[in_range], bin_index[in_range])
    r_use = P.r200m_of_M(mass)                       # cMpc/h, comoving

    # Unit check on the catalogue radius: it should agree with r200m_of_M up to
    # the rhobar_m convention. A ratio near h or 1/h means the catalogue is in
    # cMpc rather than cMpc/h; either way we use r_use, this is informational.
    finite = np.isfinite(r_cat) & (r_cat > 0)
    if np.any(finite):
        ratio = np.median(r_cat[finite] / r_use[finite])
        print(f"  catalogue r200b / r200m_of_M(M200b): median {ratio:.4f} "
              f"(1 = same units and convention; {P.H_LITTLE:.3f} or "
              f"{1 / P.H_LITTLE:.3f} = cMpc vs cMpc/h)")

    pos = np.mod(pos, P.BOX)                         # cKDTree boxsize wants [0, L)
    order = np.argsort(-mass, kind='stable')         # descending mass
    print(f"  {mass.size} centrals in [{logm_min}, {logm_max}), "
          f"r_max = {r_use.max():.3f} cMpc/h")
    return (pos[order], mass[order], r_use[order], bin_index[order],
            logM_edges)


# ==============================================================================
# 3.  Particle membership
# ==============================================================================
def assign_particles(pos_p, halo_pos, halo_r, halo_mass, m_particle_msun_h,
                     chunk_particles=5e7, nthread=4):
    """label[p] = index (into the descending-mass halo arrays) of the most
    massive central whose r200m sphere contains particle p, or -1.

    Periodic KD-tree on the particles; halos are walked in descending mass in
    chunks sized so that the expected number of member particles per chunk is
    ~chunk_particles, which bounds the transient Python-list memory of
    query_ball_point.
    """
    n_p = pos_p.shape[0]
    label = np.full(n_p, -1, dtype=np.int32)
    t0 = time.time()
    tree = cKDTree(pos_p, boxsize=P.BOX, leafsize=64, balanced_tree=False,
                   compact_nodes=False)
    print(f"    KD-tree on {n_p:.3e} particles: {time.time() - t0:.1f} s")

    expected = halo_mass / m_particle_msun_h          # rough member count
    cum = np.cumsum(expected)
    edges = np.searchsorted(cum, np.arange(0.0, cum[-1], chunk_particles))
    edges = np.unique(np.append(edges, halo_mass.size))

    n_assigned = 0
    t0 = time.time()
    for a, b in zip(edges[:-1], edges[1:]):
        lists = tree.query_ball_point(halo_pos[a:b], halo_r[a:b],
                                      workers=(nthread or 1),
                                      return_sorted=False)
        for j, idx in enumerate(lists):
            if not idx:
                continue
            idx = np.asarray(idx, dtype=np.int64)
            free = label[idx] < 0
            label[idx[free]] = a + j
            n_assigned += int(np.count_nonzero(free))
        del lists
    print(f"    membership: {n_assigned:.3e} of {n_p:.3e} particles "
          f"({n_assigned / n_p:.3%}) inside a central's r200m, "
          f"{time.time() - t0:.1f} s")
    del tree
    gc.collect()
    return label


# ==============================================================================
# 4.  Painting
# ==============================================================================
# Number of z cells used by paint(). See the note there; 3 is the smallest
# value that is safe against the _rightwrap index range.
_NZ_PAINT = 3


def paint(pos, weights, ngrid, nthread):
    """Mass on the 2-D grid (column sum), no normalisation.

    NOT a call to tsc_parallel with a 2-D grid. In abacusutils 2.1.2 the 2-D
    branch of _tsc_scatter is broken: it sets izw = int16(0) but still writes
    density[ix, iy, izw], i.e. three indices into a two-dimensional array, so
    numba fails to type it. (compute_delta_2d in utils/ hits the same bug on
    this version; if the cached delta fields were built with an older
    abacusutils, that is why they exist and this did not.)

    Instead we paint into a genuine 3-D grid with _NZ_PAINT cells along z and
    sum over z. The TSC weights in z sum to unity for every particle, so the
    column sum is EXACTLY the 2-D result and is independent of _NZ_PAINT -- it
    is not an approximation, and it is not a different smoothing. _NZ_PAINT = 1
    would be the obvious choice but is unsafe: iz = round(z/box) can be 1, and
    then izp1 = _rightwrap(2, 1) = 1 indexes past the end of a size-1 axis,
    which numba does not bounds-check.

    Weights are normalised by their mean before painting and the mean restored
    afterwards, so the float32 accumulation happens on numbers of order unity
    rather than on raw particle masses.
    """
    pos = np.ascontiguousarray(pos, dtype=np.float32)   # tsc wraps this IN PLACE
    w = np.asarray(weights, dtype=np.float64)
    w_mean = float(w.mean()) if w.size else 1.0
    if not np.isfinite(w_mean) or w_mean == 0.0:
        w_mean = 1.0
    grid = np.zeros((ngrid, ngrid, _NZ_PAINT), dtype=np.float32)
    tsc_parallel(pos, grid, P.BOX,
                 weights=np.ascontiguousarray(w / w_mean, dtype=np.float32),
                 nthread=(nthread or 1))
    return grid.sum(axis=2, dtype=np.float64) * w_mean


def tsc_window_2d(ngrid):
    """W(k) = prod_i sinc^3(k_i Delta/2) on the fft2 layout, Delta = L/ngrid.

    One factor of the TSC assignment window. Every spectrum in the pipeline is
    a product of two painted fields and carries W^2, which cancels in the ratio
    u~_m = R_star_i / (f_i P_halo_x). A stacked profile is ONE painted field and
    carries a single W that does not cancel; it must be divided out, or the
    stack reads low by 10% at k = 2 and 30% at k = 5 for ngrid = 2048.
    """
    kf = 2.0 * np.pi * np.fft.fftfreq(ngrid, d=P.BOX / ngrid)
    arg = 0.5 * kf * (P.BOX / ngrid)
    s = np.ones_like(arg)
    nz = arg != 0
    s[nz] = np.sin(arg[nz]) / arg[nz]
    return (s[:, None] * s[None, :]) ** 3


def azimuthal_mean(field, ngrid, k_bins, deconvolve_tsc=False):
    """Azimuthal average of a full fft2 layout field over the bundle's k bins.

    Self-contained (own fftfreq grid), used only for the profile transforms,
    where the absolute normalisation is fixed by dividing by the k=0 mode and
    the repo's binning helper is not needed. With deconvolve_tsc the field is
    divided by one TSC window before averaging.
    """
    if deconvolve_tsc:
        field = field / tsc_window_2d(ngrid)
    kf = 2.0 * np.pi * np.fft.fftfreq(ngrid, d=P.BOX / ngrid)
    kk = np.sqrt(kf[:, None] ** 2 + kf[None, :] ** 2)
    ib = np.digitize(kk.ravel(), k_bins) - 1
    ok = (ib >= 0) & (ib < k_bins.size - 1)
    num = np.bincount(ib[ok], weights=field.ravel()[ok], minlength=k_bins.size - 1)
    den = np.bincount(ib[ok], minlength=k_bins.size - 1).astype(float)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(den > 0, num / den, np.nan)


# ==============================================================================
# 5.  The measurement
# ==============================================================================
def measure(nbins, logm_min, logm_max, nkbins, data, skip_stack=False,
            chunk_particles=5e7):
    ngrid, nthread = P.NGRID, P.NTHREAD
    k_bins = np.asarray(data['k_bins'])
    k_min = 2.0 * np.pi / P.BOX
    k_grid = compute_k_grid_2d(ngrid, P.BOX)

    def _binned(prod):
        kb, kc, Pk = bin_power_spectrum_2d(prod, k_grid, ngrid, P.BOX,
                                           nkbins=nkbins, k_min=k_min)
        return np.asarray(kc), np.asarray(Pk) * P.BOX ** 2

    # --- tracer fields, exactly as measure_spectra builds them ----------------
    print("\nProjected tracer fields:")
    gas_fft = compute_2d_fft(P._load_or_compute_delta('gas'), ngrid)
    dm_fft = compute_2d_fft(P._load_or_compute_delta('dm'), ngrid)
    f_c, f_g = float(data['f_c']), float(data['f_g'])
    m_fft = f_c * dm_fft + f_g * gas_fft
    tracer_fft = {'gas': gas_fft, 'dm': dm_fft}

    # --- centrals --------------------------------------------------------------
    print("\nHalo catalogue:")
    (h_pos, h_mass, h_r, h_bin, logM_edges) = load_binned_centrals(
        nbins, logm_min, logm_max)
    if not np.allclose(logM_edges, data['logM_edges']):
        raise SystemExit("Mass-bin edges differ from the spectra bundle; "
                         "use the same --nbins/--logm-min/--logm-max.")
    n_halo = h_mass.size
    h_bin16 = h_bin.astype(np.int16)

    # --- particles, one species at a time --------------------------------------
    # Accumulators: mass on the grid per bin (dm + gas together), the stacked
    # profile per bin, and the bookkeeping sums.
    dens_bin = np.zeros((nbins, ngrid, ngrid), dtype=np.float64)
    stack_bin = None if skip_stack else np.zeros((nbins, ngrid, ngrid),
                                                 dtype=np.float64)
    mass_bin = np.zeros(nbins)                       # assigned mass per bin
    mass_bin_species = {s: np.zeros(nbins) for s in SPECIES}
    mass_tot_species = {}
    # Shot-noise bookkeeping. delta_m and delta_x share particles, so every
    # cross spectrum <delta_m^(i) delta_x*> and the truth <delta_m delta_x*>
    # contain a self-pair term. For mass-weighted fields it is
    #     P_shot_xx * (M_x / M_tot) * sum_{p in i, species x} m_p^2 / sum_{p in x} m_p^2,
    # so we need sum m^2 per bin and species, and P_shot_xx itself, which is
    # measured below in the pipeline's own convention (TSC window, L^2 factor,
    # binning) by painting the species at uniformly random positions.
    m2_bin_species = {s: np.zeros(nbins) for s in SPECIES}
    m2_tot_species = {}
    P_shot_species = {}
    rng = np.random.default_rng(12345)
    mass_halo_assigned = np.zeros(n_halo)            # per-halo assigned mass
    particles_file = get_particle_file_path(P.FEEDBACK, sim_name=P.SIM_NAME)

    for species in SPECIES:
        print(f"\n[{species}] loading particles ...")
        t0 = time.time()
        part = load_particle_properties(particles_file, species,
                                        requested=('mass', 'pos'),
                                        sim_name=P.SIM_NAME, Lbox=P.BOX)
        # float32 is kept for painting; cKDTree makes its own float64 copy.
        pos = np.mod(np.asarray(part['pos'], dtype=np.float32), np.float32(P.BOX))
        pos[pos >= P.BOX] -= np.float32(P.BOX)       # boxsize wants [0, L)
        mass = np.asarray(part['mass'], dtype=np.float64)
        del part
        gc.collect()
        mass_tot_species[species] = float(mass.sum())
        m2_tot_species[species] = float(np.sum(mass ** 2))
        print(f"    {pos.shape[0]:.3e} particles, {time.time() - t0:.1f} s")

        # --- shot spectrum of this species, same convention as the bundle ---
        print(f"[{species}] shot-noise spectrum (random positions) ...")
        t0 = time.time()
        pos_rand = rng.uniform(0.0, P.BOX, size=pos.shape).astype(np.float32)
        pos_rand[pos_rand >= P.BOX] -= np.float32(P.BOX)
        dens_r = paint(pos_rand, mass, ngrid, nthread)
        del pos_rand
        delta_r = dens_r / (mass_tot_species[species] / ngrid ** 2) - 1.0
        del dens_r
        fft_r = compute_2d_fft(delta_r, ngrid)
        del delta_r
        _, P_shot_species[species] = _binned(np.abs(fft_r) ** 2)
        del fft_r
        gc.collect()
        print(f"    P_shot_{species}{species}: {np.nanmedian(P_shot_species[species]):.4e} "
              f"(median over k), {time.time() - t0:.1f} s")

        # Particle-mass unit, self-calibrated: the species' total should be
        # its cosmological share of rhobar_m L^3 (short by the stellar mass).
        # Only used to size the membership chunks, so a few per cent is fine.
        omega_b = float(P.SIM_PARAMS.get('omega_b'))
        share = (P.OMEGA_M - omega_b) / P.OMEGA_M if species == 'dm' else omega_b / P.OMEGA_M
        unit = share * P.rhobar_m * P.BOX ** 3 / mass_tot_species[species]
        m_p = float(np.mean(mass)) * unit            # Msun/h per particle
        print(f"    mean particle mass ~ {m_p:.3e} Msun/h (self-calibrated)")

        print(f"[{species}] membership ...")
        label = assign_particles(pos, h_pos, h_r, h_mass, m_p,
                                 chunk_particles=chunk_particles,
                                 nthread=nthread)
        member = label >= 0
        mass_halo_assigned += np.bincount(label[member], weights=mass[member],
                                          minlength=n_halo)

        print(f"[{species}] painting per bin ...")
        t0 = time.time()
        bin_of_particle = np.full(label.size, -1, dtype=np.int16)
        bin_of_particle[member] = h_bin16[label[member]]
        for i in range(nbins):
            sel = bin_of_particle == i
            if not np.any(sel):
                continue
            m_i = mass[sel]
            mass_bin_species[species][i] = float(m_i.sum())
            m2_bin_species[species][i] = float(np.sum(m_i ** 2))
            dens_bin[i] += paint(pos[sel], m_i, ngrid, nthread)
            if stack_bin is not None:
                # every member re-centred on its own central, all centrals
                # moved to the box centre: the stacked (mass-weighted mean)
                # profile of the bin on the same grid and modes
                # Re-centre every member on its own central and put all
                # centrals at the ORIGIN, not the box centre: a stack at L/2
                # carries the phase (-1)^(nx+ny) mode by mode, which the
                # azimuthal average then cancels to ~0. At the origin the
                # transform is real and positive; the periodic wrap in tsc
                # takes care of the negative displacements.
                d = pos[sel] - h_pos[label[sel]]
                stack_bin[i] += paint(np.mod(d, P.BOX), m_i, ngrid, nthread)
        print(f"    {time.time() - t0:.1f} s")
        del pos, mass, label, member, bin_of_particle
        gc.collect()

    # --- normalisation and bookkeeping ----------------------------------------
    M_tot = mass_tot_species['dm'] + mass_tot_species['gas']
    f_c_here, f_g_here = mass_tot_species['dm'] / M_tot, mass_tot_species['gas'] / M_tot
    print(f"\nComponent fractions here f_c={f_c_here:.5f}, f_g={f_g_here:.5f}; "
          f"bundle f_c={f_c:.5f}, f_g={f_g:.5f}")
    if abs(f_c_here - f_c) > 1e-4:
        print("  WARNING: differs from the bundle -- was the bundle built from "
              "the same particle file? The truth m_fft uses the bundle values.")
    mass_bin[:] = mass_bin_species['dm'] + mass_bin_species['gas']
    f_part = mass_bin / M_tot                        # true mass fraction per bin
    counts = data['counts']
    n_i = counts / P.BOX ** 3
    f_i = n_i * data['M_mean'] / P.rhobar_m          # what the reconstruction uses

    print("\n  bin  logM   f_i (n_i M_i/rhobar)   f_part (assigned)   ratio")
    for i in range(nbins):
        if counts[i] > 0:
            r = f_part[i] / f_i[i] if f_i[i] > 0 else np.nan
            print(f"  {i:3d}  {data['logM_cen'][i]:5.2f}   {f_i[i]:.5f}"
                  f"              {f_part[i]:.5f}        {r:.4f}")
    print(f"  sum f_i = {f_i[counts > 0].sum():.4f},  sum f_part = {f_part.sum():.4f},"
          f"  true out-of-sphere fraction = {1 - f_part.sum():.4f}")
    # Per-halo check: assigned mass vs M200b (the catalogue mass counts stars
    # and the sphere assignment loses overlaps, so this is not 1 exactly).
    unit_total = P.rhobar_m * P.BOX ** 3 / M_tot     # particle unit -> Msun/h
    with np.errstate(divide='ignore', invalid='ignore'):
        q = mass_halo_assigned * unit_total / h_mass
    print(f"  assigned mass / M200b per halo: median {np.nanmedian(q):.4f}, "
          f"16-84% [{np.nanpercentile(q, 16):.4f}, {np.nanpercentile(q, 84):.4f}]")

    # --- spectra ---------------------------------------------------------------
    print("\nPer-bin cross spectra R_star_i and profiles ...")
    dens_avg = M_tot / ngrid ** 2                    # global mean per cell
    tracers = list(tracer_fft)
    R_star = {x: np.zeros((nbins, k_bins.size - 1)) for x in tracers}
    sum_fft = np.zeros_like(m_fft)
    u_stack = np.full((nbins, k_bins.size - 1), np.nan)      # TSC-deconvolved
    u_stack_raw = np.full((nbins, k_bins.size - 1), np.nan)  # as painted
    k_center = None
    for i in range(nbins):
        if mass_bin[i] == 0:
            continue
        delta_i = dens_bin[i] / dens_avg             # rho^(i)/rhobar_m
        fft_i = compute_2d_fft(delta_i, ngrid)
        sum_fft += fft_i
        for x in tracers:
            k_center, R_star[x][i] = _binned((fft_i * np.conj(tracer_fft[x])).real)
        del fft_i
        if stack_bin is not None:
            F = (np.fft.fft2(stack_bin[i]))
            F = F / F[0, 0]
            u_stack[i] = azimuthal_mean(F.real, ngrid, k_bins, deconvolve_tsc=True)
            u_stack_raw[i] = azimuthal_mean(F.real, ngrid, k_bins, deconvolve_tsc=False)
        print(f"  bin {i:2d} done")
    del dens_bin
    gc.collect()

    out_fft = m_fft - sum_fft                        # delta_m^out, exact
    U_star = {}
    for x in tracers:
        _, U_star[x] = _binned((out_fft * np.conj(tracer_fft[x])).real)

    # --- closure against the bundle's truths ----------------------------------
    for x, key, fx, Psh in (('gas', 'P_matter_gas', f_g_here, P_shot_species['gas']),
                            ('dm', 'P_matter_dm', f_c_here, P_shot_species['dm'])):
        tot = R_star[x].sum(axis=0) + U_star[x]
        with np.errstate(divide='ignore', invalid='ignore'):
            dev = np.abs(tot / data[key] - 1.0)
            frac = fx * Psh / data[key]
        print(f"  closure x={x}: |(R_star + U_star)/P_matter_x - 1| max = "
              f"{np.nanmax(dev):.2e}")
        k_c = data['k_center']
        print(f"  shot fraction f_x P_shot_xx / P_matter_x for x={x}: "
              + ", ".join(f"k={kk:g}: {frac[np.argmin(np.abs(k_c - kk))]:.3f}"
                          for kk in (1, 2, 3, 5)))

    if k_center is not None and not np.allclose(k_center, data['k_center']):
        print("  WARNING: k binning differs from the spectra bundle.")

    return dict(
        k_center=np.asarray(data['k_center']), k_bins=k_bins,
        logM_edges=logM_edges, logM_cen=data['logM_cen'], M_mean=data['M_mean'],
        counts=counts, f_i=f_i, f_part=f_part,
        mass_bin_dm=mass_bin_species['dm'], mass_bin_gas=mass_bin_species['gas'],
        M_tot_particles=M_tot, f_c=f_c_here, f_g=f_g_here,
        R_star_gas=R_star['gas'], R_star_dm=R_star['dm'],
        U_star_gas=U_star['gas'], U_star_dm=U_star['dm'],
        u_stack=u_stack, u_stack_raw=u_stack_raw, skip_stack=skip_stack,
        P_shot_dd=P_shot_species['dm'], P_shot_gg=P_shot_species['gas'],
        m2_bin_dm=m2_bin_species['dm'], m2_bin_gas=m2_bin_species['gas'],
        m2_tot_dm=m2_tot_species['dm'], m2_tot_gas=m2_tot_species['gas'],
        assigned_over_m200b_median=float(np.nanmedian(q)),
        box=P.BOX, ngrid=ngrid, redshift=P.Z_EVAL, feedback=P.FEEDBACK,
        mass_def=P.MASS_DEF, centrals_only=P.CENTRALS_ONLY,
    )


# ==============================================================================
# 6.  Derived quantities
# ==============================================================================
def shot_terms(us, data, x):
    """Self-pair (shot-noise) contributions to R_star_i, U_star and the truth
    for tracer x, in the bundle's units. Zero arrays if the cache predates the
    shot measurement.

        <delta_m^(i) delta_x*>_shot = P_shot_xx (M_x/M_tot) sum_{i,x} m^2 / sum_x m^2
        <delta_m delta_x*>_shot     = P_shot_xx (M_x/M_tot)
        U_star_shot                 = truth_shot - sum_i R_star_i_shot
    For x = matter the truth is an auto spectrum: f_c^2 P_shot_dd + f_g^2 P_shot_gg.
    """
    nk = us['k_center'].size
    nb = data['counts'].size
    zero = dict(R_i=np.zeros((nb, nk)), U=np.zeros(nk), truth=np.zeros(nk))
    if 'P_shot_dd' not in us:
        return zero
    f_c, f_g = float(us['f_c']), float(us['f_g'])
    parts = []
    if x in ('dm', 'matter'):
        parts.append((f_c if x == 'dm' else f_c ** 2, us['P_shot_dd'],
                      us['m2_bin_dm'] / float(us['m2_tot_dm'])))
    if x in ('gas', 'matter'):
        parts.append((f_g if x == 'gas' else f_g ** 2, us['P_shot_gg'],
                      us['m2_bin_gas'] / float(us['m2_tot_gas'])))
    R_i = np.zeros((nb, nk)); truth = np.zeros(nk)
    for w, Psh, share in parts:
        R_i += w * share[:, None] * Psh[None, :]
        truth += w * Psh
    return dict(R_i=R_i, U=truth - R_i.sum(axis=0), truth=truth)


def derived(us, data, z, subtract_shot=True):
    """u~_m per tracer, u_nfw per bin, rho_u, and the Eq. (59) terms.

    With subtract_shot the self-pair terms are removed from R_star_i, U_star
    and the truth before anything is formed. P_halo_x needs no correction
    (halos and particles are distinct point sets) and neither does R.
    """
    k = us['k_center']
    M_i = data['M_mean']
    counts = data['counts']
    occ = counts > 0
    f_i = us['f_i']
    c_i = P.concentration(M_i, z)
    u_nfw = np.array([P.u_nfw(k, M_i[i], c_i[i]) for i in range(M_i.size)])

    out = dict(k=k, occ=occ, u_nfw=u_nfw, u_stack=us['u_stack'], f_i=f_i,
               u_stack_raw=us.get('u_stack_raw', us['u_stack']),
               subtract_shot=subtract_shot)
    for x in ('gas', 'dm', 'matter'):
        if x == 'matter':
            f_c, f_g = float(data['f_c']), float(data['f_g'])
            R_star_i = f_c * us['R_star_dm'] + f_g * us['R_star_gas']
            U_star = f_c * us['U_star_dm'] + f_g * us['U_star_gas']
        else:
            R_star_i, U_star = us[f'R_star_{x}'], us[f'U_star_{x}']
        P_halo_x = data[P.TRACER_INFO[x]['halo_key']]
        P_true = np.asarray(data[P.TRACER_INFO[x]['truth_key']], dtype=float)
        sh = shot_terms(us, data, x)
        shot_frac = sh['truth'] / P_true
        if subtract_shot:
            R_star_i = R_star_i - sh['R_i']
            U_star = U_star - sh['U']
            P_true = P_true - sh['truth']

        with np.errstate(divide='ignore', invalid='ignore'):
            u_tilde = R_star_i / (f_i[:, None] * P_halo_x)      # Eq. (56)
        R_i = f_i[:, None] * u_nfw * P_halo_x                   # the model
        R = R_i[occ].sum(axis=0)
        R_star = R_star_i[occ].sum(axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            rho_u = R / R_star                                  # Eq. (57)
            omega = R_star_i / R_star[None, :]                  # Eq. (58)
            # Eq. (59) with the baseline flat template for U
        ref = int(np.flatnonzero(occ)[0])
        f_u_cat = 1.0 - f_i[occ].sum()
        U_flat = f_u_cat * P_halo_x[ref]
        with np.errstate(divide='ignore', invalid='ignore'):
            profile_err = (rho_u - 1.0) * R_star / (R_star + U_star)
            template_err = (U_flat - U_star) / (R_star + U_star)
            total_err = (R + U_flat) / (R_star + U_star) - 1.0
            closure = (R_star + U_star) / P_true - 1.0
        out[x] = dict(u_tilde=u_tilde, R_i=R_i, R=R, R_star_i=R_star_i,
                      R_star=R_star, U_star=U_star, U_flat=U_flat, rho_u=rho_u,
                      omega=omega, profile_err=profile_err,
                      template_err=template_err, total_err=total_err,
                      closure=closure, P_true=P_true, shot_frac=shot_frac)
    return out


# ==============================================================================
# 7.  Plot
# ==============================================================================
def make_plot(d, us, data, args):
    k = d['k']
    k_Ny = float(data['k_Nyquist'])
    logM = data['logM_cen']
    occ = d['occ']
    z = float(data['redshift'])
    tracers = ['gas', 'dm']

    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(rf"$\tilde u_m$ measured from the particles, {P.FEEDBACK}, $z={z}$"
                 + ("  [self-pair shot noise subtracted]" if d['subtract_shot']
                    else "  [shot noise NOT subtracted]"), fontsize=12)

    # (1) rho_u = R / R_star
    a = ax[0, 0]
    for x in tracers:
        a.plot(k, d[x]['rho_u'], color=TRACER_COLOR[x], lw=1.5, label=f"x = {x}")
    a.axhline(1, color='k', lw=0.6); a.axvline(k_Ny, color='grey', ls=':')
    a.set_xscale('log'); a.set_ylim(0.3, 1.4)
    a.set_xlabel(r"$k\ [h/{\rm cMpc}]$"); a.set_ylabel(r"$\rho_u = R/R_\star$")
    a.set_title(r"profile error of the resolved part, Eq. (57)")
    a.legend(fontsize=8)

    # (2) closure and U_star / P_true
    a = ax[0, 1]
    for x in tracers:
        a.plot(k, d[x]['U_star'] / d[x]['P_true'], color=TRACER_COLOR[x], lw=1.5,
               label=rf"{x}: $U_\star/P^{{mx}}$")
        a.plot(k, d[x]['U_flat'] / d[x]['P_true'], color=TRACER_COLOR[x], lw=1.2,
               ls='--', label=rf"{x}: flat $f_u^{{\rm cat}}P^{{h_r x}}/P^{{mx}}$")
    for x in tracers:
        a.plot(k, d[x]['shot_frac'], color=TRACER_COLOR[x], lw=1.0, ls=':',
               label=rf"{x}: shot $f_xP^{{\rm shot}}_{{xx}}/P^{{mx}}$ (removed)"
               if d['subtract_shot'] else rf"{x}: shot fraction (NOT removed)")
    a.axvline(k_Ny, color='grey', ls=':')
    a.set_xscale('log'); a.set_ylim(-0.2, 1.0)
    a.set_xlabel(r"$k\ [h/{\rm cMpc}]$")
    a.set_title(r"true out-of-sphere contribution vs the flat template"
                + ("" if d['subtract_shot'] else "  [shot NOT subtracted]"))
    a.legend(fontsize=7)

    # (3) Eq. (59) decomposition
    a = ax[0, 2]
    for x in tracers:
        c = TRACER_COLOR[x]
        a.plot(k, d[x]['total_err'], color=c, lw=1.8, label=f"{x}: total")
        a.plot(k, d[x]['profile_err'], color=c, lw=1.2, ls='--', label=f"{x}: profile")
        a.plot(k, d[x]['template_err'], color=c, lw=1.2, ls=':', label=f"{x}: template")
    a.axhline(0, color='k', lw=0.6); a.axvline(k_Ny, color='grey', ls=':')
    a.set_xscale('log'); a.set_ylim(-0.7, 0.7)
    a.set_xlabel(r"$k\ [h/{\rm cMpc}]$")
    a.set_title(r"$(R+U)/(R_\star+U_\star)-1$ split as Eq. (59), flat $U$")
    a.legend(fontsize=7, ncol=2)

    # (4-6) profiles in three bins: u_nfw, <u_m>_stack, u~_m for gas and dm
    show = [int(np.argmin(np.abs(logM - lm))) for lm in args.show_bins]
    show = [i for i in show if occ[i]][:3]
    for a, i in zip(ax[1], show):
        a.plot(k, d['u_nfw'][i], 'k-', lw=1.5, label=r"$u_m^{\rm NFW}$")
        if not us['skip_stack']:
            a.plot(k, d['u_stack'][i], color='grey', lw=1.5, ls='--',
                   label=r"$\langle u_m\rangle$ stacked, TSC-deconvolved")
            a.plot(k, d['u_stack_raw'][i], color='grey', lw=0.8, ls=':',
                   label=r"stacked, as painted")
        for x in tracers:
            a.plot(k, d[x]['u_tilde'][i], color=TRACER_COLOR[x], lw=1.3,
                   label=rf"$\tilde u_m$, x = {x}")
        a.axhline(1, color='k', lw=0.5); a.axhline(0, color='k', lw=0.5)
        a.axvline(k_Ny, color='grey', ls=':')
        a.set_xscale('log'); a.set_ylim(-0.3, 1.6)
        a.set_xlabel(r"$k\ [h/{\rm cMpc}]$")
        a.set_title(rf"bin $\log M = {logM[i]:.2f}$, $\langle M\rangle = "
                    rf"10^{{{np.log10(data['M_mean'][i]):.2f}}}$")
        a.legend(fontsize=8)
    for a_ in ax.flat:
        a_.grid(alpha=0.3)
    fig.tight_layout()

    stem = (f"u_tilde_{P.MASS_DEF}_nb{args.nbins}"
            f"_logM{args.logm_min:g}-{args.logm_max:g}_{args.concentration}"
            f"{'' if d['subtract_shot'] else '_noshot'}")
    out = plot_path('pme_reconstruction', P.FEEDBACK, stem=stem)
    ensure_parents(out)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n[plot] saved {out}")

    payload = dict(k=k, logM_cen=logM, occupied=occ, u_nfw=d['u_nfw'],
                   u_stack=d['u_stack'], f_i=d['f_i'], concentration=args.concentration)
    payload['subtract_shot'] = d['subtract_shot']
    for x in ('gas', 'dm', 'matter'):
        for key in ('u_tilde', 'rho_u', 'R', 'R_star', 'U_star', 'U_flat',
                    'profile_err', 'template_err', 'total_err', 'omega',
                    'shot_frac', 'P_true'):
            payload[f'{key}_{x}'] = d[x][key]
    save_plot_data(out, payload,
                   description=("Measured tracer-cross-weighted matter profile "
                                "u~_m per bin (Eq. 56), rho_u (Eq. 57) and the "
                                "Eq. 59 decomposition, per tracer"))

    print("\n  k      " + "".join(f"  rho_u({x:3s})  U*/P({x:3s})  shot({x:3s})"
                                   for x in tracers))
    for kk in (0.1, 0.5, 1.0, 2.0, 3.0, 5.0):
        j = int(np.argmin(np.abs(k - kk)))
        row = f"  {k[j]:6.3f}"
        for x in tracers:
            row += (f"  {d[x]['rho_u'][j]:10.3f}  {d[x]['U_star'][j] / d[x]['P_true'][j]:10.3f}"
                    f"  {d[x]['shot_frac'][j]:9.3f}")
        print(row)


# ==============================================================================
# 8.  Main
# ==============================================================================
def main():
    ap = argparse.ArgumentParser(
        description="Measure u~_m(k|M_i), R_star, U_star from the particles.")
    ap.add_argument('--nbins', type=int, default=P.NBINS)
    ap.add_argument('--logm-min', type=float, default=P.LOGM_MIN)
    ap.add_argument('--logm-max', type=float, default=P.LOGM_MAX)
    ap.add_argument('--nkbins', type=int, default=P.NKBINS)
    ap.add_argument('--recompute', action='store_true')
    ap.add_argument('--plot-only', action='store_true',
                    help="never measure; fail if the cache is absent")
    ap.add_argument('--skip-stack', action='store_true',
                    help="do not measure the stacked first-moment profile")
    ap.add_argument('--chunk-particles', type=float, default=5e7,
                    help="expected member particles per membership chunk")
    ap.add_argument('--concentration', choices=['powerlaw', 'colossus'],
                    default=P.CONCENTRATION_SOURCE)
    ap.add_argument('--concentration-model', default=P.COLOSSUS_CONC_MODEL)
    ap.add_argument('--show-bins', nargs='+', type=float, default=[12.5, 13.5, 14.5],
                    help="log10 M of the three bins whose profiles are drawn")
    ap.add_argument('--no-shot-subtract', action='store_true',
                    help="keep the self-pair shot-noise terms in R_star, U_star "
                         "and the truth (for comparison; the default removes them)")
    ap.add_argument('--nthread', type=int, default=None,
                    help=f"threads for the KD-tree query and TSC painting "
                         f"(default: sim_params nthread = {P.NTHREAD})")
    args = ap.parse_args()
    if args.nthread is not None:
        P.NTHREAD = int(args.nthread)

    P.CONCENTRATION_SOURCE = args.concentration
    P.COLOSSUS_CONC_MODEL = args.concentration_model

    bpath = P.bundle_path(args.nbins, args.logm_min, args.logm_max, P.NGRID, args.nkbins)
    if not bpath.exists():
        raise SystemExit(f"Spectra bundle missing:\n  {bpath}\nRun "
                         "predict_Pmx_from_Phx.py first.")
    print(f"Spectra bundle:\n  {bpath}")
    with np.load(bpath, allow_pickle=False) as f:
        data = {key: f[key] for key in f.files}
    P.derive_P_halo_matter(data)

    upath = ustar_path(args.nbins, args.logm_min, args.logm_max, P.NGRID, args.nkbins)
    if upath.exists() and not args.recompute:
        print(f"u~ cache:\n  {upath}")
        with np.load(upath, allow_pickle=False) as f:
            us = {key: f[key] for key in f.files}
    elif args.plot_only:
        raise SystemExit(f"--plot-only but no cache at\n  {upath}")
    else:
        us = measure(args.nbins, args.logm_min, args.logm_max, args.nkbins,
                     data, skip_stack=args.skip_stack,
                     chunk_particles=args.chunk_particles)
        ensure_parents(upath)
        np.savez_compressed(upath, **us)
        print(f"\nu~ cache saved:\n  {upath}")

    if 'P_shot_dd' not in us and not args.no_shot_subtract:
        print("NOTE: cache predates the shot-noise measurement; nothing is "
              "subtracted. Run with --recompute to measure it.")
    d = derived(us, data, float(data['redshift']),
                subtract_shot=not args.no_shot_subtract)
    make_plot(d, us, data, args)


if __name__ == '__main__':
    main()