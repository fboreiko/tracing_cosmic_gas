#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""u_bar(k | M_i): the MEASURED mean matter profile of a halo mass bin.

The reconstruction's R evaluates u_m from NFW plus a c(M,z) relation. This
module measures the same object off the particles instead, with no model and
no grid:

    u_bar(k | M_i) = sum_{p in i} m_p j0(k r_p) / sum_{p in i} m_p,
    r_p = |x_p - x_host(p)|,   j0(x) = sin x / x.

WHAT IS IN THE CACHE
    Shells:     W_r_dm, W_r_gas   (nbins, NX)  sum of m_p
                S_r_dm, S_r_gas   (nbins, NX)  sum of m_p r_p   [cMpc/h]
    Counts:     counts, M_mean, npart_dm, npart_gas, npart_mean
    Budget:     M_tot_particles, mass_bin_dm, mass_bin_gas,
                assigned_over_m200b_median (per bin)
    Floor:      logm_floor, bin_ok, n_halo_min, n_stack_min -- for the
                record only; resolved_bins re-derives them on load, so the
                gate can be moved without re-measuring anything
    Binning:    logM_edges, logM_cen, x_edges, nx_shells
    Provenance: box, nbins, logm_min, logm_max, bins_per_decade, redshift,
                feedback, mass_def, centrals_only, aperture, version

USAGE
    from Pmx_reconstruction.pmxlib import u_bar as ub
    cache = ub.load_or_measure_u_bar(cfg)          # measures if absent
    u = ub.u_bar_of_k(cache, k)                    # (nbins, nk), -> 1 at k -> 0
    fu = ub.f_tilde(cache)[:, None] * u            # what enters R

    python -m Pmx_reconstruction.pmxlib.u_bar --aperture 1
"""

import gc
import time

import numpy as np

from utils.catalog_loaders import load_particle_properties
from utils.pipeline_paths import (DATA_ROOT, ensure_parents,
                                  get_particle_file_path)

from Pmx_reconstruction.pmxlib.binning import log_mass_bin_edges
from Pmx_reconstruction.pmxlib.rstar_ustar import (SPECIES, assign_particles,
                                                   load_binned_centrals)

__all__ = ['CACHE_VERSION', 'NX_SHELLS', 'mass_grid', 'cache_path',
           'compute_u_bar', 'load_or_measure_u_bar', 'u_bar_of_k', 'f_tilde',
           'resolved_bins', 'measured_range', 'u_bar_interp',
           'check_against_rstar']

CACHE_VERSION = 'v1'

U_LOGM_MIN = 8.0
U_LOGM_MAX = 15.0
U_BINS_PER_DECADE = 8.0

NX_SHELLS = 128

N_HALO_MIN = 100        # halos in the bin
N_STACK_MIN = 10_000    # matter particles summed over the bin

CHUNK_RADII = 20_000_000

POS_EPS_ULPS = 4.0


# Mass grid
def mass_grid():
    """(nbins, logm_min, logm_max) of the u_bar binning -- the constants above.

    A function rather than three bare names so every caller agrees on the bin
    count without recomputing it, and so the one place that rounds does it
    once.
    """
    nbins = int(round((U_LOGM_MAX - U_LOGM_MIN) * U_BINS_PER_DECADE))
    return nbins, U_LOGM_MIN, U_LOGM_MAX


def cache_path(cfg, aperture=1.0):
    """Companion to rstar_ustar.cache_path, minus everything about k.

    No ngrid and no kbin_tag in the stem: the histogram this file caches has
    no k in it, so a bundle rebuilt on a different grid does not invalidate it.
    """
    nbins, lo, hi = mass_grid()
    ap = '' if float(aperture) == 1.0 else f'_ap{float(aperture):g}'
    stem = (f'u_bar_{CACHE_VERSION}_{cfg.feedback}_{cfg.mass_def}'
            f'_logM{lo:g}-{hi:g}_nb{nbins}_nx{NX_SHELLS}'
            f'_{"cen" if cfg.centrals_only else "all"}{ap}.npz')
    return DATA_ROOT / cfg.sim_name / 'pme_inputs' / stem


# The measurement
def _accumulate_shells(pos, mass, label, h_pos, h_r, h_bin, box, nbins,
                       W_r, S_r, N_p, M_halo, chunk=CHUNK_RADII):
    """Add sum(m), sum(m r), the particle count and the per-halo assigned mass.

    All four accumulate in place, in ONE chunked pass. Two things that pass
    buys, both of which start to matter exactly when the mass floor drops:

      - No loop over mass bins. With a seven-decade grid that would be ~56
        full-length boolean passes, so the bin index rides along inside the
        flat bincount instead.
      - No full-length `member` mask and no mass[member] / label[member]
        copies. A lower floor means more halos claim more particles, so those
        temporaries grow just when there is least room for them; here nothing
        outlives a chunk.
    """
    n_p = label.size
    r_over_R_max = 0.0        # how far past the aperture, as a fraction
    over_abs_max = 0.0        # ... and in cMpc/h, which is what gets asserted
    for a in range(0, n_p, chunk):
        b = min(a + chunk, n_p)
        lab = label[a:b]
        sel = lab >= 0
        if not np.any(sel):
            continue
        lab = lab[sel]
        # float32 pos minus float64 halo pos promotes to float64. The
        # subtraction itself is then exact; what is left is the float32
        # storage of pos, 4e-5 cMpc/h on a 681 box. That is 0.03% of r200m at
        # 10^11 Msun/h and it averages down over the stack.
        d = pos[a:b][sel] - h_pos[lab]
        d -= box * np.round(d / box)                  # minimum image
        r = np.sqrt(np.einsum('ij,ij->i', d, d))
        del d
        R = h_r[lab]

        # query_ball_point guarantees r <= R. Checking it costs two reductions
        # and catches a bad wrap, a stale sort order and a units slip at once.
        # The ratio is the readable diagnostic; the ABSOLUTE overshoot is what
        # the caller asserts on, because the float32 slop is a fixed length and
        # would look large against a 10^9 halo's radius while meaning nothing.
        r_over_R_max = max(r_over_R_max, float(np.max(r / R)))
        over_abs_max = max(over_abs_max, float(np.max(r - R)))

        ix = np.minimum((r * (NX_SHELLS / R)).astype(np.int64), NX_SHELLS - 1)
        bin_of = h_bin[lab]
        flat = bin_of * NX_SHELLS + ix
        m = mass[a:b][sel]
        size = nbins * NX_SHELLS
        W_r += np.bincount(flat, weights=m, minlength=size).reshape(nbins, -1)
        S_r += np.bincount(flat, weights=m * r,
                           minlength=size).reshape(nbins, -1)
        N_p += np.bincount(bin_of, minlength=nbins)
        M_halo += np.bincount(lab, weights=m, minlength=M_halo.size)
        del lab, sel, r, R, ix, bin_of, flat, m
    return r_over_R_max, over_abs_max


def compute_u_bar(cfg, aperture=1.0, chunk_particles=5e7):
    """Measure the stacked radial histograms. Returns the cache dict."""
    nbins, logm_lo, logm_hi = mass_grid()
    print(f"\nStacked matter profiles, logM [{logm_lo:g}, {logm_hi:g}) in "
          f"{nbins} bins ({nbins / (logm_hi - logm_lo):.3g} per decade), "
          f"{NX_SHELLS} shells, aperture = {float(aperture):g} r200m")

    print("\nHalo catalogue:")
    h_pos, h_mass, h_r, h_bin, logM_edges = load_binned_centrals(
        cfg, aperture=aperture, nbins=nbins,
        logm_min=logm_lo, logm_max=logm_hi)
    n_halo = h_mass.size
    if n_halo == 0:
        raise SystemExit(f"No centrals in [{logm_lo:g}, {logm_hi:g}).")
    h_bin = np.asarray(h_bin, dtype=np.int64)
    _, logM_cen, _ = log_mass_bin_edges(nbins, logm_lo, logm_hi)

    # The resolution floor that has nothing to do with the catalogue: particle
    # positions are float32, so a radius carries a fixed absolute error however
    # many particles the halo has. Printed against the smallest halo on the
    # grid, because that is where a seven-decade range puts it under pressure.
    pos_eps = cfg.box * 2.0 ** -23
    pos_tol = POS_EPS_ULPS * pos_eps
    r200_min = float(h_r.min()) / float(aperture)
    print(f"  float32 position quantum {pos_eps:.2e} cMpc/h = "
          f"{pos_eps / r200_min:.3%} of the smallest halo's r200m "
          f"({r200_min:.4f} cMpc/h); a random error, so it averages down over "
          f"the stack")

    counts = np.bincount(h_bin, minlength=nbins).astype(np.float64)
    M_sum = np.bincount(h_bin, weights=h_mass, minlength=nbins)
    M_mean = np.where(counts > 0, M_sum / np.maximum(counts, 1.0),
                      10.0 ** logM_cen)

    W_r = {s: np.zeros((nbins, NX_SHELLS)) for s in SPECIES}
    S_r = {s: np.zeros((nbins, NX_SHELLS)) for s in SPECIES}
    N_p = {s: np.zeros(nbins, dtype=np.int64) for s in SPECIES}
    mass_tot_species, m_p_species = {}, {}
    mass_halo_assigned = np.zeros(n_halo)
    particles_file = get_particle_file_path(cfg.feedback, sim_name=cfg.sim_name)

    for species in SPECIES:
        print(f"\n[{species}] loading particles ...")
        t0 = time.time()
        part = load_particle_properties(particles_file, species,
                                        requested=('mass', 'pos'),
                                        sim_name=cfg.sim_name, Lbox=cfg.box)
        pos = np.mod(np.asarray(part['pos'], dtype=np.float32),
                     np.float32(cfg.box))
        pos[pos >= cfg.box] -= np.float32(cfg.box)   # boxsize wants [0, L)
        mass = np.asarray(part['mass'], dtype=np.float64)
        del part
        gc.collect()
        mass_tot_species[species] = float(mass.sum())
        print(f"    {pos.shape[0]:.3e} particles, {time.time() - t0:.1f} s")

        omega_b = float(cfg.sim_params.get('omega_b'))
        share = ((cfg.omega_m - omega_b) / cfg.omega_m if species == 'dm'
                 else omega_b / cfg.omega_m)
        unit = share * cfg.rhobar_m * cfg.box ** 3 / mass_tot_species[species]
        m_p = float(np.mean(mass)) * unit            # Msun/h per particle
        m_p_species[species] = m_p
        print(f"    mean particle mass ~ {m_p:.3e} Msun/h (self-calibrated)")

        print(f"[{species}] membership ...")
        label = assign_particles(cfg, pos, h_pos, h_r, h_mass, m_p,
                                 chunk_particles=chunk_particles,
                                 nthread=cfg.threads, aperture=aperture)

        print(f"[{species}] stacking radii ...")
        t0 = time.time()
        worst, over = _accumulate_shells(pos, mass, label, h_pos, h_r, h_bin,
                                         cfg.box, nbins, W_r[species],
                                         S_r[species], N_p[species],
                                         mass_halo_assigned)
        print(f"    max r / r_aperture = {worst:.6f}, overshoot "
              f"{max(over, 0.0):.2e} cMpc/h (tolerance {pos_tol:.2e}), "
              f"{time.time() - t0:.1f} s")
        if over > pos_tol:
            raise RuntimeError(
                f"[{species}] a member particle sits {over:.3e} cMpc/h beyond "
                f"its host's aperture radius, past the {pos_tol:.3e} the "
                f"float32 positions can explain. query_ball_point cannot "
                f"return that, so the minimum image, the halo sort order or "
                f"the position units are wrong -- not a tolerance to widen.")
        del pos, mass, label
        gc.collect()

    # --- bookkeeping ----------------------------------------------------------
    M_tot = mass_tot_species['dm'] + mass_tot_species['gas']
    mass_bin = {s: W_r[s].sum(axis=1) for s in SPECIES}
    npart_bin = N_p['dm'] + N_p['gas']
    npart_mean = np.where(counts > 0, npart_bin / np.maximum(counts, 1.0), 0.0)
    f_part = (mass_bin['dm'] + mass_bin['gas']) / M_tot

    unit_total = cfg.rhobar_m * cfg.box ** 3 / M_tot
    with np.errstate(divide='ignore', invalid='ignore'):
        q = mass_halo_assigned * unit_total / h_mass
    q_med = np.full(nbins, np.nan)
    for i in np.flatnonzero(counts > 0):
        qi = q[h_bin == i]
        qi = qi[np.isfinite(qi)]
        if qi.size:
            q_med[i] = float(np.median(qi))

    ok, logm_floor = resolved_bins(dict(counts=counts, npart_dm=N_p['dm'],
                                        npart_gas=N_p['gas'],
                                        logM_edges=logM_edges))

    print("\n  bin  logM_cen   n_halo    <n_part>   stack N   sigma(u)   "
          "f_part    <M_assigned/M200b>")
    for i in np.flatnonzero(counts > 0):
        flag = '' if ok[i] else '   <- below the floor'
        sig = 1.0 / np.sqrt(npart_bin[i]) if npart_bin[i] > 0 else np.inf
        print(f"  {i:3d}   {logM_cen[i]:6.2f}  {int(counts[i]):9d}  "
              f"{npart_mean[i]:9.1f}  {npart_bin[i]:8.2e}  {sig:8.1e}  "
              f"{f_part[i]:.6f}   {q_med[i]:8.4f}{flag}")
    print(f"  sum f_part = {f_part.sum():.4f} over the whole grid")
    if ok.any():
        print(f"  measured floor: logM = {logm_floor:.2f} "
              f"(>= {N_HALO_MIN} halos and >= {N_STACK_MIN:g} particles in "
              f"the stack, i.e. sigma(u_bar) <~ "
              f"{1 / np.sqrt(N_STACK_MIN):.1%})")
    else:
        print(f"  WARNING: no bin meets the floor criteria; the catalogue "
              f"does not support this mass grid.")

    x_edges = np.linspace(0.0, float(aperture), NX_SHELLS + 1)
    return dict(
        version=CACHE_VERSION,
        W_r_dm=W_r['dm'], W_r_gas=W_r['gas'],
        S_r_dm=S_r['dm'], S_r_gas=S_r['gas'],
        counts=counts, M_mean=M_mean,
        npart_dm=N_p['dm'], npart_gas=N_p['gas'], npart_mean=npart_mean,
        mass_bin_dm=mass_bin['dm'], mass_bin_gas=mass_bin['gas'],
        M_tot_particles=M_tot, f_part=f_part,
        assigned_over_m200b_median=q_med,
        m_p_dm=m_p_species['dm'], m_p_gas=m_p_species['gas'],
        logm_floor=logm_floor, bin_ok=ok,
        n_halo_min=N_HALO_MIN, n_stack_min=N_STACK_MIN,
        logM_edges=logM_edges, logM_cen=logM_cen,
        x_edges=x_edges, nx_shells=NX_SHELLS,
        box=cfg.box, nbins=nbins, logm_min=logm_lo, logm_max=logm_hi,
        bins_per_decade=U_BINS_PER_DECADE,
        redshift=cfg.z, feedback=cfg.feedback, mass_def=cfg.mass_def,
        centrals_only=cfg.centrals_only, aperture=float(aperture),
    )


# Cache
def load_or_measure_u_bar(cfg, recompute=False, allow_compute=True,
                          aperture=1.0, chunk_particles=5e7):
    """Read the cache, or measure and write it. None if absent and not allowed."""
    path = cache_path(cfg, aperture=aperture)
    if recompute or not path.exists():
        if not allow_compute:
            return None
        print(f"No u_bar cache at\n  {path}\nMeasuring (one particle pass per "
              f"species; no painting and no FFTs) ...")
        cache = compute_u_bar(cfg, aperture=aperture,
                              chunk_particles=chunk_particles)
        ensure_parents(path)
        np.savez_compressed(path, **cache)
        print(f"\nu_bar cache saved:\n  {path}")
        return cache

    print(f"u_bar cache:\n  {path}")
    with np.load(path, allow_pickle=False) as f:
        return {key: f[key] for key in f.files}


# Read-time assembly
def _shells(cache, species=None):
    """(W, S) for 'dm', 'gas', or the matter sum when species is None."""
    if species is None:
        return (np.asarray(cache['W_r_dm'], float)
                + np.asarray(cache['W_r_gas'], float),
                np.asarray(cache['S_r_dm'], float)
                + np.asarray(cache['S_r_gas'], float))
    if species not in SPECIES:
        raise ValueError(f"species must be None, 'dm' or 'gas', not {species!r}")
    return (np.asarray(cache[f'W_r_{species}'], float),
            np.asarray(cache[f'S_r_{species}'], float))


def u_bar_of_k(cache, k, species=None):
    """u_bar(k | M_i), shape (nbins, nk). NaN for bins that hold no mass.

    k is in h/cMpc and is otherwise unconstrained: the histogram carries no
    grid, so this needs no k binning, no stitching and no interpolation.

    Each shell is transformed at its own MASS-WEIGHTED mean radius, which is
    what makes the shell width a second-order error rather than a first-order
    one. u_bar(0) = 1 identically, by construction and not by convention --
    that is the sharpest check against u_tilde, which does not.
    """
    W, S = _shells(cache, species)
    k = np.atleast_1d(np.asarray(k, dtype=float))
    with np.errstate(invalid='ignore', divide='ignore'):
        r_bar = np.where(W > 0, S / np.where(W > 0, W, 1.0), 0.0)
    M = W.sum(axis=1)
    out = np.full((W.shape[0], k.size), np.nan)
    for i in np.flatnonzero(M > 0):
        occ = W[i] > 0
        # np.sinc(x) = sin(pi x)/(pi x), so j0(k r) = np.sinc(k r / pi).
        out[i] = W[i][occ] @ np.sinc(np.outer(r_bar[i][occ], k) / np.pi) / M[i]
    return out


def f_tilde(cache, species=None):
    """Mass fraction inside the membership spheres, per bin (f_tilde_i).

    The weight that PAIRS with u_bar: both are normalised to the mass actually
    assigned inside the aperture, so f_tilde * u_bar is the measured form of
    the unnormalised f_i u_m that enters R. Pairing u_bar with n_i M_i /
    rhobar_m instead mixes two different masses and gives a wrong answer that
    looks entirely plausible.
    """
    W, _ = _shells(cache, species)
    return W.sum(axis=1) / float(cache['M_tot_particles'])


def resolved_bins(cache):
    """(bin_ok, logm_floor): which bins count as measured, and the lowest.

    Derived from `counts` and the stored particle counts rather than read back,
    so moving N_HALO_MIN or N_STACK_MIN re-gates a cache already on disk
    without re-measuring it. compute_u_bar stores its own answer too, for the
    record, but nothing reads that copy.

    bin_ok is per bin and logm_floor is only its lowest member: a bin ABOVE the
    floor can fail as well, for want of halos at the top of the mass function
    rather than for want of particles.
    """
    counts = np.asarray(cache['counts'], dtype=float)
    n_stack = (np.asarray(cache['npart_dm'], dtype=float)
               + np.asarray(cache['npart_gas'], dtype=float))
    ok = (counts >= N_HALO_MIN) & (n_stack >= N_STACK_MIN)
    edges = np.asarray(cache['logM_edges'], dtype=float)
    floor = float(edges[np.flatnonzero(ok)[0]]) if ok.any() else float('nan')
    return ok, floor


def measured_range(cache):
    """(logM_lo, logM_hi) over which u_bar is actually measured.

    The span of <M> over the bins that are both occupied and past the floor,
    because those are the nodes an interpolation has to stand on. Outside it
    there is nothing measured and the caller falls back to the model.
    """
    ok, _ = resolved_bins(cache)
    if not ok.any():
        return float('nan'), float('nan')
    lg = np.log10(np.asarray(cache['M_mean'], float)[ok])
    return float(lg.min()), float(lg.max())


def u_bar_interp(cache, k, M, species=None):
    """u_bar(k|M) on arbitrary masses. Returns (u, inside), u of shape (nM, nk).

    u is meaningless where `inside` is False; the caller decides what belongs
    there, which in the reconstruction is the NFW model. Interpolating in
    log10 M rather than demanding a shared mass grid is what lets this cache
    keep its own binning: u_bar varies with mass far more slowly than the
    0.125 dex its nodes are spaced by, so the interpolation costs nothing that
    a matched grid would have saved.
    """
    k = np.atleast_1d(np.asarray(k, dtype=float))
    M = np.atleast_1d(np.asarray(M, dtype=float))
    out = np.full((M.size, k.size), np.nan)
    ok, _ = resolved_bins(cache)
    if not ok.any():
        return out, np.zeros(M.size, dtype=bool)

    nodes = np.log10(np.asarray(cache['M_mean'], dtype=float)[ok])
    u_nodes = u_bar_of_k(cache, k, species=species)[ok]
    order = np.argsort(nodes)
    nodes, u_nodes = nodes[order], u_nodes[order]

    with np.errstate(divide='ignore', invalid='ignore'):
        lgM = np.log10(np.where(M > 0, M, np.nan))
    inside = np.isfinite(lgM) & (lgM >= nodes[0]) & (lgM <= nodes[-1])
    for ik in range(k.size):
        out[inside, ik] = np.interp(lgM[inside], nodes, u_nodes[:, ik])
    return out, inside


def check_against_rstar(cfg, cache, aperture=1.0):
    """Cross-check f_tilde over the bundle's range against the R*/U* cache.

    Both passes sort the catalogue by descending mass and assign first-wins, so
    lowering the floor only adds halos BELOW the bundle's range and cannot take
    a particle from any halo above them. The resolved halos' membership is
    therefore bit-identical between the two passes and this must agree to
    machine precision, not to a tolerance. If it does not, the two disagree
    about the catalogue, the centrals cut, the aperture or the mass definition.

    Needs the bundle's lower edge to fall on a u_bar edge; says so and skips if
    it does not. Returns None when skipped, else (here, there).
    """
    from Pmx_reconstruction.pmxlib.rstar_ustar import cache_path as ru_path
    path = ru_path(cfg, aperture=aperture)
    if not path.exists():
        print(f"  [check] no R*/U* cache at {path.name}; skipped")
        return None

    edges = np.asarray(cache['logM_edges'], float)
    hit = np.isclose(edges, cfg.logm_min, atol=1e-9)
    if not hit.any():
        print(f"  [check] the bundle's floor logM = {cfg.logm_min:g} is not a "
              f"u_bar bin edge, so the two grids do not nest and the sums "
              f"cover different mass ranges; skipped. The grid here is "
              f"{U_BINS_PER_DECADE:g} bins/decade from logM {U_LOGM_MIN:g}, "
              f"so this needs ({cfg.logm_min:g} - {U_LOGM_MIN:g}) * "
              f"{U_BINS_PER_DECADE:g} to be a whole number.")
        return None

    with np.load(path, allow_pickle=False) as f:
        there = float(np.asarray(f['f_part'], float).sum())
    here = float(f_tilde(cache)[int(np.flatnonzero(hit)[0]):].sum())
    print(f"  [check] sum f_part above logM {cfg.logm_min:g}: "
          f"u_bar {here:.12f} vs R*/U* {there:.12f}, "
          f"difference {here - there:+.3e}")
    if not np.isclose(here, there, rtol=1e-10, atol=0.0):
        print("  [check] WARNING: these should agree to machine precision. "
              "Check that both caches used the same --aperture, --mass-def, "
              "--feedback and centrals cut.")
    return here, there


# Standalone entry point
def main():
    from Pmx_reconstruction.pmxlib.config import PmxConfig, base_parser

    ap = base_parser("Measure and cache the stacked matter profile u_bar(k|M).",
                     validate=False)
    ap.add_argument('--aperture', type=float, default=1.0,
                    help="membership radius in units of r200m, matching the "
                         "R*/U* cache it will be compared against")
    ap.add_argument('--chunk-particles', type=float, default=5e7,
                    help="expected member particles per membership chunk")
    args = ap.parse_args()
    cfg = PmxConfig.from_args(args)

    cache = load_or_measure_u_bar(cfg, recompute=args.recompute,
                                  aperture=args.aperture,
                                  chunk_particles=args.chunk_particles)
    check_against_rstar(cfg, cache, aperture=args.aperture)

    k_f = 2.0 * np.pi / cfg.box
    k = np.array([k_f, 0.1, 1.0, 3.0, 9.0])
    u = u_bar_of_k(cache, k)
    counts = np.asarray(cache['counts'], float)
    print("\n  bin  logM_cen   " + "  ".join(f"u(k={kk:7.4f})" for kk in k))
    for i in np.flatnonzero(counts > 0):
        row = "  ".join(f"{v:12.5f}" for v in u[i])
        print(f"  {i:3d}   {cache['logM_cen'][i]:6.2f}   {row}")
    # u_bar(0) = 1 identically, so at the box fundamental the only departure is
    # the leading 1 - u ~ k^2 <r^2> / 6, which is ~5e-6 even for the largest
    # halo in the box. u_tilde does NOT pass this: its k -> 0 limit carries
    # Cov(M, b) within the bin. This line is the cheapest way to tell the two
    # apart, so it is worth reading every run.
    dev = float(np.nanmax(np.abs(u[:, 0] - 1.0)))
    print(f"\n  max |u_bar(k_f) - 1| = {dev:.2e} over the occupied bins "
          f"(k_f = {k_f:.5f} h/cMpc); expected ~k^2<r^2>/6, below 1e-4. "
          f"{'OK' if dev < 1e-4 else 'SUSPICIOUS -- check the centres'}")


if __name__ == '__main__':
    main()
