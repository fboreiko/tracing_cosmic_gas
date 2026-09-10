"""R*_i and U*: the measured decomposition of the matter-tracer cross spectrum.

Assign every particle to at most one catalogue central, paint the mass of each
mass bin onto its own grid, and cross each of those against the tracer field:

    P_matter_x(k) = sum_i R*_i(k) + U*(k),
    R*_i(k) = <delta_m^(i) delta_x*>,   U*(k) = <delta_m^out delta_x*>.

R*_i is the bin's TRUE contribution to the cross spectrum, and U* is everything
not assigned to any bin -- mass outside every r200m sphere, and mass in halos
below the catalogue. U* is built as delta_m - sum_i delta_m^(i) in Fourier
space, so the closure above holds to round-off by construction rather than as a
numerical coincidence.

Together they are what a reconstruction should be scored against when no bins
have been deliberately hidden:

    (R + U) / (R* + U*) - 1,

with R the reconstruction from the resolved bins and U whatever correction
stands in for the rest.

SHOT NOISE
    delta_m and delta_x share particles, so every cross spectrum here carries a
    self-pair term. It is measured, not modelled: each species is painted at
    uniformly random positions to get P_shot_xx in the pipeline's own
    convention (TSC window, L^2 factor, k binning), then split over bins by
    sum m^2. The cache stores the raw spectra, the shot ingredients AND the
    assembled shot terms, so the subtraction stays a choice at read time
    (cfg.subtract_self_pairs) instead of being baked in by whoever ran the
    measurement.

WHAT IS IN THE CACHE
    Raw:        R_star_<species>, U_star_<species>          (nbins, nk), (nk,)
    Shot:       shot_R_i_<tracer>, shot_U_<tracer>, shot_truth_<tracer>
                P_shot_dd, P_shot_gg, m2_bin_<species>, m2_tot_<species>
    Binning:    k_center, k_bins, counts, f_i, f_part, M_mean, logM_cen,
                logM_edges
    Provenance: box, ngrid, nkbins, nbins, logm_min, logm_max, redshift,
                feedback, mass_def, centrals_only, f_c, f_g, version

    'tracer' is gas, dm or matter; 'species' is only gas or dm, since matter is
    a mass-weighted combination formed at read time, never painted separately.

COST
    One pass over the particles per species (load, periodic KD-tree,
    membership), then nbins TSC paintings and FFTs. Memory is dominated by the
    nbins x ngrid^2 float64 accumulator, ~1 GB at ngrid = 2048, nbins = 30,
    plus one species' positions and the KD-tree on them.

USAGE
    from Pmx_reconstruction.pmxlib import rstar_ustar as ru
    cache = ru.load_or_compute(cfg, data)            # measures if not cached
    tot = ru.totals(cfg, cache, data, 'gas')['total']  # R* + U*, shot removed

    python -m Pmx_reconstruction.pmxlib.rstar_ustar --recompute
"""

import gc
import time

import numpy as np
from scipy.spatial import cKDTree

from utils.catalog_loaders import load_particle_properties
from utils.pipeline_paths import (DATA_ROOT, ensure_parents,
                                  get_particle_file_path)
from utils.power_spectrum_utils import (bin_power_spectrum_2d, compute_2d_fft,
                                        compute_k_grid_2d)

from Pmx_reconstruction.pmxlib.binning import load_binned_halos
from Pmx_reconstruction.pmxlib.bundle import _load_or_compute_delta
from Pmx_reconstruction.pmxlib.nfw import r200m_of_M
from Pmx_reconstruction.pmxlib.painting import paint_2d

__all__ = ['CACHE_VERSION', 'SPECIES', 'TRACERS', 'cache_path',
           'load_binned_centrals', 'assign_particles', 'compute_R_star_U_star',
           'load_or_compute', 'totals', 'load_totals']

# Bump when the cache layout or the measurement convention changes: the version
# is part of the filename, so old and new caches coexist instead of one being
# silently read as the other.
CACHE_VERSION = 'v1'

SPECIES = ('dm', 'gas')
TRACERS = ('gas', 'dm', 'matter')


# ==============================================================================
# 1.  Path
# ==============================================================================
def cache_path(cfg):
    """Companion to bundle_path: same binning knobs in the stem."""
    nk = 'default' if cfg.nkbins is None else str(cfg.nkbins)
    stem = (f'rstar_ustar_{CACHE_VERSION}_{cfg.feedback}_{cfg.mass_def}'
            f'_logM{cfg.logm_min:g}-{cfg.logm_max:g}_nb{cfg.nbins}'
            f'_ngrid{cfg.grid}_nk{nk}'
            f'_{"cen" if cfg.centrals_only else "all"}.npz')
    return DATA_ROOT / cfg.sim_name / 'pme_inputs' / stem


# ==============================================================================
# 2.  Halo catalogue, binned exactly as measure_spectra does it
# ==============================================================================
def load_binned_centrals(cfg):
    """Positions, masses, radii and bin index of the in-range centrals.

    The binning is pmxlib.binning's, the same code measure_spectra uses, so bin
    i here is bin i in the spectra bundle by construction rather than by
    comment. r200b is NOT taken from the catalogue but recomputed from M200b
    with r200m_of_M, i.e. with exactly the truncation radius the NFW profile
    assumes; the catalogue value is loaded only for a units check.
    """
    pos, mass, bin_index, logM_edges, extras = load_binned_halos(
        cfg, extra_props=('r200b',), restrict_to_range=True)
    r_cat = extras['r200b']
    r_use = r200m_of_M(mass, cfg.rhobar_m)           # cMpc/h, comoving

    finite = np.isfinite(r_cat) & (r_cat > 0)
    if np.any(finite):
        ratio = np.median(r_cat[finite] / r_use[finite])
        print(f"  catalogue r200b / r200m_of_M(M200b): median {ratio:.4f} "
              f"(1 = same units and convention; {cfg.h:.3f} or "
              f"{1 / cfg.h:.3f} = cMpc vs cMpc/h)")

    pos = np.mod(pos, cfg.box)                       # cKDTree boxsize wants [0, L)
    order = np.argsort(-mass, kind='stable')         # descending mass
    print(f"  {mass.size} centrals in [{cfg.logm_min}, {cfg.logm_max}), "
          f"r_max = {r_use.max():.3f} cMpc/h")
    return pos[order], mass[order], r_use[order], bin_index[order], logM_edges


# ==============================================================================
# 3.  Particle membership
# ==============================================================================
def assign_particles(cfg, pos_p, halo_pos, halo_r, halo_mass,
                     m_particle_msun_h, chunk_particles=5e7, nthread=4):
    """label[p] = index of the most massive central whose r200m sphere contains
    particle p, or -1.

    Periodic KD-tree on the particles; halos are walked in descending mass (the
    arrays come sorted that way) in chunks sized so the expected number of
    member particles per chunk is ~chunk_particles, which bounds the transient
    Python-list memory of query_ball_point. First assignment wins, so a
    particle in two overlapping spheres belongs to the more massive halo.
    """
    n_p = pos_p.shape[0]
    label = np.full(n_p, -1, dtype=np.int32)
    t0 = time.time()
    tree = cKDTree(pos_p, boxsize=cfg.box, leafsize=64, balanced_tree=False,
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
# 4.  Shot terms
# ==============================================================================
def _shot_terms(counts_size, nk, f_c, f_g, P_shot, m2_bin, m2_tot, tracer):
    """Self-pair contributions to R*_i, U* and the truth, in bundle units.

        <delta_m^(i) delta_x*>_shot = P_shot_xx (M_x/M_tot)
                                      sum_{p in i, species x} m^2 / sum_x m^2
        <delta_m delta_x*>_shot     = P_shot_xx (M_x/M_tot)
        U*_shot                     = truth_shot - sum_i R*_i_shot

    For tracer 'matter' the truth is an auto spectrum, hence the squared
    weights: f_c^2 P_shot_dd + f_g^2 P_shot_gg.
    """
    R_i = np.zeros((counts_size, nk))
    truth = np.zeros(nk)
    parts = []
    if tracer in ('dm', 'matter'):
        parts.append((f_c if tracer == 'dm' else f_c ** 2, P_shot['dm'],
                      m2_bin['dm'] / m2_tot['dm']))
    if tracer in ('gas', 'matter'):
        parts.append((f_g if tracer == 'gas' else f_g ** 2, P_shot['gas'],
                      m2_bin['gas'] / m2_tot['gas']))
    for w, Psh, share in parts:
        R_i += w * share[:, None] * Psh[None, :]
        truth += w * Psh
    return dict(R_i=R_i, U=truth - R_i.sum(axis=0), truth=truth)


# ==============================================================================
# 5.  The measurement
# ==============================================================================
def compute_R_star_U_star(cfg, data, chunk_particles=5e7):
    """Measure R*_i, U* and the shot terms. Returns the cache dict."""
    ngrid, nthread, nbins = cfg.grid, cfg.threads, cfg.nbins
    k_bins = np.asarray(data['k_bins'])
    k_min = 2.0 * np.pi / cfg.box
    k_grid = compute_k_grid_2d(ngrid, cfg.box)

    def _binned(prod):
        kb, kc, Pk = bin_power_spectrum_2d(prod, k_grid, ngrid, cfg.box,
                                           nkbins=cfg.nkbins, k_min=k_min)
        return np.asarray(kc), np.asarray(Pk) * cfg.box ** 2

    # --- tracer fields, exactly as measure_spectra builds them ----------------
    print("\nProjected tracer fields:")
    gas_fft = compute_2d_fft(_load_or_compute_delta(cfg, 'gas'), ngrid)
    dm_fft = compute_2d_fft(_load_or_compute_delta(cfg, 'dm'), ngrid)
    f_c, f_g = float(data['f_c']), float(data['f_g'])
    m_fft = f_c * dm_fft + f_g * gas_fft
    tracer_fft = {'gas': gas_fft, 'dm': dm_fft}

    # --- centrals -------------------------------------------------------------
    print("\nHalo catalogue:")
    h_pos, h_mass, h_r, h_bin, logM_edges = load_binned_centrals(cfg)
    if not np.allclose(logM_edges, data['logM_edges']):
        raise SystemExit("Mass-bin edges differ from the spectra bundle; "
                         "use the same --nbins/--logm-min/--logm-max.")
    n_halo = h_mass.size
    h_bin16 = h_bin.astype(np.int16)

    dens_bin = np.zeros((nbins, ngrid, ngrid), dtype=np.float64)
    mass_bin_species = {s: np.zeros(nbins) for s in SPECIES}
    m2_bin_species = {s: np.zeros(nbins) for s in SPECIES}
    mass_tot_species, m2_tot_species, P_shot_species = {}, {}, {}
    rng = np.random.default_rng(12345)
    mass_halo_assigned = np.zeros(n_halo)
    particles_file = get_particle_file_path(cfg.feedback, sim_name=cfg.sim_name)

    for species in SPECIES:
        print(f"\n[{species}] loading particles ...")
        t0 = time.time()
        part = load_particle_properties(particles_file, species,
                                        requested=('mass', 'pos'),
                                        sim_name=cfg.sim_name, Lbox=cfg.box)
        # float32 is kept for painting; cKDTree makes its own float64 copy.
        pos = np.mod(np.asarray(part['pos'], dtype=np.float32),
                     np.float32(cfg.box))
        pos[pos >= cfg.box] -= np.float32(cfg.box)   # boxsize wants [0, L)
        mass = np.asarray(part['mass'], dtype=np.float64)
        del part
        gc.collect()
        mass_tot_species[species] = float(mass.sum())
        m2_tot_species[species] = float(np.sum(mass ** 2))
        print(f"    {pos.shape[0]:.3e} particles, {time.time() - t0:.1f} s")

        # --- shot spectrum of this species, same convention as the bundle ----
        print(f"[{species}] shot-noise spectrum (random positions) ...")
        t0 = time.time()
        pos_rand = rng.uniform(0.0, cfg.box, size=pos.shape).astype(np.float32)
        pos_rand[pos_rand >= cfg.box] -= np.float32(cfg.box)
        dens_r = paint_2d(pos_rand, mass, cfg.box, ngrid, nthread)
        del pos_rand
        delta_r = dens_r / (mass_tot_species[species] / ngrid ** 2) - 1.0
        del dens_r
        fft_r = compute_2d_fft(delta_r, ngrid)
        del delta_r
        _, P_shot_species[species] = _binned(np.abs(fft_r) ** 2)
        del fft_r
        gc.collect()
        print(f"    P_shot_{species}{species}: "
              f"{np.nanmedian(P_shot_species[species]):.4e} (median over k), "
              f"{time.time() - t0:.1f} s")

        # Particle-mass unit, self-calibrated: the species' total should be its
        # cosmological share of rhobar_m L^3 (short by the stellar mass). Only
        # used to size the membership chunks, so a few per cent is fine.
        omega_b = float(cfg.sim_params.get('omega_b'))
        share = ((cfg.omega_m - omega_b) / cfg.omega_m if species == 'dm'
                 else omega_b / cfg.omega_m)
        unit = share * cfg.rhobar_m * cfg.box ** 3 / mass_tot_species[species]
        m_p = float(np.mean(mass)) * unit            # Msun/h per particle
        print(f"    mean particle mass ~ {m_p:.3e} Msun/h (self-calibrated)")

        print(f"[{species}] membership ...")
        label = assign_particles(cfg, pos, h_pos, h_r, h_mass, m_p,
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
            dens_bin[i] += paint_2d(pos[sel], m_i, cfg.box, ngrid, nthread)
        print(f"    {time.time() - t0:.1f} s")
        del pos, mass, label, member, bin_of_particle
        gc.collect()

    # --- normalisation and bookkeeping ----------------------------------------
    M_tot = mass_tot_species['dm'] + mass_tot_species['gas']
    f_c_here = mass_tot_species['dm'] / M_tot
    f_g_here = mass_tot_species['gas'] / M_tot
    print(f"\nComponent fractions here f_c={f_c_here:.5f}, f_g={f_g_here:.5f}; "
          f"bundle f_c={f_c:.5f}, f_g={f_g:.5f}")
    if abs(f_c_here - f_c) > 1e-4:
        print("  WARNING: differs from the bundle -- was the bundle built from "
              "the same particle file? The truth m_fft uses the bundle values.")

    mass_bin = mass_bin_species['dm'] + mass_bin_species['gas']
    f_part = mass_bin / M_tot                        # true mass fraction per bin
    counts = data['counts']
    n_i = counts / cfg.box ** 3
    f_i = n_i * data['M_mean'] / cfg.rhobar_m        # what the reconstruction uses

    print("\n  bin  logM   f_i (n_i M_i/rhobar)   f_part (assigned)   ratio")
    for i in range(nbins):
        if counts[i] > 0:
            r = f_part[i] / f_i[i] if f_i[i] > 0 else np.nan
            print(f"  {i:3d}  {data['logM_cen'][i]:5.2f}   {f_i[i]:.5f}"
                  f"              {f_part[i]:.5f}        {r:.4f}")
    print(f"  sum f_i = {f_i[counts > 0].sum():.4f},  "
          f"sum f_part = {f_part.sum():.4f},  "
          f"true out-of-sphere fraction = {1 - f_part.sum():.4f}")
    unit_total = cfg.rhobar_m * cfg.box ** 3 / M_tot
    with np.errstate(divide='ignore', invalid='ignore'):
        q = mass_halo_assigned * unit_total / h_mass
    print(f"  assigned mass / M200b per halo: median {np.nanmedian(q):.4f}, "
          f"16-84% [{np.nanpercentile(q, 16):.4f}, "
          f"{np.nanpercentile(q, 84):.4f}]")

    # --- spectra --------------------------------------------------------------
    print("\nPer-bin cross spectra R*_i ...")
    dens_avg = M_tot / ngrid ** 2                    # global mean per cell
    # Length from the binning function, not from k_bins: bin_power_spectrum_2d
    # applies its own k_min/k_max clipping, so len(k_bins) - 1 is an assumption
    # that silently holds until someone changes --nkbins or the box. P_shot was
    # binned through the same path above, so its length is the right one.
    nk = int(P_shot_species['dm'].size)
    R_star = {x: np.zeros((nbins, nk)) for x in tracer_fft}
    sum_fft = np.zeros_like(m_fft)
    k_center = None
    for i in range(nbins):
        if mass_bin[i] == 0:
            continue
        delta_i = dens_bin[i] / dens_avg             # rho^(i)/rhobar_m
        fft_i = compute_2d_fft(delta_i, ngrid)
        sum_fft += fft_i
        for x in tracer_fft:
            k_center, R_star[x][i] = _binned((fft_i * np.conj(tracer_fft[x])).real)
        del fft_i
        print(f"  bin {i:2d} done")
    del dens_bin
    gc.collect()

    # U* from the residual field, so the closure is exact by construction.
    out_fft = m_fft - sum_fft                        # delta_m^out
    U_star = {}
    for x in tracer_fft:
        _, U_star[x] = _binned((out_fft * np.conj(tracer_fft[x])).real)

    for x, key in (('gas', 'P_matter_gas'), ('dm', 'P_matter_dm')):
        tot = R_star[x].sum(axis=0) + U_star[x]
        truth = np.asarray(data[key], dtype=float)
        if truth.shape != tot.shape:
            print(f"  closure x={x}: SKIPPED, bundle spectrum has "
                  f"{truth.size} k bins against {tot.size} measured here")
            continue
        with np.errstate(divide='ignore', invalid='ignore'):
            dev = np.abs(tot / truth - 1.0)
        print(f"  closure x={x}: |(R* + U*)/P_matter_x - 1| max = "
              f"{np.nanmax(dev):.2e}")

    k_bundle = np.asarray(data['k_center'], dtype=float)
    if k_center is None:
        raise SystemExit("No occupied bin carried any assigned mass; nothing "
                         "was measured. Check the catalogue and the mass range.")
    if k_center.shape != k_bundle.shape or not np.allclose(k_center, k_bundle):
        print(f"  WARNING: k binning differs from the spectra bundle "
              f"({k_center.size} bins here, {k_bundle.size} there). The cache "
              f"stores the grid these spectra were measured on; callers that "
              f"combine them with bundle quantities must check it.")

    # --- assemble ------------------------------------------------------------
    out = dict(
        version=CACHE_VERSION,
        k_center=np.asarray(k_center, dtype=float), k_bins=k_bins,
        logM_edges=logM_edges, logM_cen=data['logM_cen'],
        M_mean=data['M_mean'], counts=counts, f_i=f_i, f_part=f_part,
        M_tot_particles=M_tot, f_c=f_c_here, f_g=f_g_here,
        R_star_gas=R_star['gas'], R_star_dm=R_star['dm'],
        U_star_gas=U_star['gas'], U_star_dm=U_star['dm'],
        P_shot_dd=P_shot_species['dm'], P_shot_gg=P_shot_species['gas'],
        m2_bin_dm=m2_bin_species['dm'], m2_bin_gas=m2_bin_species['gas'],
        m2_tot_dm=m2_tot_species['dm'], m2_tot_gas=m2_tot_species['gas'],
        mass_bin_dm=mass_bin_species['dm'], mass_bin_gas=mass_bin_species['gas'],
        assigned_over_m200b_median=float(np.nanmedian(q)),
        box=cfg.box, ngrid=ngrid, nkbins=(-1 if cfg.nkbins is None else cfg.nkbins),
        nbins=nbins, logm_min=cfg.logm_min, logm_max=cfg.logm_max,
        redshift=cfg.z, feedback=cfg.feedback, mass_def=cfg.mass_def,
        centrals_only=cfg.centrals_only,
    )
    # The shot terms themselves, assembled once and stored alongside the raw
    # spectra. Storing both is the point: the ingredients let the convention be
    # revisited, and these let a reader subtract without knowing it.
    for x in TRACERS:
        sh = _shot_terms(counts.size, nk, f_c_here, f_g_here, P_shot_species,
                         m2_bin_species, m2_tot_species, x)
        out[f'shot_R_i_{x}'] = sh['R_i']
        out[f'shot_U_{x}'] = sh['U']
        out[f'shot_truth_{x}'] = sh['truth']
    return out


# ==============================================================================
# 6.  Cache
# ==============================================================================
def load_or_compute(cfg, data, recompute=False, chunk_particles=5e7,
                    allow_compute=True):
    """Read the cache, or measure and write it. None if absent and not allowed.

    `allow_compute=False` turns this into a pure lookup, for callers that would
    rather degrade than trigger a full particle pass inside a plotting run.
    """
    path = cache_path(cfg)
    if path.exists() and not recompute:
        print(f"R*/U* cache:\n  {path}")
        with np.load(path, allow_pickle=False) as f:
            return {key: f[key] for key in f.files}
    if not allow_compute:
        return None
    print(f"No R*/U* cache at\n  {path}\nMeasuring (one particle pass per "
          f"species) ...")
    cache = compute_R_star_U_star(cfg, data, chunk_particles=chunk_particles)
    ensure_parents(path)
    np.savez_compressed(path, **cache)
    print(f"\nR*/U* cache saved:\n  {path}")
    return cache


# ==============================================================================
# 7.  Read-time assembly
# ==============================================================================
def totals(cfg, cache, data, tracer, subtract_shot=None):
    """R*_i, R*, U* and their sum for one tracer, shot terms optional.

    'matter' is a mass-weighted combination of the two species formed here
    rather than painted separately -- the weights are the cache's own f_c, f_g.
    Summing R*_i runs over OCCUPIED bins only, since an empty bin contributes
    nothing but can carry a shot term.
    """
    if subtract_shot is None:
        subtract_shot = bool(getattr(cfg, 'subtract_self_pairs', True))
    if tracer not in TRACERS:
        raise ValueError(f'tracer must be one of {TRACERS}, got {tracer!r}')

    if tracer == 'matter':
        f_c, f_g = float(cache['f_c']), float(cache['f_g'])
        R_star_i = f_c * cache['R_star_dm'] + f_g * cache['R_star_gas']
        U_star = f_c * cache['U_star_dm'] + f_g * cache['U_star_gas']
    else:
        R_star_i = np.array(cache[f'R_star_{tracer}'], dtype=float)
        U_star = np.array(cache[f'U_star_{tracer}'], dtype=float)

    if subtract_shot:
        R_star_i = R_star_i - cache[f'shot_R_i_{tracer}']
        U_star = U_star - cache[f'shot_U_{tracer}']

    occ = np.asarray(data['counts']) > 0
    R_star = R_star_i[occ].sum(axis=0)
    return dict(R_star_i=R_star_i, R_star=R_star, U_star=U_star,
                total=R_star + U_star, shot_subtracted=subtract_shot)


def load_totals(cfg, data, tracer, recompute=False, allow_compute=True):
    """What plotting callers want: the totals dict for `tracer`, or None.

    R_star and U_star are returned separately, not summed: the error
    decomposition needs each one. Second return value is the cache path either
    way, so a caller can name it when telling the user what to run.
    """
    path = cache_path(cfg)
    cache = load_or_compute(cfg, data, recompute=recompute,
                            allow_compute=allow_compute)
    if cache is None:
        return None, path
    k_c = np.asarray(cache['k_center'], dtype=float)
    k_b = np.asarray(data['k_center'], dtype=float)
    if k_c.shape != k_b.shape or not np.allclose(k_c, k_b, rtol=1e-6):
        print(f"[R*/U*] cache k grid does not match the bundle's "
              f"({k_c.size} vs {k_b.size} bins); ignoring it")
        return None, path
    return totals(cfg, cache, data, tracer), path


# ==============================================================================
# 8.  Standalone entry point
# ==============================================================================
def main():
    import argparse
    from Pmx_reconstruction.pmxlib.bundle import (bundle_path,
                                                  derive_P_halo_matter)
    from Pmx_reconstruction.pmxlib.config import PmxConfig, add_binning_args

    ap = argparse.ArgumentParser(
        description="Measure R*_i, U* and the self-pair terms; cache them.")
    add_binning_args(ap)
    ap.add_argument('--recompute', action='store_true',
                    help="measure even if the cache exists")
    ap.add_argument('--chunk-particles', type=float, default=5e7,
                    help="expected member particles per membership chunk")
    args = ap.parse_args()
    cfg = PmxConfig.from_args(args)

    bpath = bundle_path(cfg, cfg.nbins, cfg.logm_min, cfg.logm_max, cfg.grid,
                        cfg.nkbins)
    if not bpath.exists():
        raise SystemExit(f"Spectra bundle missing:\n  {bpath}\n"
                         "Run predict_Pmx_from_Phx.py first.")
    print(f"Spectra bundle:\n  {bpath}")
    with np.load(bpath, allow_pickle=False) as f:
        data = {key: f[key] for key in f.files}
    derive_P_halo_matter(cfg, data)

    cache = load_or_compute(cfg, data, recompute=args.recompute,
                            chunk_particles=args.chunk_particles)
    for x in TRACERS:
        t = totals(cfg, cache, data, x)
        frac = np.nanmedian(t['U_star'] / t['total'])
        print(f"  x={x}: median U*/(R*+U*) = {frac:.4f} "
              f"(shot subtracted: {t['shot_subtracted']})")


if __name__ == '__main__':
    main()