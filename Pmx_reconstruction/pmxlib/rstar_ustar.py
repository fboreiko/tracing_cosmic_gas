"""R*_i and U*: the measured decomposition of the matter-gas cross spectrum.

Assign every particle to at most one catalogue central, paint the mass of each
mass bin onto its own grid, and cross each of those against the gas field:

    P_matter_gas(k) = sum_i R*_i(k) + U*(k),
    R*_i(k) = <delta_m^(i) delta_gas*>,   U*(k) = <delta_m^out delta_gas*>.

R*_i is the bin's TRUE contribution to the cross spectrum, and U* is everything
not assigned to any bin -- mass outside every r200m sphere, and mass in halos
below the catalogue.

Together they are what a reconstruction should be scored against when no bins
have been deliberately hidden:

    (R + U) / (R* + U*) - 1,

with R the reconstruction from the resolved bins and U whatever correction
stands in for the rest.

SHOT NOISE
    delta_m and delta_gas share their gas particles, so every cross spectrum
    here carries a self-pair term. P^shot_gg comes from the bundle, which
    measured it in this same convention (TSC window, L^2 factor, k binning);
    here it is split over the bins by sum m^2. The cache stores the raw
    spectra and the ingredients; load_or_compute subtracts on the way out, so
    this file and the bundle are on the same footing and the spectra they hold
    can be differenced without anyone checking which variant is which.

WHAT IS IN THE CACHE
    Raw:        R_star_gas (nbins, nk), U_star_gas (nk,)
    Shot:       P_shot_gg (copied from the bundle), m2_bin_gas, m2_tot_gas --
                the ingredients only; the terms are derived on load and
                handed to the shared subtract_self_pairs.
    Binning:    k_center, k_bins, counts, f_i, f_part, M_mean, logM_cen,
                logM_edges
    Provenance: box, ngrid, nkbins, nbins, logm_min, logm_max, redshift,
                feedback, mass_def, centrals_only, aperture, f_c, f_g, version

    'aperture' is the membership radius in units of r200m, stored as
    provenance; nothing reads it back, the filename carries it.

    'species' is dm or gas: delta_m^(i) is painted from BOTH, since it is the
    matter in bin i, but it is only ever crossed against the gas field.

USAGE
    from Pmx_reconstruction.pmxlib import rstar_ustar as ru
    cache = ru.load_or_compute(cfg, data)        # measures if not cached
    tot = ru.totals(cfg, cache, data)['total']   # R* + U*, shot removed

    python -m Pmx_reconstruction.pmxlib.rstar_ustar --recompute
"""

import gc
import time

import numpy as np
from scipy.spatial import cKDTree

from utils.catalog_loaders import load_particle_properties
from utils.pipeline_paths import (DATA_ROOT, ensure_parents,
                                  get_particle_file_path)
from utils.power_spectrum_utils import compute_2d_fft, compute_k_grid_2d

from Pmx_reconstruction.pmxlib.binning import load_binned_halos
from Pmx_reconstruction.pmxlib.bundle import _load_or_compute_delta
from Pmx_reconstruction.pmxlib.nfw import r200m_of_M
from Pmx_reconstruction.pmxlib.painting import binned_spectrum, paint_2d
from Pmx_reconstruction.pmxlib.self_pairs import (rstar_self_pair_terms,
                                                  subtract_self_pairs)

__all__ = ['CACHE_VERSION', 'SPECIES', 'cache_path',
           'load_binned_centrals', 'assign_particles', 'compute_R_star_U_star',
           'load_or_compute', 'totals', 'load_totals']

CACHE_VERSION = 'v1'

SPECIES = ('dm', 'gas')


# Path
def cache_path(cfg, aperture=1.0):
    """Companion to bundle_path: same binning knobs in the stem.

    `aperture` is the membership radius in units of r200m (see
    compute_R_star_U_star). It enters the filename only when off the default,
    so the x = 1 cache keeps the plain stem.
    """
    nk = 'default' if cfg.nkbins is None else str(cfg.nkbins)
    ap = '' if float(aperture) == 1.0 else f'_ap{float(aperture):g}'
    stem = (f'rstar_ustar_{CACHE_VERSION}_{cfg.feedback}_{cfg.mass_def}'
            f'_logM{cfg.logm_min:g}-{cfg.logm_max:g}_nb{cfg.nbins}'
            f'_ngrid{cfg.grid}_nk{nk}'
            f'_{"cen" if cfg.centrals_only else "all"}{ap}.npz')
    return DATA_ROOT / cfg.sim_name / 'pme_inputs' / stem


# Halo catalogue, binned exactly as measure_spectra does it
def load_binned_centrals(cfg, aperture=1.0):
    """Positions, masses, radii and bin index of the in-range centrals.

    The binning is pmxlib.binning's, the same code measure_spectra uses, so bin
    i here is bin i in the spectra bundle by construction.
    """
    pos, mass, bin_index, logM_edges, extras = load_binned_halos(
        cfg, extra_props=('r200b',), restrict_to_range=True)
    r_cat = extras['r200b']
    r_use = float(aperture) * r200m_of_M(mass, cfg.rhobar_m)   # cMpc/h, comoving

    finite = np.isfinite(r_cat) & (r_cat > 0)
    if np.any(finite):
        # Divided by the aperture so the check tests the units and the mass
        # definition, not the knob: it must read 1 whatever the aperture is.
        ratio = np.median(r_cat[finite] / r_use[finite]) * float(aperture)
        print(f"  catalogue r200b / r200m_of_M(M200b): median {ratio:.4f} "
              f"(1 = same units and convention; {cfg.h:.3f} or "
              f"{1 / cfg.h:.3f} = cMpc vs cMpc/h)")

    pos = np.mod(pos, cfg.box)                       # cKDTree boxsize wants [0, L)
    order = np.argsort(-mass, kind='stable')         # descending mass
    print(f"  {mass.size} centrals in [{cfg.logm_min}, {cfg.logm_max}), "
          f"aperture = {float(aperture):g} r200m, "
          f"r_max = {r_use.max():.3f} cMpc/h")
    return pos[order], mass[order], r_use[order], bin_index[order], logM_edges


# Particle membership
def assign_particles(cfg, pos_p, halo_pos, halo_r, halo_mass,
                     m_particle_msun_h, chunk_particles=5e7, nthread=4,
                     aperture=1.0):
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

    expected = float(aperture) ** 3 * halo_mass / m_particle_msun_h
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


# The measurement
def compute_R_star_U_star(cfg, data, chunk_particles=5e7, aperture=1.0):
    """Measure R*_i and U*. Returns the cache dict.

    `aperture` changes WHICH particles carry a halo label, so how the identity
    P_matter_gas = sum_i R*_i + U* splits between the two; it does not change
    the identity. Bins stay labelled by M200b, so R*_i is comparable across
    apertures bin by bin.
    """
    ngrid, nthread, nbins = cfg.grid, cfg.threads, cfg.nbins
    k_bins = np.asarray(data['k_bins'])
    k_grid = compute_k_grid_2d(ngrid, cfg.box)

    def _binned(prod):
        """binned_spectrum, dropping the bin edges this function never uses."""
        _, kc, Pk = binned_spectrum(cfg, prod, k_grid, cfg.nkbins)
        return kc, Pk

    # --- the fields, exactly as measure_spectra builds them -------------------
    print("\nProjected fields:")
    gas_fft = compute_2d_fft(_load_or_compute_delta(cfg, 'gas'), ngrid)
    dm_fft = compute_2d_fft(_load_or_compute_delta(cfg, 'dm'), ngrid)
    f_c, f_g = float(data['f_c']), float(data['f_g'])
    m_fft = f_c * dm_fft + f_g * gas_fft
    del dm_fft
    gc.collect()

    # --- centrals -------------------------------------------------------------
    print("\nHalo catalogue:")
    h_pos, h_mass, h_r, h_bin, logM_edges = load_binned_centrals(
        cfg, aperture=aperture)
    if not np.allclose(logM_edges, data['logM_edges']):
        raise SystemExit("Mass-bin edges differ from the spectra bundle; "
                         "use the same --nbins/--logm-min/--logm-max.")
    n_halo = h_mass.size
    h_bin16 = h_bin.astype(np.int16)

    dens_bin = np.zeros((nbins, ngrid, ngrid), dtype=np.float64)
    mass_bin_species = {s: np.zeros(nbins) for s in SPECIES}
    m2_bin_gas = np.zeros(nbins)
    mass_tot_species = {}
    m2_tot_gas = 0.0
    mass_halo_assigned = np.zeros(n_halo)
    # P^shot_gg is not re-measured here: the bundle carries it already
    P_shot_gg = np.asarray(data['P_shot_gas'], dtype=float)
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
        if species == 'gas':
            m2_tot_gas = float(np.sum(mass ** 2))
        print(f"    {pos.shape[0]:.3e} particles, {time.time() - t0:.1f} s")

        omega_b = float(cfg.sim_params.get('omega_b'))
        share = ((cfg.omega_m - omega_b) / cfg.omega_m if species == 'dm'
                 else omega_b / cfg.omega_m)
        unit = share * cfg.rhobar_m * cfg.box ** 3 / mass_tot_species[species]
        m_p = float(np.mean(mass)) * unit            # Msun/h per particle
        print(f"    mean particle mass ~ {m_p:.3e} Msun/h (self-calibrated)")

        print(f"[{species}] membership ...")
        label = assign_particles(cfg, pos, h_pos, h_r, h_mass, m_p,
                                 chunk_particles=chunk_particles,
                                 nthread=nthread, aperture=aperture)
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
            if species == 'gas':
                m2_bin_gas[i] = float(np.sum(m_i ** 2))
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
    nk = int(P_shot_gg.size)
    R_star = np.zeros((nbins, nk))
    sum_fft = np.zeros_like(m_fft)
    k_center = None
    for i in range(nbins):
        if mass_bin[i] == 0:
            continue
        delta_i = dens_bin[i] / dens_avg             # rho^(i)/rhobar_m
        fft_i = compute_2d_fft(delta_i, ngrid)
        sum_fft += fft_i
        k_center, R_star[i] = _binned((fft_i * np.conj(gas_fft)).real)
        del fft_i
        print(f"  bin {i:2d} done")
    del dens_bin
    gc.collect()

    # U* from the residual field, so the closure is exact by construction.
    out_fft = m_fft - sum_fft                        # delta_m^out
    _, U_star = _binned((out_fft * np.conj(gas_fft)).real)

    tot = R_star.sum(axis=0) + U_star
    truth = np.asarray(data['P_matter_gas'], dtype=float)
    if truth.shape != tot.shape:
        print(f"  closure: SKIPPED, bundle spectrum has {truth.size} k bins "
              f"against {tot.size} measured here")
    else:
        with np.errstate(divide='ignore', invalid='ignore'):
            dev = np.abs(tot / truth - 1.0)
        print(f"  closure: |(R* + U*)/P_matter_gas - 1| max = "
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
        R_star_gas=R_star, U_star_gas=U_star,
        P_shot_gg=P_shot_gg, m2_bin_gas=m2_bin_gas, m2_tot_gas=m2_tot_gas,
        mass_bin_dm=mass_bin_species['dm'], mass_bin_gas=mass_bin_species['gas'],
        assigned_over_m200b_median=float(np.nanmedian(q)),
        box=cfg.box, ngrid=ngrid, nkbins=(-1 if cfg.nkbins is None else cfg.nkbins),
        nbins=nbins, logm_min=cfg.logm_min, logm_max=cfg.logm_max,
        redshift=cfg.z, feedback=cfg.feedback, mass_def=cfg.mass_def,
        centrals_only=cfg.centrals_only, aperture=float(aperture),
    )
    return out


# Cache
def load_or_measure_rstar_ustar(cfg, data, recompute=False, chunk_particles=5e7,
                    allow_compute=True, aperture=1.0):
    """Read the cache, or measure and write it. None if absent and not allowed."""
    path = cache_path(cfg, aperture=aperture)
    if path.exists() and not recompute:
        print(f"R*/U* cache:\n  {path}")
        with np.load(path, allow_pickle=False) as f:
            cache = {key: f[key] for key in f.files}
        return subtract_self_pairs(cache, terms=rstar_self_pair_terms(cache),
                                   report=False)
    if not allow_compute:
        return None
    print(f"No R*/U* cache at\n  {path}\nMeasuring (one particle pass per "
          f"species) ...")
    cache = compute_R_star_U_star(cfg, data, chunk_particles=chunk_particles,
                                  aperture=aperture)
    ensure_parents(path)
    np.savez_compressed(path, **cache)
    print(f"\nR*/U* cache saved:\n  {path}")
    return subtract_self_pairs(cache, terms=rstar_self_pair_terms(cache),
                               report=False)


# Read-time assembly
def totals(cfg, cache, data, mr=None):
    """The measured partition P_matter_gas = R* + U* + V*.

    R*   bins in [M_r, M_max]
    U*   matter outside every sphere, plus the bins below M_r
    V*   bins above M_max (zero by default)

    Moving a bin between the three is exact (hosts are ranked by mass) and
    needs no particle pass. mr=None puts every occupied bin in R*.

    The self-pair terms are already off: load_or_compute removes them, exactly
    as load_or_measure does for the bundle. All that happens here is the
    partition.
    """
    R_i = np.array(cache['R_star_gas'], dtype=float)
    U = np.array(cache['U_star_gas'], dtype=float)

    occ = np.asarray(data['counts']) > 0
    if mr is None:
        resolved, hidden, above = occ, np.zeros_like(occ), np.zeros_like(occ)
    else:
        resolved, hidden, above = mr.resolved, mr.hidden, mr.above
    f_part = np.asarray(cache['f_part'], dtype=float)

    R_star = R_i[resolved].sum(axis=0)
    U_star = U + R_i[hidden].sum(axis=0)
    V_star = R_i[above].sum(axis=0)
    return dict(R_star_i=np.where(resolved[:, None], R_i, 0.0),
                R_star=R_star, U_star=U_star, V_star=V_star,
                total=R_star + U_star + V_star,
                f_part=np.where(resolved, f_part, 0.0),
                f_out=1.0 - float(f_part[resolved | above].sum()))


def load_totals(cfg, data, recompute=False, allow_compute=True,
                aperture=1.0, mr=None):
    """What plotting callers want: the totals dict, or None.

    R_star and U_star are returned separately, not summed: the error
    decomposition needs each one. Second return value is the cache path either
    way, so a caller can name it when telling the user what to run.
    """
    path = cache_path(cfg, aperture=aperture)
    cache = load_or_measure_rstar_ustar(cfg, data, recompute=recompute,
                            allow_compute=allow_compute, aperture=aperture)
    if cache is None:
        return None, path
    k_c = np.asarray(cache['k_center'], dtype=float)
    k_b = np.asarray(data['k_center'], dtype=float)
    if k_c.shape != k_b.shape or not np.allclose(k_c, k_b, rtol=1e-6):
        print(f"[R*/U*] cache k grid does not match the bundle's "
              f"({k_c.size} vs {k_b.size} bins); ignoring it")
        return None, path
    return totals(cfg, cache, data, mr=mr), path


# ==============================================================================
# 8.  Standalone entry point
# ==============================================================================
def main():
    from Pmx_reconstruction.pmxlib.bundle import bundle_path
    from Pmx_reconstruction.pmxlib.config import PmxConfig, base_parser

    # base_parser brings --recompute and the binning group; the mass-range and
    # extrapolation groups it also brings are harmless here, since this entry
    # point only measures and caches. It is the same parser both experiments
    # use, so a cache can never be built on a different binning from theirs.
    ap = base_parser("Measure R*_i, U* and the self-pair terms; cache them.",
                     validate=False)
    ap.add_argument('--chunk-particles', type=float, default=5e7,
                    help="expected member particles per membership chunk")
    ap.add_argument('--aperture', type=float, default=1.0,
                    help="membership radius in units of r200m. Values above 1 "
                         "are what experiment_B.py sweeps.")
    args = ap.parse_args()
    cfg = PmxConfig.from_args(args)

    bpath = bundle_path(cfg, cfg.nbins, cfg.logm_min, cfg.logm_max, cfg.grid,
                        cfg.nkbins)
    if not bpath.exists():
        raise SystemExit(f"Spectra bundle missing:\n  {bpath}\n"
                         "Build it by running experiment_A or experiment_B.")
    print(f"Spectra bundle:\n  {bpath}")
    with np.load(bpath, allow_pickle=False) as f:
        data = {key: f[key] for key in f.files}

    cache = load_or_measure_rstar_ustar(cfg, data, recompute=args.recompute,
                            chunk_particles=args.chunk_particles,
                            aperture=args.aperture)
    t = totals(cfg, cache, data)
    frac = np.nanmedian(t['U_star'] / t['total'])
    print(f"  median U*/(R*+U*) = {frac:.4f}")


if __name__ == '__main__':
    main()