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
    spectra and the ingredients; the loader subtracts on the way out, so
    this file and the bundle are on the same footing and the spectra they hold
    can be differenced without anyone checking which variant is which.

RANDOMISED VARIANTS  (--randomise, experiment D)
    With `randomise` set, every particle that carries a halo label is moved
    inside its own host before being painted, by pmxlib.spherise, so that its
    distance from the halo centre is preserved exactly and nothing else is.
    The gas field it is crossed against is the true one. R*_i then measures
    the contribution of a SPHERISED (or randomly reoriented) halo population,
    and R*(true) - R*(randomised) is the intra-halo matter-gas correlation
    that no radial profile can carry. See pmxlib.spherise for the algebra.

    Three things follow, and all three are load-bearing:

      - Unlabelled particles do not move, so delta_m^out and therefore U* are
        identical to the production run at the same aperture. U* is still
        MEASURED here rather than copied, by painting the unassigned particles
        directly; agreeing with the production cache is then a check on the
        membership rather than an assumption about it.
      - Labels do not change, so f_i, f_part, M_mean, counts and the mass
        moments are bit-identical to the production cache. Anything else is a
        bug, not a result.
      - The self-pair correction changes shape but does NOT go away, and this
        is the easiest thing here to get wrong. A member gas particle is still
        in the gas field at its true position while its copy in delta_m^(i)
        has moved to a random direction at the same radius, so the pair is
        smeared to a separation of order r rather than removed, and what comes
        off R*_i is m^2 j0(k r)^2 instead of m^2. An UNASSIGNED gas particle
        never moved, so U* keeps exactly the term it always had. Both live in
        self_pairs.rstar_self_pair_terms, which the loader calls with the
        cache's own `randomise` key, so a randomised cache and a production
        one stay differenceable -- the entire point of deciding it in one
        place. Leaving R*_i uncorrected biases R*(randomised) HIGH by nearly
        the full self-pair at low k, which subtracts straight out of the
        intra-halo term and can turn it negative wherever the spectrum is
        shot-dominated.

    U* being untouched also means the reconstruction's U side is unaffected,
    which is why experiment D compares R against R* and not (R + U) against
    the full spectrum.

WHAT IS IN THE CACHE
    Raw:        R_star_gas (nbins, nk), U_star_gas (nk,)
    Shot:       P_shot_gg (copied from the bundle), m2_bin_gas, m2_tot_gas --
                the ingredients only; the terms are derived on load and
                handed to the shared subtract_self_pairs.
    Moments:    m2_bin_mat, m2_tot_mat -- the same second moments over dm AND
                gas. Nothing in the self-pair path wants them; they set the
                noise a randomised run adds, and experiment_D reports it.
    Smearing:   m2_shell_gas, m2r_shell_gas (nbins, selfpair_nx) -- the member
                gas m^2 against radius, RANDOMISED CACHES ONLY. This is what
                turns the coincident self-pair into m^2 j0(k r)^2 on the way
                out; without it R*_i comes off disk uncorrected.
    Binning:    k_center, k_bins, counts, f_i, f_part, M_mean, logM_cen,
                logM_edges
    Provenance: box, ngrid, kbins_per_decade, nbins, logm_min, logm_max, redshift,
                feedback, mass_def, centrals_only, aperture, randomise,
                f_c, f_g, version

    'aperture' is the membership radius in units of r200m, stored as
    provenance; nothing reads it back, the filename carries it. 'randomise' is
    read back, by cache_randomise: it decides whether the loader subtracts.

    With --ngrid-3d there is a second cache beside it, the same measurement on
    a cubic grid, carrying nmodes / k_eff / ngrid_3d as well. The loader
    splices the two at --k-split so the error budget at low k is not limited
    by the projection while P_matter_gas no longer is.

    'species' is dm or gas: delta_m^(i) is painted from BOTH, since it is the
    matter in bin i, but it is only ever crossed against the gas field.

USAGE
    from Pmx_reconstruction.pmxlib import rstar_ustar as ru
    cache = ru.load_or_measure_rstar_ustar(cfg, data)   # measures if absent
    tot = ru.measured_partition(cfg, cache, data)['total']   # shot removed

    python -m Pmx_reconstruction.pmxlib.rstar_ustar --recompute
    python -m Pmx_reconstruction.pmxlib.rstar_ustar --randomise shuffle
"""

import gc
import time

import numpy as np
from scipy.spatial import cKDTree

from utils.catalog_loaders import load_particle_properties
from utils.pipeline_paths import (DATA_ROOT, ensure_parents,
                                  get_particle_file_path)
from utils.power_spectrum_utils import (Binner3D, bin_mode_stats,
                                        compute_2d_fft, compute_3d_fft,
                                        compute_k_grid_2d)

from Pmx_reconstruction.pmxlib.binning import load_binned_halos
from Pmx_reconstruction.pmxlib.bundle import _load_or_compute_delta
from Pmx_reconstruction.pmxlib.bundle3d import stitch_on_k
from Pmx_reconstruction.pmxlib.nfw import r200m_of_M
from Pmx_reconstruction.pmxlib.painting import (binned_spectrum,
                                                binned_spectrum_3d, paint_2d,
                                                paint_3d)
from Pmx_reconstruction.pmxlib.self_pairs import (rstar_self_pair_terms,
                                                  subtract_self_pairs)
from Pmx_reconstruction.pmxlib import spherise

__all__ = ['CACHE_VERSION', 'SPECIES', 'cache_path', 'cache_randomise',
           'load_binned_centrals', 'assign_particles', 'compute_R_star_U_star',
           'load_or_measure_rstar_ustar', 'measured_partition',
           'load_measured_partition']

CACHE_VERSION = 'v1'

SPECIES = ('dm', 'gas')

# Raw particles per chunk when painting the unassigned ones. Its own constant
# rather than --chunk-particles, which counts EXPECTED MEMBERS and is sized for
# query_ball_point's list-of-lists; here the transient is a straight
# (chunk, 3) float32 copy, 240 MB at this value.
PAINT_CHUNK = 20_000_000


# Path
def cache_path(cfg, aperture=1.0, ngrid3=None, randomise=None):
    """Companion to bundle_path: same binning knobs in the stem.

    `aperture` is the membership radius in units of r200m (see
    compute_R_star_U_star). It enters the filename only when off the default,
    so the x = 1 cache keeps the plain stem.

    `ngrid3` names the cubic-grid companion of a cache. It is a SEPARATE file
    rather than extra keys in the 2-D one, so the caches already on disk stay
    valid and only what is missing gets measured.

    `randomise` likewise appends only when set, so every cache measured before
    experiment D existed keeps its name and none of them is re-measured. The
    mode is IN the stem rather than a key inside, because a randomised cache
    and the production one it is differenced against have to be able to sit
    side by side.
    """
    ap = '' if float(aperture) == 1.0 else f'_ap{float(aperture):g}'
    rnd = '' if not randomise else f'_rand-{spherise.check_mode(randomise)}'
    grid = f'ngrid{cfg.grid}' if ngrid3 is None else f'3d_n{int(ngrid3)}'
    stem = (f'rstar_ustar_{CACHE_VERSION}_{cfg.feedback}_{cfg.mass_def}'
            f'_logM{cfg.logm_min:g}-{cfg.logm_max:g}_nb{cfg.nbins}'
            f'_{grid}_{cfg.kbin_tag}'
            f'_{"cen" if cfg.centrals_only else "all"}{ap}{rnd}.npz')
    return DATA_ROOT / cfg.sim_name / 'pme_inputs' / stem


# Halo catalogue, binned exactly as measure_spectra does it
def load_binned_centrals(cfg, aperture=1.0, nbins=None, logm_min=None,
                         logm_max=None):
    """Positions, masses, radii and bin index of the in-range centrals.

    The binning is pmxlib.binning's, the same code measure_spectra uses, so bin
    i here is bin i in the spectra bundle by construction -- unless the caller
    overrides it, which only pmxlib.u_bar does and which puts it on its own
    mass grid deliberately.

    Halos come back sorted by DESCENDING mass, and assign_particles is
    first-assignment-wins, so extending the range downwards adds halos at the
    bottom of the order and cannot take a particle from any halo above them:
    the membership of the bundle's bins is bit-identical whatever the floor.
    """
    pos, mass, bin_index, logM_edges, extras = load_binned_halos(
        cfg, extra_props=('r200b',), restrict_to_range=True,
        nbins=nbins, logm_min=logm_min, logm_max=logm_max)
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
    lo = cfg.logm_min if logm_min is None else float(logm_min)
    hi = cfg.logm_max if logm_max is None else float(logm_max)
    print(f"  {mass.size} centrals in [{lo}, {hi}), "
          f"aperture = {float(aperture):g} r200m, "
          f"r_max = {r_use.max():.3f} cMpc/h")
    return pos[order], mass[order], r_use[order], bin_index[order], logM_edges


# Particle membership
def assign_particles(cfg, pos_p, halo_pos, halo_r, halo_mass,
                     m_particle_msun_h, chunk_particles=5e7, nthread=4,
                     aperture=1.0, max_halos_per_chunk=2_000_000):
    """label[p] = index of the most massive central whose r200m sphere contains
    particle p, or -1.

    Periodic KD-tree on the particles; halos are walked in descending mass (the
    arrays come sorted that way) in chunks sized so the expected number of
    member particles per chunk is ~chunk_particles, which bounds the transient
    Python-list memory of query_ball_point. First assignment wins, so a
    particle in two overlapping spheres belongs to the more massive halo.

    `max_halos_per_chunk` bounds the chunk from the other side. Sizing purely
    by expected particles is fine while the catalogue stops at 10^11, but
    n(M) ~ M^-1.9, so a floor two decades lower multiplies the halo count by
    ~60 and most of those halos expect under one particle each: the cumulative
    sum barely advances and a single chunk swallows millions of halos, whose
    list-of-lists from query_ball_point is the memory the sizing was meant to
    bound. With the bundle's own range the cap never binds.
    """
    n_p = pos_p.shape[0]
    label = np.full(n_p, -1, dtype=np.int32)
    t0 = time.time()
    tree = cKDTree(pos_p, boxsize=cfg.box, leafsize=64, balanced_tree=False,
                   compact_nodes=False)
    print(f"    KD-tree on {n_p:.3e} particles: {time.time() - t0:.1f} s")

    expected = float(aperture) ** 3 * halo_mass / m_particle_msun_h
    cum = np.cumsum(expected)
    edges = np.unique(np.concatenate([
        np.searchsorted(cum, np.arange(0.0, cum[-1], chunk_particles)),
        np.arange(0, halo_mass.size, int(max_halos_per_chunk)),
        [halo_mass.size]]))

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
def _bin_spectra(dens_bin, dens_avg, gas_fft, fft_of, binned, occupied, nk,
                 m_fft=None, dens_out=None):
    """R*_i against the gas field, and U* from whichever route is available.

    `m_fft` (production): U* is taken from the full matter transform minus the
    summed per-bin ones rather than painted separately, so
    P_matter_gas = sum_i R*_i + U* closes exactly on whichever grid this is
    called on, and _closure can say so.

    `dens_out` (randomised): the per-bin fields have been moved and the full
    matter field has not, so the residual route would measure
    U* + (in-place - moved) instead of U*. The unassigned particles are
    painted directly instead; they are the ones the randomisation leaves
    alone, so this U* is the production one re-measured by a second route.

    Exactly one of the two must be given.
    """
    if (m_fft is None) == (dens_out is None):
        raise ValueError("_bin_spectra takes m_fft (production) or dens_out "
                         "(randomised), not both and not neither")
    R_star = np.zeros((occupied.size, nk))
    sum_fft = None if m_fft is None else np.zeros_like(m_fft)
    for i in np.flatnonzero(occupied):
        fft_i = fft_of(np.asarray(dens_bin[i], dtype=np.float64) / dens_avg)
        if sum_fft is not None:
            sum_fft += fft_i
        R_star[i] = binned((fft_i * np.conj(gas_fft)).real)
        del fft_i
        print(f"  bin {i:2d} done")
    if sum_fft is None:
        out_fft = fft_of(np.asarray(dens_out, dtype=np.float64) / dens_avg)
        U_star = binned((out_fft * np.conj(gas_fft)).real)
        del out_fft
    else:
        U_star = binned(((m_fft - sum_fft) * np.conj(gas_fft)).real)
        del sum_fft
    gc.collect()
    return R_star, U_star


def _paint_unassigned(cfg, pos, mass, label, ngrid, out_2d, ngrid3, out_3d,
                      chunk=PAINT_CHUNK):
    """Add the particles with no halo label to the out-of-sphere grids.

    Chunked, because the selection is most of the box: pos[label < 0] as one
    fancy-index copy would be three quarters of the particle array again,
    which is the one allocation this pass cannot afford. Accumulating grid by
    grid keeps the transient at one chunk.
    """
    n_p = label.size
    step = max(1, int(chunk))
    n_out = 0
    for a in range(0, n_p, step):
        b = min(a + step, n_p)
        sel = label[a:b] < 0
        if not np.any(sel):
            continue
        p = pos[a:b][sel]
        m = mass[a:b][sel]
        n_out += int(m.size)
        if out_2d is not None:
            out_2d += paint_2d(p, m, cfg.box, ngrid, cfg.threads)
        if out_3d is not None:
            paint_3d(p, m, cfg.box, ngrid3, cfg.threads, out=out_3d)
        del p, m, sel
    return n_out


def _closure(tag, R_star, U_star, data, suffix='', randomise=None):
    """|(R* + U*)/P_matter_gas - 1|, the check that U*-from-the-residual closes.

    Against the RAW bundle spectrum: the identity holds for the painted fields,
    and R*/U* here have not had their self-pair term taken off yet. Comparing
    them to the corrected spectrum would report the shot fraction instead.

    There is nothing to check on a randomised run and the ratio is not printed
    as though there were. Two separate things break it: the matter field has
    been changed and the bundle's spectrum has not, and R*_i no longer carries
    a self-pair term while the bundle's spectrum still does, so the number
    would be a physical effect and an accounting mismatch added together.
    Scoring the randomised run is experiment_D's job, on corrected spectra and
    against the production cache rather than against the bundle.
    """
    if randomise is not None:
        print(f"  closure [{tag}]: not applicable to a '{randomise}' run; "
              f"score it with\n      python -m "
              f"Pmx_reconstruction.experiment_D --randomise {randomise}")
        return
    tot = R_star.sum(axis=0) + U_star
    for key in (f'P_matter_gas_raw{suffix}', f'P_matter_gas{suffix}',
                'P_matter_gas_raw', 'P_matter_gas'):
        if key in data:
            truth = np.asarray(data[key], dtype=float)
            break
    if truth.shape != tot.shape:
        print(f"  closure [{tag}]: SKIPPED, bundle spectrum has {truth.size} "
              f"k bins against {tot.size} measured here")
        return
    with np.errstate(divide='ignore', invalid='ignore'):
        dev = np.abs(tot / truth - 1.0)
    print(f"  closure [{tag}]: |(R* + U*)/P_matter_gas - 1| max = "
          f"{np.nanmax(dev):.2e}")


def compute_R_star_U_star(cfg, data, chunk_particles=5e7, aperture=1.0,
                          want_2d=True, ngrid3=None, randomise=None):
    """Measure R*_i and U*. Returns (cache_2d, cache_3d), either may be None.

    `aperture` changes WHICH particles carry a halo label, so how the identity
    P_matter_gas = sum_i R*_i + U* splits between the two; it does not change
    the identity. Bins stay labelled by M200b, so R*_i is comparable across
    apertures bin by bin.

    `randomise` ('shuffle' or 'rotate') moves each labelled particle inside its
    own host before painting, preserving its radius exactly; see the module
    docstring and pmxlib.spherise. The halo catalogue, the labels, the mass
    budget and the gas field are all untouched, so the only thing that can
    differ from the production cache is the spectra.

    The two grids are measured in the SAME pass because the cost here is the
    membership KD-tree, which is grid-independent; painting a second grid
    inside the loop that is already running is nearly free. Pass want_2d=False
    to add a cubic grid to a 2-D cache that already exists.
    """
    randomise = spherise.check_mode(randomise)
    ngrid, nthread, nbins = cfg.grid, cfg.threads, cfg.nbins
    k_grid = compute_k_grid_2d(ngrid, cfg.box)
    f_c, f_g = float(data['f_c']), float(data['f_g'])
    if not (want_2d or ngrid3):
        raise ValueError("nothing to measure: want_2d is False and ngrid3 is None")
    k_bins = cfg.k_bins
    nmodes, k_center = bin_mode_stats(k_grid, k_bins)
    if not np.allclose(k_bins, np.asarray(data['k_bins'], dtype=float)):
        raise SystemExit("The k binning here differs from the spectra bundle's. "
                         "The two are differenced against each other, so they "
                         "must share --kbins-per-decade, --kbin-wmin and "
                         "--ngrid.")

    def _binned(prod):
        return binned_spectrum(cfg, prod, k_grid)

    binner3 = None
    if ngrid3:
        binner3 = Binner3D(ngrid3, cfg.box, k_bins)
        print(f"\n3-D companion on a {ngrid3}^3 grid, k_Nyquist = "
              f"{binner3.k_Nyquist:.3f} h/cMpc, TSC window deconvolved")

    if randomise:
        print(f"\n{'=' * 70}\nRANDOMISED RUN: '{randomise}'. Every labelled "
              f"particle is moved inside its own host at fixed radius; the "
              f"gas\nfield it is crossed against is the true one. See "
              f"pmxlib.spherise.\n{'=' * 70}")

    # --- the projected fields, exactly as measure_spectra builds them ---------
    # delta_m is only needed to get U* as a residual. A randomised run cannot
    # use that route (the per-bin fields have moved and this one has not), so
    # it paints the unassigned particles instead and never reads the dm field.
    gas_fft = m_fft = None
    if want_2d:
        print("\nProjected fields:")
        gas_fft = compute_2d_fft(_load_or_compute_delta(cfg, 'gas'), ngrid)
        if randomise is None:
            dm_fft = compute_2d_fft(_load_or_compute_delta(cfg, 'dm'), ngrid)
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

    # One draw for the whole run, shared by both species: a halo's dark matter
    # and its gas have to turn together, or the halo's MATTER is not rigidly
    # rotated but two independently rotated pieces.
    quat = spherise.halo_rotations(n_halo) if randomise == 'rotate' else None

    dens_bin = (np.zeros((nbins, ngrid, ngrid), dtype=np.float64)
                if want_2d else None)
    # float32 here: tsc accumulates in float32 anyway, and the stack is
    # 2.0 GB at 256^3 against 4.0 in float64.
    dens_bin_3d = (np.zeros((nbins, ngrid3, ngrid3, ngrid3), dtype=np.float32)
                   if ngrid3 else None)
    # Only the gas field is wanted whole on a randomised run; dm enters
    # nowhere but m_fft, which that run does not build.
    species_all = SPECIES if randomise is None else ('gas',)
    dens_all_3d = ({s: np.zeros((ngrid3,) * 3, dtype=np.float32)
                    for s in species_all} if ngrid3 else None)
    dens_out_2d = (np.zeros((ngrid, ngrid), dtype=np.float64)
                   if (randomise and want_2d) else None)
    dens_out_3d = (np.zeros((ngrid3,) * 3, dtype=np.float32)
                   if (randomise and ngrid3) else None)
    mass_bin_species = {s: np.zeros(nbins) for s in SPECIES}
    m2_bin_gas = np.zeros(nbins)
    m2_bin_mat = np.zeros(nbins)
    mass_tot_species = {}
    m2_tot_gas = 0.0
    m2_tot_mat = 0.0
    n_unassigned = 0
    # Radial histogram of the member gas m^2, for the smeared self-pair that a
    # randomised run leaves behind. Gas only: dark matter shares no particle
    # with the gas field.
    m2_shell = m2r_shell = None
    mass_halo_assigned = np.zeros(n_halo)
    # P^shot_gg is not re-measured here: the bundle carries it already, and
    # carries one per grid once it has been stitched.
    P_shot_2d = np.asarray(data.get('P_shot_gas_2d', data['P_shot_gas']),
                           dtype=float)
    P_shot_3d = None
    if ngrid3:
        if 'P_shot_gas_3d' not in data:
            raise SystemExit(
                "The bundle carries no 3-D self-pair spectrum. Build it first "
                "with the same --ngrid-3d, e.g.\n"
                f"      python -m Pmx_reconstruction.experiment_A "
                f"--ngrid-3d {ngrid3}")
        P_shot_3d = np.asarray(data['P_shot_gas_3d'], dtype=float)
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
        m2_species = float(np.sum(mass ** 2))
        m2_tot_mat += m2_species
        if species == 'gas':
            m2_tot_gas = m2_species
        print(f"    {pos.shape[0]:.3e} particles, {time.time() - t0:.1f} s")

        if ngrid3 and species in dens_all_3d:
            paint_3d(pos, mass, cfg.box, ngrid3, nthread,
                     out=dens_all_3d[species])

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

        if randomise:
            # Order matters twice over: the out-of-sphere grid has to be
            # painted from the TRUE positions, and the randomisation has to
            # happen before the per-bin painting and after nothing else reads
            # `pos`, because it overwrites it.
            if species == 'gas':
                print(f"[{species}] radial histogram of m^2, for the smeared "
                      f"self-pair ...")
                t0 = time.time()
                m2_shell, m2r_shell = spherise.selfpair_shells(
                    pos, label, h_pos, h_r, h_bin.astype(np.int64),
                    mass ** 2, cfg.box, nbins)
                print(f"    {time.time() - t0:.1f} s")
            print(f"[{species}] painting the unassigned particles ...")
            t0 = time.time()
            n_unassigned += _paint_unassigned(
                cfg, pos, mass, label, ngrid, dens_out_2d, ngrid3, dens_out_3d)
            print(f"    {time.time() - t0:.1f} s")
            print(f"[{species}] randomising inside the hosts ...")
            t0 = time.time()
            info = spherise.randomise_in_place(pos, label, h_pos, cfg.box,
                                               randomise, quat=quat)
            tol = spherise.position_tolerance(cfg.box)
            if info['dr_max'] > tol:
                raise RuntimeError(
                    f"[{species}] a particle's radius moved by "
                    f"{info['dr_max']:.3e} cMpc/h, past the {tol:.3e} the "
                    f"float32 positions can explain. The randomisation is "
                    f"supposed to preserve it exactly, so this is the "
                    f"minimum image, the wrap or the host indexing -- not a "
                    f"tolerance to widen.")
            print(f"    {time.time() - t0:.1f} s")

        print(f"[{species}] painting per bin ...")
        t0 = time.time()
        bin_of_particle = np.full(label.size, -1, dtype=np.int16)
        bin_of_particle[member] = h_bin16[label[member]]
        for i in range(nbins):
            sel = bin_of_particle == i
            if not np.any(sel):
                continue
            m_i = mass[sel]
            pos_i = pos[sel]
            mass_bin_species[species][i] = float(m_i.sum())
            m2_i = float(np.sum(m_i ** 2))
            m2_bin_mat[i] += m2_i
            if species == 'gas':
                m2_bin_gas[i] = m2_i
            if want_2d:
                dens_bin[i] += paint_2d(pos_i, m_i, cfg.box, ngrid, nthread)
            if ngrid3:
                paint_3d(pos_i, m_i, cfg.box, ngrid3, nthread,
                         out=dens_bin_3d[i])
            del pos_i, m_i
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
    if randomise:
        print(f"  {n_unassigned:.3e} particles carried no halo label and were "
              f"left where they are; U* is the production one by "
              f"construction, and is re-measured from them here")
    unit_total = cfg.rhobar_m * cfg.box ** 3 / M_tot
    with np.errstate(divide='ignore', invalid='ignore'):
        q = mass_halo_assigned * unit_total / h_mass
    print(f"  assigned mass / M200b per halo: median {np.nanmedian(q):.4f}, "
          f"16-84% [{np.nanpercentile(q, 16):.4f}, "
          f"{np.nanpercentile(q, 84):.4f}]")

    shared = dict(
        version=CACHE_VERSION,
        k_bins=k_bins,
        logM_edges=logM_edges, logM_cen=data['logM_cen'],
        M_mean=data['M_mean'], counts=counts, f_i=f_i, f_part=f_part,
        M_tot_particles=M_tot, f_c=f_c_here, f_g=f_g_here,
        m2_bin_gas=m2_bin_gas, m2_tot_gas=m2_tot_gas,
        # Matter (dm + gas) second moments. Not used by the self-pair
        # subtraction, which is a gas-only affair; they are what sets the
        # noise a randomised run adds, so experiment_D can weigh its own
        # answer. Extra keys rather than a CACHE_VERSION bump, so nothing
        # already measured is invalidated by their arrival.
        m2_bin_mat=m2_bin_mat, m2_tot_mat=m2_tot_mat,
        # Only a randomised run carries these, and only a randomised run's
        # self-pair term needs them.
        **({} if m2_shell is None else
           dict(m2_shell_gas=m2_shell, m2r_shell_gas=m2r_shell,
                selfpair_nx=spherise.SELFPAIR_NX)),
        mass_bin_dm=mass_bin_species['dm'], mass_bin_gas=mass_bin_species['gas'],
        assigned_over_m200b_median=float(np.nanmedian(q)),
        randomise=str(randomise or ''),
        box=cfg.box, kbins_per_decade=cfg.kbins_per_decade,
        kbin_wmin_kf=cfg.kbin_wmin_kf,
        nbins=nbins, logm_min=cfg.logm_min, logm_max=cfg.logm_max,
        redshift=cfg.z, feedback=cfg.feedback, mass_def=cfg.mass_def,
        centrals_only=cfg.centrals_only, aperture=float(aperture),
    )

    # --- spectra, one grid at a time ------------------------------------------
    out_2d = out_3d = None
    if not np.any(mass_bin > 0):
        raise SystemExit("No occupied bin carried any assigned mass; nothing "
                         "was measured. Check the catalogue and the mass range.")

    if want_2d:
        print("\nPer-bin cross spectra R*_i (projected) ...")
        R_star, U_star = _bin_spectra(
            dens_bin, M_tot / ngrid ** 2, gas_fft,
            lambda d: compute_2d_fft(d, ngrid), _binned, mass_bin > 0,
            k_center.size, m_fft=m_fft, dens_out=dens_out_2d)
        del dens_bin, m_fft, dens_out_2d
        gc.collect()
        _closure('2d', R_star, U_star, data, suffix='_2d',
                 randomise=randomise)
        out_2d = dict(shared, k_center=k_center, nmodes=np.asarray(nmodes),
                      ngrid=ngrid, R_star_gas=R_star, U_star_gas=U_star,
                      P_shot_gg=P_shot_2d)

    if ngrid3:
        print(f"\nPer-bin cross spectra R*_i ({ngrid3}^3) ...")
        gas3 = (np.asarray(dens_all_3d['gas'], dtype=np.float64)
                / (mass_tot_species['gas'] / ngrid3 ** 3) - 1.0)
        m_fft3 = None
        if randomise is None:
            dm3 = (np.asarray(dens_all_3d['dm'], dtype=np.float64)
                   / (mass_tot_species['dm'] / ngrid3 ** 3) - 1.0)
        del dens_all_3d
        gc.collect()
        gas_fft3 = compute_3d_fft(gas3, ngrid3)
        del gas3
        gc.collect()
        if randomise is None:
            dm_fft3 = compute_3d_fft(dm3, ngrid3)
            del dm3
            gc.collect()
            m_fft3 = f_c * dm_fft3 + f_g * gas_fft3
            del dm_fft3
            gc.collect()

        R_star3, U_star3 = _bin_spectra(
            dens_bin_3d, M_tot / ngrid3 ** 3, gas_fft3,
            lambda d: compute_3d_fft(d, ngrid3),
            lambda prod: binned_spectrum_3d(cfg, prod, binner3),
            mass_bin > 0, binner3.k_center.size,
            m_fft=m_fft3, dens_out=dens_out_3d)
        del dens_bin_3d, m_fft3, gas_fft3, dens_out_3d
        gc.collect()
        _closure('3d', R_star3, U_star3, data, suffix='_3d',
                 randomise=randomise)
        out_3d = dict(shared, k_center=binner3.k_center, ngrid=ngrid3,
                      ngrid_3d=ngrid3, k_Nyquist_3d=binner3.k_Nyquist,
                      nmodes=binner3.nmodes,
                      nmodes_eff=binner3.nmodes_eff, k_eff=binner3.k_eff,
                      R_star_gas=R_star3, U_star_gas=U_star3,
                      P_shot_gg=P_shot_3d)

    return out_2d, out_3d


# Cache
def cache_randomise(cache):
    """Which randomisation, if any, a loaded cache was measured with.

    Read off the cache rather than passed in, so that a file decides for
    itself whether it wants a self-pair subtraction and no caller can put a
    randomised cache on the production footing by forgetting an argument.
    Caches written before experiment D existed carry no key and are
    production by construction.
    """
    return spherise.check_mode(str(cache.get('randomise', '')))


def _read_cache(path):
    """Load a cache and put its spectra on the corrected footing.

    A randomised cache is corrected too, but not in the same places: a moved
    gas particle no longer coincides with its own copy in the gas field, so
    R*_i has no delta-function self-pair left (what replaces it -- that
    particle's new position correlating with its true one -- is part of the
    spherised signal), while the unassigned particles never moved and U* keeps
    the term it always had. rstar_self_pair_terms is where that distinction
    lives, so both files come off disk meaning the same thing.
    """
    with np.load(path, allow_pickle=False) as f:
        cache = {key: f[key] for key in f.files}
    terms = rstar_self_pair_terms(
        cache, randomised=cache_randomise(cache) is not None)
    return subtract_self_pairs(cache, terms=terms, report=False)


def load_or_measure_rstar_ustar(cfg, data, recompute=False, chunk_particles=5e7,
                    allow_compute=True, aperture=1.0, randomise=None):
    """Read the cache, or measure and write it. None if absent and not allowed.

    With cfg.ngrid_3d set there are two caches, one per grid, each corrected
    with its own P^shot_gg; the dict handed back is the 2-D one with k <
    cfg.k_split replaced by the cubic-grid measurement. Whichever of the two
    files is missing is the only thing measured, but either way that costs a
    full particle pass, so both are written when both are absent.

    `randomise` selects a sibling cache measured with the halo interiors
    randomised (experiment D); None is the production measurement.
    """
    randomise = spherise.check_mode(randomise)
    ngrid3 = cfg.ngrid_3d
    path = cache_path(cfg, aperture=aperture, randomise=randomise)
    path3 = (cache_path(cfg, aperture=aperture, ngrid3=ngrid3,
                        randomise=randomise) if ngrid3 else None)

    want_2d = recompute or not path.exists()
    want_3d = bool(ngrid3) and (recompute or not path3.exists())

    if want_2d or want_3d:
        if not allow_compute:
            return None
        missing = " and ".join(str(p) for p, w in
                               ((path, want_2d), (path3, want_3d)) if w)
        print(f"No R*/U* cache at\n  {missing}\nMeasuring (one particle pass "
              f"per species) ...")
        out_2d, out_3d = compute_R_star_U_star(
            cfg, data, chunk_particles=chunk_particles, aperture=aperture,
            want_2d=want_2d, ngrid3=(ngrid3 if want_3d else None),
            randomise=randomise)
        for target, out in ((path, out_2d), (path3, out_3d)):
            if out is None:
                continue
            ensure_parents(target)
            np.savez_compressed(target, **out)
            print(f"\nR*/U* cache saved:\n  {target}")

    print(f"R*/U* cache:\n  {path}")
    cache = _read_cache(path)
    if not ngrid3:
        return cache
    print(f"R*/U* 3-D cache:\n  {path3}")
    return stitch_on_k(cache, _read_cache(path3), cfg.k_split,
                       label='R*/U*')


# Read-time assembly
def measured_partition(cfg, cache, data, mr=None):
    """The measured partition P_matter_gas = R* + U* (+ V*).

    R*   bins in [M_r, M_max] -- what the reconstruction claims to model
    U*   matter outside every sphere, plus the bins below M_r
    V*   bins above M_max, which nothing models
    """
    R_i = np.array(cache['R_star_gas'], dtype=float)
    U = np.array(cache['U_star_gas'], dtype=float)

    occ = np.asarray(data['counts']) > 0
    if mr is None:
        resolved = occ
        hidden = above = np.zeros_like(occ)
    else:
        resolved, hidden, above = mr.resolved, mr.hidden, mr.above
    f_part = np.asarray(cache['f_part'], dtype=float)

    R_star = R_i[resolved].sum(axis=0)
    U_star = U + R_i[hidden].sum(axis=0)
    V_star = R_i[above].sum(axis=0) if above.any() else None
    return dict(R_star_i=np.where(resolved[:, None], R_i, 0.0),
                R_star=R_star, U_star=U_star, V_star=V_star,
                total=R_star + U_star if V_star is None
                      else R_star + U_star + V_star,
                f_part=np.where(resolved, f_part, 0.0),
                f_out=1.0 - float(f_part[resolved | above].sum()))


def load_measured_partition(cfg, data, recompute=False, allow_compute=True,
                            aperture=1.0, mr=None, randomise=None):
    """What plotting callers want: the partition dict, or None."""
    path = cache_path(cfg, aperture=aperture, randomise=randomise)
    cache = load_or_measure_rstar_ustar(cfg, data, recompute=recompute,
                            allow_compute=allow_compute, aperture=aperture,
                            randomise=randomise)
    if cache is None:
        return None, path
    k_c = np.asarray(cache['k_center'], dtype=float)
    k_b = np.asarray(data['k_center'], dtype=float)
    if k_c.shape != k_b.shape or not np.allclose(k_c, k_b, rtol=1e-6):
        print(f"[R*/U*] cache k grid does not match the bundle's "
              f"({k_c.size} vs {k_b.size} bins); ignoring it")
        return None, path
    return measured_partition(cfg, cache, data, mr=mr), path


# Standalone entry point
def main():
    from Pmx_reconstruction.pmxlib.bundle import bundle_path, load_or_measure
    from Pmx_reconstruction.pmxlib.config import PmxConfig, base_parser

    ap = base_parser("Measure R*_i, U* and the self-pair terms; cache them.",
                     validate=False)
    ap.add_argument('--chunk-particles', type=float, default=5e7,
                    help="expected member particles per membership chunk")
    ap.add_argument('--aperture', type=float, default=1.0,
                    help="membership radius in units of r200m. Values above 1 "
                         "are what experiment_B.py sweeps.")
    ap.add_argument('--randomise', default=None,
                    choices=list(spherise.MODES),
                    help="measure the SIBLING cache in which every labelled "
                         "particle is moved inside its own host at fixed "
                         "radius: 'shuffle' spherises each halo, 'rotate' "
                         "turns it rigidly. Both have the same expectation; "
                         "'shuffle' is the low-noise one. experiment_D "
                         "differences the result against the production cache")
    args = ap.parse_args()
    cfg = PmxConfig.from_args(args)

    bpath = bundle_path(cfg, cfg.nbins, cfg.logm_min, cfg.logm_max, cfg.grid)
    if not bpath.exists():
        raise SystemExit(f"Spectra bundle missing:\n  {bpath}\n"
                         "Build it by running experiment_A or experiment_B.")
    # Through load_or_measure rather than straight off disk: with --ngrid-3d
    # the R*/U* pass needs the bundle's 3-D self-pair spectrum.
    data = load_or_measure(cfg, cfg.nbins, cfg.logm_min, cfg.logm_max,
                           recompute_3d=args.recompute_3d)

    cache = load_or_measure_rstar_ustar(cfg, data, recompute=args.recompute,
                            chunk_particles=args.chunk_particles,
                            aperture=args.aperture,
                            randomise=args.randomise)
    part = measured_partition(cfg, cache, data)
    frac = np.nanmedian(part['U_star'] / part['total'])
    print(f"  median U*/(R*+U*) = {frac:.4f}")
    if args.randomise:
        print(f"\nThat was the '{args.randomise}' cache. Score it against the "
              f"production one with\n      python -m "
              f"Pmx_reconstruction.experiment_D --randomise {args.randomise}")


if __name__ == '__main__':
    main()