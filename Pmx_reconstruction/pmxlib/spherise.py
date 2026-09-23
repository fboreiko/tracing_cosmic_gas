#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Randomising a halo's internal structure while holding its radial profile.

This is the geometry behind experiment D. It moves every particle that carries
a halo label to a new position inside the SAME host, chosen so that the
particle's distance from the halo centre is preserved exactly. Nothing else in
the box is touched: unlabelled particles stay where they are, halo centres stay
where they are, and the gas field the result is crossed against is the true one.

WHY THIS IS THE TEST
    R writes bin i's contribution as a single radial profile times the bin's
    measured halo-gas cross spectrum,

        R_i = f_i u_m(k|M_i) P_halo_gas(k|M_i),

    while the exact contribution of the same particles is

        R*_i = (1/M_tot) sum_{j in i} M_j^a u_j(k) c_j(k),

    with u_j the halo's OWN (non-spherical, clumpy) profile and c_j its own
    cross with the gas. Writing u_bar_m = sum_j M_j^a u_j / sum_j M_j^a and
    P_halo_gas = <c>_i, the two differ by exactly

        R*_i - R_i = (N_i / M_tot) Cov_{j in i}( M_j^a u_j(k), c_j(k) ),      (*)

    a covariance ACROSS the halos of the bin between where a halo's mass sits
    and how that halo correlates with the gas. Nothing in (*) is an
    approximation: it holds for the realised fields, bin by bin, at each
    k VECTOR.

    The pipeline's R is built from two separately shell-averaged scalars,
    u_bar_m(k) and P_halo_gas(k), so on binned spectra the same difference is
    a covariance over the halos AND over the directions in the k shell. The
    two averages agree on the mean -- the shell average of u_j(k) over k-hat
    is that halo's own spherised profile, sum_p m_p j0(k r_p) / M_j^a, which
    is exactly what u_bar stacks -- so no term is lost. What the extra
    direction average adds is the alignment of a halo's shape with the field
    around it, which is why the randomisation below removes that too and why
    "intra-halo" here means clumps and asphericity together.

    Randomising each halo's internal configuration independently of everything
    else replaces u_j by something whose expectation is the halo's own
    SPHERISED profile u_bar_j, and leaves c_j untouched. So the measurement
    made on the randomised field isolates the part of (*) that survives
    spherisation -- the halo-to-halo piece, driven by within-bin mass scatter
    and assembly bias -- and the difference between the two measurements,

        C(k) = R*(true) - R*(randomised),

    is the intra-halo piece: the correlation between a halo's clumps and the
    gas sitting in those same clumps. That is the term the reconstruction has
    no representation for, and C(k) is it, measured rather than modelled.

THE TWO MODES, AND WHY THEY HAVE THE SAME EXPECTATION
    'shuffle'   each particle is moved to a random direction at its own radius.
                The halo is spherised: all substructure, triaxiality and
                alignment go at once.
    'rotate'    the whole halo is rigidly rotated by one random rotation.
                Every clump survives intact; only the halo's orientation is
                randomised.

    For a fixed k vector, E[exp(-i k.r n)] over uniform n is j0(k r), and a
    Haar-uniform rotation gives the same: E[exp(-i k.R r)] = j0(k r). Both
    modes therefore have the SAME expectation, u_bar_j, and 'rotate' is not an
    independent physical control -- it cannot preserve the clump-gas alignment,
    because the gas is not rotated with the matter. What it is is an
    independent implementation of the same expectation with a very different
    noise model: 'shuffle' draws one direction per particle, so its error on
    u_j averages down as 1/sqrt(N_particles), while 'rotate' draws one per
    halo and averages down only as 1/sqrt(N_halos). 'shuffle' is therefore the
    estimator to use; 'rotate' is worth a run only as a cross-check, and is
    expected to agree within its (larger) noise.

    The control that WOULD preserve the clump-gas alignment has to rotate each
    halo's gas by the same rotation and re-measure P_halo_gas against the
    rotated gas field. That is a second measurement, not a mode here.

NOISE
    The random directions are uncorrelated with the gas field, so the estimator
    is unbiased; what they add is variance, at the level of the member
    particles' own shot power crossed with the gas. experiment_D prints it
    next to the signal, the same way self_pairs reports its subtraction, and
    it is the thing to check before reading the high-k end of C(k).

SELF-PAIRS, WHICH DO NOT GO AWAY
    delta_m is painted from dark matter AND gas, so a member gas particle sits
    in both fields and the two copies coincide. Randomising moves only the copy
    in delta_m, and only to the far side of the same orbit, so the coincident
    self-pair is SMEARED rather than removed: what survives in R*(randomised)
    is m^2 j0(k r)^2 where m^2 came off R*(true). Left alone it biases
    R*(randomised) high by nearly the whole self-pair on large scales, which
    subtracts straight out of C(k) and drives it negative wherever the gas
    spectrum is shot-dominated -- a "spherising increased the correlation"
    that cannot happen. selfpair_shells below histograms the member gas m^2
    against radius so self_pairs can build that term; _selfpair_test is the
    check that it works.

USAGE
    from Pmx_reconstruction.pmxlib import spherise
    quat = spherise.halo_rotations(n_halo)          # 'rotate' only
    info = spherise.randomise_in_place(pos, label, h_pos, box, 'shuffle')

    python -m Pmx_reconstruction.pmxlib.spherise     # self-test, no data needed
"""
import numpy as np

__all__ = ['MODES', 'ROTATION_SEED', 'SHUFFLE_SEED', 'CHUNK', 'SELFPAIR_NX',
           'halo_rotations', 'randomise_in_place', 'check_mode',
           'selfpair_shells', 'position_tolerance']

MODES = ('shuffle', 'rotate')

# Fixed so a randomised cache can be reproduced from the command line alone.
# The two differ so that a run of each is not secretly sharing draws.
SHUFFLE_SEED = 20260923
ROTATION_SEED = 20260924

# Particles per chunk. Sized by the transient it bounds: 'rotate' holds about
# four (chunk, 3) float64 temporaries at once, so 2e7 is ~2 GB. Rotating by
# quaternion rather than by a stacked rotation matrix is what keeps it to
# that: scipy's Rotation.apply would materialise (chunk, 3, 3) instead, twice
# the memory for the same answer, and pull scipy.spatial in besides.
CHUNK = 20_000_000


def check_mode(mode):
    """Normalise and validate a mode name. None/'' means 'do not randomise'."""
    if mode is None or mode == '':
        return None
    mode = str(mode)
    if mode not in MODES:
        raise ValueError(f"unknown randomisation mode {mode!r}; "
                         f"expected one of {', '.join(MODES)}")
    return mode


def halo_rotations(n_halo, seed=ROTATION_SEED):
    """Haar-uniform rotations, one per halo, as quaternions (n_halo, 4).

    A normalised 4-vector of independent Gaussians is uniform on the
    3-sphere, and the unit quaternions ARE the 3-sphere, so this is Haar
    measure on SO(3) with no rejection step and no scipy dependency at
    generation time.

    Quaternions rather than matrices because these live in memory for the
    whole pass while particles reach into them at random: 4 floats per halo
    against 9, which at a few times 10^7 centrals is the difference between
    half a gigabyte and one and a half.

    Generated ONCE and shared by both species, so that a halo's dark matter
    and its gas turn together -- otherwise the halo's matter is not rigidly
    rotated, it is two independently rotated pieces.
    """
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((int(n_halo), 4))
    nrm = np.sqrt(np.einsum('ij,ij->i', q, q))
    bad = nrm <= 0.0                       # measure zero; keeps it total
    if np.any(bad):
        q[bad] = np.array([0.0, 0.0, 0.0, 1.0])
        nrm[bad] = 1.0
    return (q / nrm[:, None]).astype(np.float32)


SELFPAIR_NX = 64


def selfpair_shells(pos, label, h_pos, h_r, h_bin, weights, box, nbins,
                    nx=SELFPAIR_NX, chunk=CHUNK):
    """sum(w) and sum(w r) in radial shells, per mass bin, for the members.

    This exists for one reason, and it is a correction that is easy to miss.
    A gas particle inside a halo appears TWICE in the measurement: once in
    delta_m^(i), because it is matter, and once in delta_gas. In the
    production run those two copies coincide, and what they contribute is the
    delta-function self-pair that self_pairs removes. Randomising moves only
    the first copy, so the pair does not vanish -- it is smeared to a
    separation of order r, and contributes

        m_q^2 j0(k r_q)^2   instead of   m_q^2 .

    It is the same discreteness artefact either way and has to come off either
    way, but the amount depends on where in the halo the particle sits, so it
    cannot be read off sum m^2 alone. Histogramming m^2 against radius here,
    and mass-weighting each shell's radius as u_bar does, is what lets
    self_pairs build the k dependence later without a second particle pass.

    Called with `weights` = m^2 and gas particles only; dark matter shares no
    particle with the gas field and has no such term.
    """
    W = np.zeros((nbins, nx))
    S = np.zeros((nbins, nx))
    n_p = label.size
    for a in range(0, n_p, int(chunk)):
        b = min(a + int(chunk), n_p)
        lab = label[a:b]
        sel = lab >= 0
        if not np.any(sel):
            continue
        lab = lab[sel]
        d = pos[a:b][sel] - h_pos[lab]
        d -= box * np.round(d / box)
        r = np.sqrt(np.einsum('ij,ij->i', d, d))
        R = h_r[lab]
        ix = np.minimum((r * (nx / R)).astype(np.int64), nx - 1)
        flat = h_bin[lab] * nx + ix
        w = np.asarray(weights[a:b], dtype=np.float64)[sel]
        W += np.bincount(flat, weights=w, minlength=nbins * nx).reshape(nbins, nx)
        S += np.bincount(flat, weights=w * r,
                         minlength=nbins * nx).reshape(nbins, nx)
        del lab, sel, d, r, R, ix, flat, w
    return W, S


def _rotate_by_quat(q, v):
    """Rotate each row of v by the unit quaternion in the same row of q.

    q is scalar-LAST, (n, 4) = (x, y, z, w), as halo_rotations writes it.
    The identity is

        v' = v + 2 u x (u x v + w v),      u = (x, y, z),

    which is the same rotation a 3x3 matrix would apply without ever building
    one: four (n, 3) temporaries instead of an (n, 3, 3) stack.
    """
    u = q[:, :3]
    w = q[:, 3:4]
    t = np.cross(u, v)
    t += w * v
    return v + 2.0 * np.cross(u, t)


def _random_directions(n, rng):
    """n unit vectors, uniform on the sphere, (n, 3) float64."""
    g = rng.standard_normal((n, 3))
    nrm = np.sqrt(np.einsum('ij,ij->i', g, g))
    bad = nrm <= 0.0
    if np.any(bad):
        g[bad] = np.array([0.0, 0.0, 1.0])
        nrm[bad] = 1.0
    return g / nrm[:, None]


def randomise_in_place(pos, label, h_pos, box, mode, seed=None, quat=None,
                       chunk=CHUNK, report=True):
    """Move every labelled particle inside its host. Modifies `pos` in place.

    Parameters
    ----------
    pos   : (n_p, 3) float32   particle positions in [0, box), AS THE PIPELINE
                               stores them. Overwritten for labelled particles
                               only; unlabelled ones are left untouched, which
                               is what makes U* identical to the production
                               run by construction rather than by measurement.
    label : (n_p,) int         host index, or < 0 for no host. From
                               rstar_ustar.assign_particles, so it indexes the
                               DESCENDING-mass halo arrays.
    h_pos : (n_halo, 3) float  halo centres in [0, box).
    box   : float              periodic box side [cMpc/h].
    mode  : 'shuffle' | 'rotate'
    seed  : int                'shuffle' only; SHUFFLE_SEED if None.
    quat  : (n_halo, 4)        'rotate' only, from halo_rotations. Required,
                               rather than made here, because both species
                               must turn each halo the same way.

    Returns
    -------
    dict with n_moved, r_max, dr_max (the largest change in a particle's
    radius, which must be at the float32 position quantum) and dr_rel_max.

    The radius is recomputed FROM THE WRITTEN float32 positions rather than
    from the float64 intermediates, so dr_max tests the minimum image, the
    wrap, the write-back and the dtype round trip in one number, and the
    caller can assert on it the way u_bar asserts on its aperture overshoot.
    """
    mode = check_mode(mode)
    if mode is None:
        raise ValueError("randomise_in_place needs a mode; None means the "
                         "caller should not have called it")
    if mode == 'rotate':
        if quat is None:
            raise ValueError("mode 'rotate' needs `quat` from halo_rotations: "
                             "both species must rotate each halo identically, "
                             "so the draw cannot be made here")
        quat = np.asarray(quat)
        if quat.shape != (h_pos.shape[0], 4):
            raise ValueError(f"quat has shape {quat.shape}, expected "
                             f"{(h_pos.shape[0], 4)}")

    rng = np.random.default_rng(SHUFFLE_SEED if seed is None else int(seed))
    box_f32 = np.float32(box)
    eps = float(box) * 2.0 ** -23          # the float32 position quantum
    n_p = label.size
    n_moved = 0
    r_max = dr_max = dr_rel_max = 0.0

    for a in range(0, n_p, int(chunk)):
        b = min(a + int(chunk), n_p)
        view = pos[a:b]                     # a VIEW: assignment writes through
        lab = label[a:b]
        sel = lab >= 0
        if not np.any(sel):
            continue
        lab = lab[sel]
        centre = h_pos[lab]                 # (m, 3) float64

        # float32 pos minus float64 centre promotes to float64, so the
        # subtraction is exact and the only error left is the float32 storage
        # of pos -- the same 4e-5 cMpc/h on a 681 box that u_bar lives with.
        d = view[sel] - centre
        d -= box * np.round(d / box)        # minimum image
        r = np.sqrt(np.einsum('ij,ij->i', d, d))
        r_max = max(r_max, float(r.max()))

        if mode == 'shuffle':
            d = r[:, None] * _random_directions(r.size, rng)
        else:
            d = _rotate_by_quat(quat[lab].astype(np.float64), d)

        new = np.mod(centre + d, box)
        new32 = new.astype(np.float32)
        # np.mod can land exactly on box, and the float32 cast can round up to
        # it from just below; tsc wants [0, L) either way.
        new32[new32 >= box_f32] -= box_f32
        view[sel] = new32
        n_moved += int(r.size)

        # Read back what was actually stored and re-derive the radius.
        d2 = view[sel] - centre
        d2 -= box * np.round(d2 / box)
        r2 = np.sqrt(np.einsum('ij,ij->i', d2, d2))
        dr = np.abs(r2 - r)
        dr_max = max(dr_max, float(dr.max()))
        # Relative only where a radius is long enough for a ratio to mean
        # something. A halo centre IS its most-bound particle, so members sit
        # at r = 0 exactly; against those the fixed float32 slop is an
        # arbitrarily large "relative error" and says nothing. dr_max is the
        # number that gets asserted on for this reason.
        big = r > 100.0 * eps
        if np.any(big):
            dr_rel_max = max(dr_rel_max, float(np.max(dr[big] / r[big])))
        del view, lab, sel, centre, d, d2, r, r2, dr

    info = dict(mode=mode, n_moved=n_moved, r_max=r_max, dr_max=dr_max,
                dr_rel_max=dr_rel_max)
    if report:
        print(f"    [{mode}] {n_moved:.3e} labelled particles moved; "
              f"max radius {r_max:.4f} cMpc/h, max |dr| {dr_max:.2e} cMpc/h "
              f"({dr_rel_max:.1e} relative, over r > 100 float32 quanta)")
    return info


def position_tolerance(box, ulps=4.0):
    """How far a radius may move and still be float32 storage, not a bug.

    Same convention as u_bar.POS_EPS_ULPS: the quantum is box * 2^-23 and a
    handful of them covers the round trip through a subtraction and a wrap.
    """
    return float(ulps) * float(box) * 2.0 ** -23


# Self-test
def _self_test():
    """Synthetic halos, no simulation data. Checks the four things that would
    silently ruin the measurement: radii, wrapping, host identity and the
    rigidity of 'rotate'."""
    rng = np.random.default_rng(7)
    box = 10.0
    n_halo, per_halo = 50, 200
    h_pos = rng.uniform(0.0, box, size=(n_halo, 3))
    # Put one halo hard on the origin so every chunk has particles whose
    # minimum image wraps; that is the case a naive subtraction gets wrong.
    h_pos[0] = np.array([0.0, 0.0, 0.0])

    label = np.repeat(np.arange(n_halo), per_halo)
    label = np.concatenate([label, np.full(500, -1)])
    offs = rng.normal(scale=0.3, size=(label.size, 3))
    pos = np.mod(h_pos[np.maximum(label, 0)] + offs, box).astype(np.float32)
    pos[pos >= box] -= np.float32(box)
    free = label < 0
    pos_free_before = pos[free].copy()

    def radii(p):
        d = p[label >= 0] - h_pos[label[label >= 0]]
        d -= box * np.round(d / box)
        return np.sqrt(np.einsum('ij,ij->i', d, d))

    tol = position_tolerance(box)
    quat = halo_rotations(n_halo)

    for mode in MODES:
        p = pos.copy()
        r0 = radii(p)
        info = randomise_in_place(p, label, h_pos, box, mode, quat=quat,
                                  chunk=3000, report=False)
        r1 = radii(p)
        assert info['n_moved'] == n_halo * per_halo, info
        assert np.all(p >= 0.0) and np.all(p < box), "left the box"
        assert np.allclose(p[free], pos_free_before), \
            "an unlabelled particle moved: U* would no longer be the " \
            "production one"
        assert np.max(np.abs(r1 - r0)) <= tol, \
            f"{mode}: radius changed by {np.max(np.abs(r1 - r0)):.2e} > {tol:.2e}"
        assert np.max(np.abs(p - pos)) > 0.0, f"{mode}: nothing moved"
        print(f"  [{mode}] radii preserved to "
              f"{np.max(np.abs(r1 - r0)):.2e} cMpc/h (tolerance {tol:.2e}); "
              f"unlabelled particles untouched")

    # 'rotate' is rigid: every pairwise distance inside a halo survives.
    p = pos.copy()
    randomise_in_place(p, label, h_pos, box, 'rotate', quat=quat, chunk=3000,
                       report=False)
    for j in (0, 17):
        m = np.flatnonzero(label == j)[:40]
        def _sep(q):
            d = q[m][:, None, :] - q[m][None, :, :]
            d -= box * np.round(d / box)
            return np.sqrt(np.einsum('ijk,ijk->ij', d, d))
        err = np.max(np.abs(_sep(p) - _sep(pos)))
        assert err <= 10.0 * tol, f"rotate is not rigid in halo {j}: {err:.2e}"
    print(f"  [rotate] intra-halo separations preserved to {err:.2e} cMpc/h")

    # 'shuffle' is not rigid, and its directions are isotropic. The mean of
    # n unit vectors has |mean| ~ 1/sqrt(n); 5/sqrt(n) is a loose gate that
    # still catches a hemisphere bug.
    p = pos.copy()
    randomise_in_place(p, label, h_pos, box, 'shuffle', chunk=3000,
                       report=False)
    d = p[label >= 0] - h_pos[label[label >= 0]]
    d -= box * np.round(d / box)
    n_hat = d / np.linalg.norm(d, axis=1)[:, None]
    mean = np.linalg.norm(n_hat.mean(axis=0))
    gate = 5.0 / np.sqrt(n_hat.shape[0])
    assert mean < gate, f"directions are not isotropic: |<n>| = {mean:.3f}"
    print(f"  [shuffle] |<n_hat>| = {mean:.4f} over {n_hat.shape[0]} "
          f"particles (gate {gate:.4f})")
    print("\nspherise: all checks passed")


def _selfpair_test(verbose=True):
    """The smeared self-pair: does subtracting it restore a smooth universe?

    The check _science_test does NOT make, because it gives matter and gas
    disjoint particle sets. The pipeline does not: delta_m is painted from dm
    AND gas, so a member gas particle is in both fields, and shuffling moves
    only its copy in delta_m. The coincident pair is then smeared rather than
    removed, and m^2 j0(k r)^2 survives in R*(randomised) where m^2 was taken
    off R*(true).

    Uncorrected that shows up as a NEGATIVE intra-halo term -- an apparent
    "spherising increases the correlation with the gas", which is impossible
    in a universe built with nothing inside a halo aligned. So: build exactly
    that universe, and require the artefact to appear without the correction
    and to go away with it. Two things are on trial, the j0^2 form and
    selfpair_shells' histogram of it.
    """
    rng = np.random.default_rng(5)
    L, r_halo = 100.0, 2.0
    n_h, n_dm, n_g = 400, 60, 4      # few gas particles: shot-dominated, as
                                     # a ~100x subsampled snapshot is

    def sphere(n):
        x = rng.standard_normal((n, 3))
        return x / np.linalg.norm(x, axis=1)[:, None]

    h_pos = rng.uniform(0, L, size=(n_h, 3))
    h_r = np.full(n_h, r_halo)
    h_bin = np.zeros(n_h, dtype=np.int64)
    dm = np.empty((n_h * n_dm, 3))
    gas = np.empty((n_h * n_g, 3))
    for j in range(n_h):
        dm[j * n_dm:(j + 1) * n_dm] = (h_pos[j] + (r_halo * rng.power(2., n_dm))
                                       [:, None] * sphere(n_dm))
        gas[j * n_g:(j + 1) * n_g] = (h_pos[j] + (r_halo * rng.power(2., n_g))
                                      [:, None] * sphere(n_g))
    dm, gas = np.mod(dm, L), np.mod(gas, L)
    mat = np.concatenate([dm, gas]).astype(np.float32)
    lab_m = np.concatenate([np.repeat(np.arange(n_h), n_dm),
                            np.repeat(np.arange(n_h), n_g)])
    lab_g = np.repeat(np.arange(n_h), n_g)
    n_m, n_gg = mat.shape[0], gas.shape[0]

    shuf = mat.copy()
    randomise_in_place(shuf, lab_m, h_pos, L, 'shuffle', report=False)

    def radii(p, lab):
        d = np.asarray(p, float) - h_pos[lab]
        d -= L * np.round(d / L)
        return np.sqrt(np.einsum('ij,ij->i', d, d))

    r_gas = radii(gas, lab_g)
    W, S = selfpair_shells(gas.astype(np.float32), lab_g, h_pos, h_r, h_bin,
                           np.ones(n_gg), L, 1)
    with np.errstate(invalid='ignore', divide='ignore'):
        r_shell = np.where(W[0] > 0, S[0] / np.where(W[0] > 0, W[0], 1.0), 0.0)

    def shell_k(n_int, cap=500):
        a = np.arange(-n_int, n_int + 1)
        g = np.array(np.meshgrid(a, a, a, indexing='ij')).reshape(3, -1).T
        g = g[np.abs(np.linalg.norm(g, axis=1) - n_int) < 0.5]
        if g.shape[0] > cap:
            g = g[np.random.default_rng(2).choice(g.shape[0], cap, False)]
        return g * (2 * np.pi / L)

    def ft(kv, p, chunk=256):
        p = np.asarray(p, float)
        out = np.empty(kv.shape[0], complex)
        for a in range(0, kv.shape[0], chunk):
            b = min(a + chunk, kv.shape[0])
            out[a:b] = np.exp(-1j * (kv[a:b] @ p.T)).sum(axis=1)
        return out

    def cross(kv, x, y):
        return float(np.mean((ft(kv, x) / len(x)
                              * np.conj(ft(kv, y) / len(y))).real))

    if verbose:
        print("\n  === shared matter/gas particles, smooth halos " + "=" * 12)
        print("       k   intra/R* uncorrected    corrected   shells vs direct")
    uncs, worst_cor, worst_unc, worst_hist = [], 0.0, 0.0, 0.0
    for n_int in (4, 9, 16, 25):
        kv, kmag = shell_k(n_int), n_int * 2 * np.pi / L
        # The histogrammed j0^2 sum against the direct one, particle by
        # particle: this is what self_pairs.smeared_share builds.
        j0sq_hist = float(W[0] @ np.sinc(r_shell * kmag / np.pi) ** 2 / n_gg)
        j0sq_direct = float(np.mean(np.sinc(kmag * r_gas / np.pi) ** 2))
        worst_hist = max(worst_hist, abs(j0sq_hist / j0sq_direct - 1.0))

        R_true = cross(kv, mat, gas) - 1.0 / n_m      # plain self-pair
        R_raw = cross(kv, shuf, gas)
        R_shuf = R_raw - j0sq_hist / n_m              # smeared self-pair
        unc, cor = (R_true - R_raw) / R_true, (R_true - R_shuf) / R_true
        uncs.append(unc)
        worst_unc, worst_cor = max(worst_unc, -unc), max(worst_cor, abs(cor))
        if verbose:
            print(f"    {kmag:5.2f}      {unc:+10.4f}   {cor:+10.4f}"
                  f"          {j0sq_hist / j0sq_direct - 1:+.2e}")

    assert worst_hist < 5e-3, \
        f"selfpair_shells is {worst_hist:.2e} off the direct sum"
    # Assert on the PATTERN, not a magnitude that depends on how shot-dominated
    # this particular synthetic gas field is.
    assert all(u < 0 for u in uncs), \
        "uncorrected should be negative at EVERY k: that is the artefact's sign"
    assert worst_unc > 3 * worst_cor, \
        f"the correction shrank the bias only {worst_unc / worst_cor:.1f}x"
    assert worst_cor < 0.02, f"corrected smooth case still shows {worst_cor:.3f}"
    print(f"  [selfpair] uncorrected: spurious intra-halo term, negative at "
          f"every k, down to {-worst_unc:+.3f} of R*")
    print(f"  [selfpair] corrected:   |intra/R*| <= {worst_cor:.3f}, "
          f"{worst_unc / worst_cor:.0f}x smaller, as a smooth universe must")


def _science_test(n_halo=400, n_part=80, verbose=True):
    """End-to-end: does spherising isolate the intra-halo term, and only it?

    A synthetic box with direct-sum spectra -- no grid, no TSC, no FFT, so
    nothing here can be fixed by a painting convention -- and the real
    randomise_in_place in the loop. Two universes:

      smooth  each halo is a spherical cloud whose matter and gas are drawn
              with INDEPENDENT directions: nothing inside a halo is aligned.
      clumpy  half the matter and half the gas sit in the same few clumps per
              halo, so the two are co-located clump by clump.

    What experiment D claims must then hold in both:

      smooth  R* already equals the radial-profile model, and spherising
              changes nothing. A method that reported a term here would be
              manufacturing one.
      clumpy  R* exceeds the model by a residual that GROWS with k, exactly as
              it does in FLAMINGO, and R*(shuffled) lands back on the model --
              so the difference accounts for the residual rather than merely
              correlating with it.

    'rotate' is run alongside and must agree with 'shuffle': same expectation,
    different draw.
    """
    rng = np.random.default_rng(3)
    L, r_halo = 100.0, 2.0
    n_clump, clump_frac, clump_size = 4, 0.5, 0.25

    def sphere(n):
        v = rng.standard_normal((n, 3))
        return v / np.linalg.norm(v, axis=1)[:, None]

    def build(clumpy):
        h_pos = rng.uniform(0, L, size=(n_halo, 3))
        m_pos = np.empty((n_halo * n_part, 3))
        g_pos = np.empty((n_halo * n_part, 3))
        label = np.repeat(np.arange(n_halo), n_part)
        n_c = int(clump_frac * n_part)
        for j in range(n_halo):
            dm = (r_halo * rng.power(2.0, n_part))[:, None] * sphere(n_part)
            dg = (r_halo * rng.power(2.0, n_part))[:, None] * sphere(n_part)
            if clumpy:
                cen = ((r_halo * rng.power(2.0, n_clump))[:, None]
                       * sphere(n_clump))
                dm[:n_c] = (cen[rng.integers(0, n_clump, n_c)]
                            + clump_size * rng.standard_normal((n_c, 3)))
                dg[:n_c] = (cen[rng.integers(0, n_clump, n_c)]
                            + clump_size * rng.standard_normal((n_c, 3)))
            m_pos[j * n_part:(j + 1) * n_part] = h_pos[j] + dm
            g_pos[j * n_part:(j + 1) * n_part] = h_pos[j] + dg
        return (h_pos, np.mod(m_pos, L).astype(np.float32),
                np.mod(g_pos, L), label)

    def shell(n_int, cap=600):
        a = np.arange(-n_int, n_int + 1)
        g = np.array(np.meshgrid(a, a, a, indexing='ij')).reshape(3, -1).T
        g = g[np.abs(np.linalg.norm(g, axis=1) - n_int) < 0.5]
        if g.shape[0] > cap:
            g = g[np.random.default_rng(11).choice(g.shape[0], cap, False)]
        return g * (2.0 * np.pi / L)

    def ft(kv, p, chunk=256):
        p = np.asarray(p, dtype=np.float64)
        out = np.empty(kv.shape[0], dtype=np.complex128)
        for a in range(0, kv.shape[0], chunk):
            b = min(a + chunk, kv.shape[0])
            out[a:b] = np.exp(-1j * (kv[a:b] @ p.T)).sum(axis=1)
        return out

    def cross(kv, x, y):
        return float(np.mean((ft(kv, x) / len(x)
                              * np.conj(ft(kv, y) / len(y))).real))

    n_ints = (2, 4, 6, 9, 13, 18, 25)
    for clumpy in (False, True):
        name = 'clumpy' if clumpy else 'smooth'
        h_pos, m_pos, g_pos, label = build(clumpy)
        shuf = m_pos.copy()
        randomise_in_place(shuf, label, h_pos, L, 'shuffle', report=False)
        rot = m_pos.copy()
        randomise_in_place(rot, label, h_pos, L, 'rotate',
                           quat=halo_rotations(n_halo), report=False)
        d = np.asarray(m_pos, float) - h_pos[label]
        d -= L * np.round(d / L)
        r_p = np.linalg.norm(d, axis=1)

        if verbose:
            print(f"\n  === {name} halos " + "=" * 40)
            print("     k    R*(true)   R*(shuf)    R_model   (R_m-R*)/R*"
                  "  intra/total")
        worst_smooth, deepest, shares = 0.0, 0.0, []
        for n_int in n_ints:
            kv, kmag = shell(n_int), n_int * 2.0 * np.pi / L
            R_star = cross(kv, m_pos, g_pos)
            R_shuf = cross(kv, shuf, g_pos)
            R_rot = cross(kv, rot, g_pos)
            P_he = cross(kv, h_pos, g_pos)
            R_mod = float(np.mean(np.sinc(kmag * r_p / np.pi))) * P_he
            total, intra = R_mod - R_star, R_star - R_shuf
            share = -intra / total if total != 0 else np.nan
            if verbose:
                print(f"   {kmag:5.2f} {R_star:11.3e}{R_shuf:11.3e}"
                      f"{R_mod:11.3e}   {total / R_star:+9.3f}   {share:+9.3f}")
            assert abs(R_rot / R_shuf - 1.0) < 0.05, \
                f"{name} k={kmag:.2f}: rotate and shuffle disagree by " \
                f"{abs(R_rot / R_shuf - 1.0):.3f}; they share an expectation"
            if clumpy:
                # Only score where there is a residual to account for.
                deepest = max(deepest, abs(total / R_star))
                if abs(total / R_star) > 0.02:
                    shares.append(share)
            else:
                worst_smooth = max(worst_smooth, abs(total / R_star))

        if not clumpy:
            assert worst_smooth < 0.02, \
                f"smooth halos show a {worst_smooth:.3f} residual; the model " \
                f"should already be exact when nothing inside a halo is aligned"
            print(f"  [smooth] |R_model/R* - 1| <= {worst_smooth:.4f} at every "
                  f"k, and spherising changes nothing")
        else:
            assert len(shares) >= 3, "the clumpy box grew no residual to test"
            lo, hi = float(np.min(shares)), float(np.max(shares))
            assert lo > 0.75, \
                f"spherising accounted for only {lo:.0%} of a residual that " \
                f"is intra-halo by construction"
            print(f"  [clumpy] the residual reaches {deepest:.0%} of R*, and "
                  f"spherising accounts for {lo:.0%}-{hi:.0%} of it at every "
                  f"k where it is above 2%")
    print("\nspherise: the science check passed")


if __name__ == '__main__':
    import sys
    _self_test()
    if '--science' in sys.argv:
        _science_test()
        _selfpair_test()
    else:
        print("\n(run with --science for the end-to-end check that "
              "spherising isolates the intra-halo term; ~1 min, no data)")
