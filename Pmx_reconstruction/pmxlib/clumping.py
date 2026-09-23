#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The intra-halo clumping correction to u_m, calibrated by experiment D.

Experiment D itself is NOT on this branch: pmxlib/spherise.py, experiment_D.py
and rstar_ustar's --randomise caches live on 3d-stitch-expD. What is here is
its result, which stands on its own -- nothing below imports any of them. Read
that branch, or re-derive the numbers from its experiment_D plot data, before
changing CLUMP_A, CLUMP_ALPHA or XI_MAX: they are a fit, not constants, and
the run they were fitted on is named in CALIBRATION.

R writes a halo's matter as one smooth radial profile, so the only thing it
can say about a halo's interior is how much mass lies at each radius. A real
halo keeps some of that mass in subhaloes, those subhaloes hold gas, and matter
and gas therefore sit together clump by clump and correlate more strongly than
any spherical distribution of the same radial profile does. R falls short by
exactly that, and no choice of u_m -- NFW, a refitted concentration, or the
measured stack -- can recover it, because all three are radial.

Experiment D measures the missing term directly, as

    C_i(k) = R*_i(true) - R*_i(halo interiors randomised at fixed radius),

and this module turns that measurement into a factor R can carry:

    u_m(k|M)  ->  u_m(k|M) * [1 + xi(k r200m(M))] .

Multiplying u_m rather than R_i is the same arithmetic -- R_i = f_i u_m P^he --
and puts the correction where it belongs conceptually: it is a statement about
the effective matter profile of a halo of mass M, so it should apply wherever
u_m appears, including inside U's S(k). It does essentially nothing there:
haloes below M_r have r200m <~ 0.12 cMpc/h, so k r200m < 1.2 out to Nyquist
and xi < 0.01.

WHY k r200m AND NOT k
    xi collapses on k r200m and does not collapse on k. Over the bins that
    carry the term (logM > 13) the fractional scatter across 1.5 decades of
    host mass is 0.26-0.33 against k r200m and 0.57-0.67 against k. That is
    what a self-similar subhalo population predicts: a subhalo mass function
    universal in m/M and a radial distribution universal in r/r200m leave a
    correction that can only depend on k r200m.

THE CALIBRATION, AND WHERE IT STOPS
    Fitted on strongest_AGN, z = 0.74, aperture x = 1, 30 bins over
    logM 11-15, 3-D/2-D spliced at k = 1 -- 682 (bin, k) points from the 19
    bins above logM 12.5, weighted by each bin's own R*_i:

        xi = 0.0067 (k r200m)^1.74 ,    0.2 < k r200m < 6

    The measured xi then SATURATES: the median over the logM > 13.2 bins is
    0.138 at k r200m = 5, 0.184 at 8 and 0.181 at 12, while the power law has
    reached 0.25 and 0.51. Past k r200m ~ 12 the per-bin values scatter from
    -0.1 to +1.1 and mean nothing -- those are differences of two spectra that
    are mostly self-pairs by then. So the law is capped at XI_MAX, the plateau
    the measurement actually reaches, and the cap is not cosmetic: the biggest
    haloes in a 681 cMpc/h box reach k r200m = 28 at the grid's Nyquist, where
    an uncapped 0.0067 x^1.74 asks for a 220% correction on evidence that does
    not exist.

WHAT IT BUYS, AND WHAT IT DOES NOT
    Summed back over the bins, the capped law reproduces the measured C(k) to
    0.75-0.95 over k = 1-9: it recovers most of the missing term, not all of
    it. Two known reasons, both quantified rather than guessed:

      - The fit is to xi = C_i/R*_i, the correction relative to the TRUTH,
        while what multiplies the model is C_i/R_i, and R_i is low by the very
        residual being corrected. That under-corrects by roughly xi/(1-xi) --
        10% at xi = 0.1, 25% at xi = 0.2.
      - The power law runs ~30% low around k r200m = 5 and ~35% high at 8; a
        smooth saturating form would fit better than a power law and a cap.

    Leaving the cap off (clump_max = inf) happens to reproduce the TOTAL C(k)
    better, at 0.93-1.04. That is not a reason to do it: it works by
    over-correcting the massive bins at large k r200m, where nothing was
    measured, and the two errors cancelling is a property of this box and this
    k range rather than of the physics.

    Calibrated at ONE aperture, ONE feedback model and ONE redshift. Rerunning
    experiment D at x = 2, or on fiducial, is what would say whether a, alpha
    and the cap travel.
"""
import numpy as np

from Pmx_reconstruction.pmxlib.nfw import r200m_of_M

__all__ = ['CLUMP_A', 'CLUMP_ALPHA', 'XI_MAX', 'CALIBRATION', 'APPLY_MODES',
           'xi', 'factor', 'describe']

# How xi becomes a multiplier on u_m. xi was measured as C_i / R*_i -- against
# the TRUTH, not against the model -- so inverting it is a division, not an
# addition:
#
#     R*_i = R_i + C_i = R_i + xi_i R*_i   =>   R*_i = R_i / (1 - xi_i)
#
# 'linear' is the (1 + xi) form, which is what C_i/R_i would call for and
# under-corrects by xi/(1-xi): 11% at xi = 0.1, 23% at xi = 0.19. Measured on
# the calibration run, 'self-consistent' recovers 0.84-0.99 of C(k) over
# k = 1-9 against 0.74-0.95 for 'linear', and closes R/R* - 1 at k = 7 from
# -10.0% to -2.2% against -3.3%. Kept as a switch because the two differ by
# more than the fit's own scatter only at the top of the mass range.
APPLY_MODES = ('self-consistent', 'linear')

# xi = CLUMP_A * (k r200m) ** CLUMP_ALPHA, capped at XI_MAX.
CLUMP_A = 0.0067
CLUMP_ALPHA = 1.74
XI_MAX = 0.19

# Where the numbers came from, so a run can print its own provenance.
CALIBRATION = ('experiment D (branch 3d-stitch-expD), strongest_AGN, '
               'z=0.74, x=1, nb30 logM 11-15, 3d384 k_split 1; '
               'xi measured over 0.2 < k r200m < 12')

# The cap is reached here, for reporting: (XI_MAX / CLUMP_A) ** (1 / CLUMP_ALPHA)
KR_SATURATES = (XI_MAX / CLUMP_A) ** (1.0 / CLUMP_ALPHA)


def xi(k, M, rhobar_m, a=CLUMP_A, alpha=CLUMP_ALPHA, xi_max=XI_MAX):
    """Fractional clumping correction, shape (nM, nk) to match Profile.u.

    Parameters
    ----------
    k        : (nk,)   wavenumbers [h/cMpc]
    M        : (nM,)   halo masses [Msun/h]; M <= 0 gives 0, so an empty bin's
                       placeholder mass cannot produce a correction
    rhobar_m : scalar  comoving mean matter density, for r200m(M)
    a, alpha : the power law
    xi_max   : the cap; pass np.inf for the unbounded law (read the module
               docstring first -- it is unbounded in k, not just large)

    xi -> 0 as k -> 0 for every mass, so the k -> 0 sum rule that fixes the
    mass budget is untouched: this changes the SHAPE of u_m, never its
    normalisation.
    """
    k = np.atleast_1d(np.asarray(k, dtype=float))
    M = np.atleast_1d(np.asarray(M, dtype=float))
    r = np.where(M > 0, r200m_of_M(np.where(M > 0, M, 1.0), rhobar_m), 0.0)
    kr = np.outer(r, np.abs(k))
    with np.errstate(invalid='ignore'):
        out = float(a) * kr ** float(alpha)
    return np.minimum(out, float(xi_max))


def factor(k, M, rhobar_m, a=CLUMP_A, alpha=CLUMP_ALPHA, xi_max=XI_MAX,
           apply='self-consistent'):
    """The multiplier on u_m, shape (nM, nk). 1 where xi is 0.

    'self-consistent'  1 / (1 - xi), the inversion of xi = C_i / R*_i
    'linear'           1 + xi

    xi_max < 1 is what keeps the division finite; the cap is enforced in xi()
    and this checks it rather than trusting it, because a caller passing
    xi_max = inf to see the unbounded power law would otherwise divide by a
    negative number and silently flip the sign of a bin's contribution.
    """
    if apply not in APPLY_MODES:
        raise ValueError(f"unknown clumping application {apply!r}; "
                         f"expected one of {', '.join(APPLY_MODES)}")
    v = xi(k, M, rhobar_m, a=a, alpha=alpha, xi_max=xi_max)
    if apply == 'linear':
        return 1.0 + v
    if np.any(v >= 1.0):
        raise ValueError(
            f"xi reaches {np.max(v):.2f} >= 1, so 1/(1 - xi) is not a "
            f"correction any more. Lower --clump-max below 1 (the measured "
            f"plateau is {XI_MAX:g}) or use --clump-apply linear.")
    return 1.0 / (1.0 - v)


def describe(a=CLUMP_A, alpha=CLUMP_ALPHA, xi_max=XI_MAX,
             apply='self-consistent'):
    """One line naming the law, how it is applied and where it stops."""
    form = '1/(1 - xi)' if apply == 'self-consistent' else '1 + xi'
    if not np.isfinite(xi_max):
        return (f"xi = {a:g} (k r200m)^{alpha:g} applied as {form}, UNCAPPED "
                f"-- extrapolates past where experiment D measured anything "
                f"({CALIBRATION})")
    kr = (xi_max / a) ** (1.0 / alpha)
    return (f"xi = {a:g} (k r200m)^{alpha:g} applied as {form}, capped at "
            f"{xi_max:g} (reached at k r200m = {kr:.1f}); {CALIBRATION}")


def _self_test():
    """The three things that would make this silently wrong."""
    rhobar_m = 0.306 * 2.775e11
    k = np.logspace(-2, 1.2, 60)
    M = np.array([1e11, 1e12, 1e13, 1e14, 1e15])

    v = xi(k, M, rhobar_m)
    assert v.shape == (M.size, k.size), v.shape
    # 1. k -> 0 leaves u_m, and therefore the mass budget, alone.
    assert np.all(xi(np.array([1e-8]), M, rhobar_m) < 1e-10), \
        "xi must vanish as k -> 0 or it moves the k -> 0 sum rule"
    # 2. It really is a function of k r200m and of nothing else.
    r = r200m_of_M(M, rhobar_m)
    ref = xi(np.array([1.0]), np.array([M[2]]), rhobar_m)[0, 0]
    for m, rr in zip(M, r):
        got = xi(np.array([r[2] / rr]), np.array([m]), rhobar_m)[0, 0]
        assert abs(got / ref - 1) < 1e-12, \
            f"xi at the same k r200m differs between masses: {got} vs {ref}"
    # 3. The cap binds where the measurement stopped, not before.
    assert np.isclose(xi(np.array([KR_SATURATES / r[3]]), np.array([M[3]]),
                         rhobar_m)[0, 0], XI_MAX, rtol=1e-6)
    assert np.all(v <= XI_MAX + 1e-12)
    big = xi(np.array([9.45]), np.array([1e15]), rhobar_m)[0, 0]
    unc = xi(np.array([9.45]), np.array([1e15]), rhobar_m, xi_max=np.inf)[0, 0]
    print(f"  shape {v.shape}, xi(k->0) = 0, universal in k r200m, "
          f"cap reached at k r200m = {KR_SATURATES:.1f}")
    print(f"  at k = k_Nyquist(2048, L=681) and M = 1e15: capped {big:.3f} "
          f"against {unc:.2f} uncapped -- the reason for the cap")

    # 4. Both application forms are multipliers that vanish with xi, and the
    #    self-consistent one refuses rather than flipping a sign.
    for mode in APPLY_MODES:
        fac = factor(k, M, rhobar_m, apply=mode)
        assert np.all(fac >= 1.0) and fac.shape == v.shape
        assert np.allclose(factor(np.array([1e-8]), M, rhobar_m, apply=mode),
                           1.0, atol=1e-9), f"{mode}: factor(k->0) != 1"
    lin = factor(np.array([9.45]), np.array([1e15]), rhobar_m, apply='linear')
    sc = factor(np.array([9.45]), np.array([1e15]), rhobar_m)
    print(f"  at the cap: linear {lin[0,0]:.4f}, self-consistent {sc[0,0]:.4f}")
    try:
        factor(k, M, rhobar_m, xi_max=np.inf)
    except ValueError as exc:
        print(f"  uncapped + self-consistent refused: {str(exc)[:58]}...")
    else:
        raise AssertionError("1/(1 - xi) with xi >= 1 must not be allowed")

    print(f"\n  {describe()}")
    print("\nclumping: all checks passed")


if __name__ == '__main__':
    _self_test()
