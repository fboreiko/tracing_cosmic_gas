#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The intra-halo clumping correction to u_m, calibrated by experiment D.

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

    u_m(k|M)  ->  u_m(k|M) * [1 + xi(k r200m(M))]

Multiplying u_m rather than R_i is the same arithmetic -- R_i = f_i u_m P^he --
and puts the correction where it belongs conceptually: it is a statement about
the effective matter profile of a halo of mass M, so it should apply wherever
u_m appears, including inside U's S(k). It does essentially nothing there:
haloes below M_r have r200m <~ 0.12 cMpc/h, so k r200m < 1.2 out to Nyquist
and xi < 0.01.

Measured directly, by running experiment D at two apertures and fitting
each the same way:

fit variable        x = 1              x = 1.5           ratio
k x r200m           a = 0.0067         a = 0.0074         1.10
                    alpha = 1.74       alpha = 1.75
"""
import numpy as np

from Pmx_reconstruction.pmxlib.nfw import r200m_of_M

__all__ = ['CLUMP_A', 'CLUMP_ALPHA', 'XI_MAX', 'xi', 'factor']

# HOW xi BECOMES A MULTIPLIER, AND WHY THAT IS NOT A SWITCH
# xi is measured as C_i / R*_i -- against the TRUTH, not against the model --
# so inverting it is a division, not an addition:
#
#     R*_i = R_i + C_i = R_i + xi_i R*_i   =>   R*_i = R_i / (1 - xi_i)

#       xi = CLUMP_A * (k x r200m) ** CLUMP_ALPHA, capped at XI_MAX.
CLUMP_A = 0.0074
CLUMP_ALPHA = 1.75
XI_MAX = 0.20


def xi(k, M, rhobar_m, aperture=1.0, a=CLUMP_A, alpha=CLUMP_ALPHA,
       xi_max=XI_MAX):
    """Fractional clumping correction, shape (nM, nk) to match Profile.u.

    Parameters
    ----------
    k        : (nk,)   wavenumbers [h/cMpc]
    M        : (nM,)   halo masses [Msun/h]; M <= 0 gives 0, so an empty bin's
                       placeholder mass cannot produce a correction
    rhobar_m : scalar  comoving mean matter density, for r200m(M)
    aperture : scalar  the membership radius in units of r200m. The scaling
                       variable is k * aperture * r200m, NOT k * r200m -- see
                       THE APERTURE in the module docstring. At the default
                       x = 1 the two are the same, which is where the fit was
                       made, so nothing about that calibration changes here.
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
    kr = float(aperture) * np.outer(r, np.abs(k))
    with np.errstate(invalid='ignore'):
        out = float(a) * kr ** float(alpha)
    return np.minimum(out, float(xi_max))


def factor(k, M, rhobar_m, aperture=1.0, a=CLUMP_A, alpha=CLUMP_ALPHA,
           xi_max=XI_MAX):
    """The multiplier on u_m, 1 / (1 - xi). Shape (nM, nk); 1 where xi is 0.

    xi_max < 1 is what keeps the division finite; the cap is enforced in xi()
    and this checks it rather than trusting it, because a caller passing
    xi_max = inf to see the unbounded power law would otherwise divide by a
    negative number and silently flip the sign of a bin's contribution.
    """
    v = xi(k, M, rhobar_m, aperture=aperture, a=a, alpha=alpha,
           xi_max=xi_max)
    if np.any(v >= 1.0):
        raise ValueError(
            f"xi reaches {np.max(v):.2f} >= 1, so 1/(1 - xi) is not a "
            f"correction any more. Lower --clump-max below 1; the measured "
            f"plateau is {XI_MAX:g}.")
    return 1.0 / (1.0 - v)
