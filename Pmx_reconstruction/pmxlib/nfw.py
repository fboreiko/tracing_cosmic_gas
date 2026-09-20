#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NFW profile u_m(k|M) and the concentration-mass relation c(M, z)."""
import warnings

import numpy as np
from scipy.special import sici

from colossus.halo import concentration as colossus_conc

from utils.colossus_cosmology import ensure_colossus_cosmology
from Pmx_reconstruction.pmxlib.cosmology import DELTA_HALO

_conc_warned = False


# NFW PROFILE  u_m(k|M)   (normalized: u_m -> 1 as k -> 0)

def r200m_of_M(M, rhobar_m):
    """Comoving radius r200m [Mpc/h] for M200m [Msun/h].

    M = (4/3) pi r^3 * 200 * rhobar_m  =>  r = (3M / (4 pi 200 rhobar_m))^(1/3).
    """
    return (3.0 * M / (4.0 * np.pi * 200.0 * rhobar_m)) ** (1.0 / 3.0)


def u_nfw(k, M, c, rhobar_m, trunc=1.0):
    """Normalized Fourier transform of an NFW halo of mass M, concentration c.

    k [h/Mpc] (array), M [Msun/h] (scalar), c [-] (scalar). Returns u(k).

    `trunc` moves the outer edge to trunc * r200m. Since rs is fixed by
    c = r200m/rs, a truncation radius of trunc * r200m is trunc * c in units of
    rs, so the whole generalisation is c -> c_t = trunc * c.
    """
    r200 = r200m_of_M(M, rhobar_m)
    rs = r200 / c
    c_t = trunc * c                          # truncation radius in units of rs
    mc = np.log(1.0 + c_t) - c_t / (1.0 + c_t)     # mass normalisation
    kr = np.asarray(k, dtype=float) * rs
    kr = np.where(kr > 0, kr, 1e-12)         # k=0 mode is never used, keep sici finite
    si_1c, ci_1c = sici((1.0 + c_t) * kr)
    si_0, ci_0 = sici(kr)
    term = (np.sin(kr) * (si_1c - si_0)
            - np.sin(c_t * kr) / ((1.0 + c_t) * kr)
            + np.cos(kr) * (ci_1c - ci_0))
    return term / mc


# CONCENTRATION-MASS RELATION  c(M, z)

# Published relations via colossus, which model is chosen by
# --concentration-model (diemer19, duffy08, ishiyama21, ...). 
#
# `source` is kept as a dispatch point so a relation that is not a colossus
# model (a FLAMINGO-DMO fit, say) can be added without touching every caller:
# register it in CONCENTRATION_SOURCES and give it a branch in concentration().
CONCENTRATION_SOURCES = ('colossus',)


def _concentration_colossus(M, z, colossus_model, sim_params, sim_name,
                            power_spectrum):
    global _conc_warned
    ensure_colossus_cosmology(sim_params, sim_name, power_spectrum)
    scalar = np.isscalar(M)
    M = np.atleast_1d(np.asarray(M, dtype=float))
    # The unresolved-mass integral reaches down to 10^8 Msun/h, below the
    # calibration range of most c(M,z) fits. colossus warns and extrapolates;
    # that is acceptable here because u_m ~ 1 for those halos at every k we use,
    # so the extrapolated c barely affects the answer. Surface it once rather
    # than letting it either spam the log or vanish.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        c = colossus_conc.concentration(M, f'{int(DELTA_HALO)}m', z,
                                        model=colossus_model)
    if caught and not _conc_warned:
        print(f"  [conc] {colossus_model}: some masses lie outside the "
              "model's calibration range and are extrapolated. Harmless here "
              "(u_m -> 1 for those halos), but noted.")
        _conc_warned = True
    return float(c[0]) if scalar else c


def concentration(M, z, source='colossus', colossus_model='diemer19',
                  sim_params=None, sim_name='flamingo', power_spectrum='camb'):
    """Concentration-mass relation, c(M200m, z).

    `power_spectrum` is not used for c itself -- colossus_conc.concentration
    takes no ps_args and uses the cosmology's built-in spectrum -- but it names
    the colossus cosmology, and this may be what registers it. See
    ensure_colossus_cosmology.
    """
    if source == 'colossus':
        return _concentration_colossus(M, z, colossus_model, sim_params,
                                       sim_name, power_spectrum)
    raise ValueError(f"unknown concentration source {source!r}; "
                     f"available: {', '.join(CONCENTRATION_SOURCES)}")


def conc_of(cfg, M, z):
    """concentration() with the switches taken from a PmxConfig."""
    return concentration(M, z, source=cfg.concentration_source,
                         colossus_model=cfg.colossus_conc_model,
                         sim_params=cfg.sim_params, sim_name=cfg.sim_name,
                         power_spectrum=cfg.power_spectrum)
