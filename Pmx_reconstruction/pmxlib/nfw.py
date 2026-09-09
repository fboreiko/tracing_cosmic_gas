#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NFW profile u_m(k|M) and the concentration-mass relation c(M, z).

Was sections 1 and 2 of predict_Pmx_from_Phx.py. `rhobar_m` and the two
concentration switches used to be module globals -- and the switches were
rebound from the outside by measure_u_tilde.py and plot_omega_weights.py, so
c(M,z) returned a different number depending on who had imported the module and
how far through main() the interpreter was. They are now arguments.
"""
import warnings

import numpy as np
from scipy.special import sici

from colossus.halo import concentration as colossus_conc

from utils.colossus_cosmology import ensure_colossus_cosmology
from Pmx_reconstruction.pmxlib.cosmology import DELTA_HALO

_conc_warned = False


# ==============================================================================
# 1.  NFW PROFILE  u_m(k|M)   (normalized: u_m -> 1 as k -> 0)
# ==============================================================================
def delta_vir_200m():
    """Halo mass is M200m: mean interior density = 200 * rhobar_m."""
    return 200.0


def r200m_of_M(M, rhobar_m):
    """Comoving radius r200m [Mpc/h] for M200m [Msun/h].

    M = (4/3) pi r^3 * 200 * rhobar_m  =>  r = (3M / (4 pi 200 rhobar_m))^(1/3).
    Using the comoving rhobar_m gives a comoving radius, consistent with a
    comoving-k profile transform.
    """
    return (3.0 * M / (4.0 * np.pi * delta_vir_200m() * rhobar_m)) ** (1.0 / 3.0)


def u_nfw(k, M, c, rhobar_m):
    """Normalized Fourier transform of an NFW halo of mass M, concentration c.

    k [h/Mpc] (array), M [Msun/h] (scalar), c [-] (scalar). Returns u(k).
    """
    r200 = r200m_of_M(M, rhobar_m)
    rs = r200 / c
    mc = np.log(1.0 + c) - c / (1.0 + c)     # mass normalisation
    kr = np.asarray(k, dtype=float) * rs
    kr = np.where(kr > 0, kr, 1e-12)         # k=0 mode is never used, keep sici finite
    si_1c, ci_1c = sici((1.0 + c) * kr)
    si_0, ci_0 = sici(kr)
    term = (np.sin(kr) * (si_1c - si_0)
            - np.sin(c * kr) / ((1.0 + c) * kr)
            + np.cos(kr) * (ci_1c - ci_0))
    return term / mc


# ==============================================================================
# 2.  CONCENTRATION-MASS RELATION  c(M, z)
# ==============================================================================
# A simple power law in M200m, with parameters chosen to sit on the Diemer+19 /
# FLAMINGO-DMO median at these redshifts (c ~ 5-6 at cluster scales, ~9-10 for
# small halos). Swap in an exact colossus call if you want; with x = gas the
# reconstruction is only mildly sensitive to c, since P_halo_gas already carries the
# gas profile and u_m only redistributes the *matter* weight.
#
# That last sentence does NOT survive the switch to x = dm or x = matter. There
# the measured P_halo_dm (or P_halo_matter) carries the true matter profile and
# the truth goes as u_true^2, so the reconstruction/truth ratio at high k is
# essentially u_model/u_true and this function is being tested directly. Tune
# c0/alpha before concluding anything about the halo-model identity from a
# high-k residual in either mode.
#
# 'powerlaw' is the built-in fit below and is the DEFAULT, so that upgrading
# this file does not silently move every existing plot: u_m enters the baseline
# reconstruction, not just Experiment A. 'colossus' swaps in a published
# relation and is the better choice for new work -- especially in 'matter_dm'
# and 'matter_matter' modes, where the reconstruction/truth ratio at high k is
# close to u_model/u_true and this function is what is really being tested.
def _concentration_colossus(M, z, colossus_model, sim_params, sim_name):
    global _conc_warned
    ensure_colossus_cosmology(sim_params, sim_name)
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


def concentration(M, z, source='powerlaw', colossus_model='diemer19',
                  sim_params=None, sim_name='flamingo'):
    """Concentration-mass relation, c(M200m, z)."""
    if source == 'colossus':
        return _concentration_colossus(M, z, colossus_model, sim_params, sim_name)
    c0, Mpiv, alpha, beta = 7.5, 1e12, 0.090, 0.65
    return c0 * (M / Mpiv) ** (-alpha) * (1.0 + z) ** (-beta)


def concentration_from_cfg(M, z, cfg):
    """concentration() with the switches taken from a PmxConfig."""
    return concentration(M, z, source=cfg.concentration_source,
                         colossus_model=cfg.colossus_conc_model,
                         sim_params=cfg.sim_params, sim_name=cfg.sim_name)