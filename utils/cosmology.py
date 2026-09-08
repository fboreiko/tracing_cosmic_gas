#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cosmology for the Pmx reconstruction.

Every function here is a pure function of an explicit `sim_params` dict (from
utils.sim_params.get_sim_params) rather than of module globals, so that two
callers in one process cannot disagree about the cosmology.

This module used to live inside the Experiment A section of
predict_Pmx_from_Phx.py, which is why RHO_CRIT_0 and rhobar_m were declared None
at the top of that file and only filled in 1300 lines later. Sections 1 and 2
(the NFW profile and c(M,z)) depend on them, so they belong here, ahead of both.
"""
import numpy as np
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM
from functools import lru_cache

from utils.pipeline_paths import DATA_ROOT, ensure_parents

DELTA_C = 1.686               # spherical-collapse threshold
DELTA_HALO = 200.0            # w.r.t. MEAN background: matches m200b and r200m_of_M

# Sum of neutrino masses [eV] handed to CAMB. The parameter file does not carry
# it, and Omega_m there is the TOTAL matter density, so setting a non-zero mnu
# here without also adjusting omch2 would change Omega_m. FLAMINGO's fiducial
# (DES Y3 3x2pt+all) uses 0.06 eV; set it here if you want that, and note that
# it suppresses P_lin by ~ a few per cent at the k of interest. It cancels to
# first order in T(k,M), which is a ratio, so the default of 0 is safe for
# Experiment A even though it is not the simulation's cosmology.
NEUTRINO_MASS_EV = 0.0

# CAMB table extent. It has to bracket everything colossus will ask for when it
# builds its sigma(R) interpolator, which reaches well beyond the k of the box.
CAMB_KMIN, CAMB_KMAX, CAMB_NK = 1e-6, 3e2, 1000


def astropy_cosmology(sim_params):
    """FlatLambdaCDM built from the simulation's own parameter file."""
    return FlatLambdaCDM(H0=100.0 * sim_params['h'], Om0=sim_params['omega_m'],
                         Ob0=sim_params.get('omega_b'),
                         Tcmb0=sim_params.get('tcmb0', 2.725))


def critical_density_h_units(sim_params):
    """rho_crit(z=0) in (Msun/h)/(Mpc/h)^3, from astropy.

    The h's cancel, so this is numerically the same as rho_crit in
    h^2 Msun/Mpc^3, i.e. the familiar 2.77536627e11.

    The `/ ac.h ** 2` is the whole content of this function. Dropping it gives a
    density in Msun/Mpc^3, low by a factor h^2 = 0.4638, which puts every
    reconstruction weight n_i M_i / rhobar_m out by 2.156 and every r200m out by
    +29% without raising anything.
    """
    ac = astropy_cosmology(sim_params)
    return float((ac.critical_density0 / ac.h ** 2).to(u.Msun / u.Mpc ** 3).value)


def mean_matter_density(sim_params):
    """rhobar_m = Omega_m * rho_crit(0), in (Msun/h)/(Mpc/h)^3.

    Comoving mean matter density does not evolve in these units, so this is
    redshift-independent. Numerically identical to the module-level `rhobar_m`
    that predict_Pmx_from_Phx.py used to hold.

    A function rather than a module constant on purpose: the old
    `rhobar_m = None`, filled in far below, was what made the import order of
    that file fragile.
    """
    return sim_params['omega_m'] * critical_density_h_units(sim_params)


def camb_linear_power_table(sim_params, sim_name):
    """P_lin(k, z=0) from CAMB, cached on disk as a colossus-readable table.

    Returns the path to a two-column file of log10(k [h/Mpc]), log10(P
    [(Mpc/h)^3]), normalised so that sigma_8 equals the value in the parameter
    file. CAMB is run with an arbitrary A_s and the spectrum rescaled by
    (sigma8_target / sigma8_camb)^2, which is exact in linear theory and avoids
    having to solve for A_s.

    The table is what makes the whole stack self-consistent: colossus computes
    sigma(M), the Tinker+08 mass function and the Tinker+10 bias from this same
    spectrum rather than from its built-in Eisenstein & Hu approximation.
    """
    import camb

    h_little, omega_m = sim_params['h'], sim_params['omega_m']
    ob, ns = sim_params.get('omega_b'), sim_params.get('n_s')
    s8, tcmb = sim_params.get('sigma8'), sim_params.get('tcmb0', 2.725)
    tag = (f"{omega_m:.6f}_{ob:.6f}_{h_little:.6f}_{s8:.6f}_{ns:.6f}"
           f"_{tcmb:.4f}_{NEUTRINO_MASS_EV:.4f}_{CAMB_KMAX:.0f}")
    path = DATA_ROOT / sim_name / 'pme_inputs' / f'plin_camb_{tag}.txt'
    if path.exists():
        print(f"  [camb] reusing cached linear P(k): {path.name}")
        return path

    print("  [camb] running CAMB for the linear power spectrum (one-off, ~20 s) ...")
    pars = camb.set_params(
        H0=100.0 * h_little,
        ombh2=ob * h_little ** 2,
        omch2=(omega_m - ob) * h_little ** 2,
        ns=ns, TCMB=tcmb, mnu=NEUTRINO_MASS_EV, omk=0.0, As=2.0e-9,
    )
    pars.set_matter_power(redshifts=[0.0], kmax=CAMB_KMAX * 1.7)
    pars.NonLinear = camb.model.NonLinear_none
    results = camb.get_results(pars)

    s8_camb = float(results.get_sigma8_0())
    kh, _, pk = results.get_matter_power_spectrum(
        minkh=CAMB_KMIN, maxkh=CAMB_KMAX, npoints=CAMB_NK)
    pk0 = pk[0] * (s8 / s8_camb) ** 2
    print(f"  [camb] sigma8 = {s8_camb:.5f} from A_s = 2e-9, rescaled to "
          f"{s8:.5f} (factor {(s8 / s8_camb) ** 2:.5f} in P)")

    ensure_parents(path)
    np.savetxt(path, np.column_stack([np.log10(kh), np.log10(pk0)]),
               header='log10(k [h/Mpc])  log10(P [(Mpc/h)^3])')
    print(f"  [camb] table written to {path}")
    return path


def colossus_params(sim_params):
    """Return the colossus cosmology parameter dict for a simulation."""
    return {
        'flat': True,
        'H0': 100.0 * sim_params['h'],
        'Om0': sim_params['omega_m'],
        'Ob0': sim_params.get('omega_b'),
        'sigma8': sim_params.get('sigma8'),
        'ns': sim_params.get('n_s'),
        'Tcmb0': sim_params.get('tcmb0', 2.725),
    }


@lru_cache(maxsize=None)
def _cosmo_name(sim_name, h, omega_m, omega_b, sigma8, n_s, tcmb0):
    """Stable cache key for the registered colossus cosmology."""
    ob = 'None' if omega_b is None else f'{omega_b:.8f}'
    s8 = 'None' if sigma8 is None else f'{sigma8:.8f}'
    ns = 'None' if n_s is None else f'{n_s:.8f}'
    return (
        f'{sim_name}_h{h:.8f}_om{omega_m:.8f}_ob{ob}'
        f'_s8{s8}_ns{ns}_tcmb{tcmb0:.4f}'
    )


def ensure_colossus_cosmology(sim_params, sim_name='flamingo'):
    """Register and return the colossus cosmology for ``sim_params``.

    colossus stores the active cosmology in process-global state. This helper
    makes sure the repo always installs the same cosmology for a given set of
    simulation parameters, and reuses it on subsequent calls.
    """
    from colossus.cosmology import cosmology as colossus_cosmology

    params = colossus_params(sim_params)
    name = _cosmo_name(
        sim_name,
        float(sim_params['h']),
        float(sim_params['omega_m']),
        None if sim_params.get('omega_b') is None else float(sim_params['omega_b']),
        None if sim_params.get('sigma8') is None else float(sim_params['sigma8']),
        None if sim_params.get('n_s') is None else float(sim_params['n_s']),
        float(sim_params.get('tcmb0', 2.725)),
    )

    try:
        current = colossus_cosmology.getCurrent()
    except Exception:
        current = None
    if current is not None and getattr(current, 'name', None) == name:
        return current

    try:
        colossus_cosmology.addCosmology(name, params)
    except Exception:
        # The cosmology may already exist from a previous call in this process.
        pass
    colossus_cosmology.setCosmology(name)
    return colossus_cosmology.getCurrent()