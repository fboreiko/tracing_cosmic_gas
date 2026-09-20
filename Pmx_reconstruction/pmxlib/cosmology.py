#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cosmology for the Pmx reconstruction."""
import numpy as np
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM

from utils.pipeline_paths import DATA_ROOT, ensure_parents

DELTA_C = 1.686               # spherical-collapse threshold
DELTA_HALO = 200.0            # w.r.t. MEAN background: matches m200b and r200m_of_M

NEUTRINO_MASS_EV = 0

CAMB_KMIN, CAMB_KMAX, CAMB_NK = 1e-6, 3e2, 1000


def astropy_cosmology(sim_params):
    """FlatLambdaCDM built from the simulation's own parameter file."""
    return FlatLambdaCDM(H0=100.0 * sim_params['h'], Om0=sim_params['omega_m'],
                         Ob0=sim_params.get('omega_b'),
                         Tcmb0=sim_params.get('tcmb0', 2.725))


def critical_density_h_units(sim_params):
    """rho_crit(z=0) in (Msun/h)/(Mpc/h)^3, from astropy."""
    ac = astropy_cosmology(sim_params)
    return float((ac.critical_density0 / ac.h ** 2).to(u.Msun / u.Mpc ** 3).value)


def mean_matter_density(sim_params):
    """rhobar_m = Omega_m * rho_crit(0), in (Msun/h)/(Mpc/h)^3."""
    return sim_params['omega_m'] * critical_density_h_units(sim_params)


def camb_linear_power_table(sim_params, sim_name):
    """P_lin(k, z=0) from CAMB, cached on disk as a colossus-readable table.

    Returns the path to a two-column file of log10(k [h/Mpc]), log10(P
    [(Mpc/h)^3]).
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