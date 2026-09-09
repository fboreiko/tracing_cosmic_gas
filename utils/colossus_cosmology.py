#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single registration point for the colossus cosmology.

colossus keeps its cosmology in process-global module state, and both pipelines
in this repo set it: Pmx_reconstruction (via concentration() and the halo-model
backend) and HATF (TkSZ_profiles.py, get_AP_simple_jax_batched.py). They used to
do so independently, under different names and with different parameter sets --
Pmx passed Tcmb0 and relspecies=False, HATF did not. They never collided only
because nothing imported both. This module exists so that cannot happen.

The parameter set here is the Pmx one (Tcmb0 and relspecies=False included).
"""
from colossus.cosmology import cosmology as colossus_cosmology


def ensure_colossus_cosmology(sim_params, sim_name):
    """Set the colossus cosmology from the parameter file, once.

    concentration() can be called without a HaloModel ever being built (the
    baseline reconstruction uses u_m too), so the cosmology has to be
    establishable on its own.
    """
    if colossus_cosmology.current_cosmo is not None:
        return colossus_cosmology.current_cosmo
    return colossus_cosmology.setCosmology(
        f'{sim_name}_hatf',
        **colossus_params(sim_params))


def colossus_params(sim_params):
    """The parameter dict handed to colossus.setCosmology."""
    return dict(flat=True,
                H0=100.0 * sim_params['h'],
                Om0=sim_params['omega_m'],
                Ob0=sim_params.get('omega_b'),
                sigma8=sim_params.get('sigma8'),
                ns=sim_params.get('n_s'),
                Tcmb0=sim_params.get('tcmb0', 2.725),
                relspecies=False)