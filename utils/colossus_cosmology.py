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


def ensure_colossus_cosmology(sim_params, sim_name, power_spectrum):
    """Set the colossus cosmology from the parameter file, once.

    `power_spectrum` names the linear P(k) the run is using ('camb',
    'eisenstein98') and goes into the REGISTERED NAME. colossus does not need
    it there -- the spectrum is handed to each call as ps_args -- but colossus
    PERSISTS its sigma(R) and P(k) interpolation tables under ~/.colossus,
    keyed on the cosmology name and parameters and nothing else. Two runs with
    different spectra under one name read each other's stored tables, so
    swapping the CAMB input appears to change nothing at all, which is
    indistinguishable from the change genuinely having no effect. Putting the
    spectrum in the name gives each its own cache entry.

    concentration() can be called without a HaloModel ever being built (u_m is
    in the baseline reconstruction too), so the cosmology has to be
    establishable on its own and whichever entry point runs first registers it.
    They must therefore agree: a later caller asking for a different spectrum is
    an error, not a silent no-op.
    """
    name = f'{sim_name}_hatf_{power_spectrum}'
    current = colossus_cosmology.current_cosmo
    if current is not None:
        if current.name != name:
            raise RuntimeError(
                f"colossus cosmology is already registered as {current.name!r} "
                f"but {name!r} was requested. One process cannot hold two "
                f"linear power spectra: colossus keeps the cosmology in module "
                f"state and caches derived tables against it.")
        return current
    return colossus_cosmology.setCosmology(name, **colossus_params(sim_params))


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