from astropy import units as u
from astropy.constants import codata2018 as const
import numpy as np


def compute_prefactor(total_baryon_mass, sim_params):
    """Compute tau prefactor from total baryon mass and cosmology/chemistry parameters."""
    
    box_comoving_mpc = sim_params['box_size_cMpc_h'] / sim_params['h']
    box_physical_mpc = box_comoving_mpc / (1 + sim_params['redshift'])
    
    if total_baryon_mass is not None:
        total_mass_msun = total_baryon_mass * 1e10 / sim_params['h'] # Msun
        volume_mpc3 = box_physical_mpc**3
        rho_bar_b = total_mass_msun / volume_mpc3
    else:
        H0 = sim_params['h'] * 100.0 * u.km / u.s / u.Mpc
        rho_crit_0 = (3 * H0**2 / (8 * np.pi * const.G)).to(u.Msun / u.Mpc**3)
        rho_bar_b = rho_crit_0 * sim_params['omega_b'] * (1 + sim_params['redshift'])**3

    rho_bar_b_kg_per_m3 = u.Quantity(
        rho_bar_b,
        unit=u.Unit("solMass Mpc-3"),
    ).to(u.Unit("kg m-3"))

    cell_physical_m = ((box_physical_mpc * u.Mpc) / sim_params['ngrid_default']).to(u.m)

    f_e = (sim_params['X_H'] + 0.5 * sim_params['X_He']) / const.m_p

    prefactor = (const.sigma_T * rho_bar_b_kg_per_m3 * f_e * cell_physical_m).to(
        u.dimensionless_unscaled
    ).value

    return prefactor