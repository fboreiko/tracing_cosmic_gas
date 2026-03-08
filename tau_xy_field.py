import numpy as np
from pathlib import Path
import deepdish as dd
from astropy import units as u
from astropy.constants import m_p, sigma_T
from abacusnbody.analysis.tsc import tsc_parallel
from utils import *

# Define RDS base path
rds_base = '/rds-d6/user/fb635/hpc-work/tracing_cosmic_gas'

sim_type = 'strongest_AGN'  # 'fiducial', 'strongest_AGN'
field_type = 'gas'  # 'gas' or 'dm'

# Simulation/cosmology parameters
box = 681  # cMpc/h
h = 0.681
Z_SIM = 0.74
ngrid = 2048
nthread = 4

# Hydrogen and Helium fractions
X_H = 0.76
X_He = 0.24


def compute_delta(pos, box, ngrid, weights, nthread=4):
    """Compute overdensity field using TSC assignment."""
    dens = tsc_parallel(pos, ngrid, box, weights=weights, nthread=nthread)
    dens_avg = np.sum(dens) / ngrid**3
    delta = dens / dens_avg - 1
    return delta


def compute_2d_projection(delta):
    """Project 3D overdensity field to 2D by averaging along z-axis."""
    return np.mean(delta, axis=2)


def load_particles(sim_file, particle_type='gas'):
    """Load particle positions and masses from simulation file."""
    file = dd.io.load(sim_file)
    
    if particle_type == 'gas':
        masses = file['GasParticles']['mass']  # 1e10 Msun/h
        pos = file['GasParticles']['pos']
    elif particle_type == 'dm':
        masses = file['DMParticles']['mass']  # 1e10 Msun/h
        pos = file['DMParticles']['pos']
    else:
        raise ValueError(f"Unknown particle_type: {particle_type}")
    
    masses *= 100  # Apply scaling factor
    return masses, pos


def compute_prefactor(masses, ngrid, box, h, Z_SIM, X_H, X_He):
    # Convert masses and box size
    masses_Msun = masses * 1e10 / h  # Convert to Msun
    box_comoving_Mpc = box / h  # Convert from cMpc/h to Mpc
    box_physical_Mpc = box_comoving_Mpc / (1 + Z_SIM)  # Physical size at redshift Z_SIM
    
    # Compute mean density
    total_mass_Msun = np.sum(masses_Msun)
    volume_Mpc3 = box_physical_Mpc**3
    rho_bar_Msun_per_Mpc3 = total_mass_Msun / volume_Mpc3
    
    # Convert to kg/m^3
    rho_bar_kg_per_m3 = (rho_bar_Msun_per_Mpc3 * u.Msun / u.Mpc**3).to(u.kg / u.m**3)
    
    # Cell size in physical units
    cell_physical_m = ((box_physical_Mpc * u.Mpc) / ngrid).to(u.m)
    
    # Electrons per kg (assuming ionized gas)
    f_e = (X_H + 0.5 * X_He) / m_p
    
    # Compute prefactor
    prefactor = (sigma_T * rho_bar_kg_per_m3 * f_e * cell_physical_m).to(u.dimensionless_unscaled).value
    
    return prefactor


def main():
    # Get file paths using pipeline_paths
    sim_file = particles_hdf5(sim_type)
    
    tau_out_path = tau_map_path({
        'gas_type': sim_type,
        'tau_method': 'fullFT_tau_reconstruction',
        'field_type': field_type,
    })
    
    print(f"Processing {field_type.upper()} field for {sim_type} simulation")
    print(f"Output tau map path: {tau_out_path}")
    
    # Compute 2D projected density field from particles
    print(f"\nComputing 2D projected delta field from particles...")
    
    if field_type == 'dm':
        # Load DM particles
        print("  Loading DM particles...")
        masses_dm, pos_dm = load_particles(sim_file, particle_type='dm')
        
        # Also load gas masses to compute rescaling factor
        print("  Loading gas masses for rescaling...")
        masses_gas, _ = load_particles(sim_file, particle_type='gas')
        
        # Rescale DM masses so that sum(DM masses) = sum(gas masses)
        total_dm_mass = np.sum(masses_dm)
        total_gas_mass = np.sum(masses_gas)
        rescale_factor = total_gas_mass / total_dm_mass
        
        print(f"  Total DM mass (before rescaling): {total_dm_mass:.4e} (1e10 Msun/h)")
        print(f"  Total gas mass: {total_gas_mass:.4e} (1e10 Msun/h)")
        print(f"  Rescale factor: {rescale_factor:.6f}")
        
        # Apply rescaling to DM masses
        masses_dm *= rescale_factor
        masses_for_tau = masses_dm
        
        # Compute weights for TSC
        weights = masses_dm / np.mean(masses_dm)
        
        # Compute 3D delta field
        print("  Computing 3D delta field using TSC...")
        delta_3d = compute_delta(pos_dm, box, ngrid, weights, nthread=nthread)
        
        # Project to 2D
        print("  Projecting to 2D...")
        delta_2d_field = compute_2d_projection(delta_3d)
    else:
        # Load gas particles
        print("  Loading gas particles...")
        masses_gas, pos_gas = load_particles(sim_file, particle_type='gas')
        masses_for_tau = masses_gas
        
        # Compute weights for TSC
        weights = masses_gas / np.mean(masses_gas)
        
        # Compute 3D delta field
        print("  Computing 3D delta field using TSC...")
        delta_3d = compute_delta(pos_gas, box, ngrid, weights, nthread=nthread)
        
        # Project to 2D
        print("  Projecting to 2D...")
        delta_2d_field = compute_2d_projection(delta_3d)
    
    print(f"  2D delta field computed. Shape: {delta_2d_field.shape}")
    print(f"  Stats - mean: {np.mean(delta_2d_field):.4f}, std: {np.std(delta_2d_field):.4f}, min: {np.min(delta_2d_field):.4f}, max: {np.max(delta_2d_field):.4f}")
    
    # Compute prefactor
    print("\nComputing prefactor...")
    prefactor = compute_prefactor(masses_for_tau, ngrid, box, h, Z_SIM, X_H, X_He)
    print(f"Prefactor: {prefactor:.6e}")
    
    # Compute tau map from 2D projected field (same as HATF)
    # For 2D: tau = prefactor * (1 + delta_2d) * ngrid
    # This is equivalent to: prefactor * sum((1 + delta_3d), axis=2)
    print("\nComputing tau map from 2D projected field...")
    tau_map = prefactor * (1.0 + delta_2d_field) * ngrid
    
    print(f"Tau map shape: {tau_map.shape}")
    print(f"Tau map stats - mean: {np.mean(tau_map):.6e}, std: {np.std(tau_map):.6e}, min: {np.min(tau_map):.6e}, max: {np.max(tau_map):.6e}")
    
    # Save tau map
    Path(tau_out_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(tau_out_path, tau_map)
    print(f"Tau map saved to: {tau_out_path}")


if __name__ == '__main__':
    main()