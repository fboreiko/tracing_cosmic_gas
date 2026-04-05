import numpy as np
from pathlib import Path

from utils.pipeline_paths import tau_map_path, get_particle_file_path
from utils.delta_fields import compute_delta_field_and_mass
from utils.tau_prefactor import compute_prefactor
from utils.sim_params import get_sim_params

SIM_NAME = 'abacus'
SIM_PARAMS = get_sim_params(SIM_NAME)

FEEDBACK_MODE = 'strongest_AGN'  # 'fiducial', 'strongest_AGN'
FIELD_TYPE = 'dm'  # 'gas' or 'dm'

# Simulation/cosmology parameters
box = SIM_PARAMS['box_size_cMpc_h']
ngrid = SIM_PARAMS['ngrid_default']
nthread = SIM_PARAMS['nthread_default']


def main():
    # Get file paths using pipeline_paths
    dm_particles_file = get_particle_file_path(
        FEEDBACK_MODE,
        sim_name=SIM_NAME,
    )
    gas_particles_file = get_particle_file_path(
        FEEDBACK_MODE,
        sim_name='flamingo',
    )
    
    tau_out_path = tau_map_path({
        'sim_name': SIM_NAME,
        'gas_type': FEEDBACK_MODE,
        'tau_method': 'fullFT_tau_reconstruction',
        'field_type': FIELD_TYPE,
    })
    
    print(f"Processing {FIELD_TYPE.upper()} field for {FEEDBACK_MODE} simulation")
    print(f"Output tau map path: {tau_out_path}")
    
    # Compute 2D projected density field from particles
    print(f"\nComputing 2D projected delta field from particles...")
    
    delta_2d_field, total_mass_for_tau = compute_delta_field_and_mass(
        field_type=FIELD_TYPE,
        sim_name=SIM_NAME,
        dm_particles_file=dm_particles_file,
        gas_particles_file=gas_particles_file,
        box=box,
        ngrid=ngrid,
        nthread=nthread,
    )
    
    print(f"  2D delta field computed. Shape: {delta_2d_field.shape}")
    print(f"  Stats - mean: {np.mean(delta_2d_field):.4f}, std: {np.std(delta_2d_field):.4f}, min: {np.min(delta_2d_field):.4f}, max: {np.max(delta_2d_field):.4f}")
    
    # Compute prefactor
    print("\nComputing prefactor...")
    if SIM_NAME == 'abacus':
        total_mass_for_tau = None  # Use cosmological parameters for Abacus
    prefactor = compute_prefactor(total_mass_for_tau, SIM_PARAMS)
    print(f"Prefactor: {prefactor:.6e}")
    
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