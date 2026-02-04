"""
Unified script to run multiple aperture photometry configurations sequentially.
Processes multiple gas types, projections, and other parameters without resubmitting jobs.
"""

import numpy as np
import os
import sys
from mpi4py import MPI
from datetime import datetime

# Initialize MPI once for all configurations
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Import both processing modules (they will use the same MPI comm)
import get_AP_simple_jax_batched as simple_module
import get_AP_pixell_jax_batched as pixell_module

# --------------------
# Configuration Profiles
# --------------------

# Base configurations (without mass bin specification)
PROFILES = [
    {
        "projection_type": "simple",
        "tau_method": "2D_FT_upgrade_tau_reconstruction",
        "smoothed": True,
        "gas_type": "strongest_AGN_reconstructed",
        "halo_mass_range": None,
        "ngrid": 2048,
        "JAX": True,
        "z": 0.74,
        "n_gal_density": 87e-5, #87e-5,
        "account_for_miscentering": False
    },
    {
        "projection_type": "simple",
        "tau_method": "fullFT_tau_reconstruction",
        "smoothed": True,
        "gas_type": "strongest_AGN",
        "halo_mass_range": None,
        "ngrid": 2048,
        "JAX": True,
        "z": 0.74,
        "n_gal_density": 87e-5, #87e-5,
        "account_for_miscentering": False,
    },
    {
        "projection_type": "simple",
        "tau_method": "2D_FT_upgrade_tau_reconstruction",
        "smoothed": True,
        "gas_type": "fiducial_reconstructed",
        "halo_mass_range": None,  
        "ngrid": 2048,
        "JAX": True,
        "z": 0.74,
        "n_gal_density": 87e-5, #87e-5,
        "account_for_miscentering": False
    },
    {
        "projection_type": "simple",  # 'pixell', 'pixell_cea', or 'simple'
        "tau_method": "fullFT_tau_reconstruction",  # 'fullFT' or 'histmethod', or '2D_FT', or '2D_FT_massdep'
        "smoothed": True,
        "gas_type": "fiducial",  # 'fiducial_reconstructed', 'strongest_AGN_reconstructed', 'fiducial', 'strongest_AGN'
        "halo_mass_range": None,  # int (0 to N-1) for mass bin index, or None to use n_gal_density
        "ngrid": 2048,
        "JAX": True,
        "z": 0.74,
        "n_gal_density": 87e-5, #87e-5,
        "account_for_miscentering": False,
    }
]


def run_configuration(profile, config_num, total_configs):
    
    try:
        # Build config dict (same structure for both modules)
        config = {
            'gas_type': profile['gas_type'],
            'tau_method': profile['tau_method'],
            'n_cell': profile['ngrid'],
            'z_real': profile['z'],
            'n_gal_density': profile['n_gal_density'],
            'halo_mass_range': profile['halo_mass_range'],
            'beam_smoothing': profile['smoothed'],
            'account_for_miscentering': profile['account_for_miscentering']
        }
        
        # Add A parameter if present in profile
        if 'A' in profile:
            config['A'] = profile['A']
        
        if profile['projection_type'] == 'simple':
            # Call simple projection with config dict
            if rank == 0:
                print("Running simple projection processing...")
            simple_module.get_AP_simple(config)
            
        elif profile['projection_type'] == 'pixell':
            # Call pixell projection with config dict (CAR projection)
            config['projection_type'] = 'car'
            if rank == 0:
                print("Running pixell (CAR) projection processing...")
            pixell_module.get_AP_pixell(config)
            
        elif profile['projection_type'] == 'pixell_cea':
            # Call pixell projection with config dict (CEA projection)
            config['projection_type'] = 'cea'
            if rank == 0:
                print("Running pixell (CEA) projection processing...")
            pixell_module.get_AP_pixell(config)
        
        else:
            raise ValueError(f"Unknown projection_type: {profile['projection_type']}")
        
        if rank == 0:
            print(f"\n✓ Configuration {config_num}/{total_configs} completed successfully!")
            print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    except Exception as e:
        if rank == 0:
            print(f"\n✗ Configuration {config_num}/{total_configs} failed with error:")
            print(f"  {type(e).__name__}: {str(e)}")
        raise


def main():
    """Main execution loop for processing all configurations."""
    if rank == 0:
        print("\n" + "=" * 80)
        print("MULTI-CONFIGURATION BATCH PROCESSOR")
        print("=" * 80)
        print(f"Total configurations: {len(PROFILES)}")
        print(f"MPI processes: {size}")
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
    
    # Process each configuration sequentially
    for i, profile in enumerate(PROFILES, start=1):
        run_configuration(profile, i, len(PROFILES))
        
        # Barrier to ensure all processes finish before moving to next config
        comm.Barrier()
    
    if rank == 0:
        print("\n" + "=" * 80)
        print("ALL CONFIGURATIONS COMPLETED")
        print("=" * 80)
        print(f"Total configurations processed: {len(PROFILES)}")
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)


if __name__ == "__main__":
    main()
