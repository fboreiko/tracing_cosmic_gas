"""
Conductor script for running multiple profile configurations of the get_AP 
pipeline in sequence.

Profile configs are essentially designed to compare HATF reconstructed and 
true tau fields via computing their kSZ statistics for the same set of halos. 
This set of halos has to be the same as the one used for HATF construction, 
and the pipeline ensures this. 

Somewhat confusingly, the HATF reconstructed profiles are referred to as 
"2D_FT_upgrade_tau_reconstruction" with "strongest_AGN_reconstructed" or 
"fiducial_reconstructed" gas specifications.

The true profiles are referred to as "fullFT_tau_reconstruction" with 
"strongest_AGN" or "fiducial" gas specifications.

This naming convention is legacy and should be cleaned up in the future.
"""

from mpi4py import MPI
from datetime import datetime
import gc

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

import get_AP_simple_jax_batched as simple_module
import get_AP_pixell_jax_batched as pixell_module

# Simulation constants
Z_REAL = 0.74   # redshift of observed sample — never changes, this is independent of simulation intrinsic redshift


from utils.profile_configs import PROFILE_CONFIGS, build_profiles

# Which of the shared configs this script actually runs.
ACTIVE_PROFILE_NAMES = ['flamingo_cen_massbin']

PROFILES = [p for p in build_profiles(PROFILE_CONFIGS)
            if p['name'] in ACTIVE_PROFILE_NAMES]


def run_configuration(profile, config_num, total_configs):
    if rank == 0:
        print(f"\n{'='*70}")
        print(f"Config {config_num}/{total_configs}")
        print(f"  profile   : {profile.get('name', 'N/A')}")
        print(f"  sim_name  : {profile.get('sim_name', 'flamingo')}")
        print(f"  projection: {profile.get('projection_type', 'N/A')}")
        print(f"  gas_type  : {profile['gas_type']}")
        print(f"  tau_method: {profile['tau_method']}")
        print(f"  field_type: {profile['field_type']}")
        print(f"  sel_mode  : {profile.get('selection_mode', 'N/A')}")
        print(f"  sel_mass  : {profile.get('selection_mass_def', 'N/A')}")
        _sf = profile.get('sat_frac')
        print(f"  sat_frac  : {'N/A' if _sf is None else f'{_sf*100:.0f}%'}")
        print(f"  Start     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        config = {
            'sim_name': profile.get('sim_name', 'flamingo'),
            'gas_type': profile['gas_type'],
            'tau_method': profile['tau_method'],
            'field_type': profile.get('field_type', 'gas'),
            'z_real': Z_REAL,
            'selection_mode': profile['selection_mode'],
            'selection_mass_def': profile['selection_mass_def'],
            'select_nonzero_masses': profile['select_nonzero_masses'],
            'upper_mass_cut': profile['upper_mass_cut'],
            'upper_radius_cut': profile['upper_radius_cut'],
        }

        config['target_mean_mass'] = profile.get('target_mean_mass')

        if profile.get('upper_mass_cut'):
            config['max_mass'] = profile['max_mass']

        # Always include these for cache detection
        config['n_gal_density'] = profile.get('n_gal_density')
        config['halo_mass_range'] = profile.get('halo_mass_range')

        if profile.get('selection_mode') in ('mixed', 'sat'):
            config['sat_frac'] = profile['sat_frac']

        config['convergence_mode'] = profile.get('convergence_mode', 'ngal')

        if 'A' in profile:
            config['A'] = profile['A']

        if profile['projection_type'] == 'simple':
            if rank == 0:
                print("  Running simple projection...")
            simple_module.get_AP_simple(config)
        elif profile['projection_type'] == 'pixell':
            if rank == 0:
                print("  Running pixell projection...")
            pixell_module.get_AP_pixell(config)
        else:
            raise ValueError(f"Unknown projection_type: {profile['projection_type']}")

        if rank == 0:
            print(f"✓ Config {config_num}/{total_configs} completed  "
                  f"[{datetime.now().strftime('%H:%M:%S')}]")
        
        # Explicit garbage collection and MPI barrier to sync all ranks
        gc.collect()
        comm.Barrier()

    except Exception as e:
        if rank == 0:
            print(f"✗ Config {config_num}/{total_configs} FAILED: {type(e).__name__}: {e}")
        raise


def main():
    if rank == 0:
        print("\n" + "=" * 70)
        print("MULTI-CONFIGURATION BATCH PROCESSOR")
        print("=" * 70)
        print(f"Total configurations : {len(PROFILES)}")
        print(f"\nProfile configurations:")
        for i, config in enumerate(PROFILE_CONFIGS, start=1):
            print(f"\n  Config {i}: {config['name']}")
            print(f"    Selection mode    : {config['selection_params']['selection_mode']}")
            print(f"    Mass def          : {config['selection_params']['selection_mass_def']}")
            print(f"    Target mass       : {config['selection_params'].get('target_mean_mass', 'N/A')}")
            if config['sat_fracs'] is not None:
                print(f"    Satellite fracs   : {[f'{s*100:.0f}%' for s in config['sat_fracs']]} (swept)")
            else:
                print(f"    Satellite fracs   : N/A (not swept; mode={config['selection_params']['selection_mode']})")
            print(f"    Tau methods       : {len(config['tau_methods'])} (recon, fullFT gas, fullFT dm)")
            print(f"    Convergence mode  : {config['convergence_mode']}")
        print(f"\nMPI processes        : {size}")
        print(f"Start time           : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)

    for i, profile in enumerate(PROFILES, start=1):
        run_configuration(profile, i, len(PROFILES))
        comm.Barrier()

    if rank == 0:
        print("\n" + "=" * 70)
        print("ALL CONFIGURATIONS COMPLETED")
        print(f"Total processed : {len(PROFILES)}")
        print(f"End time        : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)


if __name__ == "__main__":
    main()