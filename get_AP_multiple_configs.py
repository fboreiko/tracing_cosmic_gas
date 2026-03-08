from mpi4py import MPI
from datetime import datetime
from utils import *

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

import get_AP_simple_jax_batched as simple_module

# Simulation constants
Z_SIM = 0.74   # simulation snapshot redshift — never changes

SAT_FRACS = [0.01, 0.10, 0.20, 0.30]

PROFILE_CONFIGS = [
    {
        'name': 'mstell_mixed',
        'selection_params': {
            'selection_mode': 'mixed',
            'selection_mass_def': 'mstell',
            'select_nonzero_masses': True,
            'upper_mass_cut': False,
            'max_mass': 1e14,
            'upper_radius_cut': False,
            'target_mean_mass': 13.2,  # Informational only; convergence done in HATF
        },
        'convergence_mode': 'ngal',
        'sat_fracs': SAT_FRACS,
        'tau_methods': [
            {
                "projection_type": "simple",
                "tau_method": "2D_FT_upgrade_tau_reconstruction",
                "gas_type": "strongest_AGN_reconstructed",
                "field_type": "gas",
            },
            {
                "projection_type": "simple",
                "tau_method": "fullFT_tau_reconstruction",
                "gas_type": "strongest_AGN",
                "field_type": "gas",
            },
            {
                "projection_type": "simple",
                "tau_method": "fullFT_tau_reconstruction",
                "gas_type": "strongest_AGN",
                "field_type": "dm",
            },
        ],
    },
]

"""{
        'name': 'm200b_centrals',
        'selection_params': {
            'selection_mode': 'cen',
            'selection_mass_def': 'm200b',
            'select_nonzero_masses': False,
            'upper_mass_cut': False,
            'max_mass': 1e14,
            'upper_radius_cut': False,
            'target_mean_mass': 13.5,  # Informational only; convergence done in HATF
        },
        'convergence_mode': 'massbin',
        'sat_fracs': None,
        'tau_methods': [
            {
                "projection_type": "simple",
                "tau_method": "2D_FT_upgrade_tau_reconstruction",
                "gas_type": "strongest_AGN_reconstructed",
                "field_type": "gas",
            },
            {
                "projection_type": "simple",
                "tau_method": "fullFT_tau_reconstruction",
                "gas_type": "strongest_AGN",
                "field_type": "gas",
            },
            {
                "projection_type": "simple",
                "tau_method": "fullFT_tau_reconstruction",
                "gas_type": "strongest_AGN",
                "field_type": "dm",
            },
        ],
    },"""

def build_profiles():
    """Expand profile configurations over all sat_frac values and tau methods."""
    profiles = []
    
    for config in PROFILE_CONFIGS:
        sel_params = dict(config['selection_params'])
        convergence_mode = config['convergence_mode']
        
        # Determine if we should iterate over sat_fracs
        if config['sat_fracs'] is not None:
            sat_frac_values = config['sat_fracs']
        else:
            # For non-mixed modes, use a single placeholder value
            # (won't appear in filenames anyway)
            sat_frac_values = [0.10]
        
        for sf in sat_frac_values:
            for base in config['tau_methods']:
                p = dict(base)
                p.update(sel_params)
                # Don't set values - let resolve_converged_param find the right files
                # based on the specified convergence mode
                p['n_gal_density']   = None
                p['halo_mass_range'] = None
                p['sat_frac']        = sf
                p = resolve_converged_param(p, mode=convergence_mode)  # Finds and parses the converged parameters
                profiles.append(p)
    
    return profiles

PROFILES = build_profiles()


def run_configuration(profile, config_num, total_configs):
    if rank == 0:
        print(f"\n{'='*70}")
        print(f"Config {config_num}/{total_configs}")
        print(f"  gas_type  : {profile['gas_type']}")
        print(f"  tau_method: {profile['tau_method']}")
        print(f"  field_type: {profile['field_type']}")
        print(f"  sel_mode  : {profile.get('selection_mode', 'N/A')}")
        print(f"  sel_mass  : {profile.get('selection_mass_def', 'N/A')}")
        print(f"  sat_frac  : {profile['sat_frac']*100:.0f}%")
        if profile.get('n_gal_density') is not None:
            print(f"  n_gal_dens: {profile['n_gal_density']:.3e} (converged)")
        if profile.get('halo_mass_range') is not None:
            if isinstance(profile['halo_mass_range'], list):
                print(f"  mass_range: start={profile['halo_mass_range'][0]}, size={profile['halo_mass_range'][1]} (converged)")
            else:
                print(f"  mass_range: {profile['halo_mass_range']} (converged)")
        print(f"  Start     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        config = {
            'gas_type':              profile['gas_type'],
            'tau_method':            profile['tau_method'],
            'field_type':            profile.get('field_type', 'gas'),
            'z_real':                Z_SIM,
            'n_gal_density':         profile['n_gal_density'],
            'halo_mass_range':       profile['halo_mass_range'],
            'selection_mode':        profile['selection_mode'],
            'selection_mass_def':    profile['selection_mass_def'],
            'select_nonzero_masses': profile['select_nonzero_masses'],
            'upper_mass_cut':        profile['upper_mass_cut'],
            'max_mass':              profile['max_mass'],
            'upper_radius_cut':      profile['upper_radius_cut'],
            'sat_frac':              profile['sat_frac'],
            'target_mean_mass':      profile.get('target_mean_mass'),
        }

        if 'A' in profile:
            config['A'] = profile['A']

        if profile['projection_type'] == 'simple':
            if rank == 0:
                print("  Running simple projection...")
            simple_module.get_AP_simple(config)
        else:
            raise ValueError(f"Unknown projection_type: {profile['projection_type']}")

        if rank == 0:
            print(f"✓ Config {config_num}/{total_configs} completed  "
                  f"[{datetime.now().strftime('%H:%M:%S')}]")

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

