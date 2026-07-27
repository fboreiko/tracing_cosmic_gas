"""
Single source of truth for pipeline profile configurations.

Both the runner (`get_AP_multiple_configs.py`) and the plotter
(`TkSZ_profiles.py`) import PROFILE_CONFIGS and build_profiles from here, so
"what the runner computed" and "what the plotter plots" cannot drift apart.

Each script selects its own working subset via its own ACTIVE_PROFILE_NAMES
constant.

This module imports the standard library and utils.pipeline_config only — no
mpi4py, no matplotlib, no pipeline stage — so it stays cheap to import.
"""

from utils.pipeline_config import PipelineConfig

SAT_FRACS = [0.10, 0.20, 0.30]

# Profile configurations: each specifies selection parameters, convergence
# mode, and which tau methods / field types to include.
#
# Entries 1-2 were moved verbatim from get_AP_multiple_configs.py;
# entries 3-5 verbatim from TkSZ_profiles.py.
PROFILE_CONFIGS = [
    {
        'name': 'flamingo_cen_massbin',
        'sim_name': 'flamingo',
        'selection_params': {
            'selection_mode': 'cen',
            'selection_mass_def': 'm200b',
            'select_nonzero_masses': True,
            'upper_mass_cut': False,
            'max_mass': 1e14,
            'upper_radius_cut': False,
            'target_mean_mass': 12.2,
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
    },
    {
        'name': 'flamingo_cen_massbin',
        'sim_name': 'flamingo',
        'selection_params': {
            'selection_mode': 'cen',
            'selection_mass_def': 'm200b',
            'select_nonzero_masses': True,
            'upper_mass_cut': False,
            'max_mass': 1e14,
            'upper_radius_cut': False,
            'target_mean_mass': 13.8,
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
    },   
    {
        'name': 'flamingo_m200b_cen_massbin',
        'sim_name': 'flamingo',
        'selection_params': {
            'selection_mode': 'cen',
            'selection_mass_def': 'm200b',
            'select_nonzero_masses': True,
            'upper_mass_cut': False,
            'max_mass': 1e14,
            'upper_radius_cut': False,
            'target_mean_mass': 13.2,  # Informational only; convergence done in HATF
        },
        'convergence_mode': 'massbin',
        'sat_fracs': None,  # Not used for centrals-only mode; sat_frac doesn't affect selection or filenames
        'tau_methods': [
            {
                "projection_type": "simple",
                "tau_method": "fullFT_tau_reconstruction",
                "gas_type": "strongest_AGN",
                "field_type": "gas",
            },

        ],
    },
    {
        'name': 'flamingo_mstell_mixed_ngal',
        'sim_name': 'flamingo',
        'selection_params': {
            'selection_mode': 'mixed',
            'selection_mass_def': 'mstell',
            'select_nonzero_masses': True,
            'upper_mass_cut': False,
            'max_mass': 1e14,
            'upper_radius_cut': False,
            'target_mean_mass': 13.2,
        },
        'convergence_mode': 'ngal',
        'sat_fracs': SAT_FRACS,  # Not used for centrals-only mode; sat_frac doesn't affect selection or filenames
        'tau_methods': [
            {
                "projection_type": "simple",
                "tau_method": "fullFT_tau_reconstruction",
                "gas_type": "strongest_AGN",
                "field_type": "gas",
            },

        ],
    },
    {
        'name': 'flamingo_mstell_cen_ngal',
        'sim_name': 'flamingo',
        'selection_params': {
            'selection_mode': 'cen',
            'selection_mass_def': 'mstell',
            'select_nonzero_masses': True,
            'upper_mass_cut': False,
            'max_mass': 1e14,
            'upper_radius_cut': False,
            'target_mean_mass': 13.2,  # Informational only; convergence done in HATF
        },
        'convergence_mode': 'ngal',
        'sat_fracs': None,  # Not used for centrals-only mode; sat_frac doesn't affect selection or filenames
        'tau_methods': [
            {
                "projection_type": "simple",
                "tau_method": "fullFT_tau_reconstruction",
                "gas_type": "strongest_AGN",
                "field_type": "gas",
            },
        ],
        },
]


def build_profiles(configs, *, default_sim_name='flamingo'):
    """Expand profile configurations over all sat_frac values and tau methods.

    Replaces the two near-identical build_profiles() implementations that
    previously lived in the runner and the plotter. Where those two disagreed,
    the resolution is:

      * placeholder sat_frac is None (not the runner's 0.10, nor the plotter's
        0.0) when no sweep is requested — the sf token never enters a filename
        for those modes, so None is both schema-valid and honest;
      * the profile name is attached under 'name' (the plotter's key).

    Every produced profile is validated against PipelineConfig, so a malformed
    profile fails at definition time in both scripts identically.
    """
    profiles = []

    for config in configs:
        sel_params = dict(config['selection_params'])

        # Determine if we should iterate over sat_fracs
        # Only iterate when sat_fracs is not None (typically for 'mixed' mode)
        if config['sat_fracs'] is not None:
            sat_frac_values = config['sat_fracs']
        else:
            # A sat_frac sweep is mandatory for modes whose tag carries the sf
            # token; without it the runner and plotter could disagree on the
            # placeholder and silently look for different files.
            if sel_params.get('selection_mode') in ('mixed', 'sat'):
                raise ValueError(
                    f"Profile config {config.get('name', 'unnamed_profile')!r} has "
                    f"selection_mode={sel_params.get('selection_mode')!r} but "
                    f"sat_fracs is None; a sat_frac sweep is mandatory for "
                    f"'mixed' and 'sat' modes."
                )
            # For other modes sat_frac never appears in a filename, so carry an
            # honest None rather than a placeholder number.
            sat_frac_values = [None]

        for sf in sat_frac_values:
            for base in config['tau_methods']:
                p = dict(base)
                p.update(sel_params)
                # Set convergence parameters based on convergence_mode
                if config['convergence_mode'] == 'massbin':
                    # For massbin mode: halo_mass_range must be non-None
                    # Use placeholder [0, 1] since indices will be loaded from halo_indices file
                    p['halo_mass_range'] = [0, 1]
                    p['n_gal_density']   = None
                else:
                    # For ngal mode: both can be None (indices loaded from file)
                    p['n_gal_density']   = None
                    p['halo_mass_range'] = None
                p['sat_frac']        = sf
                p['name']            = config.get('name', 'unnamed_profile')
                p['convergence_mode'] = config.get('convergence_mode', 'ngal')
                p.setdefault('sim_name', config.get('sim_name', default_sim_name))
                profiles.append(p)

    # Schema handshake — fail at definition time, identically in both scripts.
    for index, profile in enumerate(profiles):
        try:
            PipelineConfig.from_dict(profile, require_selection=True)
        except ValueError as exc:
            raise ValueError(
                f"Profile {index} ({profile.get('name', 'unnamed_profile')!r}) "
                f"failed schema validation: {exc}"
            ) from exc

    return profiles


# --- Inactive historical configs (moved from get_AP_multiple_configs.py and TkSZ_profiles.py) ---

# From get_AP_multiple_configs.py:
"""
    {
        'name': 'flamingo_mstell_sat',
        'sim_name': 'flamingo',
        'selection_params': {
            'selection_mode': 'sat',
            'selection_mass_def': 'mstell',
            'select_nonzero_masses': True,
            'upper_mass_cut': False,
            'max_mass': 1e14,
            'upper_radius_cut': False,
            'target_mean_mass': None,
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
    {
        'name': 'abacus_m200b_cen_ngal',
        'sim_name': 'abacus',
        'selection_params': {
            'selection_mode': 'cen',
            'selection_mass_def': 'm200b',
            'select_nonzero_masses': True,
            'upper_mass_cut': False,
            'max_mass': 1e14,
            'upper_radius_cut': False,
            'target_mean_mass': None,  # Informational only; convergence done in HATF
        },
        'convergence_mode': 'ngal',
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
                "field_type": "dm",
            },
            {
                "projection_type": "simple",
                "tau_method": "2D_FT_upgrade_tau_reconstruction",
                "gas_type": "fiducial_reconstructed",
                "field_type": "gas",
            },
            {
                "projection_type": "simple",
                "tau_method": "fullFT_tau_reconstruction",
                "gas_type": "fiducial",
                "field_type": "dm",
            },
        ],
    }, """

# From TkSZ_profiles.py:
"""     {
        'name': 'abacus_m200b_cen_ngal',
        'sim_name': 'abacus',
        'selection_params': {
            'selection_mode': 'cen',
            'selection_mass_def': 'm200b',
            'select_nonzero_masses': True,
            'upper_mass_cut': False,
            'max_mass': 1e14,
            'upper_radius_cut': False,
            'target_mean_mass': None,  # Informational only; convergence done in HATF
        },
        'convergence_mode': 'ngal',
        'sat_fracs': None,  # Not used for centrals-only mode; sat_frac doesn't affect selection or filenames
        'tau_methods': [
            {
                "projection_type": "simple",
                "tau_method": "2D_FT_upgrade_tau_reconstruction",
                "gas_type": "strongest_AGN_reconstructed",
                "field_type": "gas",
            },
            {
                "projection_type": "simple",
                "tau_method": "2D_FT_upgrade_tau_reconstruction",
                "gas_type": "fiducial_reconstructed",
                "field_type": "gas",
            },
        ],
    },
    {
        'name': 'flamingo_mstell_cen_ngal',
        'sim_name': 'flamingo',
        'selection_params': {
            'selection_mode': 'cen',
            'selection_mass_def': 'mstell',
            'select_nonzero_masses': True,
            'upper_mass_cut': False,
            'max_mass': 1e14,
            'upper_radius_cut': False,
            'target_mean_mass': 13.2,  # Informational only; convergence done in HATF
        },
        'convergence_mode': 'ngal',
        'sat_fracs': None,  # Not used for centrals-only mode; sat_frac doesn't affect selection or filenames
        'tau_methods': [
            {
                "projection_type": "simple",
                "tau_method": "fullFT_tau_reconstruction",
                "gas_type": "strongest_AGN",
                "field_type": "gas",
            },
            {
                "projection_type": "simple",
                "tau_method": "fullFT_tau_reconstruction",
                "gas_type": "fiducial",
                "field_type": "gas",
            },
        ],
    },    
        {
        'name': 'flamingo_m200b_cen_massbin',
        'sim_name': 'flamingo',
        'selection_params': {
            'selection_mode': 'cen',
            'selection_mass_def': 'm200b',
            'select_nonzero_masses': True,
            'upper_mass_cut': False,
            'max_mass': 1e14,
            'upper_radius_cut': False,
            'target_mean_mass': 13.2,  # Informational only; convergence done in HATF
        },
        'convergence_mode': 'massbin',
        'sat_fracs': None,  # Not used for centrals-only mode; sat_frac doesn't affect selection or filenames
        'tau_methods': [
            {
                "projection_type": "simple",
                "tau_method": "fullFT_tau_reconstruction",
                "gas_type": "strongest_AGN",
                "field_type": "gas",
            },

        ],
    },
    {
        'name': 'flamingo_mstell_mixed_ngal',
        'sim_name': 'flamingo',
        'selection_params': {
            'selection_mode': 'mixed',
            'selection_mass_def': 'mstell',
            'select_nonzero_masses': True,
            'upper_mass_cut': False,
            'max_mass': 1e14,
            'upper_radius_cut': False,
            'target_mean_mass': 13.2,
        },
        'convergence_mode': 'ngal',
        'sat_fracs': SAT_FRACS,  # Not used for centrals-only mode; sat_frac doesn't affect selection or filenames
        'tau_methods': [
            {
                "projection_type": "simple",
                "tau_method": "fullFT_tau_reconstruction",
                "gas_type": "strongest_AGN",
                "field_type": "gas",
            },

        ],
    },
    {
        'name': 'abacus_m200b_cen_ngal',
        'sim_name': 'abacus',
        'selection_params': {
            'selection_mode': 'cen',
            'selection_mass_def': 'm200b',
            'select_nonzero_masses': True,
            'upper_mass_cut': False,
            'max_mass': 1e14,
            'upper_radius_cut': False,
            'target_mean_mass': None,  # Informational only; convergence done in HATF
        },
        'convergence_mode': 'ngal',
        'sat_fracs': None,  # Not used for centrals-only mode; sat_frac doesn't affect selection or filenames
        'tau_methods': [
            {
                "projection_type": "simple",
                "tau_method": "2D_FT_upgrade_tau_reconstruction",
                "gas_type": "strongest_AGN_reconstructed",
                "field_type": "gas",
            },
            {
                "projection_type": "simple",
                "tau_method": "2D_FT_upgrade_tau_reconstruction",
                "gas_type": "fiducial_reconstructed",
                "field_type": "gas",
            },
        ],
    },
"""