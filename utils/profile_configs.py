"""
Single source of truth for pipeline profile configurations.

Both the runner (`get_AP_multiple_configs.py`) and the plotter
(`TkSZ_profiles.py`) import PROFILE_CONFIGS and build_profiles from here, so
"what the runner computed" and "what the plotter plots" cannot drift apart.

v2 changes (see NAMING_REFACTOR.md)
-----------------------------------
* 'tau_methods' -> 'tau_variants'; each entry is now
      {'tau_source': 'truth'|'recon', 'tracer': 'gas'|'dm'}
  The feedback variant is declared ONCE per profile, not repeated on every
  entry with a '_reconstructed' suffix that had to be kept in sync by hand.
* 'name' is now enforced unique - it is the key ACTIVE_PROFILE_NAMES selects
  on. Two v1 entries both called 'flamingo_cen_massbin' meant selecting that
  name silently ran six profiles across two different target masses.
"""

from utils.pipeline_config import PipelineConfig

SAT_FRACS = [0.10, 0.20, 0.30]

# The comparison that nearly every profile wants: reconstructed gas against
# true gas, plus the true DM field as a reference.
TRUTH_VS_RECON = [
    {'tau_source': 'recon', 'tracer': 'gas'},
    {'tau_source': 'truth', 'tracer': 'gas'},
    {'tau_source': 'truth', 'tracer': 'dm'},
]

TRUTH_GAS_ONLY = [
    {'tau_source': 'truth', 'tracer': 'gas'},
]

PROFILE_CONFIGS = [
    {
        'name': 'flamingo_cen_massbin_tm12p2',
        'sim_name': 'flamingo',
        'feedback': 'strongest_AGN',
        'projection_type': 'simple',
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
        'tau_variants': TRUTH_VS_RECON,
    },
    {
        'name': 'flamingo_cen_massbin_tm13p8',
        'sim_name': 'flamingo',
        'feedback': 'strongest_AGN',
        'projection_type': 'simple',
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
        'tau_variants': TRUTH_VS_RECON,
    },
    {
        'name': 'flamingo_m200b_cen_massbin_tm13p2',
        'sim_name': 'flamingo',
        'feedback': 'strongest_AGN',
        'projection_type': 'simple',
        'selection_params': {
            'selection_mode': 'cen',
            'selection_mass_def': 'm200b',
            'select_nonzero_masses': True,
            'upper_mass_cut': False,
            'max_mass': 1e14,
            'upper_radius_cut': False,
            'target_mean_mass': 13.2,
        },
        'convergence_mode': 'massbin',
        'sat_fracs': None,
        'tau_variants': TRUTH_GAS_ONLY,
    },
    {
        'name': 'flamingo_mstell_mixed_ngal',
        'sim_name': 'flamingo',
        'feedback': 'strongest_AGN',
        'projection_type': 'simple',
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
        'sat_fracs': SAT_FRACS,
        'tau_variants': TRUTH_GAS_ONLY,
    },
    {
        'name': 'flamingo_mstell_cen_ngal',
        'sim_name': 'flamingo',
        'feedback': 'strongest_AGN',
        'projection_type': 'simple',
        'selection_params': {
            'selection_mode': 'cen',
            'selection_mass_def': 'mstell',
            'select_nonzero_masses': True,
            'upper_mass_cut': False,
            'max_mass': 1e14,
            'upper_radius_cut': False,
            'target_mean_mass': 13.2,
        },
        'convergence_mode': 'ngal',
        'sat_fracs': None,
        'tau_variants': TRUTH_GAS_ONLY,
    },
]


def build_profiles(configs, *, default_sim_name='flamingo'):
    """Expand profile configurations over all sat_frac values and tau variants.

    Every produced profile is validated against PipelineConfig, so a malformed
    profile fails at definition time in both the runner and the plotter.
    """
    names = [c['name'] for c in configs]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(
            f"Duplicate profile name(s): {duplicates}. 'name' is the key that "
            f"ACTIVE_PROFILE_NAMES selects on and must be unique."
        )

    profiles = []
    for config in configs:
        sel_params = dict(config['selection_params'])

        if config['sat_fracs'] is not None:
            sat_frac_values = config['sat_fracs']
        else:
            if sel_params.get('selection_mode') in ('mixed', 'sat'):
                raise ValueError(
                    f"Profile {config['name']!r} has selection_mode="
                    f"{sel_params.get('selection_mode')!r} but sat_fracs is None; "
                    f"a sat_frac sweep is mandatory for 'mixed' and 'sat' modes."
                )
            sat_frac_values = [None]

        for sf in sat_frac_values:
            for variant in config['tau_variants']:
                p = dict(variant)
                p.update(sel_params)

                if config['convergence_mode'] == 'massbin':
                    # Placeholder; the real indices come from the halo_indices file.
                    p['halo_mass_range'] = [0, 1]
                else:
                    p['halo_mass_range'] = None
                p['n_gal_density'] = None

                p['sat_frac'] = sf
                p['name'] = config['name']
                p['convergence_mode'] = config['convergence_mode']
                p['feedback'] = config['feedback']
                p['projection_type'] = config.get('projection_type', 'simple')
                p.setdefault('sim_name', config.get('sim_name', default_sim_name))
                profiles.append(p)

    for index, profile in enumerate(profiles):
        try:
            PipelineConfig.from_dict(profile, require_selection=True)
        except ValueError as exc:
            raise ValueError(
                f"Profile {index} ({profile['name']!r}) failed schema validation: {exc}"
            ) from exc

    return profiles
