import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import rcParams
from astropy.cosmology import FlatLambdaCDM
from astropy.constants import codata2018 as const
import astropy.units as u
from datetime import datetime
from utils.pipeline_paths import (
    ap_output_path, plot_path, ensure_parents, get_halo_file_path,
    halo_props_cache_path as _halo_props_cache_path,
    is_massbin_config,
)
from utils.catalog_loaders import load_halo_properties
from colossus.halo import concentration
from colossus.halo import mass_defs
from colossus.cosmology import cosmology
from utils.sim_params import get_sim_params, require_sim_param

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Computer Modern']
rcParams['text.usetex'] = False  # Disable LaTeX rendering (requires system TeX installation)

# --------------------
# Configuration
# --------------------
C = 2.99792e5 # km/s

PLOT_DATA = False
RATIO_MODE = True

SAT_FRACS = [0.01, 0.10, 0.20, 0.30]

# Define profile configurations: each configuration specifies selection parameters,
# convergence mode, and which tau methods/field types to include
PROFILE_CONFIGS = [
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
            {
                "projection_type": "simple",
                "tau_method": "fullFT_tau_reconstruction",
                "gas_type": "fiducial",
                "field_type": "gas",
            },
        ],
    },
    {
        'name': 'abacus_m200b_cen_massbin',
        'sim_name': 'abacus',
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
]

"""{
        'name': 'm200b_massbin',
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
                "tau_method": "2D_FT_upgrade_tau_reconstruction",
                "gas_type": "strongest_AGN_reconstructed",
                "field_type": "gas",
            },
        ],
    },
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
        ],
    },
    {
        'name': 'm200b_cen',
        'selection_params': {
            'selection_mode': 'cen',
            'selection_mass_def': 'm200b',
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
                "tau_method": "2D_FT_upgrade_tau_reconstruction",
                "gas_type": "fiducial_reconstructed",
                "field_type": "gas",
            },
            {
                "projection_type": "simple",
                "tau_method": "fullFT_tau_reconstruction",
                "gas_type": "fiducial",
                "field_type": "gas",
            },
        ],
    },"""

def build_profiles():
    """Expand profile configurations over all sat_frac values and tau methods."""
    profiles = []
    
    for config in PROFILE_CONFIGS:
        sel_params = dict(config['selection_params'])
        
        # Determine if we should iterate over sat_fracs
        # Only iterate when sat_fracs is not None (typically for 'mixed' mode)
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
                p['name']            = config['name']  # Add name from config for simplified labels
                p.setdefault('sim_name', config.get('sim_name', 'flamingo'))
                profiles.append(p)
    
    return profiles

PROFILES = build_profiles()


def M200c_to_M200m(M200c, z):
    """
    Convert M200c to M200m using colossus.
    
    Parameters:
    -----------
    M200c : float
        Halo mass in M200c definition (Msun/h)
    z : float
        Redshift
    
    Returns:
    --------
    M200m : float
        Halo mass in M200m definition (Msun/h)
    """
    c200c = concentration.concentration(
        M200c, '200c', z, model='diemer19'
    )
    M200m, R200m, c200m = mass_defs.changeMassDefinition(
        M200c, c200c, z,
        mdef_in='200c',
        mdef_out='200m'
    )
    return M200m


def get_halo_data_for_dataset(config):
    """
    Load halo data from unified HDF5 source based on configuration.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary with keys:
        - gas_type: str
    
    Returns:
    --------
    all_m200b : ndarray
        Halo masses in m200b (m200m) definition
    all_m200c : ndarray
        Halo masses in M200c definition
    all_vels : ndarray
        Halo velocities
    all_mstell : ndarray
        Stellar masses
    all_r200b : ndarray
        Halo r200b values from catalog
    """
    
    # Extract config values
    GAS_TYPE = config['gas_type']
    sim_name = config.get('sim_name', 'flamingo')
    
    # Get halo data path using pipeline_paths
    halo_galaxy_data_path = str(get_halo_file_path(GAS_TYPE, sim_name=sim_name))
    
    requested_props = ['m200b', 'm200c', 'hvel_200b', 'r200b']
    try:
        halo_props = load_halo_properties(
            halo_galaxy_data_path,
            requested_props + ['mstell_50kpc'],
            sim_name=sim_name,
        )
    except KeyError:
        halo_props = load_halo_properties(
            halo_galaxy_data_path,
            requested_props,
            sim_name=sim_name,
        )
    if halo_props is None:
        raise RuntimeError(f'Failed to load halo properties from {halo_galaxy_data_path}')
    halo_props['mstell_50kpc'] = np.full_like(halo_props['m200b'], np.nan, dtype=float)
    all_m200b = halo_props['m200b']
    all_m200c = halo_props['m200c']
    all_vels = halo_props['hvel_200b']
    all_mstell = halo_props['mstell_50kpc']
    all_r200b = halo_props['r200b']
    
    return all_m200b, all_m200c, all_vels, all_mstell, all_r200b


def load_halo_data_for_massbin(config):
    """
    Load halo data from the massbin props cache (no catalog open).

    Returns the same 5-tuple as get_halo_data_for_dataset:
        all_m200b, all_m200c, all_vels, all_mstell, all_r200b

    The arrays are already the selected subset.
    all_m200c and all_mstell are set to NaN arrays (not stored in cache).
    """
    cache_path = _halo_props_cache_path(config)
    if not cache_path.exists():
        raise FileNotFoundError(
            f"massbin halo props cache not found: {cache_path}\n"
            f"Run HATF with convergence_mode='massbin' first."
        )
    cached = np.load(cache_path)
    m200b  = cached['m200b']
    vels   = cached['vel']
    r200b  = cached['r200b']
    m200c  = m200b
    mstell = m200b
    return m200b, m200c, vels, mstell, r200b

def compute_temperature_signal_pixell(tau_xy_inner, tau_xy_outer, t_cmb_microK):
    delta_tau = tau_xy_inner - tau_xy_outer  # (cMpc/h)^2
    return -t_cmb_microK * delta_tau


def compute_temperature_signal_simple(tau_signals, t_cmb_microK):
    return -t_cmb_microK * tau_signals


def weighted_stacking(t_signals, vel_los, sigma_i=1.0, r_v=1.0):
    # Constants
    v_rms = np.std(vel_los) # RMS of the velocity catalog
    
    # Weights w_i = 1 / sigma_i^2
    weights = 1.0 / (sigma_i**2)
    
    # Term: (v_i / c)
    v_term = vel_los / C

    numerator = np.sum(t_signals * weights, axis=0) * np.std((vel_los[:, np.newaxis] / C) * v_term[:, np.newaxis])
    denominator = np.sum((v_term**2) * weights, axis=0)
    
    if denominator == 0:
        return 0.0
        
    T_stacked = - (1.0 / r_v) * (v_rms / C) * (numerator / denominator)
    
    return T_stacked


def compute_distances(cosmo, z, h):
    d_L = cosmo.luminosity_distance(z).to(u.Mpc).value
    d_C = d_L / (1.0 + z)                      # Mpc
    d_A = d_L / (1.0 + z) ** 2                 # Mpc
    d_C *= h                                   # Mpc/h
    d_A *= h                                   # Mpc/h
    return d_A, d_C


def compute_profile_errors(T_signals, N_halos):
    std_per_aperture = np.std(T_signals, axis=0)
    errors = std_per_aperture / np.sqrt(N_halos)
    return errors


def get_kSZ_profile(config, use_dm=False):
    """
    Get kSZ profile for a single configuration.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary with keys required by pipeline_paths
    use_dm : bool
        If True, load DM-only profile (changes field_type to 'dm')
    """
    # Prepare config for ap_output_path
    load_config = dict(config)

    sim_name = config.get('sim_name', 'flamingo')
    h = require_sim_param(sim_name, 'h')
    om_m = require_sim_param(sim_name, 'omega_m')
    om_b = require_sim_param(sim_name, 'omega_b')
    sigma8 = require_sim_param(sim_name, 'sigma8')
    n_s = require_sim_param(sim_name, 'n_s')
    redshift = require_sim_param(sim_name, 'redshift')
    x_h = require_sim_param(sim_name, 'X_H')
    x_he = require_sim_param(sim_name, 'X_He')
    t_cmb_microK = require_sim_param(sim_name, 'tcmb0') * 1e6

    params = {
        'flat': True,
        'H0': h * 100,
        'Om0': om_m,
        'Ob0': om_b,
        'sigma8': sigma8,
        'ns': n_s,
    }
    cosmology.setCosmology('myCosmo', params)
    
    if use_dm:
        load_config['field_type'] = 'dm'
        load_config['tau_method'] = 'fullFT_tau_reconstruction'
        # For DM, strip '_reconstructed' from gas_type
        load_config['gas_type'] = config['gas_type'].replace('_reconstructed', '')
    
    # Determine method based on projection_type
    projection_type = config.get('projection_type', 'simple')
    method = 'pixell' if projection_type in ['pixell', 'pixell_cea'] else 'simple'
    
    # Get path using pipeline_paths
    tau_apertures_path = str(ap_output_path(load_config, method=method))
    print(f"Loading tau data from: {tau_apertures_path}")
    
    try:
        data = np.load(tau_apertures_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Tau data file not found: {tau_apertures_path}")
    
    if config['projection_type'] in ['pixell', 'pixell_cea']:
        tau_xy_inner = data['tau_xy_inner']
        tau_xy_outer = data['tau_xy_outer']
        inds_sub = data['inds_sub']
        aperture_radii = data.get('r_comoving_mpc_h', np.linspace(0.1, 3.0, 9))  # cMpc/h

        if is_massbin_config(config):
            # Massbin mode: load from cache (already selected subset)
            print("Loading halo data from massbin cache")
            all_m200b_full, all_m200c_full, all_vels_full, all_mstell_full, all_r200b_full = \
                load_halo_data_for_massbin(config)
            # Cache arrays are already the selected subset, no indexing needed
            halo_vels = all_vels_full.copy()
            halo_m200b = all_m200b_full
            halo_m200c = all_m200c_full
            halo_mstell = all_mstell_full
            halo_r200b = all_r200b_full
        else:
            # Ngal mode: load from full catalog and subset with inds_sub
            all_m200b_full, all_m200c_full, all_vels_full, all_mstell_full, all_r200b_full = \
                get_halo_data_for_dataset(config)
            halo_vels = all_vels_full[inds_sub].copy()  # Copy to avoid modifying original array
            halo_m200b = all_m200b_full[inds_sub]
            halo_m200c = all_m200c_full[inds_sub]
            halo_mstell = all_mstell_full[inds_sub]
            halo_r200b = all_r200b_full[inds_sub]
        
        los_vels = halo_vels[:, 2]  # km/s

        print(f"Mean stellar mass: {np.log10(np.mean(halo_mstell)):.2e} log10(Msun/h)")
        
        T_signals = compute_temperature_signal_pixell(tau_xy_inner, tau_xy_outer, t_cmb_microK)
        stacked_kSZ_signal = weighted_stacking(T_signals, los_vels)

        one_halo_signal, mean_r200b_mpc_comoving = get_one_halo_term(
            halo_m200c,
            halo_vels,
            halo_r200b,
            redshift,
            h,
            om_m,
            om_b,
            x_h,
            x_he,
            t_cmb_microK,
        )
        
        mean_halo_m200b = np.mean(halo_m200b)  # Msun/h
        N_halos = len(inds_sub)

    else:
        tau_signals = data['tau_signals']
        inds_sub = data['inds_sub']
        aperture_radii = data.get('r_comoving_mpc_h')

        if is_massbin_config(config):
            print("Loading halo data from massbin cache")
            all_m200b_full, all_m200c_full, all_vels_full, all_mstell_full, all_r200b_full = \
                load_halo_data_for_massbin(config)
            # Cache arrays are already the selected subset, no indexing needed
            halo_vels = all_vels_full.copy()
            halo_m200b = all_m200b_full
            halo_m200c = all_m200c_full
            halo_mstell = all_mstell_full
            halo_r200b = all_r200b_full
        else:
            # Ngal mode: load from full catalog and subset with inds_sub
            all_m200b_full, all_m200c_full, all_vels_full, all_mstell_full, all_r200b_full = \
                get_halo_data_for_dataset(config)
            halo_vels = all_vels_full[inds_sub].copy()  # Copy to avoid modifying original array
            halo_m200b = all_m200b_full[inds_sub]
            halo_m200c = all_m200c_full[inds_sub]
            halo_mstell = all_mstell_full[inds_sub]
            halo_r200b = all_r200b_full[inds_sub]
        
        los_vels = halo_vels[:, 2]  # km/s

        print(f"Mean stellar mass: {np.log10(np.mean(halo_mstell)):.2e} log10(Msun/h)")

        T_signals = compute_temperature_signal_simple(tau_signals, t_cmb_microK)
        stacked_kSZ_signal = weighted_stacking(T_signals, los_vels)

        one_halo_signal, mean_r200b_mpc_comoving = get_one_halo_term(
            halo_m200c,
            halo_vels,
            halo_r200b,
            redshift,
            h,
            om_m,
            om_b,
            x_h,
            x_he,
            t_cmb_microK,
        )
        
        mean_halo_m200b = np.mean(halo_m200b)  # Msun/h
        N_halos = len(inds_sub)

    return aperture_radii, stacked_kSZ_signal, one_halo_signal, mean_r200b_mpc_comoving, mean_halo_m200b, N_halos, T_signals

def get_one_halo_term(halo_m200c, halo_vels, halo_r200b, z, h, om_m, om_b, x_h, x_he, t_cmb_microK):

    mean_halo_m200c = np.mean(halo_m200c) / h # Msun
    vel_rms = np.std(halo_vels[:, 2])   # km/s 

    print(f"Mean m200c: {np.log10(mean_halo_m200c):.2e} Msun, Velocity RMS: {vel_rms:.2f} km/s")

    mean_r200b_mpc_comoving = np.mean(halo_r200b)
    print(f"Mean r200b from catalog: {mean_r200b_mpc_comoving:.3f} cMpc/h")

    mean_gas_mass = (om_b / om_m) * mean_halo_m200c * u.Msun # Msun
    f_e = (x_h + 0.5 * x_he) / const.m_p # 1/kg
    N_e = (mean_gas_mass.to(u.kg) * f_e).to(u.dimensionless_unscaled).value
    v_factor = (vel_rms / C)
    sigma_T_mpc2 = const.sigma_T.value / (u.Mpc.to(u.m) ** 2) # Mpc^2
    delta_T_halo = t_cmb_microK * v_factor * sigma_T_mpc2 * N_e * ((1 + z) * h)**2 # microK (cMpc/h)^2
    
    return delta_T_halo, mean_r200b_mpc_comoving


def generate_comparison_filename(profiles):
    """
    Generate a descriptive filename based on common and varying profile parameters.
    Automatically detects the comparison subject (varying parameter).
    
    Args:
        profiles: List of profile dictionaries
    
    Returns:
        tuple: (filename_string, comparison_subject_string, is_single_profile)
            - filename_string: Filename without extension
            - comparison_subject_string: The parameter(s) being compared (e.g., 'ngrid' or 'ngrid+tau_method')
            - is_single_profile: True if only one profile is being plotted
    """
    if len(profiles) == 0:
        return "kSZ_profile_comparison", "unknown", False
    
    is_single_profile = len(profiles) == 1
    
    # Define all possible keys (removed constants: ngrid, JAX, z, smoothed, selection_regime)
    keys = ['projection_type', 'tau_method', 'gas_type', 'halo_mass_range', 'sat_frac', 'A', 
            'selection_mode', 'selection_mass_def', 'upper_mass_cut', 'upper_radius_cut']
    
    # Find common and varying parameters
    common_params = {}
    varying_params = {}
    
    for key in keys:
        values = [str(p.get(key)) for p in profiles]
        unique_values = list(dict.fromkeys(values))  # Preserve order, remove duplicates
        
        # Skip if all values are None
        if len(unique_values) == 1 and unique_values[0] == 'None':
            continue
        
        if len(unique_values) == 1:
            # All profiles have the same value for this parameter
            common_params[key] = unique_values[0]
        else:
            # Profiles differ in this parameter
            varying_params[key] = unique_values
    
    # Determine comparison subject (what's varying)
    # Priority order for naming when multiple params vary
    comparison_priority = ['sat_frac', 'halo_mass_range', 'gas_type', 'tau_method', 'projection_type', 
                          'selection_mode', 'selection_mass_def', 'upper_mass_cut', 'upper_radius_cut', 'A']
    comparison_subjects = [key for key in comparison_priority if key in varying_params]
    
    if len(comparison_subjects) == 0:
        comparison_subject = "single" if is_single_profile else "identical"
    elif len(comparison_subjects) == 1:
        comparison_subject = comparison_subjects[0]
    else:
        comparison_subject = "+".join(comparison_subjects)
    
    # Build filename parts
    filename_parts = ['kSZ_over_DM'] if RATIO_MODE else ['kSZ']
    
    # For single profile, include all relevant parameters
    if is_single_profile:
        param_order = ['projection_type', 'tau_method', 'gas_type', 'halo_mass_range', 'sat_frac', 'A',
                      'selection_mode', 'selection_mass_def', 'upper_mass_cut', 'upper_radius_cut']
        for key in param_order:
            value = common_params.get(key)
            if value is None:
                continue
            if key == 'halo_mass_range' and value != 'None':
                filename_parts.append(f"massbin{value}")
            elif key == 'sat_frac':
                filename_parts.append(f"sf{int(float(value)*100):02d}")
            elif key == 'A':
                filename_parts.append(f"A{value}")
            elif key in ['selection_mode', 'selection_mass_def', 'upper_mass_cut', 'upper_radius_cut']:
                # Skip internal selection parameters in filename (they're in the path)
                continue
            elif key not in ['halo_mass_range']:
                filename_parts.append(value)
    else:
        # For multiple profiles, add common parameters
        param_order = ['projection_type', 'tau_method', 'gas_type', 'halo_mass_range', 'sat_frac', 'A']
        for key in param_order:
            if key in common_params:
                value = common_params[key]
                if key == 'halo_mass_range' and value != 'None':
                    filename_parts.append(f"massbin{value}")
                elif key == 'sat_frac':
                    filename_parts.append(f"sf{int(float(value)*100):02d}")
                elif key == 'A':
                    filename_parts.append(f"A{value}")
                elif key not in ['halo_mass_range']:
                    filename_parts.append(value)
        
        # Add varying parameters with their values
        for key in ['sat_frac', 'projection_type', 'tau_method', 'gas_type', 'halo_mass_range', 'A',
                   'selection_mode', 'selection_mass_def', 'upper_mass_cut', 'upper_radius_cut']:
            if key in varying_params:
                values_str = '-'.join(varying_params[key])
                if key == 'sat_frac':
                    # Convert to percentages: 0.10 -> 10, 0.20 -> 20
                    pct_values = [f"{int(float(v)*100):02d}" for v in varying_params[key]]
                    filename_parts.append(f"sf_{'_'.join(pct_values)}")
                elif key == 'halo_mass_range':
                    filename_parts.append(f"massbin_{values_str}")
                elif key == 'A':
                    filename_parts.append(f"A_{values_str}")
                elif key in ['selection_mode', 'selection_mass_def', 'upper_mass_cut', 'upper_radius_cut']:
                    # Skip internal selection parameters (they're in the path)
                    continue
                else:
                    filename_parts.append(f"{key}_{values_str}")
    
    filename = '_'.join(filename_parts)
    return filename, comparison_subject, is_single_profile


def _format_sat_frac(sf_value):
    if sf_value is None:
        return '0'
    sf_float = float(sf_value)
    if np.isclose(sf_float, 0.0):
        return '0'
    return f"{sf_float:.2f}"


def format_profile_label(profile, all_profiles):
    """Build plot labels focused on selection type and satellite fraction."""
    profile_name = profile.get('name', '')
    selection_mode = profile.get('selection_mode')

    if profile_name == 'm200b_massbin':
        base_label = 'Mass bin selected sample, sf=0'
    elif profile_name == 'mstell_cen':
        base_label = 'Number density selected sample, sf = 0'
    elif profile_name == 'mstell_mixed' or selection_mode == 'mixed':
        sat_frac_label = _format_sat_frac(profile.get('sat_frac', 0.0))
        base_label = f"Number density selected sample, sf = {sat_frac_label}"
    else:
        sat_frac_label = _format_sat_frac(profile.get('sat_frac', 0.0))
        base_label = f"{profile_name.replace('_', ' ')} sample, sf={sat_frac_label}"

    all_strongest_agn_reconstructed = (
        len(all_profiles) > 0
        and all(p.get('gas_type') == 'strongest_AGN_reconstructed' for p in all_profiles)
    )

    if all_strongest_agn_reconstructed:
        return base_label

    gas_type = profile.get('gas_type', 'profile')
    gas_label_map = {
        'fiducial': 'Fiducial',
        'strongest_AGN': 'Strongest AGN',
    }
    is_reconstructed = gas_type.endswith('_reconstructed')
    base_gas_type = gas_type.replace('_reconstructed', '')
    gas_label = gas_label_map.get(base_gas_type, base_gas_type.replace('_', ' '))
    state_label = 'reconstructed' if is_reconstructed else 'true'

    return f"{base_label} ({gas_label}, {state_label})"


def main():
    # Print configuration summary
    print("\n" + "=" * 70)
    print("TkSZ PROFILES PLOTTER")
    print("=" * 70)
    print(f"Total profiles        : {len(PROFILES)}")
    print(f"Ratio mode            : {RATIO_MODE}")
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
    print(f"\nStart time            : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70 + "\n")
    
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)

    figure_content = np.load("/home/fb635/fedirfiles/tracing_cosmic_gas/data/Fig2_sim.npz")

    theta_arcmin = figure_content['theta_arcmins']
    signal = figure_content['signal']  # μK arcmin²
    noise = figure_content['noise']    # μK arcmin²
    gas_illustris = figure_content['gas_illustris']
    dm_tng = figure_content['dm_tng']

    data_sim = PROFILES[0].get('sim', 'flamingo') if len(PROFILES) > 0 else 'flamingo'
    data_h = require_sim_param(data_sim, 'h')
    data_om_m = require_sim_param(data_sim, 'omega_m')
    data_tcmb0 = require_sim_param(data_sim, 'tcmb0')
    data_z_sim = require_sim_param(data_sim, 'redshift')
    cosmo = FlatLambdaCDM(H0=data_h * 100, Om0=data_om_m, Tcmb0=data_tcmb0)
    _, d_C = compute_distances(cosmo, data_z_sim, data_h)  # cMpc/h
    theta_rad = theta_arcmin * (np.pi / 180) / 60  # radians
    r_comoving_mpc = d_C * theta_rad # cMpc/h

    conversion_factor = (r_comoving_mpc / theta_arcmin) ** 2  # (cMpc/h)² per arcmin²
    signal_converted = signal * conversion_factor  # μK (cMpc/h)²
    noise_converted = noise * conversion_factor    # μK (cMpc/h)²

    if PLOT_DATA:
        ax.errorbar(r_comoving_mpc, signal_converted, yerr=noise_converted, fmt='o', label='Observed Signal with Error Bars', color='red', ecolor='red', elinewidth=2, capsize=3)
    
    # PLOTTING PROFILES
    
    # Auto-detect comparison subject
    filename, comparison_subject, is_single_profile = generate_comparison_filename(PROFILES)
    print(f"Detected comparison subject: {comparison_subject}")

    # Get mean mass from first profile
    first_profile_data = get_kSZ_profile(PROFILES[0])
    mean_mass_m200b_msun_h = first_profile_data[4]  # Msun/h
    mean_mass_log = np.log10(mean_mass_m200b_msun_h)

    # Store one-halo terms and mean r200 for each profile
    profile_one_halo_terms = []
    profile_mean_r200b_values = []
    profile_colors = []

    # Strong, high-contrast color palette for profiles (exclude brown)
    profile_palette_base = [
        'tab:blue',
        'tab:pink',
        'tab:orange',
        'tab:red',
        'darkred',
        'tab:cyan',
        'tab:purple',
        'tab:olive',
        'tab:gray',
    ]
    if len(PROFILES) == 2:
        profile_palette = ['tab:red', 'tab:blue']
    else:
        profile_palette = [
            profile_palette_base[i % len(profile_palette_base)]
            for i in range(len(PROFILES))
        ]

    for i, profile in enumerate(PROFILES):
        print(f"Processing sim_name: {profile['sim_name']}")
        radii, kSZ_signal, one_halo_signal, mean_r200b_mpc_comoving, mean_halo_m200b, N_halos, T_signals = get_kSZ_profile(profile)

        print(f"Mean m200b: {np.log10(mean_halo_m200b)} log10(Msun/h)")
        
        # If RATIO_MODE is enabled, compute the ratio with DM profile
        if RATIO_MODE:
            radii_dm, kSZ_signal_dm, one_halo_signal_dm, _, _, _, T_signals_dm = get_kSZ_profile(profile, use_dm=True)
            # Compute ratio: CAP kSZ / CAP DM
            plot_signal = kSZ_signal / kSZ_signal_dm
            one_halo_signal_plot = one_halo_signal / one_halo_signal_dm
            # Propagate errors for ratio (using same observed errors scaled by ratio)
            error_bars = noise_converted / np.abs(kSZ_signal_dm)
        else:
            plot_signal = kSZ_signal
            one_halo_signal_plot = one_halo_signal
            error_bars = noise_converted

        label = format_profile_label(profile, PROFILES)
        line_color = 'red' if is_single_profile else profile_palette[i]
        
        # Use noise from data as error bars
        #profile_errors = compute_profile_errors(T_signals, N_halos)
        line = ax.errorbar(
            radii,
            plot_signal,
            yerr=error_bars,
            fmt='-',
            label=label,
            color=line_color,
            ecolor=line_color,
            linewidth=2.3,
            elinewidth=1.4,
            alpha=1.0,
        )
        
        # Store one-halo term and mean r200 with matching color
        profile_one_halo_terms.append(one_halo_signal_plot)
        profile_mean_r200b_values.append(mean_r200b_mpc_comoving)
        profile_colors.append(line[0].get_color())
    
    # Add mean mass as a legend entry
    ax.plot([], [], ' ', label=f"$\\langle M_{{200m}} \\rangle = 10^{{{mean_mass_log:.2f}}}$ M$_\\odot$/h")
    
    # Plot one-halo term and mean r200b (only from first profile)
    if len(profile_one_halo_terms) > 0:
        if RATIO_MODE:
            ax.axhline(y=1.0, linestyle='--', color='gray', alpha=0.7, linewidth=1.5)
        else:
            ax.axhline(y=profile_one_halo_terms[0], linestyle='--', color='gray', alpha=0.7, linewidth=1.5, label='One-halo term line')
        ax.axvline(x=profile_mean_r200b_values[0], linestyle='--', color='gray', alpha=0.7, linewidth=1.5)
    
    ax.set_xlabel(r"Aperture Radius $R$ [cMpc/h]", fontsize=14)
    if RATIO_MODE:
        ax.set_ylabel(r"CAP kSZ / CAP DM", fontsize=14)
        ax.set_ylim(0, 2)
    else:
        ax.set_ylabel(r"$T_{\mathrm{kSZ}}$ ($\mu$K (cMpc/h)$^{2}$)", fontsize=14)
    ax.legend(fontsize=6, loc="lower right")
    if not RATIO_MODE:
        ax.set_yscale('log')
    fig.tight_layout()
    
    # Determine gas_type and tau_method directory components
    all_gas_types   = list(dict.fromkeys(p['gas_type']   for p in PROFILES))
    all_tau_methods = list(dict.fromkeys(p['tau_method'] for p in PROFILES))
    gas_dir    = all_gas_types[0]   if len(all_gas_types)   == 1 else 'multi'
    method_dir = all_tau_methods[0] if len(all_tau_methods) == 1 else 'multi'
    
    category = 'ksz_profiles/cap_ksz_over_dm' if RATIO_MODE else 'ksz_profiles/cap_ksz'
    output_path = plot_path(category, gas_dir, method_dir, stem=filename)
    ensure_parents(output_path)
    
    fig.savefig(str(output_path))
    print(f"Plot saved to: {output_path}")


if __name__ == "__main__":
    main()

