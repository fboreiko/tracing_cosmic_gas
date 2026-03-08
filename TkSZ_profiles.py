import numpy as np
import deepdish as dd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from astropy.cosmology import FlatLambdaCDM
from astropy.constants import m_p, sigma_T
import astropy.units as u
from datetime import datetime
from utils import *
from colossus.halo import concentration
from colossus.halo import mass_defs
from colossus.cosmology import cosmology

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Computer Modern']
rcParams['text.usetex'] = False  # Disable LaTeX rendering (requires system TeX installation)

# --------------------
# Configuration
# --------------------
# Simulation / cosmology
LBOX = 681                 # cMpc/h
C = 2.99792e5 # km/s
T_CMB = 2.726e6 # microK
h = 0.681
OM_M = 0.306
OM_B = 0.0486
X_H = 0.76
X_He = 0.24
Z_SIM = 0.74               # simulation snapshot redshift — never changes

PLOT_DATA = True
RATIO_MODE = False 

SAT_FRACS = [0.01, 0.10, 0.20, 0.30]

# Define profile configurations: each configuration specifies selection parameters,
# convergence mode, and which tau methods/field types to include
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
    },"""

def build_profiles():
    """Expand profile configurations over all sat_frac values and tau methods."""
    profiles = []
    
    for config in PROFILE_CONFIGS:
        sel_params = dict(config['selection_params'])
        convergence_mode = config['convergence_mode']
        
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
                # Don't set values - let resolve_converged_param find the right files
                # based on the specified convergence mode
                p['n_gal_density']   = None
                p['halo_mass_range'] = None
                p['sat_frac']        = sf
                p['name']            = config['name']  # Add name from config for simplified labels
                p = resolve_converged_param(p, mode=convergence_mode)  # Finds and parses the converged parameters
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
    """
    
    # Extract config values
    GAS_TYPE = config['gas_type']
    
    # Get halo data path using pipeline_paths
    halo_galaxy_data_path = str(halo_galaxy_hdf5(GAS_TYPE))
    
    # Load only needed fields using deepdish selective loading
    all_m200b = dd.io.load(halo_galaxy_data_path, '/galaxies/m200b')  # m200b is already m200m
    all_m200c = dd.io.load(halo_galaxy_data_path, '/galaxies/m200c')
    all_vels = dd.io.load(halo_galaxy_data_path, '/galaxies/vel_fof')
    
    return all_m200b, all_m200c, all_vels


def compute_temperature_signal_pixell(tau_xy_inner, tau_xy_outer, vel_los, v_rms=300):
    # kSZ temperature fluctuation formula
    delta_tau = tau_xy_inner - tau_xy_outer  # (cMpc/h)^2
    T_signals = -T_CMB * (vel_los[:, np.newaxis] / C) * delta_tau  # microK (cMpc/h)^2
    #T_signals = -T_CMB * (v_rms / C) * delta_tau  # microK (cMpc/h)^2
    
    return T_signals


def compute_temperature_signal_simple(tau_signals, vel_los, v_rms=300):
    return -T_CMB * (vel_los[:, np.newaxis] / C) * tau_signals  # microK
    #return -T_CMB * (v_rms / C) * tau_signals  # microK


def weighted_stacking(t_signals, vel_los, sigma_i=1.0, r_v=1.0):
    # Constants
    v_rms = np.std(vel_los) # RMS of the velocity catalog

    #sigma_i = np.random.normal(loc=1.0, scale=0.1, size=len(vel_los))
    
    # Weights w_i = 1 / sigma_i^2
    weights = 1.0 / (sigma_i**2)
    
    # Term: (v_i / c)
    v_term = vel_los / C

    numerator = np.sum(t_signals * v_term[:, np.newaxis] * weights, axis=0)
    denominator = np.sum((v_term**2) * weights, axis=0)
    
    if denominator == 0:
        return 0.0
        
    T_stacked = - (1.0 / r_v) * (v_rms / C) * (numerator / denominator)
    
    return T_stacked


def weighted_stacking_rms(t_signals, vel_los, sigma_i=1.0, r_v=1.0, v_rms=300):
    """
    Weighted stacking with RMS velocity instead of individual halo velocities.
    Computes: <rho_gas> * v_rms
    
    If v_rms is None, it will be computed from the velocity catalog.
    """
    # If v_rms not provided, compute from velocity catalog
    if v_rms is None:
        v_rms = np.std(vel_los)  # km/s
    
    # Weights w_i = 1 / sigma_i^2
    weights = 1.0 / (sigma_i**2)
    
    numerator = np.sum(t_signals * weights, axis=0)
    denominator = np.sum(weights)
    
    if denominator == 0:
        print("Denominator is zero in weighted_stacking_rms!")
        return np.zeros_like(t_signals[0])
    
    T_avg = numerator / denominator
    
    # Multiply by v_rms: <rho_gas> * v_rms
    T_stacked = - (1.0 / r_v) * (v_rms / C) * T_avg
    
    return T_stacked


def compute_distances(cosmo, z):
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
    
    if use_dm:
        load_config['field_type'] = 'dm'
        load_config['tau_method'] = 'fullFT_tau_reconstruction'
        # For DM, strip '_reconstructed' from gas_type
        load_config['gas_type'] = config['gas_type'].replace('_reconstructed', '')
    
    # Get path using pipeline_paths
    tau_apertures_path = str(ap_output_path(load_config))
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

        halo_m200b, halo_m200c, halo_vels = get_halo_data_for_dataset(config)

        halo_vels = halo_vels[inds_sub]
        halo_m200b = halo_m200b[inds_sub]
        halo_m200c = halo_m200c[inds_sub]
        los_vels = halo_vels[:, 2]  # km/s
        
        T_signals = compute_temperature_signal_pixell(tau_xy_inner, tau_xy_outer, los_vels) 
        stacked_kSZ_signal = weighted_stacking(T_signals, los_vels)

        one_halo_signal, mean_r200c_mpc_comoving = get_one_halo_term(halo_m200c, halo_vels, Z_SIM)
        
        mean_halo_m200b = np.mean(halo_m200b)  # Msun/h
        N_halos = len(inds_sub)

    else:
        tau_signals = data['tau_signals']
        inds_sub = data['inds_sub']
        aperture_radii = data.get('r_comoving_mpc_h')

        halo_m200b, halo_m200c, halo_vels = get_halo_data_for_dataset(config)

        halo_vels = halo_vels[inds_sub]
        halo_m200b = halo_m200b[inds_sub]
        halo_m200c = halo_m200c[inds_sub]
        los_vels = halo_vels[:, 2]  # km/s

        T_signals = compute_temperature_signal_simple(tau_signals, los_vels)
        stacked_kSZ_signal = weighted_stacking(T_signals, los_vels)

        one_halo_signal, mean_r200c_mpc_comoving = get_one_halo_term(halo_m200c, halo_vels, Z_SIM)
        
        mean_halo_m200b = np.mean(halo_m200b)  # Msun/h
        N_halos = len(inds_sub)

    return aperture_radii, stacked_kSZ_signal, one_halo_signal, mean_r200c_mpc_comoving, mean_halo_m200b, N_halos, T_signals

def get_one_halo_term(halo_m200c, halo_vels, z):

    mean_halo_m200c = np.mean(halo_m200c) / h # Msun
    vel_rms = np.std(halo_vels[:, 2])   # km/s 

    print(f"Mean m200c: {np.log10(mean_halo_m200c):.2e} Msun, Velocity RMS: {vel_rms:.2f} km/s")

    cosmo = FlatLambdaCDM(H0=h*100, Om0=OM_M, Tcmb0=2.725)
    rho_crit_z = cosmo.critical_density(z).to(u.Msun / u.Mpc**3).value  # Msun/Mpc^3 at redshift z
    mean_r200c = (3.0 * mean_halo_m200c / (4.0 * np.pi * 200.0 * rho_crit_z))**(1/3)  # Mpc
    mean_r200c_mpc_comoving = mean_r200c * (1 + z) * h  # cMpc/h

    mean_gas_mass = (OM_B / OM_M) * mean_halo_m200c * u.Msun # Msun
    f_e = (X_H + 0.5 * X_He) / m_p # 1/kg
    N_e = (mean_gas_mass.to(u.kg) * f_e).to(u.dimensionless_unscaled).value
    v_factor = (vel_rms / C)
    sigma_T_mpc2 = sigma_T.to(u.Mpc**2).value # Mpc^2
    delta_T_halo = T_CMB * v_factor * sigma_T_mpc2 * N_e * ((1 + z) * h)**2 # microK (cMpc/h)^2
    
    return delta_T_halo, mean_r200c_mpc_comoving


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
    
    # Initialize colossus cosmology
    params = {'flat': True, 'H0': h*100, 'Om0': OM_M, 'Ob0': OM_B, 'sigma8': 0.807, 'ns': 0.967}
    cosmology.setCosmology('myCosmo', params)

    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)

    figure_content = np.load("/home/fb635/fedirfiles/tracing_cosmic_gas/data/Fig2_sim.npz")

    theta_arcmin = figure_content['theta_arcmins']
    signal = figure_content['signal']  # μK arcmin²
    noise = figure_content['noise']    # μK arcmin²
    gas_illustris = figure_content['gas_illustris']
    dm_tng = figure_content['dm_tng']

    cosmo = FlatLambdaCDM(H0=h*100, Om0=OM_M, Tcmb0=2.725)
    _, d_C = compute_distances(cosmo, Z_SIM)  # cMpc/h
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
    first_profile_data = get_kSZ_profile(PROFILES[2])
    mean_mass_m200b_msun_h = first_profile_data[4]  # Msun/h
    mean_mass_log = np.log10(mean_mass_m200b_msun_h)

    for profile in PROFILES:
        radii, kSZ_signal, one_halo_signal, mean_r200c_mpc_comoving, mean_halo_m200b, N_halos, T_signals = get_kSZ_profile(profile)

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

        # Build label based on whether it's a single profile or comparison
        # For all cases, use the simple config name + sat_frac only if mixed mode
        if comparison_subject == "sat_frac":
            # When sat_frac is the varying parameter, use name + sat_frac (only for mixed mode)
            label = profile.get('name', 'profile')
            if profile.get('selection_mode') == 'mixed':
                label += f" (sf={profile['sat_frac']:.2f})"
        elif is_single_profile or comparison_subject in ["identical", "single"]:
            # Single profile or identical profiles - use name
            label = profile.get('name', 'profile')
            if profile.get('selection_mode') == 'mixed':
                label += f" (sf={profile['sat_frac']:.2f})"
        elif "+" in comparison_subject:
            # Multiple varying parameters - use name for simplicity
            label = profile.get('name', 'profile')
            if profile.get('selection_mode') == 'mixed':
                label += f" (sf={profile['sat_frac']:.2f})"
        else:
            # Other single varying parameter - use name + parameter value
            label = f"{profile.get('name', 'profile')} ({comparison_subject}={profile[comparison_subject]})"
        
        # Use noise from data as error bars
        #profile_errors = compute_profile_errors(T_signals, N_halos)
        ax.errorbar(radii, plot_signal, yerr=error_bars, fmt='-', label=label)
    
    # Add mean mass as a legend entry
    ax.plot([], [], ' ', label=f"$\\langle M_{{200m}} \\rangle = 10^{{{mean_mass_log:.2f}}}$ M$_\\odot$/h")
    
    ax.axhline(y=one_halo_signal_plot, linestyle='--')
    ax.axvline(x=mean_r200c_mpc_comoving, linestyle=':', color='gray')
    
    ax.set_xlabel(r"Apperture Radius $R$ [cMpc/h]", fontsize=14)
    if RATIO_MODE:
        ax.set_ylabel(r"CAP kSZ / CAP DM", fontsize=14)
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
    
    fig.savefig(output_path)
    print(f"Plot saved to: {output_path}")


if __name__ == "__main__":
    main()

