import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import rcParams
from astropy.cosmology import FlatLambdaCDM
from astropy.constants import m_p, sigma_T
import astropy.units as u
from datetime import datetime
import os
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

PLOT_DATA = True

# Multiple profiles to plot
PROFILES = [
    {
        "projection_type": "simple",  # 'pixell', 'pixell_cea', or 'simple'
        "tau_method": "fullFT",  # 'fullFT' or 'histmethod', or '2D_FT', or '2D_FT_massdep'
        "smoothed": True,
        "gas_type": "fiducial",  # 'fiducial_reconstructed', 'strongest_AGN_reconstructed', 'fiducial', 'strongest_AGN'
        "halo_mass_range": None,  # int (0 to N-1) for mass bin index, or None to use n_gal_density
        "ngrid": 2048,
        "JAX": True,
        "z": 0.74,
        "n_gal_density": 87e-5, #87e-5,  # cMpc/h^-3
        "account_for_miscentering": False,
        "A": 0.00
    },
    {
        "projection_type": "simple",
        "tau_method": "2D_FT_upgrade",
        "smoothed": True,
        "gas_type": "fiducial_reconstructed",
        "halo_mass_range": None,
        "ngrid": 2048,
        "JAX": True,
        "z": 0.74,
        "n_gal_density": 87e-5, #87e-5,
        "account_for_miscentering": False,
        "A": 0.00
    },
    {
        "projection_type": "simple",
        "tau_method": "fullFT",
        "smoothed": True,
        "gas_type": "strongest_AGN",
        "halo_mass_range": None,
        "ngrid": 2048,
        "JAX": True,
        "z": 0.74,
        "n_gal_density": 87e-5, #87e-5,
        "account_for_miscentering": False,
        "A": 0.00
    },
    {
        "projection_type": "simple",
        "tau_method": "2D_FT_upgrade",
        "smoothed": True,
        "gas_type": "strongest_AGN_reconstructed",
        "halo_mass_range": None,
        "ngrid": 2048,
        "JAX": True,
        "z": 0.74,
        "n_gal_density": 87e-5, #87e-5,
        "account_for_miscentering": False,
        "A": 0.00
    },
]


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
    Load halo data based on configuration.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary with keys:
        - gas_type: str
        - n_gal_density: float (galaxy number density in cMpc/h^-3)
        - halo_mass_range: int or None (mass bin index)
        - account_for_miscentering: bool
    """
    
    # Extract config values
    ACCOUNT_FOR_MISCENTERING = config.get('account_for_miscentering', False)
    GAS_TYPE = config['gas_type']

    if ACCOUNT_FOR_MISCENTERING:
        # With miscentering: use DMO for reconstructed, hydro for actual
        if GAS_TYPE in ['fiducial', 'strongest_AGN']:
            data_type = 'hydro'
            variant = GAS_TYPE
        else:
            data_type = 'dmo'
            variant = None
    else:
        # Without miscentering: always use hydro
        data_type = 'hydro'
        if GAS_TYPE in ['fiducial_reconstructed', 'fiducial']:
            variant = 'fiducial'
        else:
            variant = 'strongest_AGN'
    
    # Set halo mass and position paths
    if data_type == 'dmo':
        HALO_M200C_PATH = "/home/fb635/fedirfiles/tracing_cosmic_gas/data/halo_data/halo_data_dmo/halo_mass.npy"  # Msun/h
        HALO_POS_PATH = "/home/fb635/fedirfiles/tracing_cosmic_gas/data/halo_data/halo_data_dmo/halo_pos.npy"  # cMpc/h
    else:  # hydro
        HALO_M200C_PATH = f"/home/fb635/fedirfiles/tracing_cosmic_gas/data/halo_data/halo_data_hydro/halo_mass_{variant}.npy"  # Msun/h
        HALO_POS_PATH = f"/home/fb635/fedirfiles/tracing_cosmic_gas/data/halo_data/halo_data_hydro/halo_pos_{variant}.npy"  # cMpc/h
    
    # Velocity paths: always use hydro catalog matching gas_type (keep original logic)
    if GAS_TYPE in ['fiducial', 'fiducial_reconstructed']:
        HALO_VELS_PATH = "/home/fb635/fedirfiles/tracing_cosmic_gas/data/halo_data/halo_data_hydro/halo_vels_fiducial.npy"   # km/s
    else:
        HALO_VELS_PATH = "/home/fb635/fedirfiles/tracing_cosmic_gas/data/halo_data/halo_data_hydro/halo_vels_strongest_AGN.npy"   # km/s
    
    # Load data
    halo_m200c = np.load(HALO_M200C_PATH)
    halo_pos = np.load(HALO_POS_PATH)
    halo_vels = np.load(HALO_VELS_PATH)
    
    return halo_m200c, halo_vels

#### I BUTCHER THE CODE HERE 
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
    #plt.hist(T_signals[:, 0].flatten(), bins=50)
    #plt.savefig("test_hist.png")
    std_per_aperture = np.std(T_signals, axis=0)
    errors = std_per_aperture / np.sqrt(N_halos)
    return errors


def get_kSZ_profile(config):
    """
    Get kSZ profile for a single configuration.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary with keys:
        - projection_type: str ('simple', 'pixell', or 'pixell_cea')
        - tau_method: str (e.g., 'fullFT_tau_reconstruction')
        - smoothed: bool
        - gas_type: str
        - halo_mass_range: int or None (mass bin index)
        - ngrid: int
        - JAX: bool
        - z: float
        - n_gal_density: float
        - account_for_miscentering: bool
    """
    # Determine base directory and tau method directory
    if config['projection_type'] == 'pixell':
        base_dir = "pixell_CAP_code"
    elif config['projection_type'] == 'pixell_cea':
        base_dir = "pixell_cea_CAP_code"
    else:
        base_dir = "simple_CAP_code"
    
    # Parse tau_method
    tau_method = config['tau_method']
    # Remove _tau_reconstruction suffix if present for backwards compatibility
    if not tau_method.endswith('_tau_reconstruction'):
        tau_method_dir = f"{tau_method}_tau_reconstruction"
    else:
        tau_method_dir = tau_method
    
    # Add _smoothed suffix if smoothing is enabled
    if config.get('smoothed', False):
        tau_method_dir += "_smoothed"
    
    # Construct subdir and filename based on halo_mass_range and JAX
    misc_str = "_acc_for_misc" if config.get('account_for_miscentering', False) else ""
    
    # Add A parameter to filename ONLY if using mass-dependent reconstruction
    A_str = f"_A_{config['A']:.5f}" if config.get('tau_method') == '2D_FT_massdep' and 'A' in config else ""
    
    if config.get('halo_mass_range') is not None:
        subdir = f"tau_apertures_{config['gas_type']}{A_str}_massbin_{config['halo_mass_range']}_ngrid_{config['ngrid']}{misc_str}"
    else:
        subdir = f"tau_apertures_{config['gas_type']}{A_str}_ngrid_{config['ngrid']}{misc_str}"
    
    # Add JAX suffix if using JAX implementation
    if config.get('JAX', False):
        subdir += "_JAX_MPI"
    
    # Add z specification for pixell projections
    if config['projection_type'] in ['pixell', 'pixell_cea']:
        filename = f"tau_apertures_z_{config['z']}.npz"
    else:
        filename = "tau_apertures.npz"
    
    tau_apertures_path = f"/home/fb635/fedirfiles/tracing_cosmic_gas/data/{base_dir}/{tau_method_dir}/{subdir}/{filename}"

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

        halo_m200c, halo_vels = get_halo_data_for_dataset(config)

        halo_vels = halo_vels[inds_sub]
        halo_m200c = halo_m200c[inds_sub]
        los_vels = halo_vels[:, 2]  # km/s
        
        T_signals = compute_temperature_signal_pixell(tau_xy_inner, tau_xy_outer, los_vels) 
        stacked_kSZ_signal = weighted_stacking(T_signals, los_vels)

        one_halo_signal, mean_r200c_mpc_comoving = get_one_halo_term(halo_m200c, halo_vels, config['z'])
        mean_halo_mass = np.mean(halo_m200c)  # Msun/h
        N_halos = len(inds_sub)

    else:
        tau_signals = data['tau_signals']
        inds_sub = data['inds_sub']
        aperture_radii = data.get('r_comoving_mpc_h')

        halo_m200c, halo_vels = get_halo_data_for_dataset(config)

        halo_vels = halo_vels[inds_sub]
        halo_m200c = halo_m200c[inds_sub]
        los_vels = halo_vels[:, 2]  # km/s

        T_signals = compute_temperature_signal_simple(tau_signals, los_vels)
        stacked_kSZ_signal = weighted_stacking(T_signals, los_vels)

        one_halo_signal, mean_r200c_mpc_comoving = get_one_halo_term(halo_m200c, halo_vels, config['z'])
        mean_halo_mass = np.mean(halo_m200c)  # Msun/h
        N_halos = len(inds_sub)

    return aperture_radii, stacked_kSZ_signal, one_halo_signal, mean_r200c_mpc_comoving, mean_halo_mass, N_halos, T_signals

def get_one_halo_term(halo_m200c, halo_vels, z):

    mean_halo_m200c = np.mean(halo_m200c) / h # Msun
    print(np.log10(mean_halo_m200c))
    vel_rms = np.std(halo_vels[:, 2])   # km/s 

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
    
    # Define all possible keys
    keys = ['projection_type', 'tau_method', 'smoothed', 'gas_type', 'halo_mass_range', 'ngrid', 'z', 'JAX', 'account_for_miscentering', 'A']
    
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
    comparison_priority = ['ngrid', 'halo_mass_range', 'gas_type', 'tau_method', 'smoothed', 'projection_type', 'z', 'JAX', 'account_for_miscentering', 'A']
    comparison_subjects = [key for key in comparison_priority if key in varying_params]
    
    if len(comparison_subjects) == 0:
        comparison_subject = "single" if is_single_profile else "identical"
    elif len(comparison_subjects) == 1:
        comparison_subject = comparison_subjects[0]
    else:
        comparison_subject = "+".join(comparison_subjects)
    
    # Build filename parts
    filename_parts = ['kSZ']
    
    # For single profile, include all relevant parameters
    if is_single_profile:
        param_order = ['projection_type', 'tau_method', 'gas_type', 'smoothed', 'z', 'halo_mass_range', 'ngrid', 'JAX', 'account_for_miscentering', 'A']
        for key in param_order:
            value = common_params.get(key)
            if value is None:
                continue
            if key == 'z':
                filename_parts.append(f"z{value}")
            elif key == 'halo_mass_range' and value != 'None':
                filename_parts.append(f"massbin{value}")
            elif key == 'smoothed' and value == 'True':
                filename_parts.append('smoothed')
            elif key == 'JAX' and value == 'True':
                filename_parts.append('JAX')
            elif key == 'account_for_miscentering' and value == 'True':
                filename_parts.append('acc_for_misc')
            elif key == 'A':
                filename_parts.append(f"A{value}")
            elif key == 'ngrid':
                filename_parts.append(f"ngrid{value}")
            elif key != 'halo_mass_range' and key != 'smoothed' and key != 'JAX' and key != 'account_for_miscentering':
                filename_parts.append(value)
    else:
        # For multiple profiles, add common parameters
        param_order = ['projection_type', 'tau_method', 'smoothed', 'gas_type', 'z', 'halo_mass_range', 'JAX', 'account_for_miscentering', 'A']
        for key in param_order:
            if key in common_params:
                value = common_params[key]
                if key == 'z':
                    filename_parts.append(f"z{value}")
                elif key == 'halo_mass_range' and value != 'None':
                    filename_parts.append(f"massbin{value}")
                elif key == 'smoothed' and value == 'True':
                    filename_parts.append('smoothed')
                elif key == 'JAX' and value == 'True':
                    filename_parts.append('JAX')
                elif key == 'account_for_miscentering' and value == 'True':
                    filename_parts.append('acc_for_misc')
                elif key == 'A':
                    filename_parts.append(f"A{value}")
                elif key != 'halo_mass_range' and key != 'smoothed' and key != 'JAX' and key != 'account_for_miscentering':
                    filename_parts.append(value)
        
        # Add varying parameters with their values
        for key in ['ngrid', 'projection_type', 'tau_method', 'smoothed', 'gas_type', 'z', 'halo_mass_range', 'JAX', 'account_for_miscentering', 'A']:
            if key in varying_params:
                values_str = '-'.join(varying_params[key])
                if key == 'z':
                    filename_parts.append(f"z_{values_str}")
                elif key == 'halo_mass_range':
                    filename_parts.append(f"massbin_{values_str}")
                elif key == 'smoothed':
                    filename_parts.append(f"smoothed_{values_str}")
                elif key == 'JAX':
                    # Only add if contains 'True'
                    if 'True' in varying_params[key]:
                        filename_parts.append(f"JAX_{values_str}")
                elif key == 'account_for_miscentering':
                    # Only add if contains 'True'
                    if 'True' in varying_params[key]:
                        filename_parts.append(f"acc_for_misc_{values_str}")
                elif key == 'A':
                    filename_parts.append(f"A_{values_str}")
                else:
                    filename_parts.append(f"{key}_{values_str}")
    
    filename = '_'.join(filename_parts)
    return filename, comparison_subject, is_single_profile


def main():
    OUTPUT_DIR = "/home/fb635/fedirfiles/tracing_cosmic_gas/plots"

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
    _, d_C = compute_distances(cosmo, PROFILES[0]['z']) # cMpc/h
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
    mean_mass_m200c_msun_h = first_profile_data[4]  # Msun/h
    z_profile = PROFILES[2]['z']
    
    # Convert M200c to M200m
    mean_mass_m200m_msun_h = M200c_to_M200m(mean_mass_m200c_msun_h, z_profile)
    mean_mass_log = np.log10(mean_mass_m200m_msun_h)

    for profile in PROFILES:
        radii, kSZ_signal, one_halo_signal, mean_r200c_mpc_comoving, _, N_halos, T_signals = get_kSZ_profile(profile)

        # Build label based on whether it's a single profile or comparison
        if is_single_profile:
            # For single profile, create a descriptive label with key parameters
            label_parts = []
            if profile['gas_type']:
                label_parts.append(profile['gas_type'])
            if profile['tau_method']:
                label_parts.append(profile['tau_method'])
            if profile['smoothed']:
                label_parts.append('smoothed')
            label_parts.append(f"ngrid={profile['ngrid']}")
            label_parts.append(f"z={profile['z']}")
            label = ", ".join(label_parts)
        elif "+" in comparison_subject:
            # Multiple varying parameters - show all
            label_parts = [f"{key}={profile[key]}" for key in comparison_subject.split("+")]
            label = ", ".join(label_parts)
        else:
            # Single varying parameter
            label = f"{comparison_subject}={profile[comparison_subject]}"
        
        if PLOT_DATA:
            # Use noise from data as error bars
            ax.errorbar(radii, kSZ_signal, yerr=noise_converted, fmt='-', label=label)
        else:
            # Compute error bars from profile measurements
            profile_errors = compute_profile_errors(T_signals, N_halos)
            ax.errorbar(radii, kSZ_signal, yerr=profile_errors, fmt='-', label=label)
    
    # Add mean mass as a legend entry
    ax.plot([], [], ' ', label=f"$\\langle M_{{200m}} \\rangle = 10^{{{mean_mass_log:.2f}}}$ M$_\\odot$/h")
    
    ax.axhline(y=one_halo_signal, linestyle='--')
    ax.axvline(x=mean_r200c_mpc_comoving, linestyle=':', color='gray')
    
    ax.set_xlabel(r"Apperture Radius $R$ [cMpc/h]", fontsize=14)
    ax.set_ylabel(r"$T_{\mathrm{kSZ}}$ ($\mu$K (cMpc/h)$^{2}$)", fontsize=14)
    ax.legend(fontsize=4, loc="lower right")
    ax.set_yscale('log')
    fig.tight_layout()
    
    # Save with auto-generated filename
    output_path = f"{OUTPUT_DIR}/{filename}.png"
    fig.savefig(output_path)
    print(f"Plot saved to: {output_path}")


if __name__ == "__main__":
    main()
