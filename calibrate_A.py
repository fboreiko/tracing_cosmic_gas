import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from astropy.cosmology import FlatLambdaCDM
from astropy.constants import m_p, sigma_T
import astropy.units as u
from colossus.halo import concentration
from colossus.halo import mass_defs
from colossus.cosmology import cosmology

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Computer Modern']
rcParams['text.usetex'] = False

# --------------------
# Configuration
# --------------------
LBOX = 681                 # cMpc/h
C = 2.99792e5             # km/s
T_CMB = 2.726e6           # microK
h = 0.681
OM_M = 0.306
OM_B = 0.0486
X_H = 0.76
X_He = 0.24

# Halo mass bin to analyze
HALO_BIN = 6

# A parameter range for calibration
A_VALUES = np.linspace(-0.0235, -0.0232, 21)

def M200c_to_M200m(M200c, z):
    """Convert M200c to M200m using colossus."""
    c200c = concentration.concentration(M200c, '200c', z, model='diemer19')
    M200m, R200m, c200m = mass_defs.changeMassDefinition(
        M200c, c200c, z, mdef_in='200c', mdef_out='200m'
    )
    return M200m


def get_halo_data_for_dataset(config):
    """Load halo data based on configuration."""
    ACCOUNT_FOR_MISCENTERING = config.get('account_for_miscentering', False)
    GAS_TYPE = config['gas_type']
    
    if ACCOUNT_FOR_MISCENTERING:
        if GAS_TYPE in ['fiducial', 'strongest_AGN']:
            data_type = 'hydro'
            variant = GAS_TYPE
        else:
            data_type = 'dmo'
            variant = None
    else:
        data_type = 'hydro'
        if GAS_TYPE in ['fiducial_reconstructed', 'fiducial']:
            variant = 'fiducial'
        else:
            variant = 'strongest_AGN'
    
    # Set halo mass and velocity paths
    if data_type == 'dmo':
        HALO_M200C_PATH = "/home/fb635/fedirfiles/tracing_cosmic_gas/data/halo_data/halo_data_dmo/halo_mass.npy"
        HALO_VELS_PATH = "/home/fb635/fedirfiles/tracing_cosmic_gas/data/halo_data/halo_data_dmo/halo_vels.npy"
    else:
        HALO_M200C_PATH = f"/home/fb635/fedirfiles/tracing_cosmic_gas/data/halo_data/halo_data_hydro/halo_mass_{variant}.npy"
        if GAS_TYPE in ['fiducial', 'fiducial_reconstructed']:
            HALO_VELS_PATH = "/home/fb635/fedirfiles/tracing_cosmic_gas/data/halo_data/halo_data_hydro/halo_vels_fiducial.npy"
        else:
            HALO_VELS_PATH = "/home/fb635/fedirfiles/tracing_cosmic_gas/data/halo_data/halo_data_hydro/halo_vels_strongest_AGN.npy"
    
    halo_m200c = np.load(HALO_M200C_PATH)
    halo_vels = np.load(HALO_VELS_PATH)
    
    return halo_m200c, halo_vels


def compute_temperature_signal_simple(tau_signals, vel_los):
    """Compute temperature signals from tau signals and line-of-sight velocities."""
    return -T_CMB * (vel_los[:, np.newaxis] / C) * tau_signals  # microK


def weighted_stacking(t_signals, vel_los, sigma_i=1.0, r_v=1.0):
    """Weighted stacking of temperature signals."""
    v_rms = np.std(vel_los)
    weights = 1.0 / (sigma_i**2)
    v_term = vel_los / C
    
    numerator = np.sum(t_signals * v_term[:, np.newaxis] * weights, axis=0)
    denominator = np.sum((v_term**2) * weights, axis=0)
    
    if denominator == 0:
        return 0.0
    
    T_stacked = -(1.0 / r_v) * (v_rms / C) * (numerator / denominator)
    
    return T_stacked


def compute_profile_errors(T_signals, N_halos):
    """Compute error bars from profile measurements."""
    std_per_aperture = np.std(T_signals, axis=0)
    errors = std_per_aperture / np.sqrt(N_halos)
    return errors


def get_tau_profile(config):
    """
    Load tau profile for a single configuration.
    
    Returns:
        tuple: (aperture_radii, tau_signals, inds_sub)
    """
    # Determine base directory
    if config['projection_type'] == 'pixell':
        base_dir = "pixell_CAP_code"
    elif config['projection_type'] == 'pixell_cea':
        base_dir = "pixell_cea_CAP_code"
    else:
        base_dir = "simple_CAP_code"
    
    # Parse tau_method
    tau_method = config['tau_method']
    if not tau_method.endswith('_tau_reconstruction'):
        tau_method_dir = f"{tau_method}_tau_reconstruction"
    else:
        tau_method_dir = tau_method
    
    # Add smoothed suffix
    if config.get('smoothed', False):
        tau_method_dir += "_smoothed"
    
    # Construct subdir with A parameter if present
    misc_str = "_acc_for_misc" if config.get('account_for_miscentering', False) else ""
    A_str = f"_A_{config['A']:.5f}" if 'A' in config else ""
    
    if config.get('halo_mass_range') is not None:
        subdir = f"tau_apertures_{config['gas_type']}{A_str}_massbin_{config['halo_mass_range']}_ngrid_{config['ngrid']}{misc_str}"
    else:
        subdir = f"tau_apertures_{config['gas_type']}{A_str}_ngrid_{config['ngrid']}{misc_str}"
    
    # Add JAX suffix
    if config.get('JAX', False):
        subdir += "_JAX_MPI"
    
    # Filename
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
    
    tau_signals = data['tau_signals']
    inds_sub = data['inds_sub']
    aperture_radii = data.get('r_comoving_mpc_h')
    
    return aperture_radii, tau_signals, inds_sub


def compute_kSZ_profile(config):
    """
    Compute kSZ profile from tau signals.
    
    Returns:
        tuple: (aperture_radii, kSZ_signal, mean_halo_mass, N_halos, T_signals)
    """
    aperture_radii, tau_signals, inds_sub = get_tau_profile(config)
    
    # Load halo data
    halo_m200c, halo_vels = get_halo_data_for_dataset(config)
    
    # Select halos
    halo_vels = halo_vels[inds_sub]
    halo_m200c = halo_m200c[inds_sub]
    los_vels = halo_vels[:, 2]  # km/s
    
    # Compute temperature signals
    T_signals = compute_temperature_signal_simple(tau_signals, los_vels)
    stacked_kSZ_signal = weighted_stacking(T_signals, los_vels)
    
    mean_halo_mass = np.mean(halo_m200c)  # Msun/h
    N_halos = len(inds_sub)
    
    return aperture_radii, stacked_kSZ_signal, mean_halo_mass, N_halos, T_signals


def compute_chi2(signal_recon, signal_true, cov_inv):
    """
    Compute chi-squared statistic.
    
    Args:
        signal_recon: Reconstructed signal
        signal_true: True signal
        cov_inv: Inverse of covariance matrix
    
    Returns:
        chi2: Chi-squared value
    """
    diff = signal_recon - signal_true
    chi2 = diff @ cov_inv @ diff
    return chi2


def get_profile_config(A_value):
    """Generate profile configuration for a given A value."""
    return [
        {
            "projection_type": "simple",
            "tau_method": "fullFT",
            "smoothed": True,
            "gas_type": "strongest_AGN",
            "halo_mass_range": HALO_BIN,
            "ngrid": 2048,
            "JAX": True,
            "z": 0.74,
            "n_gal_density": 87e-5,
            "account_for_miscentering": False
        },
        {
            "projection_type": "simple",
            "tau_method": "2D_FT_massdep",
            "smoothed": True,
            "gas_type": "strongest_AGN_reconstructed",
            "halo_mass_range": HALO_BIN,
            "ngrid": 2048,
            "JAX": True,
            "z": 0.74,
            "n_gal_density": 87e-5,
            "account_for_miscentering": True,
            "A": A_value
        },
    ]


def main():
    OUTPUT_DIR = "/home/fb635/fedirfiles/tracing_cosmic_gas/plots"
    
    # Initialize colossus cosmology
    params = {'flat': True, 'H0': h*100, 'Om0': OM_M, 'Ob0': OM_B, 'sigma8': 0.807, 'ns': 0.967}
    cosmology.setCosmology('myCosmo', params)
    
    print("=" * 70)
    print("Mass-Dependent Reconstruction Calibration (A Parameter)")
    print("=" * 70)
    print(f"Halo mass bin: {HALO_BIN}")
    print(f"A values to test: {A_VALUES}")
    
    # Load true profile (only once)
    print("\nLoading TRUE strongest AGN profile...")
    BASE_PROFILES = get_profile_config(A_VALUES[0])  # A doesn't matter for true profile
    radii_true, kSZ_true, mass_true, N_halos_true, T_signals_true = compute_kSZ_profile(BASE_PROFILES[0])
    
    # Compute covariance matrix from true profile measurements
    print(f"Computing covariance matrix from {N_halos_true} halos...")
    cov = np.cov(T_signals_true.T) / N_halos_true
    cov_inv = np.linalg.inv(cov)
    
    # Convert M200c to M200m for display
    z_profile = BASE_PROFILES[0]['z']
    mean_mass_m200m = M200c_to_M200m(mass_true, z_profile)
    mean_mass_log = np.log10(mean_mass_m200m)
    
    print(f"Mean halo mass (M200m): 10^{mean_mass_log:.2f} M☉/h")
    print(f"Number of halos: {N_halos_true}")
    print(f"Covariance matrix shape: {cov.shape}")
    
    # Loop through A values and compute chi2
    chi2_values = []
    
    print("\n" + "=" * 70)
    print("Computing chi-squared for different A values:")
    print("=" * 70)
    print(f"{'A':>10} {'chi2':>15} {'chi2/dof':>15}")
    print("-" * 70)
    
    for A in A_VALUES:
        # Load reconstructed profile with this A value
        BASE_PROFILES = get_profile_config(A)
        try:
            radii_recon, kSZ_recon, _, _, _ = compute_kSZ_profile(BASE_PROFILES[1])
            
            # Verify radii match
            if not np.allclose(radii_true, radii_recon):
                print(f"WARNING: Aperture radii don't match for A={A:.5f}!")
                continue
            
            # Compute chi2
            chi2 = compute_chi2(kSZ_recon, kSZ_true, cov_inv)
            chi2_values.append(chi2)
            
            # Degrees of freedom = number of data points - 1 parameter (A)
            dof = len(radii_true) - 1
            
            print(f"{A:>10.5f} {chi2:>15.2f} {chi2/dof:>15.2f}")
            
        except FileNotFoundError as e:
            print(f"A={A:.5f}: File not found - {e}")
            chi2_values.append(np.nan)
    
    print("-" * 70)
    
    # Convert to array and filter out NaNs
    chi2_values = np.array(chi2_values)
    valid_mask = ~np.isnan(chi2_values)
    valid_A = A_VALUES[valid_mask]
    valid_chi2 = chi2_values[valid_mask]
    
    if len(valid_chi2) == 0:
        print("\nERROR: No valid chi2 values computed!")
        return
    
    # Find optimal A
    optimal_idx = np.argmin(valid_chi2)
    optimal_A = valid_A[optimal_idx]
    optimal_chi2 = valid_chi2[optimal_idx]
    
    print("\n" + "=" * 70)
    print("Optimal A Parameter:")
    print("=" * 70)
    print(f"A_optimal = {optimal_A:.5f}")
    print(f"chi2_min = {optimal_chi2:.5f}")
    print(f"chi2_min/dof = {optimal_chi2/(len(radii_true)-1):.5f}")
    print("=" * 70)
    
    # Create chi2 vs A plot
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    ax.plot(valid_A, valid_chi2, 'o-', color='blue', linewidth=2, markersize=8, label='Chi-squared')
    ax.axvline(x=optimal_A, color='red', linestyle='--', linewidth=2, 
               label=f'Optimal A = {optimal_A:.5f}')
    
    # Mark the minimum
    ax.plot(optimal_A, optimal_chi2, 'r*', markersize=20, label=f'Min chi2 = {optimal_chi2:.2f}')
    
    ax.set_xlabel('A Parameter', fontsize=14)
    ax.set_ylabel(r'$\chi^2$', fontsize=14)
    ax.set_title(f'Mass-Dependent Reconstruction Calibration (Mass Bin {HALO_BIN})\n'
                 f'$\\langle M_{{200m}} \\rangle = 10^{{{mean_mass_log:.2f}}}$ M$_\\odot$/h, '
                 f'$N_{{halos}} = {N_halos_true}$', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12, loc='best')
    
    fig.tight_layout()
    
    # Save plot
    filename = f"massdep_calibration_massbin{HALO_BIN}_ngrid{BASE_PROFILES[0]['ngrid']}"
    output_path = f"{OUTPUT_DIR}/{filename}.png"
    fig.savefig(output_path)
    print(f"\nPlot saved to: {output_path}")
    
    # Save chi2 data to file
    data_output_path = f"{OUTPUT_DIR}/{filename}_data.npz"
    np.savez(data_output_path,
             A_values=valid_A,
             chi2_values=valid_chi2,
             optimal_A=optimal_A,
             optimal_chi2=optimal_chi2,
             mass_bin=HALO_BIN,
             mean_mass_log=mean_mass_log,
             N_halos=N_halos_true)
    print(f"Chi2 data saved to: {data_output_path}")


if __name__ == "__main__":
    main()