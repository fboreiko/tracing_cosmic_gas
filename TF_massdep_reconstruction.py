import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from jax.scipy.ndimage import map_coordinates
from scipy.fft import rfftn, rfftfreq, fftfreq, rfft2
from abacusnbody.analysis.tsc import tsc_parallel
from jax import numpy as jnp
import deepdish as dd
import h5py
from pathlib import Path
from astropy import units as u
from astropy.constants import m_p, sigma_T
import gc

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Computer Modern']
rcParams['text.usetex'] = False

def compute_delta(pos, box, ngrid, weights, nthread=4):
    # Compute density field using TSC assignment
    dens = tsc_parallel(pos, ngrid, box, weights=weights, nthread=nthread)

    # Compute overdensity field delta
    dens_avg = np.sum(dens) / ngrid**3
    delta = dens / dens_avg - 1

    return delta

def compute_2d_power_spectrum(delta_2d, box, ngrid, nkbins):
    """
    Computes the 2D power spectrum of a projected field.
    """
    # 2D FFT: Returns (ngrid, ngrid//2 + 1)
    delta_fft = rfft2(delta_2d)
    delta_fft *= (1.0 / ngrid ** 2) # Normalise for 2D

    # Compute k-space grid for 2D
    kx = fftfreq(ngrid, d=box/ngrid) * 2 * np.pi
    ky = rfftfreq(ngrid, d=box/ngrid) * 2 * np.pi
    k_Nyquist = np.pi * ngrid / box

    # Create k_magnitude grid
    k_mag = np.sqrt(kx[:, np.newaxis]**2 + ky[np.newaxis, :]**2)
    
    k_max = np.max(k_mag)
    k_bins = np.linspace(0, k_max, nkbins)
    k_center = 0.5 * (k_bins[:-1] + k_bins[1:])

    power_spectrum = np.zeros(len(k_bins)-1)
    mode_counts = np.zeros(len(k_bins)-1)

    # Compute power magnitude
    P_2d_field = (delta_fft * np.conj(delta_fft)).real

    # Binning the 2D modes
    for i in range(len(k_bins)-1):
        mask = (k_mag >= k_bins[i]) & (k_mag < k_bins[i+1])
        P_bin = P_2d_field[mask]
        if P_bin.size > 0:
            power_spectrum[i] = np.mean(P_bin)
    
    # Normalise by "Area" (Box^2) for 2D power spectrum
    power_spectrum *= box ** 2 

    return delta_fft, k_mag, k_center, k_Nyquist, power_spectrum

labels = ['dm', 'strongest_AGN'] # 'dm', 'fiducial', 'strongest_AGN'
box = 681
ngrid = 2048
nkbins = ngrid // 2

# A parameter range for mass-dependent transfer function
A_values = np.linspace(-0.0182, -0.018, 21)

print(f"Reconstructing the {labels[1]} gas field from dark matter field...")
print(f"Will vary A parameter from {A_values[0]} to {A_values[-1]} with {len(A_values)} values")

# Simulations
rds_base = '/rds-d6/user/fb635/hpc-work/tracing_cosmic_gas'
dm_sim_file = f'{rds_base}/FLAMINGO_particles_L1000N1800_DMO_FIDUCIAL_snap_77_diluted_100_lean.hdf5'
fiducial_sim_file = f'{rds_base}/FLAMINGO_particles_L1000N1800_HYDRO_FIDUCIAL_snap_77_diluted_100.hdf5'
strongest_AGN_sim_file = f'{rds_base}/FLAMINGO_particles_L1000N1800_HYDRO_STRONGEST_AGN_snap_77_diluted_100.hdf5'

Power_spectra = {}
delta_ks = {}
k_mag, k_center, k_Nyquist = None, None, None

for label in labels:
    print(f"\nProcessing {label}...")
    if label == 'dm':
        file = dd.io.load(dm_sim_file)
        masses = file['DMParticles']['mass']
        pos = file['DMParticles']['pos']
    elif label == 'fiducial':
        file = dd.io.load(fiducial_sim_file)
        masses = file['GasParticles']['mass']
        pos = file['GasParticles']['pos']
    else:
        file = dd.io.load(strongest_AGN_sim_file)
        masses = file['GasParticles']['mass']
        pos = file['GasParticles']['pos']
        
    weights = None
    if any(masses != masses[0]):
        weights = masses / np.mean(masses)

    delta = compute_delta(pos, box, ngrid, weights, nthread=4)

    delta_2d = np.mean(delta, axis=2)

    del delta  # Free memory
    del masses, pos, file  # Free memory
    gc.collect()  # Force garbage collection

    delta_k, k_mag_i, k_center_i, k_Nyquist_i, Power_spectrum = compute_2d_power_spectrum(delta_2d, box, ngrid, nkbins)
    Power_spectra[label] = Power_spectrum
    delta_ks[label] = delta_k
    if k_mag is None:
        k_mag = k_mag_i
        k_center = k_center_i
        k_Nyquist = k_Nyquist_i
    
    del delta_2d, k_mag_i, k_center_i, k_Nyquist_i  # Free memory
    gc.collect()  # Force garbage collection

print("\nReconstructing gas field from dark matter field...")

# Compute base transfer function
T_k = jnp.sqrt(Power_spectra[labels[1]] / Power_spectra['dm'])

coords = (k_mag - k_center[0]) / (k_center[-1] - k_center[0]) * (len(k_center) - 1)
T_k_interp = map_coordinates(T_k, [coords], order=1)

del coords  # Free memory
gc.collect()  # Force garbage collection

# Determine which sim_file to use for tau map computation
sim_type = labels[1]
if sim_type == 'fiducial':
    sim_file = fiducial_sim_file
elif sim_type == 'strongest_AGN':
    sim_file = strongest_AGN_sim_file
else:
    sim_file = strongest_AGN_sim_file  # default

# Load particle data for tau map computation
print(f"\nLoading particle data from {sim_file}...")
file = dd.io.load(sim_file)
masses = file['GasParticles']['mass']  # 1e10 Msun/h
pos = file['GasParticles']['pos']  # cMpc/h
masses *= 100  # Convert to proper units

# Simulation/cosmology parameters for tau map
h = 0.681
z = 0.74

masses = masses * 1e10 / h  # Convert to Msun
box_comoving_Mpc = box / h  # Convert from cMpc/h to Mpc
box_physical_Mpc = box_comoving_Mpc / (1 + z)  # Physical size at z=0.74

total_mass_Msun = np.sum(masses)
volume_Mpc3 = box_physical_Mpc**3
rho_bar_Msun_per_Mpc3 = total_mass_Msun / volume_Mpc3  # Mean density in Msun/Mpc^3

rho_bar_kg_per_m3 = (rho_bar_Msun_per_Mpc3 * u.Msun / u.Mpc**3).to(u.kg / u.m**3)

cell_physical_m = ((box_physical_Mpc * u.Mpc) / ngrid).to(u.m)

X_H = 0.76
X_He = 0.24
f_e = (X_H + 0.5 * X_He) / m_p  # electrons per kg

prefactor = (sigma_T * rho_bar_kg_per_m3 * f_e * cell_physical_m).to(u.dimensionless_unscaled).value

# Clean up intermediate variables no longer needed
del file, rho_bar_kg_per_m3, cell_physical_m, f_e, total_mass_Msun, volume_Mpc3, rho_bar_Msun_per_Mpc3
del Power_spectra  # No longer needed after computing T_k
gc.collect()  # Force garbage collection

# Create output directory for tau maps
data_dir = Path(f'{rds_base}')
tau_output_dir = Path('/home/fb635/fedirfiles/tracing_cosmic_gas/data/tau_maps/2D_FT_massdep_tau_reconstruction')
tau_output_dir.mkdir(parents=True, exist_ok=True)

# Loop over A parameter values
for A in A_values:
    print(f"\n{'='*60}")
    print(f"Processing A = {A:.5f}")
    print(f"{'='*60}")
    
    # Apply mass-dependent transfer function: TF_new = TF * (1 + A*k^2)
    T_k_modified = T_k_interp * (1.0 + A * k_mag**2)
    
    # 4. RECONSTRUCT GAS FIELD IN FOURIER SPACE
    delta_k_gas_recon = delta_ks['dm'] * T_k_modified
    
    # 5. INVERSE FFT TO GET REAL SPACE FIELD
    delta_gas_reconstructed_2d = np.fft.irfft2(delta_k_gas_recon) * (ngrid ** 2)
    
    # 6. COMPUTE TAU MAP DIRECTLY
    # For 2D_FT method: tau_xy_map = prefactor * (1.0 + delta_gas) * ngrid
    tau_xy_map = prefactor * (1.0 + delta_gas_reconstructed_2d) * ngrid
    
    # Save tau map
    tau_out_path = tau_output_dir / f'tau_map_{sim_type}_reconstructed_A_{A:.5f}.npy'
    np.save(tau_out_path, tau_xy_map)
    
    print(f"Tau map statistics for A = {A:.5f}:")
    print(f"  Mean: {np.mean(tau_xy_map):.6e}")
    print(f"  Std: {np.std(tau_xy_map):.6e}")
    print(f"  Min: {np.min(tau_xy_map):.6e}")
    print(f"  Max: {np.max(tau_xy_map):.6e}")
    print(f"Tau map saved to: {tau_out_path}")
    
    # Clean up memory after each iteration
    del T_k_modified, delta_k_gas_recon, delta_gas_reconstructed_2d, tau_xy_map
    gc.collect()  # Force garbage collection

print("\n" + "="*60)
print(f"All {len(A_values)} reconstructions completed successfully!")
print(f"Tau maps saved to: {tau_output_dir}")
print("="*60)