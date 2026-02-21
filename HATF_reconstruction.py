import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.fft import rfft2, rfftfreq, fftfreq
from abacusnbody.analysis.tsc import tsc_parallel
from jax.scipy.ndimage import map_coordinates
from jax import numpy as jnp
import deepdish as dd
import h5py
from pathlib import Path
from astropy import units as u
from astropy.constants import m_p, sigma_T
import gc
import sys
sys.path.insert(0, '/Users/fedorboreiko/Documents/Cambridge/project_github/utils')
from sample_selection import select_halos
from colossus.cosmology import cosmology
from colossus.halo import mass_defs, concentration

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Computer Modern']
rcParams['text.usetex'] = False

def compute_delta(pos, box, ngrid, weights, nthread=4):
    """Compute overdensity field using TSC assignment."""
    dens = tsc_parallel(pos, ngrid, box, weights=weights, nthread=nthread)
    dens_avg = np.sum(dens) / ngrid**3
    delta = dens / dens_avg - 1
    return delta

def compute_2d_projection(delta):
    """Project 3D overdensity field to 2D by averaging along z-axis."""
    return np.mean(delta, axis=2)

def compute_2d_fft(delta_2d, ngrid):
    """Compute 2D FFT of projected field."""
    delta_fft = rfft2(delta_2d)
    delta_fft *= (1.0 / ngrid ** 2)  # Normalize for 2D
    return delta_fft

def compute_k_grid(ngrid, box):
    """Compute k-space grid for 2D FFT."""
    kx = fftfreq(ngrid, d=box/ngrid) * 2 * np.pi
    ky = rfftfreq(ngrid, d=box/ngrid) * 2 * np.pi
    k_mag = np.sqrt(kx[:, np.newaxis]**2 + ky[np.newaxis, :]**2)
    return k_mag

# =============================================================================
# CONFIGURATION
# =============================================================================
gas_type = 'fiducial'  # 'fiducial', 'strongest_AGN'

# Regime: determines which mass to use for selection
regime = 'mgal_sel'  # 'mhalo_sel' (halo mass) or 'mgal_sel' (stellar mass)

# Central/satellite mode: determines filtering by central/satellite status
if regime == 'mhalo_sel':
    cen_sat_mode = 'cen'          
    mass_type = 'm200c'            # Use m200c for ranking
    require_nonzero_mass = False
    upper_cut_mode = False
elif regime == 'mgal_sel':
    cen_sat_mode = 'sat'     
    mass_type = 'm200b'           # Use stellar mass for ranking
    require_nonzero_mass = False
    upper_cut_mode = True

# Selection method: determines how to select from ranked objects
halo_mass_bin = None   # select specific mass bin (0, 1, 2, ...), satellites test uncut - 20, satellites test with upper cut 14.2 - 29, satellites test with upper cut 13.8 - 16 (30000)
n_gal_density = 4.4e-3   # (cMpc/h)^-3, mhalo_sel - 1.22e-3, mgal_sel - 2e-2, mixed - 5.5e-3, satellites test uncut - 5.8e-3, satellites test with upper cut 14.2 - 6.6e-3, satellites test with upper cut 13.8 - 4.4e-3

# Determine selection method for file naming
if halo_mass_bin is not None:
    method = 'mass_bin'
elif n_gal_density is not None:
    method = 'number_density'
else:
    method = 'all_halos'

# File paths
particles_sim_file = f'/rds-d6/user/fb635/hpc-work/tracing_cosmic_gas/FLAMINGO_particles_L1000N1800_HYDRO_{gas_type.upper()}_snap_77_diluted_100.hdf5'
haloes_sim_file = f'/home/fb635/rds/hpc-work/tracing_cosmic_gas/FLAMINGO_ext_L1000N1800_HYDRO_{gas_type.upper()}_snap_77.hdf5'

box = 681  # cMpc/h
ngrid = 2048
nthread = 4

# Cosmology parameters
h = 0.681
z = 0.74

# Output directories
tau_output_dir = Path('/home/fb635/fedirfiles/tracing_cosmic_gas/data/tau_maps/2D_FT_upgrade_tau_reconstruction')
tau_output_dir.mkdir(parents=True, exist_ok=True)

delta_fields_dir = Path('/home/fb635/fedirfiles/tracing_cosmic_gas/data/delta_fields')
delta_fields_dir.mkdir(parents=True, exist_ok=True)

print("="*60)
print("Loading/computing 2D projected fields")
print(f"Regime: {regime} | Method: {method} | Mass type: {mass_type}")
print("="*60)

delta_2d_fields = {}
field_labels = ['dm', 'gas', 'halos']

for label in field_labels:
    # Construct filename for stored delta_2d field
    if label == 'halos':
        if method == 'mass_bin':
            halo_suffix = f'_massbin{halo_mass_bin}'
        elif method == 'number_density':
            halo_suffix = f'_ngal{n_gal_density:.0e}'.replace('+', '').replace('-', 'm')
        else:  # all_halos
            halo_suffix = '_allhalos'
        # Add regime suffix (only add if mgal_sel, keep unchanged for mhalo_sel)
        if regime == 'mgal_sel':
            halo_suffix += '_mgal_sel'
    else:
        halo_suffix = ''
    delta_2d_path = delta_fields_dir / f'delta_2d_{label}_{gas_type}_ngrid{ngrid}{halo_suffix}.npy'
    
    if delta_2d_path.exists():
        # Load existing field
        print(f"\nLoading {label} field from {delta_2d_path}...")
        delta_2d_fields[label] = np.load(delta_2d_path)
        print(f"  Loaded successfully. Shape: {delta_2d_fields[label].shape}")
    else:
        # Compute field
        print(f"\n{label} field not found. Computing...")
        
        if label == 'dm':
            file = dd.io.load(particles_sim_file)
            masses = file['DMParticles']['mass']
            pos = file['DMParticles']['pos']
        elif label == 'gas':
            file = dd.io.load(particles_sim_file)
            masses = file['GasParticles']['mass']
            pos = file['GasParticles']['pos']
        elif label == 'halos':
            # Load all data from HDF5 file
            print(f"  Loading data from {haloes_sim_file}...")
            # Load only needed fields to minimize memory usage
            all_mstell = dd.io.load(haloes_sim_file, '/galaxies/mstell')
            all_centrals = dd.io.load(haloes_sim_file, '/galaxies/centrals')
            all_m200b = dd.io.load(haloes_sim_file, '/galaxies/m200b')  # m200b is already m200m
            all_m200c = dd.io.load(haloes_sim_file, '/galaxies/m200c')
            all_pos = dd.io.load(haloes_sim_file, '/galaxies/pos')
            
            # Map selection_mass_def to actual mass array
            mass_options = {
                'm200c': all_m200c,
                'm200b': all_m200b,
                'mstell': all_mstell
            }
            
            if mass_type not in mass_options:
                raise ValueError(f"Unknown mass_type: {mass_type}. Must be 'm200c', 'm200b', or 'mstell'.")
            
            selection_masses = mass_options[mass_type]
            
            print(f"  Using {mass_type} for mass-based selection")
            print(f"  Selection mode: {cen_sat_mode}")
            
            # Apply selection method
            selected_indices = select_halos(
                selection_masses,
                all_centrals,
                n_gal_density,
                halo_mass_bin,
                min_mass=1e9 if cen_sat_mode is None and mass_type == 'mstell' else 1e13,
                select_nonzero_masses=require_nonzero_mass,
                selection_mode=cen_sat_mode,
                LBOX=box,
                upper_cut_mode=upper_cut_mode,
            )
        
            print(f"  Total objects: {len(selected_indices)}")
            
            # Extract selected data
            pos = all_pos[selected_indices]
            masses = all_m200b[selected_indices]  # m200b values
        
            print(f"  Log10 m200b (m200m) range: [{np.log10(np.min(masses)):.2f}, {np.log10(np.max(masses)):.2f}]")
            print(f"  Log10 Mean m200b (m200m): {np.log10(np.mean(masses)):.2f} Msun/h")
            
            # Clean up
            del all_mstell, all_centrals, all_m200b, all_m200c, all_pos
            del selection_masses, selected_indices
        
        if label == 'halos':
            weights = None
        else:
            weights = None
            if not np.all(masses == masses[0]):
                weights = masses / np.mean(masses)
        
        delta_3d = compute_delta(pos, box, ngrid, weights, nthread=nthread)
        delta_2d_fields[label] = compute_2d_projection(delta_3d)

        # Save computed field
        np.save(delta_2d_path, delta_2d_fields[label])
        print(f"  Computed and saved to {delta_2d_path}")
        
        # Clean up
        del delta_3d, masses, pos, weights
        gc.collect()

print("\nAll 2D fields loaded/computed and stored in memory.")
print(f"Memory usage: DM: {delta_2d_fields['dm'].nbytes / 1e9:.2f} GB")
print(f"              Gas: {delta_2d_fields['gas'].nbytes / 1e9:.2f} GB")
print(f"              Halos: {delta_2d_fields['halos'].nbytes / 1e9:.2f} GB")


print("\n" + "="*60)
print("Computing Fourier transforms and transfer function")
print("="*60)

# Compute FFTs
delta_k_dm = compute_2d_fft(delta_2d_fields['dm'], ngrid)
delta_k_gas = compute_2d_fft(delta_2d_fields['gas'], ngrid)
delta_k_halos = compute_2d_fft(delta_2d_fields['halos'], ngrid)

# Compute k-space grid
k_mag = compute_k_grid(ngrid, box)

# Setup k-bins for binning power spectra (linearly spaced)
nkbins = ngrid // 10  # Reduced number of bins for wider bins
k_max = np.max(k_mag)
k_min = 2 * np.pi / box  # Fundamental mode
k_bins = np.linspace(k_min, k_max, nkbins)
k_center = 0.5 * (k_bins[:-1] + k_bins[1:])  # Arithmetic mean for linear bins

# Compute power spectra from halos
P_k_gas_halos = (delta_k_gas * np.conj(delta_k_halos)).real
P_k_dm_halos = (delta_k_dm * np.conj(delta_k_halos)).real

# Bin the cross power spectra by k magnitude
P_gas_halos_binned = []
P_dm_halos_binned = []

for i in range(len(k_bins) - 1):
    mask = (k_mag >= k_bins[i]) & (k_mag < k_bins[i + 1])
    
    P_gas_bin = P_k_gas_halos[mask]
    P_gas_halos_binned.append(np.mean(P_gas_bin) if P_gas_bin.size > 0 else 0)
    
    P_dm_bin = P_k_dm_halos[mask]
    P_dm_halos_binned.append(np.mean(P_dm_bin) if P_dm_bin.size > 0 else 0)

P_gas_halos_binned = np.array(P_gas_halos_binned)
P_dm_halos_binned = np.array(P_dm_halos_binned)

# Compute transfer function as a function of k magnitude
T_k_1d = P_gas_halos_binned / (P_dm_halos_binned)

# Interpolate transfer function back to 2D k-space grid using JAX
# Map k_mag values to indices in k_center array
coords = (k_mag - k_center[0]) / (k_center[-1] - k_center[0]) * (len(k_center) - 1)
T_k_new = map_coordinates(jnp.array(T_k_1d), [coords], order=1)

print(f"Transfer function computed and interpolated to 2D k-space.")
print(f"T_k_1d statistics:")
print(f"  Mean: {np.mean(np.abs(T_k_1d)):.6e}")
print(f"  Std: {np.std(np.abs(T_k_1d)):.6e}")
print(f"  Min: {np.min(np.abs(T_k_1d)):.6e}")
print(f"  Max: {np.max(np.abs(T_k_1d)):.6e}")

# Clean up fields no longer needed
del delta_k_gas, delta_k_halos, P_k_gas_halos, P_k_dm_halos
del P_gas_halos_binned, P_dm_halos_binned, coords
gc.collect()

print("\n" + "="*60)
print("Loading/computing tau map prefactor")
print("="*60)

# Check if prefactor already exists
prefactor_dir = Path('/home/fb635/fedirfiles/tracing_cosmic_gas/data/tau_map_prefactors')
prefactor_dir.mkdir(parents=True, exist_ok=True)
prefactor_path = prefactor_dir / f'tau_prefactor_{gas_type}_ngrid{ngrid}.npy'

if prefactor_path.exists():
    # Load existing prefactor
    print(f"\nLoading prefactor from {prefactor_path}...")
    prefactor = np.load(prefactor_path)
    print(f"  Prefactor loaded: {prefactor:.6e}")
else:
    # Compute prefactor
    print(f"\nPrefactor not found. Computing from particle data...")
    print("  (This may take a while - loading from RDS...)")
    
    file = dd.io.load(particles_sim_file)
    gas_masses = file['GasParticles']['mass']  # 1e10 Msun/h
    gas_masses *= 100  # Adjust for dilution factor
    
    gas_masses_Msun = gas_masses * 1e10 / h  # Convert to Msun
    box_comoving_Mpc = box / h  # Convert from cMpc/h to Mpc
    box_physical_Mpc = box_comoving_Mpc / (1 + z)  # Physical size at z
    
    total_mass_Msun = np.sum(gas_masses_Msun)
    volume_Mpc3 = box_physical_Mpc**3
    rho_bar_Msun_per_Mpc3 = total_mass_Msun / volume_Mpc3
    
    rho_bar_kg_per_m3 = (rho_bar_Msun_per_Mpc3 * u.Msun / u.Mpc**3).to(u.kg / u.m**3)
    cell_physical_m = ((box_physical_Mpc * u.Mpc) / ngrid).to(u.m)
    
    X_H = 0.76
    X_He = 0.24
    f_e = (X_H + 0.5 * X_He) / m_p  # electrons per kg
    
    prefactor = (sigma_T * rho_bar_kg_per_m3 * f_e * cell_physical_m).to(u.dimensionless_unscaled).value
    
    # Save prefactor
    np.save(prefactor_path, prefactor)
    print(f"  Prefactor computed and saved: {prefactor:.6e}")
    print(f"  Saved to: {prefactor_path}")
    
    del file, gas_masses, gas_masses_Msun, rho_bar_kg_per_m3, cell_physical_m, f_e
    del total_mass_Msun, volume_Mpc3, rho_bar_Msun_per_Mpc3
    gc.collect()

print("\n" + "="*60)
print("Reconstructing gas field and computing tau map")
print("="*60)

# Apply transfer function to DM field
delta_k_gas_recon = delta_k_dm * T_k_new

# Inverse FFT to get real space field
delta_gas_reconstructed_2d = np.fft.irfft2(delta_k_gas_recon) * (ngrid ** 2)

# Compute tau map: tau_xy_map = prefactor * (1.0 + delta_gas) * ngrid
tau_xy_map = prefactor * (1.0 + delta_gas_reconstructed_2d) * ngrid

# Construct output filename with regime suffix
regime_suffix = '_mgal_sel' if regime == 'mgal_sel' else ''
tau_out_path = tau_output_dir / f'tau_map_{gas_type}_reconstructed{regime_suffix}.npy'
np.save(tau_out_path, tau_xy_map)


# ============================================================================
# DIAGNOSTIC PLOTS FOR NEW TRANSFER FUNCTION
# ============================================================================
print("\n" + "="*60)
print("Creating diagnostic plots for new transfer function")
print("="*60)

# k_bins and k_center already defined from transfer function computation
k_Nyquist = np.pi * ngrid / box

# Recompute FFTs for diagnostic (some were deleted)
delta_k_gas_true = compute_2d_fft(delta_2d_fields['gas'], ngrid)
delta_k_halos_true = compute_2d_fft(delta_2d_fields['halos'], ngrid)

# Compute true P^gas,halo
P_k_gas_halos_true = (delta_k_gas_true * np.conj(delta_k_halos_true)).real

# Compute reconstructed P^gas,halo
P_k_gas_halos_recon = (delta_k_gas_recon * np.conj(delta_k_halos_true)).real

# Bin the power spectra
P_gas_halos_true_binned = []
P_gas_halos_recon_binned = []

for i in range(len(k_bins) - 1):
    mask = (k_mag >= k_bins[i]) & (k_mag < k_bins[i + 1])
    
    P_true_bin = P_k_gas_halos_true[mask]
    P_gas_halos_true_binned.append(np.mean(P_true_bin) if P_true_bin.size > 0 else 0)
    
    P_recon_bin = P_k_gas_halos_recon[mask]
    P_gas_halos_recon_binned.append(np.mean(P_recon_bin) if P_recon_bin.size > 0 else 0)

P_gas_halos_true_binned = np.array(P_gas_halos_true_binned) * box ** 2
P_gas_halos_recon_binned = np.array(P_gas_halos_recon_binned) * box ** 2

# Plot 1: P^gas,halo comparison
plot_output_dir = Path('/home/fb635/fedirfiles/tracing_cosmic_gas/plots')
plot_output_dir.mkdir(parents=True, exist_ok=True)

if method == 'mass_bin':
    plot_suffix = f'_massbin{halo_mass_bin}'
elif method == 'number_density':
    plot_suffix = f'_ngal{n_gal_density:.0e}'.replace('+', '').replace('-', 'm')
else:  # all_halos
    plot_suffix = '_allhalos'

# Add regime suffix
if regime == 'mgal_sel':
    plot_suffix += '_mgal_sel'

plt.figure(figsize=(6, 4), dpi=300)
plt.loglog(k_center, P_gas_halos_true_binned, c='blue', alpha=0.7, linewidth=1, label='True $P^{gas,halo}$')
plt.loglog(k_center, P_gas_halos_recon_binned, c='red', alpha=0.7, linewidth=1, label='Reconstructed $P^{gas,halo}$')
plt.axvline(k_Nyquist, c='blue', linestyle='--', label='$k_{Nyquist}$')
plt.xlabel('$k$ [h/cMpc]', fontsize=12)
plt.ylabel('$P^{gas,halo}(k)$', fontsize=12)
plt.title('Cross Power Spectrum: Gas-Halo', fontsize=14)
plt.legend()
plt.savefig(plot_output_dir / f'P_gas_halos_recon_vs_true_{gas_type}{plot_suffix}.png', bbox_inches='tight', dpi=300)
plt.close()
print(f"Saved: {plot_output_dir / f'P_gas_halos_recon_vs_true_{gas_type}{plot_suffix}.png'}")

# ============================================================================
# COMPUTE AUTO POWER SPECTRA FOR P^gas,gas COMPARISON
# ============================================================================
print("\nComputing auto power spectra P^gas,gas and P^dm,dm...")

# Compute power spectra on 2D grid
P_gas_gas_true = (delta_k_gas_true * np.conj(delta_k_gas_true)).real
P_gas_gas_recon = (delta_k_gas_recon * np.conj(delta_k_gas_recon)).real
P_dm_dm = (delta_k_dm * np.conj(delta_k_dm)).real

# Bin the auto power spectra by k magnitude
P_gas_gas_true_binned = []
P_gas_gas_recon_binned = []
P_dm_dm_binned = []

for i in range(len(k_bins) - 1):
    mask = (k_mag >= k_bins[i]) & (k_mag < k_bins[i + 1])
    
    P_gas_true_bin = P_gas_gas_true[mask]
    P_gas_gas_true_binned.append(np.mean(P_gas_true_bin) if P_gas_true_bin.size > 0 else 0)
    
    P_gas_recon_bin = P_gas_gas_recon[mask]
    P_gas_gas_recon_binned.append(np.mean(P_gas_recon_bin) if P_gas_recon_bin.size > 0 else 0)
    
    P_dm_bin = P_dm_dm[mask]
    P_dm_dm_binned.append(np.mean(P_dm_bin) if P_dm_bin.size > 0 else 0)

P_gas_gas_true_binned = np.array(P_gas_gas_true_binned) * box ** 2
P_gas_gas_recon_binned = np.array(P_gas_gas_recon_binned) * box ** 2
P_dm_dm_binned = np.array(P_dm_dm_binned) * box ** 2

# Plot 2: P^gas,gas comparison
plt.figure(figsize=(6, 4), dpi=300)
plt.loglog(k_center, P_gas_gas_true_binned, c='blue', alpha=0.7, linewidth=1, label='True $P^{gas,gas}$')
plt.loglog(k_center, P_gas_gas_recon_binned, c='red', alpha=0.7, linewidth=1, label='Reconstructed $P^{gas,gas}$')
plt.axvline(k_Nyquist, c='blue', linestyle='--', label='$k_{Nyquist}$')
plt.xlabel('$k$ [h/cMpc]', fontsize=12)
plt.ylabel('$P^{gas,gas}(k)$', fontsize=12)
plt.title('Auto Power Spectrum: Gas', fontsize=14)
plt.legend()
plt.savefig(plot_output_dir / f'P_gas_gas_recon_vs_true_{gas_type}{plot_suffix}.png', bbox_inches='tight', dpi=300)
plt.close()
print(f"Saved: {plot_output_dir / f'P_gas_gas_recon_vs_true_{gas_type}{plot_suffix}.png'}")

"""# Plot 2: Cross-correlation between reconstructed and true P^gas,halo
P_k_cross_recon_true = (delta_k_gas_recon * np.conj(delta_k_gas_true)).real

P_cross_recon_true_binned = []
for i in range(len(k_bins) - 1):
    mask = (k_mag >= k_bins[i]) & (k_mag < k_bins[i + 1])
    P_cross_bin = P_k_cross_recon_true[mask]
    P_cross_recon_true_binned.append(np.mean(P_cross_bin) if P_cross_bin.size > 0 else 0)

P_cross_recon_true_binned = np.array(P_cross_recon_true_binned) * box ** 2

# Compute auto power spectra for normalization
P_k_gas_recon_auto = (delta_k_gas_recon * np.conj(delta_k_gas_recon)).real
P_k_gas_true_auto = (delta_k_gas_true * np.conj(delta_k_gas_true)).real

P_gas_recon_auto_binned = []
P_gas_true_auto_binned = []

for i in range(len(k_bins) - 1):
    mask = (k_mag >= k_bins[i]) & (k_mag < k_bins[i + 1])
    
    P_recon_auto_bin = P_k_gas_recon_auto[mask]
    P_gas_recon_auto_binned.append(np.mean(P_recon_auto_bin) if P_recon_auto_bin.size > 0 else 0)
    
    P_true_auto_bin = P_k_gas_true_auto[mask]
    P_gas_true_auto_binned.append(np.mean(P_true_auto_bin) if P_true_auto_bin.size > 0 else 0)

P_gas_recon_auto_binned = np.array(P_gas_recon_auto_binned) * box ** 2
P_gas_true_auto_binned = np.array(P_gas_true_auto_binned) * box ** 2

# Cross-correlation coefficient
r_recon_vs_true = P_cross_recon_true_binned / np.sqrt(P_gas_recon_auto_binned * P_gas_true_auto_binned)

plt.figure(figsize=(6, 4), dpi=300)
plt.semilogx(k_center, r_recon_vs_true, c='red', alpha=0.7, label='Reconstructed vs True Gas')
plt.axhline(1.0, c='grey', linestyle='--', label='Perfect Correlation')
plt.axvline(k_Nyquist, c='blue', linestyle='--', label='$k_{Nyquist}$')
plt.xlabel('$k$ [h/cMpc]', fontsize=12)
plt.ylabel('$r(k)$', fontsize=12)
plt.title('Cross-Correlation Coefficient', fontsize=14)
plt.ylim([0, 1.1])
plt.legend()
plt.savefig(plot_output_dir / f'r_recon_vs_true_{gas_type}.png', bbox_inches='tight', dpi=300)
plt.close()
print(f"Saved: {plot_output_dir / f'r_recon_vs_true_{gas_type}.png'}")

# Clean up
del delta_k_gas_true, delta_k_halos_true, P_k_gas_halos_true, P_k_gas_halos_recon
del P_k_cross_recon_true, P_k_gas_recon_auto, P_k_gas_true_auto
del P_gas_halos_true_binned, P_gas_halos_recon_binned, P_cross_recon_true_binned
del P_gas_recon_auto_binned, P_gas_true_auto_binned, r_recon_vs_true
gc.collect()"""

# ============================================================================
# OPTIONAL: PLOT T'(k) vs T(k) = T_old(k) * (1 + Ak^2)
# ============================================================================
plot_comparison = True  # Set to False to skip this step

if plot_comparison:
    print("\n" + "="*60)
    print("Plotting T'(k) vs T(k) with mass-dependent correction")
    print("="*60)
    
    # Set A parameter
    A1 = -0.02341
    A2 = -0.0000182
    
    # Compute transfer function from binned power spectra (already computed above)
    # Remove box^2 normalization for transfer function computation
    T_k_old_1d = np.sqrt(P_gas_gas_true_binned / P_dm_dm_binned)
    
    # Interpolate transfer function back to 2D k-space grid using JAX
    coords_old = (k_mag - k_center[0]) / (k_center[-1] - k_center[0]) * (len(k_center) - 1)
    T_k_old = map_coordinates(jnp.array(T_k_old_1d), [coords_old], order=1)
    
    # Compute mass-dependent transfer function: T(k) = T_old(k) * (1 + Ak^2)
    one_plus_Ak2 = 1.0 + A1 * k_mag**2
    one_plus_Ak4 = 1.0 + A2 * k_mag**4
    T_k_massdep = T_k_old * one_plus_Ak2
    
    # Bin all three transfer functions and the correction factors by k
    T_k_new_binned = []
    T_k_old_binned = []
    T_k_massdep_binned = []
    one_plus_Ak2_binned = []
    one_plus_Ak4_binned = []
    
    for i in range(len(k_bins) - 1):
        mask = (k_mag >= k_bins[i]) & (k_mag < k_bins[i + 1])
        
        T_new_bin = T_k_new[mask]
        T_k_new_binned.append(np.mean(T_new_bin) if T_new_bin.size > 0 else 0)
        
        T_old_bin = T_k_old[mask]
        T_k_old_binned.append(np.mean(T_old_bin) if T_old_bin.size > 0 else 0)
        
        T_massdep_bin = T_k_massdep[mask]
        T_k_massdep_binned.append(np.mean(T_massdep_bin) if T_massdep_bin.size > 0 else 0)
        
        correction_k2_bin = one_plus_Ak2[mask]
        one_plus_Ak2_binned.append(np.mean(correction_k2_bin) if correction_k2_bin.size > 0 else 0)
        
        correction_k4_bin = one_plus_Ak4[mask]
        one_plus_Ak4_binned.append(np.mean(correction_k4_bin) if correction_k4_bin.size > 0 else 0)
    
    T_k_new_binned = np.array(T_k_new_binned)
    T_k_old_binned = np.array(T_k_old_binned)
    T_k_massdep_binned = np.array(T_k_massdep_binned)
    one_plus_Ak2_binned = np.array(one_plus_Ak2_binned)
    one_plus_Ak4_binned = np.array(one_plus_Ak4_binned)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 7))
    
    ax.loglog(k_center, T_k_new_binned, c='blue', linewidth=2, label=r"$T'(k) = P^{gas,halo} / P^{dm,halo}$")
    ax.loglog(k_center, T_k_old_binned, c='green', linewidth=2, linestyle='-.', 
              label=r"$T_{\rm old}(k) = \sqrt{P^{gas,gas} / P^{dm,dm}}$")
    ax.loglog(k_center, T_k_massdep_binned, c='red', linewidth=2, linestyle='--', 
              label=r"$T(k) = T_{\rm old}(k) \times (1 + Ak^2)$")
    ax.loglog(k_center, one_plus_Ak2_binned, c='orange', linewidth=2, linestyle=':', 
              label=r"$(1 + Ak^2)$")
    ax.loglog(k_center, one_plus_Ak4_binned, c='purple', linewidth=2, linestyle=':', 
              label=r"$(1 + Ak^4)$")
    ax.axvline(k_Nyquist, c='grey', linestyle=':', linewidth=1.5, label='$k_{Nyquist}$')
    
    ax.set_xlabel(r'$k$ [h/cMpc]', fontsize=14)
    ax.set_ylabel(r"Transfer Function", fontsize=14)
    ax.set_title(r"Comparison: New vs Mass-Dependent Transfer Functions (A1 = {:.5f}, A2 = {:.7f})".format(A1, A2), fontsize=16)
    ax.legend(fontsize=12)
    
    # Save plot
    plot_output_dir = Path('/home/fb635/fedirfiles/tracing_cosmic_gas/plots')
    plot_output_dir.mkdir(parents=True, exist_ok=True)
    if method == 'mass_bin':
        plot_suffix_massdep = f'_massbin{halo_mass_bin}'
    elif method == 'number_density':
        plot_suffix_massdep = f'_ngal{n_gal_density:.0e}'.replace('+', '').replace('-', 'm')
    else:  # all_halos
        plot_suffix_massdep = '_allhalos'
    
    # Add regime suffix
    if regime == 'mgal_sel':
        plot_suffix_massdep += '_mgal_sel'
    
    plot_path = plot_output_dir / f'T_new_vs_T_massdep_{gas_type}{plot_suffix_massdep}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_path}")
    plt.close()
    
    # Clean up
    del T_k_old_1d, T_k_old, coords_old, one_plus_Ak2, one_plus_Ak4, T_k_massdep
    del T_k_new_binned, T_k_old_binned, T_k_massdep_binned, one_plus_Ak2_binned, one_plus_Ak4_binned
    gc.collect()

# Clean up remaining diagnostic variables
del delta_k_gas_true, delta_k_halos_true, P_k_gas_halos_true, P_k_gas_halos_recon
del P_gas_halos_true_binned, P_gas_halos_recon_binned
del P_gas_gas_true, P_gas_gas_recon, P_dm_dm
del P_gas_gas_true_binned, P_gas_gas_recon_binned, P_dm_dm_binned
gc.collect()

print("\n" + "="*60)
print("Reconstruction completed successfully!")
print("="*60)