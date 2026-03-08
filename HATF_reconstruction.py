import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.fft import rfft2, rfftfreq, fftfreq
from abacusnbody.analysis.tsc import tsc_parallel
from jax.scipy.ndimage import map_coordinates
from jax import numpy as jnp
import deepdish as dd
from pathlib import Path
from astropy import units as u
from astropy.constants import m_p, sigma_T
import gc
from utils import *

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

gas_type = 'strongest_AGN'  # 'fiducial', 'strongest_AGN'

# Regime: determines which mass to use for selection
regime = 'mgal_sel'  # 'mhalo_sel' (halo mass) or 'mgal_sel' (stellar mass)

# Central/satellite mode: determines filtering by central/satellite status
if regime == 'mhalo_sel':
    cen_sat_mode = 'cen'          
    mass_type = 'm200c'            # Use m200c for ranking
    require_nonzero_mass = False
    upper_mass_cut = False
    upper_radius_cut = False
elif regime == 'mgal_sel':
    cen_sat_mode = 'mixed'     # Use mixed mode for centrals+satellites
    mass_type = 'mstell'           # Use stellar mass for ranking
    require_nonzero_mass = True
    upper_mass_cut = False
    upper_radius_cut = False        # Apply r200 radius cut for satellites

# Selection method: determines how to select from ranked objects
# To run either, set the other to None.
halo_mass_range = None #[0, 1000]   
n_gal_density = 2e-3   # (cMpc/h)^-3, mhalo_sel - 1.22e-3, mgal_sel - 2e-2, 
                       # mixed with nonzero filter and upper cut 14 - 1.42e-3, 
                       # satellites test uncut - 5.8e-3, satellites test with upper cut 14.2 - 6.6e-3, 
                       # satellites test with upper cut 13.8 - 4.4e-3, 
                       # mixed with nonzero filter and r200 cut - 7e-3
                       # mfof selection - 7e-3

# Target mean mass mode: if set, iteratively adjust selection to match this target log10(mean mass)
target_mean_mass = 13.2  # 13.2 to target log10(<M>) = 13.2, or None to use fixed n_gal_density/halo_mass_bin
target_mass_tolerance = 0.005  # tolerance for convergence (in log10 units)
mass_bin_halfwidth = 0.02  # ±dex span target for convergence
mass_bin_halfwidth_tol = 0.001
max_iterations = 100  # maximum iterations for target mass matching

# Satellite fraction for 'mixed' mode
sat_frac = 0.30  # 0.00, 0.10, 0.20, 0.30 for the sweep experiment

# Determine selection method for file naming
if target_mean_mass is not None:
    method = 'target_mass'
    target_mass_str = f'{target_mean_mass:.1f}'.replace('.', 'p')
elif halo_mass_range is not None and target_mean_mass is None:
    method = 'mass_bin'
elif n_gal_density is not None and target_mean_mass is None:
    method = 'number_density'
else:
    method = 'all_halos'

# File paths from pipeline_paths
particles_sim_file = particles_hdf5(gas_type)
haloes_sim_file = halo_galaxy_hdf5(gas_type)

box = 681  # cMpc/h
ngrid = 2048
nthread = 4

# Cosmology parameters
h = 0.681
Z_SIM = 0.74

# Build unified config dict for pipeline_paths
_config = dict(
    gas_type=gas_type + '_reconstructed',  # Add _reconstructed suffix for tau map paths
    tau_method='2D_FT_upgrade_tau_reconstruction',
    selection_mode=cen_sat_mode,
    selection_mass_def=mass_type,
    select_nonzero_masses=require_nonzero_mass,
    upper_mass_cut=upper_mass_cut,
    upper_radius_cut=upper_radius_cut,
    sat_frac=sat_frac,
    n_gal_density=n_gal_density,
    halo_mass_range=halo_mass_range,  # Now [start_idx, win_size] or None
    target_mean_mass=target_mean_mass,
)

# Output directories
delta_fields_dir = Path('/home/fb635/fedirfiles/tracing_cosmic_gas/data/delta_fields')
delta_fields_dir.mkdir(parents=True, exist_ok=True)

print("="*60)
print("Loading/computing 2D projected fields")
print(f"Regime: {regime} | Method: {method} | Mass type: {mass_type}")
print("="*60)

delta_2d_fields = {}
field_labels = ['dm', 'gas', 'halos']

for label in field_labels:
    # Use pipeline_paths to get the correct path
    # Update _config to remove _reconstructed suffix for delta fields
    delta_config = dict(_config)
    delta_config['gas_type'] = gas_type  # Remove _reconstructed suffix
    delta_2d_path_val = delta_2d_path(delta_config, label)
    
    if delta_2d_path_val.exists():
        # Load existing field
        print(f"\nLoading {label} field from {delta_2d_path_val}...")
        delta_2d_fields[label] = np.load(delta_2d_path_val)
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
            all_mstell = dd.io.load(haloes_sim_file, '/galaxies/mstell_50kpc')
            all_mstell_fof = dd.io.load(haloes_sim_file, '/galaxies/mstell_fof')
            all_centrals = dd.io.load(haloes_sim_file, '/galaxies/centrals')
            all_m200b = dd.io.load(haloes_sim_file, '/galaxies/m200b')  # m200b is already m200m
            all_m200c = dd.io.load(haloes_sim_file, '/galaxies/m200c')
            all_mfof = dd.io.load(haloes_sim_file, '/galaxies/TotalMass_fof')
            all_r200b = dd.io.load(haloes_sim_file, '/galaxies/r200b')
            all_pos = dd.io.load(haloes_sim_file, '/galaxies/pos')
            all_haloID = dd.io.load(haloes_sim_file, '/galaxies/haloID')

            # Sum Mfof for each halo (central + satellites) and assign to all members
            if mass_type == 'mstell_fof' or mass_type == 'mfof':
                print("  Summing Mfof for each halo group (central + satellites)...")
                unique_haloIDs, inverse_indices = np.unique(all_haloID, return_inverse=True)
                mfof_sums = np.bincount(inverse_indices, weights=all_mfof)
                all_mfof = mfof_sums[inverse_indices]
                print(f"    Processed {len(unique_haloIDs)} unique halos")

            # Map selection_mass_def to actual mass array
            mass_options = {
                'm200c': all_m200c,
                'm200b': all_m200b,
                'mstell': all_mstell,
                'mstell_fof': all_mstell_fof,
                'mfof': all_mfof,
            }
            
            if mass_type not in mass_options:
                raise ValueError(f"Unknown mass_type: {mass_type}. Must be 'm200c', 'm200b', 'mstell', 'mstell_fof', or 'mfof'.")
            
            selection_masses = mass_options[mass_type]
            
            # Validate consistency for mass bin selection mode
            if target_mean_mass is not None and halo_mass_range is not None:
                if mass_type in ('mstell', 'mstell_fof'):
                    raise ValueError(
                        f"Mass bin selection (halo_mass_range) requires HALO mass types. "
                        f"Got mass_type='{mass_type}' (stellar mass). "
                        f"Use 'm200b', 'm200c', or 'mfof' for consistency between sorting and binning."
                    )
                if cen_sat_mode == 'mixed':
                    raise ValueError(
                        f"Mass bin selection (halo_mass_range) is incompatible with cen_sat_mode='mixed'. "
                        f"Use 'cen' or 'sat' mode to preserve mass-sorted order."
                    )
            
            print(f"    Using {mass_type} for mass-based selection")
            print(f"    Selection mode: {cen_sat_mode}")
            
            # Determine if we're in target mass mode or fixed selection mode
            if target_mean_mass is not None:
                print(f"\n    TARGET MASS MODE: Iteratively adjusting to reach log10(<M>) = {target_mean_mass:.2f}")
                
                # Separate handling for mass_bin vs n_gal_density
                if halo_mass_range is not None:
    
                    param_type = 'mass_bin'
                    print(f"    Using halo_mass_range (window into filtered pool)")
                    print(f"\n    log10(<M>) = {target_mean_mass:.2f} "
                        f"±{mass_bin_halfwidth:.2f} dex")

                    # --- Step 0: obtain the full filtered, ordered pool once ---
                    pool_indices = select_halos(
                        selection_masses, all_centrals,
                        n_gal_density=None, halo_mass_range=None,
                        min_mass=1e9 if mass_type in ('mstell', 'mstell_fof') else 1e13,
                        select_nonzero_masses=require_nonzero_mass,
                        selection_mode=cen_sat_mode, sat_frac=sat_frac, LBOX=box,
                        upper_mass_cut=upper_mass_cut, m200b=all_m200b,
                        upper_radius_cut=upper_radius_cut, pos=all_pos,
                        haloID=all_haloID, r200=all_r200b,
                    )
        
                    pool_masses = selection_masses[pool_indices]
                    pool_log_masses = np.log10(pool_masses)
                    pool_centrals = all_centrals[pool_indices]

                    N_pool = len(pool_indices)
                    win_size = min(1000, N_pool)
                    win_step = 100
                    win_size_step = 100
                    
                    for iteration in range(max_iterations):
                        # Slide window from end to start to find position with target mean mass
                        start = N_pool - win_size
                        window_sum = np.sum(pool_masses[start:start + win_size])
                        best_start, best_error = start, np.inf

                        while start >= 0:
                            window_mean_log = np.log10(window_sum / win_size)
                            mass_error = abs(window_mean_log - target_mean_mass)

                            if mass_error < best_error:
                                best_error, best_start = mass_error, start

                            if mass_error < target_mass_tolerance:
                                break

                            step = min(win_step, start)
                            if step == 0:
                                break
                            window_sum -= np.sum(pool_masses[start + win_size - step:start + win_size])
                            window_sum += np.sum(pool_masses[start - step:start])
                            start -= step

                        w_start = best_start
                        w_end = w_start + win_size
                        
                        # Check span at this position
                        window_masses = pool_masses[w_start:w_end]
                        window_log = pool_log_masses[w_start:w_end]
                        window_mean_log = np.log10(np.mean(window_masses))
                        mass_error = window_mean_log - target_mean_mass
                        span_lo = window_mean_log - float(np.min(window_log))
                        span_hi = float(np.max(window_log)) - window_mean_log

                        # Check convergence: either span reaches target
                        if span_lo >= mass_bin_halfwidth or span_hi >= mass_bin_halfwidth:
                            print(f"    ✓ Converged: win_start={w_start}, win_size={win_size}, "
                                  f"log10(<M>)={window_mean_log:.4f}, span=[-{span_lo:.3f}, +{span_hi:.3f}] dex")
                            break

                        # Grow window and repeat
                        win_size = min(N_pool, win_size + win_size_step)
                        if win_size >= N_pool:
                            w_start = 0
                            win_size = N_pool
                            print(f"    ⚠ Window reached full pool size: {N_pool}")
                            break
                    else:
                        print(f"    ⚠ Max iterations reached, using: win_start={w_start}, win_size={win_size}")

                    # --- Extract selected indices directly from pool (no redundant select_halos call) ---
                    selected_indices = pool_indices[w_start:w_start + win_size]
                    
                    halo_mass_range = [int(w_start), int(win_size)]
                    _config['halo_mass_range'] = halo_mass_range

                elif n_gal_density is not None:
            
                    param_type = 'n_gal_density'
                    current_param = n_gal_density
                    print(f"    Adjusting n_gal_density (starting from {current_param:.2e})")
                    
                    for iteration in range(max_iterations):
                        selected_indices = select_halos(
                            selection_masses, all_centrals, current_param, None,
                            min_mass=1e9 if (mass_type == 'mstell' or mass_type == 'mstell_fof') else 1e13,
                            select_nonzero_masses=require_nonzero_mass,
                            selection_mode=cen_sat_mode, sat_frac=sat_frac, LBOX=box,
                            upper_mass_cut=upper_mass_cut, m200b=all_m200b,
                            upper_radius_cut=upper_radius_cut, pos=all_pos,
                            haloID=all_haloID, r200=all_r200b,
                            verbose=False,  # Suppress prints during convergence iterations
                        )
                    
                        # Compute mean mass
                        if mass_type == 'mstell_fof' or mass_type == 'mfof':
                            masses_iter = all_mfof[selected_indices]
                        else:
                            masses_iter = all_m200b[selected_indices]
                        
                        mean_log_mass = np.log10(np.mean(masses_iter))
                        mass_error = mean_log_mass - target_mean_mass
                    
                        # Check convergence
                        if abs(mass_error) < target_mass_tolerance:
                            print(f"    ✓ Converged: n_gal_density={current_param:.3e}, log10(<M>)={mean_log_mass:.4f}")
                            break
                    
                        adjustment_factor = 1.0 + 0.5 * abs(mass_error)
                        if mass_error > 0:  # Mean mass too high, increase density
                            current_param *= adjustment_factor
                        else:  # Mean mass too low, decrease density
                            current_param /= adjustment_factor
                    else:
                        print(f"    ⚠ Max iterations reached: n_gal_density={current_param:.3e}, log10(<M>)={mean_log_mass:.4f}")
                    
                    # Update config for file naming
                    n_gal_density = current_param
                    _config['n_gal_density'] = n_gal_density
                
                else:
                    raise ValueError("Either halo_mass_range or n_gal_density must be set for target mass mode")
                
            else:
                # Fixed selection mode (original behavior)
                print(f"\n  FIXED SELECTION MODE")
                selected_indices = select_halos(
                    selection_masses, all_centrals, n_gal_density, halo_mass_range,
                    min_mass=1e9 if (mass_type == 'mstell' or mass_type == 'mstell_fof') else 1e13,
                    select_nonzero_masses=require_nonzero_mass,
                    selection_mode=cen_sat_mode, sat_frac=sat_frac, LBOX=box,
                    upper_mass_cut=upper_mass_cut, m200b=all_m200b,
                    upper_radius_cut=upper_radius_cut, pos=all_pos,
                    haloID=all_haloID, r200=all_r200b,
                )
        
            print(f"\n  Total objects selected: {len(selected_indices)}")
            
            # Extract selected data
            pos = all_pos[selected_indices]
            # Use the same mass_type as selection_masses for consistency
            if mass_type == 'mstell_fof' or mass_type == 'mfof':
                masses = all_mfof[selected_indices]
                mass_type_label = 'Mfof (FOF total)'  # Label for reporting
            else:
                masses = all_m200b[selected_indices] 
                mass_type_label = 'm200b'  # Label for reporting
            
            print(f"  Log10 {mass_type_label} range: [{np.log10(np.min(masses)):.2f}, {np.log10(np.max(masses)):.2f}]")
            print(f"  Log10 Mean {mass_type_label}: {np.log10(np.mean(masses)):.2f} Msun/h")
            sat_mask = (all_centrals[selected_indices] == 0)
            print(f"  Satellite fraction: {np.sum(sat_mask) / len(sat_mask):.3f}")
            
            # Clean up
            del all_mstell, all_centrals, all_m200b, all_m200c, all_pos, all_haloID, all_r200b, all_mfof, all_mstell_fof
            if mass_type == 'mstell_fof':
                del unique_haloIDs, inverse_indices, mfof_sums
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
        ensure_parents(delta_2d_path_val)
        np.save(delta_2d_path_val, delta_2d_fields[label])
        print(f"  Computed and saved to {delta_2d_path_val}")
        
        # Clean up
        del delta_3d, masses, pos, weights
        gc.collect()

# Update _config to include converged values in subsequent filenames (tau map)
# Remove target_mean_mass so _selection_tag treats this as fixed selection mode
if target_mean_mass is not None:
    _config.pop('target_mean_mass', None)
    print(f"\nConverged parameters now in _config for tau map filename:")
    if _config.get('halo_mass_range') is not None:
        mr = _config['halo_mass_range']
        print(f"  halo_mass_range: start={mr[0]}, size={mr[1]}")
    if _config.get('n_gal_density') is not None:
        print(f"  n_gal_density: {_config['n_gal_density']:.3e}")

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

# Use pipeline_paths to get prefactor path
prefactor_config = dict(_config)
prefactor_config['gas_type'] = gas_type  # Remove _reconstructed suffix
prefactor_path = tau_prefactor_path(prefactor_config)
ensure_parents(prefactor_path)

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
    box_physical_Mpc = box_comoving_Mpc / (1 + Z_SIM)  # Physical size at Z_SIM
    
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

# Use pipeline_paths to get output path
tau_out_path = tau_map_path(_config)
ensure_parents(tau_out_path)
np.save(tau_out_path, tau_xy_map)
print(f"Tau map saved to: {tau_out_path}")

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
sel_tag = _selection_tag(_config).lstrip('_')

plt.figure(figsize=(6, 4), dpi=300)
plt.loglog(k_center, P_gas_halos_true_binned, c='blue', alpha=0.7, linewidth=1, label='True $P^{gas,halo}$')
plt.loglog(k_center, P_gas_halos_recon_binned, c='red', alpha=0.7, linewidth=1, label='Reconstructed $P^{gas,halo}$')
plt.axvline(k_Nyquist, c='blue', linestyle='--', label='$k_{Nyquist}$')
plt.xlabel('$k$ [h/cMpc]', fontsize=12)
plt.ylabel('$P^{gas,halo}(k)$', fontsize=12)
plt.title('Cross Power Spectrum: Gas-Halo', fontsize=14)
plt.legend()
p1 = plot_path('hatf/power_spectra', gas_type, stem=f'P_gas_halo_{sel_tag}')
ensure_parents(p1)
plt.savefig(p1, bbox_inches='tight', dpi=300)
plt.close()
print(f"Saved: {p1}")

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
p2 = plot_path('hatf/power_spectra', gas_type, stem=f'P_gas_auto_{sel_tag}')
ensure_parents(p2)
plt.savefig(p2, bbox_inches='tight', dpi=300)
plt.close()
print(f"Saved: {p2}")

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
plot_comparison = False  # Set to False to skip this step

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
    p3 = plot_path('hatf/transfer_fn', gas_type, stem=f'T_k_{sel_tag}')
    ensure_parents(p3)
    plt.savefig(p3, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {p3}")
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