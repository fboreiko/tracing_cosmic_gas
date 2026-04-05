import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from jax.scipy.ndimage import map_coordinates
from jax import numpy as jnp
from pathlib import Path
import gc
from utils.catalog_loaders import load_particle_properties
from utils.delta_fields import compute_delta_field_and_mass, compute_selected_halo_delta_2d
from utils.power_spectrum_utils import (
    compute_2d_fft,
    compute_k_grid_2d,
    bin_power_spectrum_2d,
    interp_extrapolate_loglog,
)
from utils.tau_prefactor import compute_prefactor
from utils.pipeline_paths import (
    delta_2d_path as _delta_2d_path,
    tau_prefactor_path as _prefactor_path,
    tau_map_path as _tau_map_path,
    halo_indices_path as _halo_indices_path,
    halo_props_cache_path as _halo_props_cache_path,
    ensure_parents,
    plot_path,
    _selection_tag,
    get_particle_file_path,
)
from utils.sim_params import get_sim_params

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Computer Modern']
rcParams['text.usetex'] = False

SIM_NAME = 'abacus'
SIM_PARAMS = get_sim_params(SIM_NAME)

FEEDBACK_MODE = 'fiducial'  # 'fiducial', 'strongest_AGN'

# Regime: determines which mass to use for selection
regime = 'mhalo_sel'  # 'mhalo_sel' (halo mass) or 'mgal_sel' (stellar mass)

# Central/satellite mode: determines filtering by central/satellite status
if regime == 'mhalo_sel':
    cen_sat_mode = 'cen'          
    mass_type = 'm200b'            # Use m200b for ranking
    require_nonzero_mass = True
    upper_mass_cut = False
    upper_radius_cut = False
elif regime == 'mgal_sel':
    cen_sat_mode = 'cen'    
    mass_type = 'm200b'           # Use stellar mass for ranking
    require_nonzero_mass = True
    upper_mass_cut = False
    upper_radius_cut = False        # Apply r200 radius cut for satellites
else:
    raise ValueError(f"Unknown regime: {regime}. Use 'mhalo_sel' or 'mgal_sel'.")

# Selection method: determines how to select from ranked objects
# To run either, set the other to None.
halo_mass_range = [0, 1000]   
n_gal_density = None   # (cMpc/h)^-3, mhalo_sel - 1.22e-3, mgal_sel - 2e-2, 
                       # mixed with nonzero filter and upper cut 14 - 1.42e-3, 
                       # satellites test uncut - 5.8e-3, satellites test with upper cut 14.2 - 6.6e-3, 
                       # satellites test with upper cut 13.8 - 4.4e-3, 
                       # mixed with nonzero filter and r200 cut - 7e-3
                       # mfof selection - 7e-3

_convergence_mode = 'massbin' if (halo_mass_range is not None and n_gal_density is None) else 'ngal'

# Target mean mass mode: if set, iteratively adjust selection to match this target log10(mean mass)
target_mean_mass = 13.2  # 13.2 to target log10(<M>) = 13.2, or None to use fixed n_gal_density/halo_mass_bin
target_mass_tolerance = 0.005  # tolerance for convergence (in log10 units)
mass_bin_halfwidth = 0.02  # ±dex span target for convergence
mass_bin_halfwidth_tol = 0.001
max_iterations = 1000  # maximum iterations for target mass matching

# Satellite fraction for 'mixed' mode
sat_frac = 0.10  # 0.00, 0.10, 0.20, 0.30 for the sweep experiment

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

if SIM_NAME == 'abacus':
    if mass_type not in ('m200b'):
        raise ValueError(
            f"For sim='abacus', mass_type must be 'm200b' (mass-only catalog). Got: {mass_type}"
        )
if cen_sat_mode in ('sat', 'mixed', 'cens_sat'):
    raise ValueError(
        f"For sim='abacus', cen_sat_mode='{cen_sat_mode}' is unsupported (no satellite labels available). "
        "Use cen_sat_mode='cen' or None."
    )

dm_particles_file_abacus = get_particle_file_path(
    FEEDBACK_MODE,
    sim_name='abacus',
)
dm_particles_file_flamingo = get_particle_file_path(
    FEEDBACK_MODE,
    sim_name='flamingo',
)
gas_particles_file_flamingo = get_particle_file_path(
    FEEDBACK_MODE,
    sim_name='flamingo',
)

box_dm = SIM_PARAMS['box_size_cMpc_h'] 
box_gas = get_sim_params('flamingo')['box_size_cMpc_h']

ngrid_dm = SIM_PARAMS['ngrid_default']
ngrid_gas = get_sim_params('flamingo')['ngrid_default']

print("\nParameter summary:")
print(f"box_dm: {box_dm} cMpc/h, ngrid_dm: {ngrid_dm}")
print(f"box_gas: {box_gas} cMpc/h, ngrid_gas: {ngrid_gas}")

nthread = SIM_PARAMS['nthread_default']

# Build unified config dict for pipeline_paths
_config = dict(
    sim_name=SIM_NAME,
    gas_type=FEEDBACK_MODE,  # Add _reconstructed suffix for tau map paths
    tau_method='2D_FT_upgrade_tau_reconstruction',
    selection_mode=cen_sat_mode,
    selection_mass_def=mass_type,
    select_nonzero_masses=require_nonzero_mass,
    upper_mass_cut=upper_mass_cut,
    upper_radius_cut=upper_radius_cut,
    sat_frac=sat_frac,
    n_gal_density=n_gal_density,
    halo_mass_range=halo_mass_range,
    target_mean_mass=target_mean_mass,
)

# Output directories
delta_fields_dir = _delta_2d_path({'sim_name': SIM_NAME, 'gas_type': FEEDBACK_MODE}, 'dm').parent
delta_fields_dir.mkdir(parents=True, exist_ok=True)

# Path where selected halo indices will be saved/loaded
# Use the sim-specific config for each branch (abacus/flamingo)
_halo_idx_path_flamingo = _halo_indices_path(dict(_config, sim_name='flamingo'))
_halo_idx_path_abacus = _halo_indices_path(dict(_config, sim_name='abacus')) if SIM_NAME == 'abacus' else None

_halo_props_cache_path_flamingo = (
    _halo_props_cache_path(dict(_config, sim_name='flamingo'))
    if _convergence_mode == 'massbin' else None
)
_halo_props_cache_path_abacus = (
    _halo_props_cache_path(dict(_config, sim_name='abacus'))
    if (_convergence_mode == 'massbin' and SIM_NAME == 'abacus') else None
)

print("="*60)
print("Loading/computing 2D projected fields")
print(f"Regime: {regime} | Method: {method} | Mass type: {mass_type}")
print("="*60)

delta_2d_fields = {}

print("\nPreparing projected fields...")

gas_config = dict(_config)
gas_config['sim_name'] = 'flamingo'
gas_path = _delta_2d_path(gas_config, 'gas')

if gas_path.exists():
    print(f"Loading gas field from {gas_path}...")
    delta_2d_fields['gas'] = np.load(gas_path)
    print(f"  Loaded successfully. Shape: {delta_2d_fields['gas'].shape}")
else:
    print("Gas field not found. Computing FLAMINGO gas field...")
    delta_2d_fields['gas'], _ = compute_delta_field_and_mass(
        field_type='gas',
        sim_name='flamingo',
        dm_particles_file=None,
        gas_particles_file=gas_particles_file_flamingo,
        box=box_gas,
        ngrid=ngrid_gas,
        nthread=nthread,
    )
    ensure_parents(gas_path)
    np.save(gas_path, delta_2d_fields['gas'])
    print(f"  Computed and saved to {gas_path}")
    gc.collect()

if SIM_NAME == 'abacus':
    print("\n--- ABACUS delta field branch ---")

    # DM field for Abacus
    abacus_dm_config = dict(_config)
    abacus_dm_config['gas_type'] = 'strongest_AGN'  # Use the same DM field for both feedback modes since it's identical
    abacus_dm_path = _delta_2d_path(abacus_dm_config, 'dm')

    if abacus_dm_path.exists():
        print(f"Loading dm field from {abacus_dm_path}...")
        delta_2d_fields['dm'] = np.load(abacus_dm_path)
        print(f"  Loaded successfully. Shape: {delta_2d_fields['dm'].shape}")
    else:
        print("DM field not found. Computing Abacus DM field...")
        delta_2d_fields['dm'], _ = compute_delta_field_and_mass(
            field_type='dm',
            sim_name='abacus',
            dm_particles_file=dm_particles_file_abacus,
            gas_particles_file=None,
            box=box_dm,
            ngrid=ngrid_dm,
            nthread=nthread,
        )
        ensure_parents(abacus_dm_path)
        np.save(abacus_dm_path, delta_2d_fields['dm'])
        print(f"  Computed and saved to {abacus_dm_path}")
        gc.collect()
    
    # Halo field for Abacus
    print("Loading/computing halo fields...")
    delta_2d_fields['halos_abacus'] = compute_selected_halo_delta_2d(
        sim_for_halos='abacus',
        gas_type=FEEDBACK_MODE,
        box=box_dm,
        ngrid=ngrid_dm,
        selection_config=_config,
        save_path=_delta_2d_path(_config, 'halos'),
        halo_indices_save_path=_halo_idx_path_abacus,
        halo_props_cache_save_path=_halo_props_cache_path_abacus,
        convergence_mode=_convergence_mode,
        nthread=nthread,
    )

    _config_flamingo = dict(_config)
    _config_flamingo['sim_name'] = 'flamingo'

    delta_2d_fields['halos_flamingo'] = compute_selected_halo_delta_2d(
        sim_for_halos='flamingo',
        gas_type=FEEDBACK_MODE,
        box=box_gas,
        ngrid=ngrid_gas,
        selection_config=_config_flamingo,
        save_path=_delta_2d_path(_config_flamingo, 'halos'),
        halo_indices_save_path=_halo_idx_path_flamingo,
        halo_props_cache_save_path=_halo_props_cache_path_flamingo,
        convergence_mode=_convergence_mode,
        nthread=nthread,
    )
    delta_2d_fields['halos'] = delta_2d_fields['halos_abacus']

    # FLAMINGO DM field used for the transfer-function construction
    flamingo_dm_path = _delta_2d_path(_config_flamingo, 'dm')
    if flamingo_dm_path.exists():
        print(f"Loading dm_flamingo field from {flamingo_dm_path}...")
        delta_2d_fields['dm_flamingo'] = np.load(flamingo_dm_path)
        print(f"  Loaded successfully. Shape: {delta_2d_fields['dm_flamingo'].shape}")
    else:
        print("DM field (FLAMINGO) not found. Computing...")
        delta_2d_fields['dm_flamingo'], _ = compute_delta_field_and_mass(
            field_type='dm',
            sim_name='flamingo',
            dm_particles_file=dm_particles_file_flamingo,
            gas_particles_file=None,
            box=box_gas,
            ngrid=ngrid_gas,
            nthread=nthread,
        )
        ensure_parents(flamingo_dm_path)
        np.save(flamingo_dm_path, delta_2d_fields['dm_flamingo'])
        print(f"  Computed and saved to {flamingo_dm_path}")
        gc.collect()

else:
    print("\n--- FLAMINGO delta field branch ---")

    # DM field for Flamingo
    flamingo_dm_path = _delta_2d_path(_config, 'dm')

    if flamingo_dm_path.exists():
        print(f"Loading dm field from {flamingo_dm_path}...")
        delta_2d_fields['dm'] = np.load(flamingo_dm_path)
        print(f"  Loaded successfully. Shape: {delta_2d_fields['dm'].shape}")
    else:
        print("DM field not found. Computing Flamingo DM field...")
        delta_2d_fields['dm'], _ = compute_delta_field_and_mass(
            field_type='dm',
            sim_name='flamingo',
            dm_particles_file=dm_particles_file_flamingo,
            gas_particles_file=None,
            box=box_gas,
            ngrid=ngrid_gas,
            nthread=nthread,
        )
        ensure_parents(flamingo_dm_path)
        np.save(flamingo_dm_path, delta_2d_fields['dm'])
        print(f"  Computed and saved to {flamingo_dm_path}")
        gc.collect()

    # Halo field for Flamingo
    delta_2d_fields['halos'] = compute_selected_halo_delta_2d(
        sim_for_halos='flamingo',
        gas_type=FEEDBACK_MODE,
        box=box_gas,
        ngrid=ngrid_gas,
        selection_config=_config,
        save_path=_delta_2d_path(_config, 'halos'),
        halo_indices_save_path=_halo_idx_path_flamingo,
        halo_props_cache_save_path=_halo_props_cache_path_flamingo,
        convergence_mode=_convergence_mode,
        nthread=nthread,
    )

    gc.collect()

print("\nAll 2D fields loaded/computed and stored in memory.")
print(f"Memory usage: DM: {delta_2d_fields['dm'].nbytes / 1e9:.2f} GB")
print(f"              Gas: {delta_2d_fields['gas'].nbytes / 1e9:.2f} GB")
print(f"              Halos: {delta_2d_fields['halos'].nbytes / 1e9:.2f} GB")

print("\n" + "="*60)
print("Computing Fourier transforms and transfer function")
print("="*60)

dm_field_fft_foundation = compute_2d_fft(delta_2d_fields['dm'], ngrid_dm)
gas_field_fft_target = compute_2d_fft(delta_2d_fields['gas'], ngrid_gas)
halo_field_fft = compute_2d_fft(delta_2d_fields['halos'], ngrid_dm)

# Compute k-space grids
k_grid_dm_foundation = compute_k_grid_2d(ngrid_dm, box_dm)
k_grid_gas_target = compute_k_grid_2d(ngrid_gas, box_gas)

if SIM_NAME == 'abacus':
    print("\n--- ABACUS transfer function branch ---")

    flamingo_dm_field_fft = compute_2d_fft(delta_2d_fields['dm_flamingo'], ngrid_gas)
    flamingo_halo_field_fft = compute_2d_fft(delta_2d_fields['halos_flamingo'], ngrid_gas)

    flamingo_k_bins, flamingo_k_center, flamingo_P_dm_halos_binned = bin_power_spectrum_2d(
        (flamingo_dm_field_fft * np.conj(flamingo_halo_field_fft)).real,
        k_grid_gas_target,
        ngrid_gas,
        box_gas,
    )
    _, _, flamingo_P_gas_halos_binned = bin_power_spectrum_2d(
        (gas_field_fft_target * np.conj(flamingo_halo_field_fft)).real,
        k_grid_gas_target,
        ngrid_gas,
        box_gas,
    )

    flamingo_P_dm_halos_binned = np.array(flamingo_P_dm_halos_binned) * box_gas ** 2
    flamingo_P_gas_halos_binned = np.array(flamingo_P_gas_halos_binned) * box_gas ** 2

    eps = 1e-30
    flamingo_T_k_1d = np.maximum(np.abs(flamingo_P_gas_halos_binned), eps) / np.maximum(np.abs(flamingo_P_dm_halos_binned), eps)

    # we matching flamingo number of bins here, this is safer, but it doesn't really matter I guess
    k_bins = np.linspace(2 * np.pi / box_dm, float(np.max(k_grid_dm_foundation)), len(flamingo_k_bins))
    k_center = 0.5 * (k_bins[:-1] + k_bins[1:])
    T_k_1d = interp_extrapolate_loglog(flamingo_k_center, flamingo_T_k_1d, k_center)

    k_Nyquist_dm = np.pi * ngrid_dm / box_dm
    k_Nyquist_gas = np.pi * ngrid_gas / box_gas

    plt.figure(figsize=(6, 4), dpi=300)
    plt.loglog(flamingo_k_center, np.abs(flamingo_T_k_1d), c='tab:orange', linewidth=1.2, label='FLAMINGO $T(k)$')
    plt.loglog(k_center, np.abs(T_k_1d), c='tab:blue', linewidth=1.2, label='Extrapolated to Abacus $k$')
    plt.axvline(k_Nyquist_gas, c='tab:orange', linestyle='--', linewidth=1.0, label='$k_{Nyquist}^{gas}$')
    plt.axvline(k_Nyquist_dm, c='tab:blue', linestyle='--', linewidth=1.0, label='$k_{Nyquist}^{dm}$')
    plt.xlabel('$k$ [h/cMpc]', fontsize=12)
    plt.ylabel('$T(k)$', fontsize=12)
    plt.title('Transfer Function: FLAMINGO to Abacus extrapolation', fontsize=14)
    plt.legend(fontsize=9)
    p_ps_cmp = plot_path('hatf/power_spectra', FEEDBACK_MODE, stem='transfer_flamingo_to_abacus')
    ensure_parents(p_ps_cmp)
    plt.savefig(p_ps_cmp, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved: {p_ps_cmp}")

    # Keep FFT intermediates in cache for diagnostics reuse.
    del flamingo_T_k_1d
    gc.collect()

else:
    print("\n--- FLAMINGO transfer function branch ---")
    
    k_bins, k_center, P_dm_halos_foundation_binned = bin_power_spectrum_2d(
        (dm_field_fft_foundation * np.conj(halo_field_fft)).real,
        k_grid_dm_foundation,
        ngrid_dm,
        box_dm,
        k_min=2 * np.pi / box_dm,
    )
    _, _, P_gas_halos_target_binned = bin_power_spectrum_2d(
        (gas_field_fft_target * np.conj(halo_field_fft)).real,
        k_grid_gas_target,
        ngrid_gas,
        box_gas,
        k_min=2 * np.pi / box_gas,
    )

    P_dm_halos_foundation_binned = np.array(P_dm_halos_foundation_binned) * box_dm ** 2
    P_gas_halos_target_binned = np.array(P_gas_halos_target_binned) * box_gas ** 2
    
    eps = 1e-30
    T_k_1d = np.maximum(np.abs(P_gas_halos_target_binned), eps) / np.maximum(np.abs(P_dm_halos_foundation_binned), eps)

    gc.collect()

# Interpolate transfer function back to 2D k-space grid using JAX
# Map k_mag values to indices in k_center array
coords = (k_grid_dm_foundation - k_center[0]) / (k_center[-1] - k_center[0]) * (len(k_center) - 1)
T_k_3d = map_coordinates(jnp.array(T_k_1d), [coords], order=1)

print(f"Transfer function computed and interpolated to 2D k-space.")
print(f"T_k_1d statistics:")
print(f"  Mean: {np.mean(np.abs(T_k_1d)):.6e}")
print(f"  Std: {np.std(np.abs(T_k_1d)):.6e}")
print(f"  Min: {np.min(np.abs(T_k_1d)):.6e}")
print(f"  Max: {np.max(np.abs(T_k_1d)):.6e}")

# Clean up temporary interpolation coordinate array
del coords
gc.collect()

print("\n" + "="*60)
print("Loading/computing tau map prefactor")
print("="*60)

# Use pipeline_paths to get prefactor path
prefactor_path = _prefactor_path(_config)
ensure_parents(prefactor_path)

if prefactor_path.exists():
    # Load existing prefactor
    print(f"\nLoading prefactor from {prefactor_path}...")
    prefactor = np.load(prefactor_path)
    print(f"  Prefactor loaded: {prefactor:.6e}")
else:
    # Compute prefactor
    print(f"\nPrefactor not found. Computing from particle data...")
    
    if SIM_NAME == 'flamingo':

        gas_particles = load_particle_properties(
            gas_particles_file_flamingo,
            'gas',
            requested=('mass',),
            sim_name='flamingo',
        )
        if gas_particles is None:
            raise RuntimeError('Failed to load gas particle properties for prefactor computation')
        gas_masses = gas_particles['mass']

        prefactor = compute_prefactor(
            total_baryon_mass=np.sum(gas_masses),
            sim_params=SIM_PARAMS,
        )
        
        del gas_particles, gas_masses
        gc.collect()

    else:
        prefactor = compute_prefactor(
            total_baryon_mass=None,
            sim_params=SIM_PARAMS,
        )

    # Save prefactor
    np.save(prefactor_path, prefactor)
    print(f"  Prefactor computed and saved: {prefactor:.6e}")
    print(f"  Saved to: {prefactor_path}")

print("\n" + "="*60)
print("Reconstructing gas field and computing tau map")
print("="*60)

# Apply transfer function to DM field
gas_field_fft_reconstructed = dm_field_fft_foundation * T_k_3d

# Inverse FFT to get real space field
gas_field_reconstructed = np.fft.irfft2(gas_field_fft_reconstructed) * (ngrid_dm ** 2)

# Compute tau map: tau_xy_map = prefactor * (1.0 + delta_gas) * ngrid
tau_xy_map = prefactor * (1.0 + gas_field_reconstructed) * ngrid_dm

# Use pipeline_paths to get output path
_config['gas_type'] = FEEDBACK_MODE  + '_reconstructed'  # Add _reconstructed suffix for tau map paths
tau_out_path = _tau_map_path(_config)
ensure_parents(tau_out_path)
np.save(tau_out_path, tau_xy_map)
print(f"Tau map saved to: {tau_out_path}")

# Build selection tag for filenames
sel_tag = _selection_tag(_config).lstrip('_')

print("\n" + "="*60)
print("Creating diagnostic plots for new transfer function")
print("="*60)

# k_bins and k_center already defined from transfer function computation
k_Nyquist = np.pi * ngrid_dm / box_dm

if SIM_NAME == 'abacus':
    print("\n--- ABACUS diagnostics branch ---")

    _, _, abacus_P_gas_halos_binned = bin_power_spectrum_2d(
        (gas_field_fft_reconstructed * np.conj(halo_field_fft)).real,
        k_grid_dm_foundation,
        ngrid_gas, # this is to match the number of bins which we used in T_k_1d
        box_dm,
    )

    abacus_P_gas_halos_binned = np.array(abacus_P_gas_halos_binned) * box_dm ** 2

    _, _, abacus_P_dm_halos_binned = bin_power_spectrum_2d(
        (dm_field_fft_foundation * np.conj(halo_field_fft)).real,
        k_grid_dm_foundation,
        ngrid_gas,
        box_dm,
    )
    
    abacus_P_dm_halos_binned = np.array(abacus_P_dm_halos_binned) * box_dm ** 2

    plt.figure(figsize=(6, 4), dpi=300)
    plt.loglog(flamingo_k_center, flamingo_P_gas_halos_binned * box_gas, c='blue', alpha=0.7, linewidth=1,
               label='FLAMINGO true $P^{gas,halo}$')
    plt.loglog(k_center, abacus_P_gas_halos_binned * box_dm, c='red', alpha=0.7, linewidth=1,
               label='Abacus reconstructed $P^{gas,halo}$')
    plt.axvline(np.pi * ngrid_gas / box_gas, c='blue', linestyle='--', linewidth=1.0, label='$k_{Nyquist}^{gas}$')
    plt.axvline(np.pi * ngrid_dm / box_dm, c='red', linestyle='--', linewidth=1.0, label='$k_{Nyquist}^{dm}$')
    plt.xlabel('$k$ [h/cMpc]', fontsize=12)
    plt.ylabel(r'$P^{gas,halo}(k)\,L_{\rm box}$', fontsize=12)
    plt.title('Cross Power Spectrum: Gas-Halo (Abacus branch)', fontsize=14)
    plt.legend(fontsize=8)
    p1 = plot_path('hatf/power_spectra', FEEDBACK_MODE, stem=f'P_gas_halo_{sel_tag}_abacus_native')
    ensure_parents(p1)
    plt.savefig(p1, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved: {p1}")

    plt.figure(figsize=(6, 4), dpi=300)
    plt.loglog(flamingo_k_center, flamingo_P_dm_halos_binned * box_gas, c='blue', alpha=0.7, linewidth=1,
               label='FLAMINGO $P^{dm,halo}$')
    plt.loglog(k_center, abacus_P_dm_halos_binned * box_dm, c='red', alpha=0.7, linewidth=1,
               label='Abacus $P^{dm,halo}$')
    plt.axvline(np.pi * ngrid_gas / box_gas, c='blue', linestyle='--', linewidth=1.0, label='$k_{Nyquist}^{gas}$')
    plt.axvline(np.pi * ngrid_dm / box_dm, c='red', linestyle='--', linewidth=1.0, label='$k_{Nyquist}^{dm}$')
    plt.xlabel('$k$ [h/cMpc]', fontsize=12)
    plt.ylabel(r'$P^{dm,halo}(k)\,L_{\rm box}$', fontsize=12)
    plt.title('Cross Power Spectrum: DM-Halo (Abacus branch)', fontsize=14)
    plt.legend(fontsize=8)
    p1_dm = plot_path('hatf/power_spectra', FEEDBACK_MODE, stem=f'P_dm_halo_{sel_tag}_abacus_native')
    ensure_parents(p1_dm)
    plt.savefig(p1_dm, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved: {p1_dm}")

    _, _, P_gas_gas_target_binned = bin_power_spectrum_2d(
        (gas_field_fft_target * np.conj(gas_field_fft_target)).real, 
         k_grid_gas_target, 
         ngrid_gas, 
         box_gas)
    
    _, _, P_gas_gas_reconstructed_binned = bin_power_spectrum_2d(
        (gas_field_fft_reconstructed * np.conj(gas_field_fft_reconstructed)).real, 
        k_grid_dm_foundation, 
        ngrid_gas, 
        box_dm)
    
    _, _, P_dm_dm_foundation_binned = bin_power_spectrum_2d(
        (dm_field_fft_foundation * np.conj(dm_field_fft_foundation)).real, 
        k_grid_dm_foundation, 
        ngrid_gas, 
        box_dm)

    _, _, flamingo_P_dm_dm_binned = bin_power_spectrum_2d(
        (flamingo_dm_field_fft * np.conj(flamingo_dm_field_fft)).real, 
        k_grid_gas_target, 
        ngrid_gas, 
        box_gas)


    P_gas_gas_target_binned = np.array(P_gas_gas_target_binned) * box_gas ** 2
    P_gas_gas_reconstructed_binned = np.array(P_gas_gas_reconstructed_binned) * box_dm ** 2
    P_dm_dm_foundation_binned = np.array(P_dm_dm_foundation_binned) * box_dm ** 2
    flamingo_P_dm_dm_binned = np.array(flamingo_P_dm_dm_binned) * box_gas ** 2

    plt.figure(figsize=(6, 4), dpi=300)
    plt.loglog(flamingo_k_center, P_gas_gas_target_binned * box_gas, c='blue', alpha=0.7, linewidth=1, label='FLAMINGO $P^{gas,gas}$')
    plt.loglog(k_center, P_gas_gas_reconstructed_binned * box_dm, c='red', alpha=0.7, linewidth=1, label='Reconstructed $P^{gas,gas}$')
    plt.axvline(k_Nyquist, c='red', linestyle='--', label='$k_{Nyquist}^{dm}$')
    plt.xlabel('$k$ [h/cMpc]', fontsize=12)
    plt.ylabel(r'$P^{gas,gas}(k)\,L_{\rm box}$', fontsize=12)
    plt.title('Auto Power Spectrum: Gas (Abacus branch)', fontsize=14)
    plt.legend()
    p2 = plot_path('hatf/power_spectra', FEEDBACK_MODE, stem=f'P_gas_auto_{sel_tag}_abacus')
    ensure_parents(p2)
    plt.savefig(p2, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved: {p2}")

    plt.figure(figsize=(6, 4), dpi=300)
    plt.loglog(flamingo_k_center, flamingo_P_dm_dm_binned * box_gas, c='blue', alpha=0.7, linewidth=1, label='FLAMINGO $P^{dm,dm}$')
    plt.loglog(k_center, P_dm_dm_foundation_binned * box_dm, c='red', alpha=0.7, linewidth=1, label='Abacus $P^{dm,dm}$')
    plt.axvline(k_Nyquist, c='red', linestyle='--', label='$k_{Nyquist}^{dm}$')
    plt.xlabel('$k$ [h/cMpc]', fontsize=12)
    plt.ylabel(r'$P^{dm,dm}(k)\,L_{\rm box}$', fontsize=12)
    plt.title('Auto Power Spectrum: DM (Abacus branch)', fontsize=14)
    plt.legend()
    p2_dm = plot_path('hatf/power_spectra', FEEDBACK_MODE, stem=f'P_dm_auto_{sel_tag}_abacus')
    ensure_parents(p2_dm)
    plt.savefig(p2_dm, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved: {p2_dm}")

else:
    print("\n--- FLAMINGO diagnostics branch ---")
    # Compute reconstructed and target P^gas,halo
    _, _, P_gas_halos_reconstructed_binned = bin_power_spectrum_2d(
        (gas_field_fft_reconstructed * np.conj(halo_field_fft)).real,
        k_grid_dm_foundation,
        ngrid_dm,
        box_dm,
    )

    P_gas_halos_reconstructed_binned = np.array(P_gas_halos_reconstructed_binned) * box_dm ** 2

    # Plot 1: P^gas,halo comparison
    plt.figure(figsize=(6, 4), dpi=300)
    plt.loglog(k_center, P_gas_halos_target_binned * box_dm, c='blue', alpha=0.7, linewidth=1, label='True $P^{gas,halo}$')
    plt.loglog(k_center, P_gas_halos_reconstructed_binned * box_dm, c='red', alpha=0.7, linewidth=1, label='Reconstructed $P^{gas,halo}$')
    plt.axvline(k_Nyquist, c='blue', linestyle='--', label='$k_{Nyquist}$')
    plt.xlabel('$k$ [h/cMpc]', fontsize=12)
    plt.ylabel(r'$P^{gas,halo}(k)\,L_{\rm box}$', fontsize=12)
    plt.title('Cross Power Spectrum: Gas-Halo', fontsize=14)
    plt.legend()
    p1 = plot_path('hatf/power_spectra', FEEDBACK_MODE, stem=f'P_gas_halo_{sel_tag}')
    ensure_parents(p1)
    plt.savefig(p1, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved: {p1}")

    print("\nComputing auto power spectra P^gas,gas and P^dm,dm...")

    # Bin the auto power spectra using the shared utility
    _, _, P_gas_gas_target_binned = bin_power_spectrum_2d(
        (gas_field_fft_target * np.conj(gas_field_fft_target)).real,
        k_grid_gas_target,
        ngrid_gas,
        box_gas,
    )
    _, _, P_gas_gas_reconstructed_binned = bin_power_spectrum_2d(
        (gas_field_fft_reconstructed * np.conj(gas_field_fft_reconstructed)).real,
        k_grid_dm_foundation,
        ngrid_dm,
        box_dm,
    )
    _, _, P_dm_dm_foundation_binned = bin_power_spectrum_2d(
        (dm_field_fft_foundation * np.conj(dm_field_fft_foundation)).real,
        k_grid_dm_foundation,
        ngrid_dm,
        box_dm,
    )

    P_gas_gas_target_binned = np.array(P_gas_gas_target_binned) * box_gas ** 2
    P_gas_gas_reconstructed_binned = np.array(P_gas_gas_reconstructed_binned) * box_dm ** 2
    P_dm_dm_foundation_binned = np.array(P_dm_dm_foundation_binned) * box_dm ** 2

    # Plot 2: P^gas,gas comparison
    plt.figure(figsize=(6, 4), dpi=300)
    plt.loglog(k_center, P_gas_gas_target_binned * box_gas, c='blue', alpha=0.7, linewidth=1, label='True $P^{gas,gas}$')
    plt.loglog(k_center, P_gas_gas_reconstructed_binned * box_dm, c='red', alpha=0.7, linewidth=1, label='Reconstructed $P^{gas,gas}$')
    plt.axvline(k_Nyquist, c='blue', linestyle='--', label='$k_{Nyquist}$')
    plt.xlabel('$k$ [h/cMpc]', fontsize=12)
    plt.ylabel(r'$P^{gas,gas}(k)\,L_{\rm box}$', fontsize=12)
    plt.title('Auto Power Spectrum: Gas', fontsize=14)
    plt.legend()
    p2 = plot_path('hatf/power_spectra', FEEDBACK_MODE, stem=f'P_gas_auto_{sel_tag}')
    ensure_parents(p2)
    plt.savefig(p2, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved: {p2}")

    # Plot 2: Cross-correlation between reconstructed and true gas fields
    _, _, P_cross_reconstructed_target_binned = bin_power_spectrum_2d(
        (gas_field_fft_reconstructed * np.conj(gas_field_fft_target)).real,
        k_grid_gas_target,
        ngrid_gas,
        box_gas,
    )
    P_cross_reconstructed_target_binned = np.array(P_cross_reconstructed_target_binned) * box_gas ** 2

    # Cross-correlation coefficient
    r_reconstructed_vs_target = P_cross_reconstructed_target_binned / np.sqrt(P_gas_gas_reconstructed_binned * P_gas_gas_target_binned)

    plt.figure(figsize=(6, 4), dpi=300)
    plt.semilogx(k_center, r_reconstructed_vs_target, c='red', alpha=0.7, label='Reconstructed vs Target Gas')
    plt.axhline(1.0, c='grey', linestyle='--', label='Perfect Correlation')
    plt.axvline(k_Nyquist, c='blue', linestyle='--', label='$k_{Nyquist}$')
    plt.xlabel('$k$ [h/cMpc]', fontsize=12)
    plt.ylabel('$r(k)$', fontsize=12)
    plt.title('Cross-Correlation Coefficient', fontsize=14)
    plt.ylim([0, 1.1])
    plt.legend()
    p_corr = plot_path('hatf/power_spectra', FEEDBACK_MODE, stem=f'r_recon_vs_true_{sel_tag}')
    ensure_parents(p_corr)
    plt.savefig(p_corr, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved: {p_corr}")

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
    T_k_old_1d = np.sqrt(P_gas_gas_target_binned / P_dm_dm_foundation_binned)
    
    # Interpolate transfer function back to 2D k-space grid using JAX
    coords_old = (k_grid_dm_foundation - k_center[0]) / (k_center[-1] - k_center[0]) * (len(k_center) - 1)
    T_k_old = map_coordinates(jnp.array(T_k_old_1d), [coords_old], order=1)
    
    # Compute mass-dependent transfer function: T(k) = T_old(k) * (1 + Ak^2)
    one_plus_Ak2 = 1.0 + A1 * k_grid_dm_foundation**2
    one_plus_Ak4 = 1.0 + A2 * k_grid_dm_foundation**4
    T_k_massdep = T_k_old * one_plus_Ak2
    
    # Bin all three transfer functions and the correction factors by k
    T_k_new_binned = []
    T_k_old_binned = []
    T_k_massdep_binned = []
    one_plus_Ak2_binned = []
    one_plus_Ak4_binned = []
    
    for i in range(len(k_bins) - 1):
        mask = (k_grid_dm_foundation >= k_bins[i]) & (k_grid_dm_foundation < k_bins[i + 1])
        
        T_new_bin = T_k_3d[mask]
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
    p3 = plot_path('hatf/transfer_fn', FEEDBACK_MODE, stem=f'T_k_{sel_tag}')
    ensure_parents(p3)
    plt.savefig(p3, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {p3}")
    plt.close()

print("\n" + "="*60)
print("Reconstruction completed successfully!")
print("="*60)