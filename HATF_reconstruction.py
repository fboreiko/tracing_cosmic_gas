"""
This is a self-contained script for reconstructing the gas field from the DM 
field using the HATF method. It handles both the Abacus and Flamingo branches 
of the pipeline, computing or loading the necessary 2D projected fields, 
Fourier transforms, transfer functions, and tau maps.
"""

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
from utils.pipeline_config import PipelineConfig
from utils.pipeline_paths import (
    delta_2d_path as _delta_2d_path,
    tau_prefactor_path as _prefactor_path,
    tau_map_path as _tau_map_path,
    halo_indices_path as _halo_indices_path,
    halo_props_cache_path as _halo_props_cache_path,
    ensure_parents,
    plot_path,
    selection_tag,
    get_particle_file_path,
)
from utils.sample_selection import selection_defaults
from utils.sim_params import get_sim_params
from utils.plot_data import save_plot_data

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Computer Modern']
rcParams['text.usetex'] = False

SIM_NAME = 'flamingo' # this defines the "branch" of the pipeline to run: 'abacus' or 'flamingo'
SIM_PARAMS = get_sim_params(SIM_NAME) # pull default sim parameters

FEEDBACK = 'strongest_AGN'  # 'fiducial', 'strongest_AGN'

# Regime: determines which mass to use for halo (galaxy) selection
regime = 'mgal_sel'  # 'mhalo_sel' (halo mass) or 'mgal_sel' (stellar mass)

# Selection parameters come from the ONE shared definition in pipeline_paths.
# This file used to re-derive them inline, and the two copies had drifted
# (upper_mass_cut differed), so HATF and the standalone get_AP main() built
# different selection tags for the same nominal selection.
_regime_params = selection_defaults(regime)

cen_sat_mode         = _regime_params['selection_mode']
mass_type            = _regime_params['selection_mass_def']
require_nonzero_mass = _regime_params['select_nonzero_masses']
upper_mass_cut       = _regime_params['upper_mass_cut']
upper_radius_cut     = _regime_params['upper_radius_cut']
max_mass             = _regime_params['max_mass']

# Selection method: determines how to select from ranked objects
# To run either, set the other to None.
halo_mass_range = None #[0, 1000]   
n_gal_density = 5e-4   # (cMpc/h)^-3, 
                       # for "sat" mode (to get the satellites of "mixed" mode):
                       #         sf=0.1: 7.501e-05
                       #         sf=0.2: 2.346e-04
                       #         sf=0.3: 6.97e-04

_convergence_mode = 'massbin' if (halo_mass_range is not None and n_gal_density is None) else 'ngal'

# Target mean mass mode: if set, iteratively adjusts selection to match this target log10(mean mass).
# If None, the selection is based on the fixed halo_mass_range or n_gal_density defined above.
target_mean_mass = 13.2 # 13.2 to target log10(<M>) = 13.2 
target_mass_tolerance = 0.005  # tolerance for convergence (in log10 units)
mass_bin_halfwidth = 0.02  # ±dex span target for convergence
mass_bin_halfwidth_tol = 0.001
max_iterations = 1000  # maximum iterations for target mass matching

# Satellite fraction to be achieved in the 'mixed' mode
sat_frac = 0.3  # 0.01, 0.10, 0.20, 0.30

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
    FEEDBACK,
    sim_name='abacus',
)
dm_particles_file_flamingo = get_particle_file_path(
    FEEDBACK,
    sim_name='flamingo',
)
gas_particles_file_flamingo = get_particle_file_path(
    FEEDBACK,
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

# Build the pipeline config ONCE, immutably.
#
# `truth_cfg` addresses everything that does not depend on the reconstruction:
# halo indices, halo-property caches, delta fields, prefactors.
# `recon_cfg` differs in exactly one field and addresses the tau map this
# script writes. They are separate objects, so which one a call site uses is
# visible at that call site.

_config = PipelineConfig.from_dict(dict(
    sim_name=SIM_NAME,
    feedback=FEEDBACK,
    tau_source='truth',
    selection_mode=cen_sat_mode,
    selection_mass_def=mass_type,
    select_nonzero_masses=require_nonzero_mass,
    upper_mass_cut=upper_mass_cut,
    max_mass=max_mass,
    upper_radius_cut=upper_radius_cut,
    sat_frac=sat_frac,
    n_gal_density=n_gal_density,
    halo_mass_range=halo_mass_range,
    target_mean_mass=target_mean_mass,
    convergence_mode=_convergence_mode,
    target_mass_tolerance=target_mass_tolerance,
    mass_bin_halfwidth=mass_bin_halfwidth,
    mass_bin_halfwidth_tol=mass_bin_halfwidth_tol,
    max_iterations=max_iterations,
))
truth_cfg = _config
recon_cfg = _config.replace(tau_source='recon')

# Output directories
delta_fields_dir = _delta_2d_path({'sim_name': SIM_NAME, 'feedback': FEEDBACK}, 'dm').parent
delta_fields_dir.mkdir(parents=True, exist_ok=True)

# Path where selected halo indices will be saved/loaded
# Use the sim-specific config for each branch (abacus/flamingo)
_halo_idx_path_flamingo = _halo_indices_path(_config.replace(sim_name='flamingo'))
_halo_idx_path_abacus = _halo_indices_path(_config.replace(sim_name='abacus')) if SIM_NAME == 'abacus' else None

_halo_props_cache_path_flamingo = _halo_props_cache_path(_config.replace(sim_name='flamingo'))
_halo_props_cache_path_abacus = _halo_props_cache_path(_config.replace(sim_name='abacus')) if SIM_NAME == 'abacus' else None

print("="*60)
print("Loading/computing 2D projected fields")
print(f"Regime: {regime} | Method: {method} | Mass type: {mass_type}")
print("="*60)

delta_2d_fields = {}
total_mass_for_tau = None

print("\nPreparing projected fields...")

# For either branch, we need the FLAMINGO gas field for the transfer function construction
gas_config = _config.replace(sim_name='flamingo')
gas_path = _delta_2d_path(gas_config, 'gas')

if gas_path.exists():
    print(f"Loading gas field from {gas_path}...")
    delta_2d_fields['gas'] = np.load(gas_path)
    print(f"  Loaded successfully. Shape: {delta_2d_fields['gas'].shape}")
else:
    print("Gas field not found. Computing FLAMINGO gas field...")
    delta_2d_fields['gas'], total_mass_for_tau = compute_delta_field_and_mass(
        tracer='gas',
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
    # The Abacus DM field is identical across feedback variants, so always
    # address it with a single canonical feedback label.
    abacus_dm_config = _config.replace(feedback='strongest_AGN')
    abacus_dm_path = _delta_2d_path(abacus_dm_config, 'dm')

    if abacus_dm_path.exists():
        print(f"Loading dm field from {abacus_dm_path}...")
        delta_2d_fields['dm'] = np.load(abacus_dm_path)
        print(f"  Loaded successfully. Shape: {delta_2d_fields['dm'].shape}")
    else:
        print("DM field not found. Computing Abacus DM field...")
        delta_2d_fields['dm'], total_mass_for_tau = compute_delta_field_and_mass(
            tracer='dm',
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
        feedback=FEEDBACK,
        box=box_dm,
        ngrid=ngrid_dm,
        selection_config=_config,
        save_path=_delta_2d_path(_config, 'halos'),
        halo_indices_save_path=_halo_idx_path_abacus,
        halo_props_cache_save_path=_halo_props_cache_path_abacus,
        nthread=nthread,
    )

    # In the Abacus branch we also require the FLAMINGO halo field for the transfer 
    # function construction
    _config_flamingo = _config.replace(sim_name='flamingo')

    delta_2d_fields['halos_flamingo'] = compute_selected_halo_delta_2d(
        sim_for_halos='flamingo',
        feedback=FEEDBACK,
        box=box_gas,
        ngrid=ngrid_gas,
        selection_config=_config_flamingo,
        save_path=_delta_2d_path(_config_flamingo, 'halos'),
        halo_indices_save_path=_halo_idx_path_flamingo,
        halo_props_cache_save_path=_halo_props_cache_path_flamingo,
        nthread=nthread,
    )
    delta_2d_fields['halos'] = delta_2d_fields['halos_abacus']

    # We also need the FLAMINGO DM field for the transfer function construction
    flamingo_dm_path = _delta_2d_path(_config_flamingo, 'dm')
    if flamingo_dm_path.exists():
        print(f"Loading dm_flamingo field from {flamingo_dm_path}...")
        delta_2d_fields['dm_flamingo'] = np.load(flamingo_dm_path)
        print(f"  Loaded successfully. Shape: {delta_2d_fields['dm_flamingo'].shape}")
    else:
        print("DM field (FLAMINGO) not found. Computing...")
        delta_2d_fields['dm_flamingo'], _ = compute_delta_field_and_mass(
            tracer='dm',
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
            tracer='dm',
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
        feedback=FEEDBACK,
        box=box_gas,
        ngrid=ngrid_gas,
        selection_config=_config,
        save_path=_delta_2d_path(_config, 'halos'),
        halo_indices_save_path=_halo_idx_path_flamingo,
        halo_props_cache_save_path=_halo_props_cache_path_flamingo,
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
    p_ps_cmp = plot_path('hatf/power_spectra', FEEDBACK, stem='transfer_flamingo_to_abacus')
    ensure_parents(p_ps_cmp)
    plt.savefig(p_ps_cmp, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved: {p_ps_cmp}")
    # Save plot data bundle
    save_plot_data(p_ps_cmp, {
        'flamingo_k_center': flamingo_k_center,
        'flamingo_T_k_1d': flamingo_T_k_1d,
        'k_center': k_center,
        'T_k_1d': T_k_1d,
        'k_Nyquist_gas': k_Nyquist_gas,
        'k_Nyquist_dm': k_Nyquist_dm,
    }, description='Transfer Function: FLAMINGO to Abacus extrapolation')

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

    prefactor = compute_prefactor(
            total_baryon_mass=total_mass_for_tau,
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
# The reconstructed tau map is addressed by recon_cfg, built at the top of
# this file. Nothing is mutated here.
tau_out_path = _tau_map_path(recon_cfg)
ensure_parents(tau_out_path)
np.save(tau_out_path, tau_xy_map)
print(f"Tau map saved to: {tau_out_path}")

# Build selection tag for filenames
sel_tag = selection_tag(_config)

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

    # Publication-quality 1x2 plots for Gas
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    
    # Left: Cross Power Spectrum Gas-Halo
    axes[0].loglog(flamingo_k_center, flamingo_P_gas_halos_binned * box_gas, c='blue', alpha=0.7, linewidth=2,
                   label=r'True $C_{gas,\,halo}$')
    axes[0].loglog(k_center, abacus_P_gas_halos_binned * box_dm, c='red', alpha=0.7, linewidth=2,
                   label=r'Reconstructed $\tilde C_{gas,\,halo}$')
    axes[0].axvline(np.pi * ngrid_gas / box_gas, c='blue', linestyle='--', linewidth=1.5, alpha=0.6)
    axes[0].axvline(np.pi * ngrid_dm / box_dm, c='red', linestyle='--', linewidth=1.5, alpha=0.6)
    axes[0].set_xlabel('$k$ [h/cMpc]', fontsize=13)
    axes[0].set_ylabel(r'$\hat C_{gas,\,halo}(k)\,L_{\rm box}$', fontsize=13)
    axes[0].legend(fontsize=11, loc='best')
    
    # Right: Auto Power Spectrum Gas-Gas
    axes[1].loglog(flamingo_k_center, P_gas_gas_target_binned * box_gas, c='blue', alpha=0.7, linewidth=2,
                   label=r'True $C_{gas,\,gas}$')
    axes[1].loglog(k_center, P_gas_gas_reconstructed_binned * box_dm, c='red', alpha=0.7, linewidth=2,
                   label=r'Reconstructed $\tilde C_{gas,\,gas}$')
    axes[1].axvline(np.pi * ngrid_gas / box_gas, c='blue', linestyle='--', linewidth=1.5, alpha=0.6)
    axes[1].axvline(np.pi * ngrid_dm / box_dm, c='red', linestyle='--', linewidth=1.5, alpha=0.6)
    axes[1].set_xlabel('$k$ [h/cMpc]', fontsize=13)
    axes[1].set_ylabel(r'$\hat C_{gas,\,gas}(k)\,L_{\rm box}$', fontsize=13)
    axes[1].legend(fontsize=11, loc='best')
    
    plt.tight_layout()
    p_gas = plot_path('hatf/power_spectra', FEEDBACK, stem=f'P_gas_combined_{sel_tag}_abacus')
    ensure_parents(p_gas)
    plt.savefig(p_gas, bbox_inches='tight', dpi=300, format='pdf')
    plt.close()
    print(f"Saved: {p_gas}")
    # Save plot data bundle
    save_plot_data(p_gas, {
        'flamingo_k_center': flamingo_k_center,
        'P_gas_halos_target_binned': flamingo_P_gas_halos_binned,
        'P_gas_halos_reconstructed_binned': abacus_P_gas_halos_binned,
        'P_gas_gas_target_binned': P_gas_gas_target_binned,
        'P_gas_gas_reconstructed_binned': P_gas_gas_reconstructed_binned,
        'k_center': k_center,
        'box_gas': box_gas,
        'box_dm': box_dm,
        'ngrid_gas': ngrid_gas,
        'ngrid_dm': ngrid_dm,
    }, description='Gas Power Spectra Comparison (Abacus vs Flamingo-reconstructed)')

    # Publication-quality 1x2 plots for DM
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    
    # Left: Cross Power Spectrum DM-Halo
    axes[0].loglog(flamingo_k_center, flamingo_P_dm_halos_binned * box_gas, c='blue', alpha=0.7, linewidth=2,
                   label=r'FLAMINGO $C_{dm,\,halo}$')
    axes[0].loglog(k_center, abacus_P_dm_halos_binned * box_dm, c='red', alpha=0.7, linewidth=2,
                   label=r'Abacus $C_{dm,\,halo}$')
    axes[0].axvline(np.pi * ngrid_gas / box_gas, c='blue', linestyle='--', linewidth=1.5, alpha=0.6)
    axes[0].axvline(np.pi * ngrid_dm / box_dm, c='red', linestyle='--', linewidth=1.5, alpha=0.6)
    axes[0].set_xlabel('$k$ [h/cMpc]', fontsize=13)
    axes[0].set_ylabel(r'$\hat C_{dm,\,halo}(k)\,L_{\rm box}$', fontsize=13)
    axes[0].legend(fontsize=11, loc='best')
    
    # Right: Auto Power Spectrum DM-DM
    axes[1].loglog(flamingo_k_center, flamingo_P_dm_dm_binned * box_gas, c='blue', alpha=0.7, linewidth=2,
                   label=r'FLAMINGO $C_{dm,\,dm}$')
    axes[1].loglog(k_center, P_dm_dm_foundation_binned * box_dm, c='red', alpha=0.7, linewidth=2,
                   label=r'Abacus $C_{dm,\,dm}$')
    axes[1].axvline(np.pi * ngrid_gas / box_gas, c='blue', linestyle='--', linewidth=1.5, alpha=0.6)
    axes[1].axvline(np.pi * ngrid_dm / box_dm, c='red', linestyle='--', linewidth=1.5, alpha=0.6)
    axes[1].set_xlabel('$k$ [h/cMpc]', fontsize=13)
    axes[1].set_ylabel(r'$\hat C_{dm,\,dm}(k)\,L_{\rm box}$', fontsize=13)
    axes[1].legend(fontsize=11, loc='best')
    
    plt.tight_layout()
    p_dm = plot_path('hatf/power_spectra', FEEDBACK, stem=f'P_dm_combined_{sel_tag}_abacus')
    ensure_parents(p_dm)
    plt.savefig(p_dm, bbox_inches='tight', dpi=300, format='pdf')
    plt.close()
    print(f"Saved: {p_dm}")
    # Save plot data bundle
    save_plot_data(p_dm, {
        'flamingo_k_center': flamingo_k_center,
        'flamingo_P_dm_halos_binned': flamingo_P_dm_halos_binned,
        'abacus_P_dm_halos_binned': abacus_P_dm_halos_binned,
        'flamingo_P_dm_dm_binned': flamingo_P_dm_dm_binned,
        'P_dm_dm_foundation_binned': P_dm_dm_foundation_binned,
        'k_center': k_center,
        'box_gas': box_gas,
        'box_dm': box_dm,
        'ngrid_gas': ngrid_gas,
        'ngrid_dm': ngrid_dm,
    }, description='Dark Matter Power Spectra Comparison (Abacus vs Flamingo)')
    
    # Save tau map for Abacus branch
    print("\nSaving tau map for Abacus branch...")
    tau_bundle_path = plot_path('hatf/tau_maps', FEEDBACK, stem=f'tau_map_{sel_tag}')
    tau_bundle_path = tau_bundle_path.with_suffix('.npz')
    ensure_parents(tau_bundle_path)
    np.savez_compressed(
        tau_bundle_path,
        tau_map_reconstructed=tau_xy_map,
        ngrid_dm=ngrid_dm,
        box_dm=box_dm,
    )
    print(f"Saved: {tau_bundle_path}")

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
    
    _, _, P_dm_halos_foundation_binned = bin_power_spectrum_2d(
        (dm_field_fft_foundation * np.conj(halo_field_fft)).real,
        k_grid_dm_foundation,
        ngrid_dm,
        box_dm,
    )

    P_gas_gas_target_binned = np.array(P_gas_gas_target_binned) * box_gas ** 2
    P_gas_gas_reconstructed_binned = np.array(P_gas_gas_reconstructed_binned) * box_dm ** 2
    P_dm_dm_foundation_binned = np.array(P_dm_dm_foundation_binned) * box_dm ** 2
    P_dm_halos_foundation_binned = np.array(P_dm_halos_foundation_binned) * box_dm ** 2

    # Publication-quality 1x2 plots for Gas
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    
    # Left: Cross Power Spectrum Gas-Halo
    axes[0].loglog(k_center, P_gas_halos_target_binned * box_dm, c='blue', alpha=0.7, linewidth=2,
                   label=r'True $C_{gas,\,halo}$')
    axes[0].loglog(k_center, P_gas_halos_reconstructed_binned * box_dm, c='red', alpha=0.7, linewidth=2,
                   label=r'Reconstructed $\tilde C_{gas,\,halo}$')
    axes[0].axvline(k_Nyquist, c='blue', linestyle='--', linewidth=1.5, alpha=0.6)
    axes[0].set_xlabel('$k$ [h/cMpc]', fontsize=13)
    axes[0].set_ylabel(r'$\hat C_{gas,\,halo}(k)\,L_{\rm box}$', fontsize=13)
    axes[0].legend(fontsize=11, loc='best')
    
    # Right: Auto Power Spectrum Gas-Gas
    axes[1].loglog(k_center, P_gas_gas_target_binned * box_gas, c='blue', alpha=0.7, linewidth=2,
                   label=r'True $C_{gas,\,gas}$')
    axes[1].loglog(k_center, P_gas_gas_reconstructed_binned * box_dm, c='red', alpha=0.7, linewidth=2,
                   label=r'Reconstructed $\tilde C_{gas,\,gas}$')
    axes[1].axvline(k_Nyquist, c='blue', linestyle='--', linewidth=1.5, alpha=0.6)
    axes[1].set_xlabel('$k$ [h/cMpc]', fontsize=13)
    axes[1].set_ylabel(r'$\hat C_{gas,\,gas}(k)\,L_{\rm box}$', fontsize=13)
    axes[1].legend(fontsize=11, loc='best')
    
    plt.tight_layout()
    p_gas = plot_path('hatf/power_spectra', FEEDBACK, stem=f'P_gas_combined_{sel_tag}')
    ensure_parents(p_gas)
    plt.savefig(p_gas, bbox_inches='tight', dpi=300, format='pdf')
    plt.close()
    print(f"Saved: {p_gas}")
    # Save plot data bundle
    save_plot_data(p_gas, {
        'k_center': k_center,
        'P_gas_halos_target_binned': P_gas_halos_target_binned,
        'P_gas_halos_reconstructed_binned': P_gas_halos_reconstructed_binned,
        'P_gas_gas_target_binned': P_gas_gas_target_binned,
        'P_gas_gas_reconstructed_binned': P_gas_gas_reconstructed_binned,
        'k_Nyquist': k_Nyquist,
        'box_dm': box_dm,
        'box_gas': box_gas,
    }, description='Gas Power Spectra (FLAMINGO diagnostics)')
    
    # Save tau maps for FLAMINGO branch
    print("\nSaving tau maps for FLAMINGO branch...")
    gas_field_target = np.fft.irfft2(gas_field_fft_target) * (ngrid_gas ** 2)
    tau_map_target = prefactor * (1.0 + gas_field_target) * ngrid_gas
    tau_map_reconstructed = prefactor * (1.0 + gas_field_reconstructed) * ngrid_dm
    
    tau_bundle_path = plot_path('hatf/tau_maps', FEEDBACK, stem=f'tau_maps_{sel_tag}')
    tau_bundle_path = tau_bundle_path.with_suffix('.npz')
    ensure_parents(tau_bundle_path)
    np.savez_compressed(
        tau_bundle_path,
        tau_map_target=tau_map_target,
        tau_map_reconstructed=tau_map_reconstructed,
        ngrid_gas=ngrid_gas,
        ngrid_dm=ngrid_dm,
        box_gas=box_gas,
        box_dm=box_dm,
    )
    print(f"Saved: {tau_bundle_path}")

# Plot 3: Cross-correlation between reconstructed and true gas fields (for both branches)
_, _, P_cross_reconstructed_target_binned = bin_power_spectrum_2d(
    (gas_field_fft_reconstructed * np.conj(gas_field_fft_target)).real,
    k_grid_gas_target,
    ngrid_gas,
    box_gas,
)
P_cross_reconstructed_target_binned = np.array(P_cross_reconstructed_target_binned) * box_gas ** 2

# Cross-correlation coefficient
r_reconstructed_vs_target = P_cross_reconstructed_target_binned / np.sqrt(P_gas_gas_reconstructed_binned * P_gas_gas_target_binned)

plt.figure(figsize=(7, 5), dpi=300)
plt.semilogx(k_center, r_reconstructed_vs_target, c='red', alpha=0.8, linewidth=2.5)
plt.axhline(1.0, c='grey', linestyle='--', linewidth=2, label='Perfect Correlation')
plt.axvline(k_Nyquist, c='blue', linestyle='--', linewidth=1.5, label='$k_{Nyquist}$')
plt.xlabel('$k$ [h/cMpc]', fontsize=13)
plt.ylabel('$r(k)$', fontsize=13)
plt.ylim([0, 1.1])
plt.legend(fontsize=11)
p_corr = plot_path('hatf/power_spectra', FEEDBACK, stem=f'r_recon_vs_true_{sel_tag}')
ensure_parents(p_corr)
plt.savefig(p_corr, bbox_inches='tight', dpi=300, format='pdf')
plt.close()
print(f"Saved: {p_corr}")
# Save plot data bundle
save_plot_data(p_corr, {
    'k_center': k_center,
    'r_reconstructed_vs_target': r_reconstructed_vs_target,
    'k_Nyquist': k_Nyquist,
}, description='Cross-correlation between reconstructed and true gas fields')

# OPTIONAL: Plotting of transfer function comparison with mass-dependent correction
plot_comparison = True  # Set to False to skip this step

if plot_comparison:
    print("\n" + "="*60)
    print("Plotting Transfer Function Comparison")
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
    
    ax.loglog(k_center, T_k_new_binned, c='blue', linewidth=2, label=r"$T'(k) = \hat C_{gas\,halo} / \hat C_{dm\,halo}$")
    ax.loglog(k_center, T_k_old_binned, c='green', linewidth=2, linestyle='-.', 
              label=r"$T_{\rm old}(k) = \sqrt{\hat C_{gas\,gas} / \hat C_{dm\,dm}}$")
    ax.loglog(k_center, T_k_massdep_binned, c='red', linewidth=2, linestyle='--', 
              label=r"$T(k) = T_{\rm old}(k) \times (1 + Ak^2)$")
    ax.loglog(k_center, one_plus_Ak2_binned, c='orange', linewidth=2, linestyle=':', 
              label=r"$(1 + Ak^2)$")
    ax.loglog(k_center, one_plus_Ak4_binned, c='purple', linewidth=2, linestyle=':', 
              label=r"$(1 + Ak^4)$")
    ax.axvline(k_Nyquist, c='grey', linestyle=':', linewidth=1.5, label='$k_{Nyquist}$')
    
    ax.set_xlabel(r'$k$ [h/cMpc]', fontsize=14)
    ax.set_ylabel(r"Transfer Function", fontsize=14)
    ax.legend(fontsize=12)
    
    # Save plot
    p3 = plot_path('hatf/transfer_fn', FEEDBACK, stem=f'T_k_{sel_tag}')
    ensure_parents(p3)
    plt.savefig(p3, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {p3}")
    plt.close()
    # Save plot data bundle
    save_plot_data(p3, {
        'k_center': k_center,
        'T_k_1d': T_k_new_binned,
        'box_dm': box_dm,
        'ngrid_dm': ngrid_dm,
    }, description='Transfer function comparison')

print("\n" + "="*60)
print("Reconstruction completed successfully!")
print("="*60)