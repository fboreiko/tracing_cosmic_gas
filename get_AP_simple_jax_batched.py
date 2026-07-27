"""
JAX + MPI implementation for aperture photometry on tau maps. Processes galaxy 
catalogs in batches with MPI parallelization. This is a simple implementation, 
with no geometric corrections accounting for the curvature of the sky.
"""

import numpy as np
import gc
import jax
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial
from mpi4py import MPI
from tqdm import tqdm
import os
import sys
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
sys.path.insert(0, '/home/fb635/fedirfiles/tracing_cosmic_gas')
from utils.catalog_loaders import load_halo_properties
from utils.pipeline_paths import (
    get_halo_file_path,
    tau_map_path as _tau_map_path,
    ap_output_path,
    halo_indices_path as _halo_indices_path,
    resolve_halo_indices,
    halo_props_cache_path as _halo_props_cache_path,
)
from colossus.halo import mass_defs, concentration
from utils.sim_params import get_sim_params, require_sim_param
import matplotlib.pyplot as plt

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

BEAM_SMOOTHED = True   # beam smoothing always applied


def M200c_to_M200m(M200c, z):
    c200c = concentration.concentration(
        M200c, '200c', z, model='diemer19'
    )
    M200m, R200m, c200m = mass_defs.changeMassDefinition(
        M200c, c200c, z,
        mdef_in='200c',
        mdef_out='200m'
    )
    return M200m


def get_cutout_dims(r_max, dx):
    """Calculate cutout dimensions based on maximum radius."""
    half_size = int(np.ceil(1.2 * r_max / dx))
    cutout_size = 2 * half_size + 1
    return cutout_size


def load_data(tau_map_path, halo_galaxy_data_path, sim='flamingo', halo_indices=None):
    """Load tau map and catalog data from unified HDF5 source.
    
    Args:
        tau_map_path: Path to tau map
        halo_galaxy_data_path: Path to HDF5 file with all galaxy/halo data
        sim: Simulation name ('flamingo' or 'abacus')
        halo_indices: Optional array of halo indices to load. If None, loads all halos.
    
    Returns:
        tau_map: Tau map array
        all_data: Dict with all mstell, m200b, m200c, pos, centrals
    """
    tau_map = np.load(tau_map_path)

    if sim == 'abacus':
        requested_props = ['m200b', 'x_L2com', 'v_L2com']
        halo_props = load_halo_properties(halo_galaxy_data_path, requested_props, sim_name=sim, halo_indices=halo_indices)
        if halo_props is None:
            raise RuntimeError('Failed to load halo properties')
        all_data = {
            'm200b': halo_props['m200b'],
            'pos': halo_props['x_L2com'],
            'm200c': None,
            'mfof': None,
        }
        return tau_map, all_data

    requested_props = ['m200b', 'm200c', 'pos', 'mfof']

    halo_props = load_halo_properties(halo_galaxy_data_path, requested_props, sim_name=sim, halo_indices=halo_indices)
    if halo_props is None:
        raise RuntimeError('Failed to load halo properties')
    
    all_data = {
        'm200b': halo_props['m200b'],
        'm200c': halo_props['m200c'],
        'mfof': halo_props.get('mfof'),
        'pos': halo_props['pos'],
    }
    
    return tau_map, all_data


def load_halo_data_from_cache(cache_path):
    """
    Load halo properties directly from the massbin props cache.

    Returns a dict with keys: pos, vel, m200b, r200b.
    All arrays are already the selected subset — no index subsetting needed.
    """
    cached = np.load(cache_path)
    return {
        'pos':   cached['pos'],    # (N, 3) float32
        'vel':   cached['vel'],    # (N, 3) float32 — velocities
        'm200b': cached['m200b'],  # (N,)   float32
        'r200b': cached['r200b'],  # (N,)   float32
        'm200c': None,             # not stored in cache
        'mfof':  None,
    }


@partial(jit, static_argnums=(1,))
def bilinear_interpolate_grid(cutout, scale_factor):
    """
    Bilinear interpolation to increase resolution using JAX.
    
    Args:
        cutout: Input array (H, W)
        scale_factor: Integer upsampling factor (static)
    
    Returns:
        Upsampled array (H*scale_factor, W*scale_factor)
    """
    if scale_factor == 1:
        return cutout
    
    H, W = cutout.shape
    H_new, W_new = H * scale_factor, W * scale_factor
    
    # Create coordinate grids for the new resolution
    # Coordinates in the original grid space
    y_new = jnp.arange(H_new) / scale_factor
    x_new = jnp.arange(W_new) / scale_factor
    
    # Get integer parts and fractional parts
    y0 = jnp.floor(y_new).astype(jnp.int32)
    x0 = jnp.floor(x_new).astype(jnp.int32)
    y1 = jnp.minimum(y0 + 1, H - 1)
    x1 = jnp.minimum(x0 + 1, W - 1)
    
    # Fractional parts
    fy = y_new - y0
    fx = x_new - x0
    
    # Broadcast for 2D interpolation
    Y0, X0 = jnp.meshgrid(y0, x0, indexing='ij')
    Y1, X1 = jnp.meshgrid(y1, x1, indexing='ij')
    FY, FX = jnp.meshgrid(fy, fx, indexing='ij')
    
    # Bilinear interpolation
    interpolated = (
        cutout[Y0, X0] * (1 - FY) * (1 - FX) +
        cutout[Y0, X1] * (1 - FY) * FX +
        cutout[Y1, X0] * FY * (1 - FX) +
        cutout[Y1, X1] * FY * FX
    )
    
    return interpolated


@partial(jit, static_argnums=(4, 5))
def extract_cutout_jax(tau_map, halo_x, halo_y, dx, half_size, n_cell):
    """
    Extract cutout centered on halo position using JAX.
    
    Args:
        tau_map: Full tau map (n_cell, n_cell)
        halo_x, halo_y: Halo position in cMpc/h
        dx: Pixel size in cMpc/h
        half_size: Half size of cutout in pixels (static)
        n_cell: Grid resolution (static)
    
    Returns:
        cutout: Extracted cutout
        center_x_frac: Fractional x position within cutout
        center_y_frac: Fractional y position within cutout
    """
    # Convert to pixel coordinates
    pix_x = halo_x / dx
    pix_y = halo_y / dx
    
    # Center pixel
    center_x = jnp.round(pix_x).astype(jnp.int32)
    center_y = jnp.round(pix_y).astype(jnp.int32)
    
    # Calculate cutout size (must be static for JAX)
    cutout_size = 2 * half_size + 1
    
    # Calculate start indices with boundary handling
    # Clamp the start indices so the cutout stays within bounds
    x_start = jnp.clip(center_x - half_size, 0, n_cell - cutout_size)
    y_start = jnp.clip(center_y - half_size, 0, n_cell - cutout_size)
    
    # Extract the cutout with fixed size
    cutout = jax.lax.dynamic_slice(
        tau_map,
        (x_start, y_start),
        (cutout_size, cutout_size)
    )
    
    # Calculate fractional position within cutout
    halo_x_cutout = pix_x - x_start
    halo_y_cutout = pix_y - y_start
    
    return cutout, halo_x_cutout, halo_y_cutout


@jit
def compute_aperture_signal_single(cutout_highres, true_x, true_y, dx_highres, r_ap):
    """
    Compute aperture signal for a single radius using JAX.
    
    Args:
        cutout_highres: High-resolution cutout
        true_x, true_y: True halo position in high-res pixels
        dx_highres: High-res pixel size
        r_ap: Aperture radius in cMpc/h
    
    Returns:
        tau_signal: Integrated tau signal
    """
    n_pix = cutout_highres.shape[0]
    
    # Create pixel coordinate grids
    pixel_indices = jnp.arange(n_pix)
    pixel_coords_x = pixel_indices * dx_highres - true_x * dx_highres
    pixel_coords_y = pixel_indices * dx_highres - true_y * dx_highres
    
    # Create 2D distance grid
    X, Y = jnp.meshgrid(pixel_coords_x, pixel_coords_y, indexing='ij')
    R = jnp.sqrt(X**2 + Y**2)
    
    # Create masks for inner and outer annuli
    mask_inner = R <= r_ap
    mask_outer = (R > r_ap) & (R <= jnp.sqrt(2) * r_ap)
    
    # Compute mean tau values
    tau_inner = jnp.sum(cutout_highres * mask_inner) / jnp.sum(mask_inner)
    tau_outer = jnp.sum(cutout_highres * mask_outer) / jnp.sum(mask_outer)
    
    # Integrated signal
    tau_signal = (tau_inner - tau_outer) * jnp.pi * r_ap**2
    
    return tau_signal


@partial(jit, static_argnums=(4, 6, 7))
def process_single_halo(tau_map, halo_x, halo_y, dx, half_size, r_apertures, n_cell, res_increase):
    """
    Process a single halo and compute all aperture signals.
    
    Args:
        tau_map: Full tau map
        halo_x, halo_y: Halo position
        dx: Pixel size
        half_size: Cutout half size (static)
        r_apertures: Array of aperture radii
        n_cell: Grid resolution (static)
        res_increase: Resolution increase factor (static)
    
    Returns:
        signals: Array of tau signals for each aperture
    """
    # Extract cutout
    cutout, true_x, true_y = extract_cutout_jax(tau_map, halo_x, halo_y, dx, half_size, n_cell)
    
    # Pad cutout if needed
    cutout_size = 2 * half_size + 1
    pad_x = cutout_size - cutout.shape[0]
    pad_y = cutout_size - cutout.shape[1]
    
    cutout = jnp.pad(cutout, ((0, pad_x), (0, pad_y)), mode='constant', constant_values=0)
    
    # Upsample to higher resolution
    if res_increase > 1:
        cutout_highres = bilinear_interpolate_grid(cutout, res_increase)
    else:
        cutout_highres = cutout
    
    # Adjust halo position for high-res grid
    true_x_highres = true_x * res_increase
    true_y_highres = true_y * res_increase
    dx_highres = dx / res_increase
    
    # Compute signals for all apertures
    def compute_for_radius(r_ap):
        return compute_aperture_signal_single(cutout_highres, true_x_highres, true_y_highres, dx_highres, r_ap)
    
    signals = vmap(compute_for_radius)(r_apertures)
    
    return signals


def process_batch_numpy(tau_map_jax, halo_positions, dx, half_size, r_apertures_jax, 
                        n_cell, batch_size=100, res_increase=8, rank=0):
    """
    Process a batch of halos using NumPy loop (for large batches that don't fit in GPU memory).
    
    Args:
        tau_map_jax: Tau map as JAX array
        halo_positions: Array of halo positions (N, 2)
        dx: Pixel size
        half_size: Cutout half size
        r_apertures_jax: Aperture radii as JAX array
        n_cell: Grid resolution
        batch_size: Internal batch size for processing
        res_increase: Resolution increase factor for upsampling
        rank: MPI rank (for progress bar display)
    
    Returns:
        signals: Array of signals (N, n_apertures)
    """
    n_halos = len(halo_positions)
    n_apertures = len(r_apertures_jax)
    signals = np.zeros((n_halos, n_apertures))
    
    # Create progress bar for rank 0
    iterator = range(0, n_halos, batch_size)
    if rank == 0:
        iterator = tqdm(iterator, desc=f"Processing halos (rank {rank})", total=(n_halos + batch_size - 1) // batch_size)
    
    for i in iterator:
        end_idx = min(i + batch_size, n_halos)
        batch_pos = halo_positions[i:end_idx]
        
        # Process each halo in the batch
        for j, (halo_x, halo_y) in enumerate(batch_pos):
            halo_signals = process_single_halo(tau_map_jax, halo_x, halo_y, dx, half_size, r_apertures_jax, n_cell, res_increase)
            signals[i + j] = np.array(halo_signals)
    
    return signals


def split_workload(n_total, n_ranks):
    """Split workload among MPI ranks."""
    base_size = n_total // n_ranks
    remainder = n_total % n_ranks
    
    # Calculate start and end indices for each rank
    counts = np.array([base_size + (1 if i < remainder else 0) for i in range(n_ranks)])
    displacements = np.array([sum(counts[:i]) for i in range(n_ranks)])
    
    return counts, displacements


def compute_distances(cosmo, z, h):
    d_L = cosmo.luminosity_distance(z).to(u.Mpc).value
    d_C = d_L / (1.0 + z)                      # Mpc
    d_A = d_L / (1.0 + z) ** 2                 # Mpc
    d_C *= h                                   # Mpc/h
    d_A *= h                                   # Mpc/h
    return d_A, d_C


def translate_grid_params_to_degrees(z, n_cell, sim_params):
    h = sim_params['h']
    om_m = sim_params['omega_m']
    tcmb0 = sim_params['tcmb0']
    lbox = sim_params['box_size_cMpc_h']

    a = 1.0 / (1.0 + z)
    cosmo = FlatLambdaCDM(H0=h * 100.0, Om0=om_m, Tcmb0=tcmb0)
    d_A, _ = compute_distances(cosmo, z, h)
    Lbox_rad = (a * lbox) / d_A                     # rad
    Lbox_deg = Lbox_rad * 180.0 / np.pi          # degrees
    cell_size_deg = Lbox_deg / n_cell

    return Lbox_rad, Lbox_deg, cell_size_deg

def gauss_beam(ellsq, fwhm):
    """
    Gaussian beam of size fwhm
    """
    tht_fwhm = np.deg2rad(fwhm/60.)
    return np.exp(-(tht_fwhm**2.)*(ellsq)/(2.*8.*np.log(2.)))

def get_smooth_density(D, fwhm, pixsizedeg, Lboxdeg, N_dim):
    """
    Smooth density map D ((0, Lbox] and N_dim^2 cells) with Gaussian beam of FWHM
    """
    kstep = 2*np.pi/(N_dim*np.pi/180*pixsizedeg)
    #karr = np.fft.fftfreq(N_dim, d=Lbox/(2*np.pi*N_dim)) # physical (not correct)
    karr = np.fft.fftfreq(N_dim, d=Lboxdeg*np.pi/180./(2*np.pi*N_dim)) # angular
    #print("kstep = ", kstep, karr[1]-karr[0]) # N_dim/d gives kstep

    # fourier transform the map and apply gaussian beam
    dfour = np.fft.fftn(D)
    dksmo = np.zeros((N_dim, N_dim), dtype=complex)
    ksq = np.zeros((N_dim, N_dim), dtype=complex)
    ksq[:, :] = karr[None, :]**2+karr[:,None]**2
    dksmo[:, :] = gauss_beam(ksq, fwhm)*dfour
    drsmo = np.real(np.fft.ifftn(dksmo))

    return drsmo

def match_aperture_radii(z, sim_params):
    """Get aperture radii matching the observational data."""
    figure_content = np.load("/home/fb635/fedirfiles/tracing_cosmic_gas/data/Fig2_sim.npz")
    theta_arcmin = figure_content['theta_arcmins']
    h = sim_params['h']
    om_m = sim_params['omega_m']
    tcmb0 = sim_params['tcmb0']
    cosmo = FlatLambdaCDM(H0=h * 100, Om0=om_m, Tcmb0=tcmb0)
    _, d_C = compute_distances(cosmo, z, h)
    theta_rad = theta_arcmin * (np.pi / 180) / 60  # radians
    r_comoving_mpc = d_C * theta_rad  # cMpc/h
    return r_comoving_mpc


def get_paths_from_config(config):
    """Derive file paths from configuration using pipeline_paths."""
    sim = config.get('sim_name', 'flamingo')
    return {
        'tau_map_path':          str(_tau_map_path(config)),
        'halo_galaxy_data_path': str(get_halo_file_path(config['gas_type'], sim_name=sim)),
        'output_file':           str(ap_output_path(config, method='simple')),
    }


def get_AP_simple(config, fwhm_beam_arcmin=1.6, batch_size=100, res_increase=8):
    """Main processing pipeline with MPI parallelization.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary with keys:
        - gas_type: str (e.g., 'fiducial', 'fiducial_reconstructed', 'strongest_AGN', 'strongest_AGN_reconstructed')
        - tau_method: str (e.g., 'fullFT_tau_reconstruction', 'histmethod_tau_reconstruction')
        - field_type: str ('gas' or 'dm', default 'gas') - use 'dm' for dark matter fictitious tau maps
        - n_cell: int (grid resolution, cells per side)
        - z_real: float (redshift)
        - n_gal_density: float (galaxy number density in cMpc/h^-3, e.g., 87e-5 or 1e-4) - only used when halo_mass_range is None
        - halo_mass_range: int or None (mass bin index 0 to N-1, where N bins are created with ~5000 halos per bin between 10^13 and max mass)
        - beam_smoothing: bool (whether to apply beam smoothing, default True)
        - selection_regime: str ('mhalo_sel' or 'mgal_sel', default 'mhalo_sel')
    fwhm_beam_arcmin : float, optional
        Beam FWHM in arcminutes (default 1.6)
    batch_size : int, optional
        Batch size for processing (default 100)
    res_increase : int, optional
        Resolution increase factor for upsampling (default 8)
    """
    
    config = dict(config)
    sim_name = config.get('sim_name', 'flamingo')
    sim_params = get_sim_params(sim_name)
    L_box = require_sim_param(sim_name, 'box_size_cMpc_h')
    n_cell = require_sim_param(sim_name, 'ngrid_default')
    z = config.get('z_real', 0.74)
    beam_smoothing = BEAM_SMOOTHED
    
    # Get paths from config
    paths = get_paths_from_config(config)
    
    cache_path = _halo_props_cache_path(config)

    if cache_path.exists():
        if rank == 0:
            print(f"Loading halo data from cache {cache_path}")
        all_data = load_halo_data_from_cache(cache_path)
        if rank == 0:
            print(f"✓ Successfully loaded {len(all_data['m200b'])} halo properties from cache")
            print(f"  - pos shape: {all_data['pos'].shape}")
            print(f"  - vel shape: {all_data['vel'].shape}")
            print(f"  - m200b range: [{np.log10(np.min(all_data['m200b'])):.2f}, {np.log10(np.max(all_data['m200b'])):.2f}]")
        
        halo_indices = resolve_halo_indices(config)
        
        gc.collect()
        if rank == 0:
            print(f"Loading tau map from {paths['tau_map_path']}")
        tau_map = np.load(paths['tau_map_path'])
    else:
        # no cache: load selected subset from source catalog
        halo_indices = resolve_halo_indices(config)
        if rank == 0:
            print(f"Loaded {len(halo_indices)} halo indices from halo_indices file.")
        
        # Load data on all ranks, subsetting to selected halo indices
        if rank == 0:
            print("\nLoading data...")

        tau_map, all_data = load_data(
            paths['tau_map_path'],
            paths['halo_galaxy_data_path'],
            sim=sim_name,
            halo_indices=halo_indices,
        )

    if beam_smoothing:
        _, Lbox_deg_real, cell_size_deg_real = translate_grid_params_to_degrees(z, n_cell, sim_params)
        tau_map = get_smooth_density(tau_map, fwhm_beam_arcmin, cell_size_deg_real, Lbox_deg_real, n_cell)
    
    # Extract selected positions and masses
    halo_pos_selected = all_data['pos'][:, :2]

    halo_m200b = all_data['m200b']
    if rank == 0:
        print(f"Selected {len(halo_indices)} objects")
        print(f"  Log10 m200b (m200m) range: [{np.log10(np.min(halo_m200b)):.2f}, {np.log10(np.max(halo_m200b)):.2f}]")
        print(f"  Mean m200b (m200m): {np.log10(np.mean(halo_m200b)):.2f} Msun/h")
    
    # Clean up
    del all_data
    
    if rank == 0:
        print("Starting aperture photometry...")
    
    # Convert to JAX arrays
    tau_map_jax = jnp.array(tau_map)
    R_comoving_mpc = match_aperture_radii(z, sim_params)
    r_apertures_jax = jnp.array(R_comoving_mpc)
    
    # Calculate cutout parameters
    dx = L_box / n_cell
    cutout_size = get_cutout_dims(np.max(R_comoving_mpc), dx)
    half_size = cutout_size // 2
    
    if rank == 0:
        print(f"Cutout size: {cutout_size} x {cutout_size} coarse pixels")
        print(f"High-res cutout: {cutout_size * res_increase} x {cutout_size * res_increase} pixels")
        print(f"Total halos: {len(halo_pos_selected)}")
        print(f"MPI ranks: {size}")
    
    # Split workload among MPI ranks
    n_halos_total = len(halo_pos_selected)
    counts, displacements = split_workload(n_halos_total, size)
    
    # Get this rank's subset of halos
    start_idx = displacements[rank]
    end_idx = start_idx + counts[rank]
    my_halo_positions = halo_pos_selected[start_idx:end_idx]
    
    if rank == 0:
        print(f"\nWork distribution:")
        for r in range(size):
            print(f"  Rank {r}: {counts[r]} halos (indices {displacements[r]}-{displacements[r]+counts[r]-1})")
    
    # Process this rank's halos
    if rank == 0:
        print(f"\nRank {rank}: Processing {counts[rank]} halos...")
    
    my_signals = process_batch_numpy(
        tau_map_jax, 
        my_halo_positions, 
        dx, 
        half_size, 
        r_apertures_jax,
        n_cell,
        batch_size=batch_size,
        res_increase=res_increase,
        rank=rank
    )
    
    if rank == 0:
        print(f"Rank {rank}: Completed processing")
    
    # Gather results on rank 0
    if rank == 0:
        all_signals = np.zeros((n_halos_total, len(R_comoving_mpc)))
    else:
        all_signals = None
    
    # Use Gatherv to collect variable-sized arrays
    sendcounts = counts * len(R_comoving_mpc)
    displacements_flat = displacements * len(R_comoving_mpc)
    
    comm.Gatherv(
        sendbuf=my_signals.flatten(),
        recvbuf=(all_signals, sendcounts, displacements_flat, MPI.DOUBLE) if rank == 0 else None,
        root=0
    )
    
    # Free large arrays to avoid memory issues
    del tau_map_jax
    gc.collect()
    
    # Save results on rank 0
    if rank == 0:
        print("\nSaving results...")
        
        output_file = paths['output_file']
        assert all_signals is not None
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        print("Aperture radii (cMpc/h):", R_comoving_mpc)
        
        np.savez(
            output_file,
            tau_signals=all_signals,
            inds_sub=halo_indices,
            r_comoving_mpc_h=R_comoving_mpc,
        )
        
        print(f"Results saved to: {output_file}")
        print(f"Shape: {all_signals.shape}")
        print(f"Min signal: {all_signals.min():.3e}")
        print(f"Max signal: {all_signals.max():.3e}")


def main():
    """Main function for standalone execution using default config."""
    from utils.pipeline_paths import selection_defaults
    
    config = {
        'sim_name': 'flamingo',
        'gas_type': 'strongest_AGN_reconstructed',
        'tau_method': 'fullFT_tau_reconstruction',
        'n_gal_density': 1e-4,
        'halo_mass_range': None,
        **selection_defaults('mgal_sel'),  # Use explicit parameters
    }
    get_AP_simple(config)


if __name__ == "__main__":
    main()