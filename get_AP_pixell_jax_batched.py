"""
Advanced implementation of aperture photometry in JAX with MPI parallelization.
This version supports geometric corrections for the curvature of the sky
through the "cea" projection option. Default is set to "car", which is a simple 
Cartesian projection, following the get_AP_simple approach, because tests show 
that the difference is negligible. Still worth keeping the option for future use.
"""

import os
import gc
import numpy as np
from pixell import enmap, utils, enplot
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import jax
import jax.numpy as jnp
from jax import vmap, jit
import sys
sys.path.insert(0, '/home/fb635/fedirfiles/tracing_cosmic_gas')
from utils.tools import (
    cutoutGeometry,
    bilinear_interpolate_batch,
    extractStamp_jax,
    compute_all_apertures_jax,
    build_aperture_templates_jax,
)
from utils.rotfuncs import recenter
from utils.pipeline_paths import (
    get_halo_file_path,
    tau_map_path as _tau_map_path,
    ap_output_path,
    halo_indices_path as _halo_indices_path,
    resolve_halo_indices,
    halo_props_cache_path as _halo_props_cache_path,
)
from utils.catalog_loaders import load_halo_properties
from utils.sim_params import get_sim_params, require_sim_param

# MPI imports
try:
    from mpi4py import MPI
    USE_MPI = True
except ImportError:
    USE_MPI = False
    print("Warning: mpi4py not found. Running in serial mode.")

# Initialize MPI
if USE_MPI:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
else:
    comm = None
    rank = 0
    size = 1

# Configure JAX - assign different GPU to each MPI rank if available
if USE_MPI and jax.default_backend() == 'gpu':
    jax.config.update("jax_default_device", jax.devices()[rank % len(jax.devices())])

jax.config.update("jax_enable_x64", True)
if rank == 0:
    print(f"Running with {size} MPI processes")
    print(f"JAX devices: {jax.devices()}")
    print(f"JAX backend: {jax.default_backend()}")


BEAM_SMOOTHED = True   # beam smoothing always applied

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

    # fourier transform the map and apply gaussian beam
    dfour = np.fft.fftn(D)
    dksmo = np.zeros((N_dim, N_dim), dtype=complex)
    ksq = np.zeros((N_dim, N_dim), dtype=complex)
    ksq[:, :] = karr[None, :]**2+karr[:,None]**2
    dksmo[:, :] = gauss_beam(ksq, fwhm)*dfour
    drsmo = np.real(np.fft.ifftn(dksmo))

    return drsmo

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
            'm200c': None,
            'mfof': None,
            'pos': halo_props['x_L2com'],
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


def match_aperture_radii(z_real, sim_params):
    """Get aperture radii matching the observational data."""
    figure_content = np.load("/home/fb635/fedirfiles/tracing_cosmic_gas/data/Fig2_sim.npz")
    theta_arcmin = figure_content['theta_arcmins']
    h = sim_params['h']
    om_m = sim_params['omega_m']
    tcmb0 = sim_params['tcmb0']
    cosmo = FlatLambdaCDM(H0=h * 100, Om0=om_m, Tcmb0=tcmb0)
    _, d_C = compute_distances(cosmo, z_real, h)
    theta_rad = theta_arcmin * (np.pi / 180) / 60  # radians
    r_comoving_mpc = d_C * theta_rad  # cMpc/h
    return r_comoving_mpc

def compute_theta_arcmins_comoving(r_comoving_mpc_h, z, sim_params):
    h = sim_params['h']
    om_m = sim_params['omega_m']
    tcmb0 = sim_params['tcmb0']
    cosmo = FlatLambdaCDM(H0=h * 100.0, Om0=om_m, Tcmb0=tcmb0)
    _, d_C_mpc_h = compute_distances(cosmo, z, h)
    theta_rad = r_comoving_mpc_h / d_C_mpc_h
    return theta_rad * 180.0 / np.pi * 60.0


@jit
def compute_apertures_batch(stamps_batch, inners, outers, r_comoving_mpc):
    """
    Compute apertures for a batch of stamps.
    
    stamps_batch: [batch, ny, nx] stamps
    inners: [n_radii, ny, nx] boolean masks
    outers: [n_radii, ny, nx] boolean masks
    r_comoving_mpc: [n_radii] radii
    
    Returns: tau_inner [batch, n_radii], tau_outer [batch, n_radii]
    """
    def compute_single_stamp(stamp):
        def compute_single_aperture(inner, outer, r_com):
            # Inner aperture
            mean_inner = jnp.sum(stamp * inner) / jnp.sum(inner)
            a_inn = mean_inner * jnp.pi * r_com ** 2 
            
            # Outer aperture
            mean_outer = jnp.sum(stamp * outer) / jnp.sum(outer)
            a_out = mean_outer * jnp.pi * r_com ** 2 
            
            return a_inn, a_out
        
        # Vectorize over apertures
        a_inns, a_outs = vmap(compute_single_aperture)(inners, outers, r_comoving_mpc)
        return a_inns, a_outs
    
    # Vectorize over batch
    tau_inners, tau_outers = vmap(compute_single_stamp)(stamps_batch)
    return tau_inners, tau_outers


def prepare_pixel_coordinates_batch(stamp_template, ras, decs, cmbMap, theta_arcmins=None):
    """
    Prepare pixel coordinates for a batch of positions.
    
    Returns: [batch, 2, ny, nx] pixel coordinates
    """
    batch_size = len(ras)
    opos = stamp_template.posmap()
    ny, nx = opos.shape[1:]
    
    coords_batch = np.zeros((batch_size, 2, ny, nx))
    
    for i, (ra, dec) in enumerate(zip(ras, decs)):
        sourcecoord = np.array([ra, dec]) * utils.degree
        
        ipos = recenter(opos[::-1], [0, 0, sourcecoord[0], sourcecoord[1]])[::-1]
        
        # Convert to pixel coordinates
        pix_coords = cmbMap.sky2pix(ipos, safe=False)
        coords_batch[i] = pix_coords

        #HERE I BUTCHER TO PLOT
        #stamp_template[:, :] = cmbMap.at(ipos, order=1)
        #plots = enplot.plot(enmap.upgrade(stamp_template, 5), grid=True)        
        #enplot.write(f"plots/temporary_plots_pixell/plt_ra_{ra:.2f}_dec_{dec:.2f}.png", plots)
    
    return coords_batch


def process_batch(cmbMap_jax, coords_batch, inners_jax, outers_jax, r_comoving_mpc_jax):
    """
    Process a complete batch: extract stamps and compute apertures.
    
    Returns: tau_inner, tau_outer for the batch
    """
    # Extract all stamps in one JAX call
    stamps_batch = bilinear_interpolate_batch(cmbMap_jax, coords_batch)
    
    # Compute all apertures in one JAX call
    tau_inner, tau_outer = compute_apertures_batch(
        stamps_batch, inners_jax, outers_jax, r_comoving_mpc_jax
    )
    
    return tau_inner, tau_outer


def run_batched_aperture_photometry(
    tau_map, halo_pos, theta_arcmins, r_comoving_mpc,
    Lbox_rad, Lbox_deg, cell_size_deg, n_cell, LBOX,
    r_ap_max_arcmin, res_cutout_arcmin, proj_cutout,
    batch_size=100, rank=0, size=1
):
    """
    Main function for fully batched aperture photometry with MPI support.
    
    Parameters:
    -----------
    tau_map : ndarray
        Full tau map
    halo_pos : ndarray [N, 3]
        Halo positions in cMpc/h (only local subset for this rank)
    theta_arcmins : ndarray
        Angular aperture radii
    r_comoving_mpc : ndarray
        Comoving aperture radii
    batch_size : int
        Number of halos to process simultaneously
    rank : int
        MPI rank
    size : int
        Number of MPI processes
    
    Returns:
    --------
    tau_inner, tau_outer : ndarrays [N_local, n_radii]
        Results for local halos assigned to this rank
    """
    
    # Create container map
    box = np.array([[0.0, 0.0], [Lbox_deg, Lbox_deg]]) * utils.degree
    shape, wcs = enmap.geometry(pos=box, res=cell_size_deg * utils.degree, proj="car")
    tau_xy_map = enmap.zeros(shape, wcs=wcs)
    tau_xy_map[:] = tau_map
    
    # Convert to JAX
    cmbMap_jax = jnp.array(np.array(tau_xy_map))
    
    # Convert positions to degrees
    cell_size = LBOX / n_cell
    pos_deg = halo_pos * (cell_size_deg / cell_size)
    DEC = pos_deg[:, 0]
    RA = pos_deg[:, 1]
    
    n_halos = len(RA)
    n_radii = len(theta_arcmins)
    
    # Build aperture templates
    if rank == 0:
        print("Building aperture templates...")
    stamp_template = cutoutGeometry(
        rApMaxArcmin=r_ap_max_arcmin,
        resCutoutArcmin=res_cutout_arcmin,
        projCutout=proj_cutout
    )
    inners_jax, outers_jax = build_aperture_templates_jax(stamp_template, theta_arcmins)
    r_comoving_mpc_jax = jnp.array(r_comoving_mpc)
    
    if rank == 0:
        print(f"Template shape: {stamp_template.shape}")
        print(f"Processing {n_halos} halos on rank {rank} in batches of {batch_size}")
    
    # Initialize output arrays
    tau_inner = np.zeros((n_halos, n_radii))
    tau_outer = np.zeros((n_halos, n_radii))
    
    # Process in batches
    n_batches = (n_halos + batch_size - 1) // batch_size
    
    # Only show progress bar on rank 0
    iterator = tqdm(range(n_batches), desc=f"Rank {rank}") if rank == 0 else range(n_batches)
    
    for batch_idx in iterator:
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_halos)
        current_batch_size = end_idx - start_idx
        
        # Get batch coordinates
        ras_batch = RA[start_idx:end_idx]
        decs_batch = DEC[start_idx:end_idx]
        
        # Prepare pixel coordinates
        coords_batch = prepare_pixel_coordinates_batch(
            stamp_template, ras_batch, decs_batch, tau_xy_map, theta_arcmins
        )
        coords_batch_jax = jnp.array(coords_batch)
        
        # Process entire batch
        tau_inn_batch, tau_out_batch = process_batch(
            cmbMap_jax, coords_batch_jax,
            inners_jax, outers_jax, r_comoving_mpc_jax
        )
        
        # Store results
        tau_inner[start_idx:end_idx] = np.array(tau_inn_batch)
        tau_outer[start_idx:end_idx] = np.array(tau_out_batch)

        exit()
    
    return tau_inner, tau_outer
    

def get_paths_from_config(config):
    """Derive file paths from configuration using pipeline_paths."""
    sim = config.get('sim_name', 'flamingo')
    return {
        'tau_map_path':          str(_tau_map_path(config)),
        'halo_galaxy_data_path': str(get_halo_file_path(config['gas_type'], sim_name=sim)),
        'output_file':           str(ap_output_path(config, method='pixell')),
    }


def get_AP_pixell(config, z_fict=3.0, cutout_pixel_dim=150, fwhm_beam_arcmin=1.6, batch_size=100,
                  proj_cutout='car'):
    """Main processing pipeline with MPI parallelization.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary with keys:
        - gas_type: str (e.g., 'fiducial', 'fiducial_reconstructed', 'strongest_AGN', 'strongest_AGN_reconstructed')
        - tau_method: str (e.g., 'fullFT_tau_reconstruction', 'histmethod_tau_reconstruction')
        - field_type: str ('gas' or 'dm', default 'gas') - use 'dm' for dark matter fictitious tau maps
        - n_cell: int (grid resolution, cells per side)
        - z_real: float (real redshift)
        - n_gal_density: float (galaxy number density in cMpc/h^-3, e.g., 87e-5 or 5e-4) - only used when halo_mass_range is None
        - halo_mass_range: int or None (mass bin index 0 to N-1, where N bins are created with ~5000 halos per bin between 10^13 and max mass)
        - selection_regime: str ('mhalo_sel' or 'mgal_sel', default 'mhalo_sel')
    z_fict : float, optional
        Fictitious redshift for projection (default 3.0)
    cutout_pixel_dim : int, optional
        Fixed cutout pixel dimensions (default 150)
    fwhm_beam_arcmin : float, optional
        Beam FWHM in arcminutes (default 1.6)
    batch_size : int, optional
        Batch size for processing (default 100)
    proj_cutout : str, optional
        Cutout map projection passed to pixell, 'car' or 'cea' (default 'car').
        This is NOT the config key 'projection_type', which selects the AP
        method ('simple' / 'pixell' / 'pixell_cea') in the conductor and plotter.
    """
    
    sim_name = config.get('sim_name', 'flamingo')
    config = dict(config)
    config['sim_name'] = sim_name
    sim_params = get_sim_params(sim_name)
    L_box = require_sim_param(sim_name, 'box_size_cMpc_h')
    n_cell = require_sim_param(sim_name, 'ngrid_default')
    z_real = config.get('z_real', 0.74)
    beam_smoothing = BEAM_SMOOTHED
    
    # Get paths from config
    paths = get_paths_from_config(config)
    
    cache_path = _halo_props_cache_path(config)

    if cache_path.exists():
        if rank == 0:
            print(f"loading halo data from cache {cache_path}")
        all_data = load_halo_data_from_cache(cache_path)
        
        inds_sub = resolve_halo_indices(config)
        
        gc.collect()
        tau_map = np.load(paths['tau_map_path'])
    else:
        # no cache: load selected subset from source catalog
        inds_sub = resolve_halo_indices(config)
        if rank == 0:
            print(f"Loaded {len(inds_sub)} halo indices from halo_indices file.")

        # Load data on all ranks, subsetting to selected halo indices
        if rank == 0:
            print(f"\nLoading data from:")
            print(f"  Tau map: {paths['tau_map_path']}")
            print(f"  Catalog: {paths['halo_galaxy_data_path']}")
            print(f"  (Loading only {len(inds_sub)} selected halos to reduce memory usage)")
        
        tau_map, all_data = load_data(
            paths['tau_map_path'],
            paths['halo_galaxy_data_path'],
            sim=sim_name,
            halo_indices=inds_sub,
        )
    
    # Apply beam smoothing if needed
    if beam_smoothing:
        _, Lbox_deg_real, cell_size_deg_real = translate_grid_params_to_degrees(z_real, n_cell, sim_params)
        tau_map = get_smooth_density(tau_map, fwhm_beam_arcmin, cell_size_deg_real, Lbox_deg_real, n_cell)
    
    # Extract selected positions (now all_data only contains selected halos)
    halo_pos_full = all_data['pos'][:, :2]
    
    if rank == 0:
        print(f"Selected {len(inds_sub)} objects")
    
    # Clean up
    del all_data
    
    # Distribute halos across MPI ranks
    n_halos_total = len(halo_pos_full)
    halos_per_rank = n_halos_total // size
    remainder = n_halos_total % size
    
    # Calculate start and end indices for this rank
    if rank < remainder:
        start_idx = rank * (halos_per_rank + 1)
        end_idx = start_idx + halos_per_rank + 1
    else:
        start_idx = rank * halos_per_rank + remainder
        end_idx = start_idx + halos_per_rank
    
    # Get local subset of halos
    halo_pos_local = halo_pos_full[start_idx:end_idx]
    inds_sub_local = inds_sub[start_idx:end_idx]
    
    if rank == 0:
        print(f"Total halos: {n_halos_total}")
        print(f"Halos per rank: ~{halos_per_rank}")
    print(f"Rank {rank}: processing {len(halo_pos_local)} halos (indices {start_idx}-{end_idx})")
    
    # Get aperture radii
    R_COMOVING_MPC = match_aperture_radii(z_real, sim_params)
    THETA_ARCMINS = compute_theta_arcmins_comoving(R_COMOVING_MPC, z_fict, sim_params)
    
    # Get geometry
    Lbox_rad_fict, Lbox_deg_fict, cell_size_deg_fict = translate_grid_params_to_degrees(z_fict, n_cell, sim_params)
    
    r_ap_max_arcmin = THETA_ARCMINS.max()
    RES_CUTOUT_ARCMIN = 2 * r_ap_max_arcmin / cutout_pixel_dim
    
    # Run batched processing on local subset
    if rank == 0:
        print(f"\nRunning batched processing with batch_size={batch_size}")
    
    tau_inner_local, tau_outer_local = run_batched_aperture_photometry(
        tau_map, halo_pos_local, THETA_ARCMINS, R_COMOVING_MPC,
        Lbox_rad_fict, Lbox_deg_fict, cell_size_deg_fict, n_cell, L_box,
        r_ap_max_arcmin, RES_CUTOUT_ARCMIN, proj_cutout,
        batch_size=batch_size, rank=rank, size=size
    )
    
    # Gather results from all ranks to rank 0
    if USE_MPI:
        assert comm is not None
        # Gather sizes
        local_size = len(tau_inner_local)
        all_sizes = comm.gather(local_size, root=0)
        
        if rank == 0:
            # Prepare receive buffers
            tau_inner_full = np.zeros((n_halos_total, len(R_COMOVING_MPC)))
            tau_outer_full = np.zeros((n_halos_total, len(R_COMOVING_MPC)))
            inds_sub_full = np.zeros(n_halos_total, dtype=int)
            
            # Gather from all ranks
            for r in range(size):
                if r < remainder:
                    r_start = r * (halos_per_rank + 1)
                    r_end = r_start + halos_per_rank + 1
                else:
                    r_start = r * halos_per_rank + remainder
                    r_end = r_start + halos_per_rank
                
                if r == 0:
                    tau_inner_full[r_start:r_end] = tau_inner_local
                    tau_outer_full[r_start:r_end] = tau_outer_local
                    inds_sub_full[r_start:r_end] = inds_sub_local
                else:
                    tau_inner_full[r_start:r_end] = comm.recv(source=r, tag=0)
                    tau_outer_full[r_start:r_end] = comm.recv(source=r, tag=1)
                    inds_sub_full[r_start:r_end] = comm.recv(source=r, tag=2)
        else:
            # Send results to rank 0
            comm.send(tau_inner_local, dest=0, tag=0)
            comm.send(tau_outer_local, dest=0, tag=1)
            comm.send(inds_sub_local, dest=0, tag=2)
    else:
        # No MPI - just use local results
        tau_inner_full = tau_inner_local
        tau_outer_full = tau_outer_local
        inds_sub_full = inds_sub_local
    
    # Free large arrays to avoid memory issues
    del tau_map
    gc.collect()
    
    # Save results (only rank 0)
    if rank == 0:
        output_file = paths['output_file']
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        np.savez(
            output_file,
            tau_xy_inner=tau_inner_full,
            tau_xy_outer=tau_outer_full,
            inds_sub=inds_sub_full,
            r_comoving_mpc_h=R_COMOVING_MPC,
            theta_arcmins=THETA_ARCMINS,
            redshift_real=z_real,
            redshift_fictitious=z_fict
        )
        
        print(f"\n✓ Results saved: {output_file}")
        print("=" * 70)


def main():
    """Main function for standalone execution using default config."""
    from utils.pipeline_paths import selection_defaults

    config = {
        'sim_name': 'flamingo',
        'gas_type': 'strongest_AGN_reconstructed',
        'tau_method': 'fullFT_tau_reconstruction',
        'field_type': 'gas',
        'n_gal_density': 1e-4,
        'halo_mass_range': None,
        'convergence_mode': 'ngal',
        'target_mean_mass': None,
        **selection_defaults('mgal_sel'),  # Use explicit parameters
    }
    get_AP_pixell(config)


if __name__ == "__main__":
    main()