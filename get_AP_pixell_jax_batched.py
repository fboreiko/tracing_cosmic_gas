"""
Advanced JAX implementation with full batch processing and MPI parallelization.

This version extracts and processes multiple stamps simultaneously,
maximizing GPU utilization for even better performance.
MPI parallelization distributes halos across multiple processes/GPUs.
"""
import os
import numpy as np
from pixell import enmap, utils
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import deepdish as dd
from tqdm import tqdm
import jax
import jax.numpy as jnp
from jax import vmap, jit
from utils import *
from utils import rotfuncs

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


# --------------------
# Simulation Constants (never change)
# --------------------
LBOX = 681     # cMpc/h
H = 0.681
OM_M = 0.306

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

def compute_distances(cosmo, z):
    d_L = cosmo.luminosity_distance(z).to(u.Mpc).value
    d_C = d_L / (1.0 + z)                      # Mpc
    d_A = d_L / (1.0 + z) ** 2                 # Mpc
    d_C *= H                                   # Mpc/h
    d_A *= H                                   # Mpc/h
    return d_A, d_C

def translate_grid_params_to_degrees(z, n_cell):
    a = 1.0 / (1.0 + z)
    cosmo = FlatLambdaCDM(H0=H * 100.0, Om0=OM_M, Tcmb0=2.725)
    d_A, _ = compute_distances(cosmo, z)
    Lbox_rad = (a * LBOX) / d_A                     # rad
    Lbox_deg = Lbox_rad * 180.0 / np.pi          # degrees
    cell_size_deg = Lbox_deg / n_cell

    return Lbox_rad, Lbox_deg, cell_size_deg

def load_data(tau_map_path, halo_galaxy_data_path, selection_regime):
    """Load tau map and catalog data from unified HDF5 source.
    
    Args:
        tau_map_path: Path to tau map
        halo_galaxy_data_path: Path to HDF5 file with all galaxy/halo data
        selection_regime: 'mhalo_sel' or 'mgal_sel'
    
    Returns:
        tau_map: Tau map array
        filtered_data: Dict with filtered mstell, m200b, pos, centrals based on regime
    """
    tau_map = np.load(tau_map_path)
    
    # Load all data from HDF5 using deepdish
    with dd.io.load(halo_galaxy_data_path) as data:
        all_mstell = data['galaxies/mstell']
        all_centrals = data['galaxies/centrals']
        all_m200b = data['galaxies/m200b']  # m200b is already m200m
        all_pos = data['galaxies/pos']
    
    # Filter based on selection regime
    if selection_regime == 'mhalo_sel':
        # Use only central galaxies
        central_mask = all_centrals.astype(bool)
        filtered_mstell = all_mstell[central_mask]
        filtered_m200b = all_m200b[central_mask]
        filtered_pos = all_pos[central_mask]
        filtered_centrals = None  # Not needed for mhalo_sel
    else:  # mgal_sel
        # Use all galaxies
        filtered_mstell = all_mstell
        filtered_m200b = all_m200b
        filtered_pos = all_pos
        filtered_centrals = all_centrals  # Keep centrals for further filtering
    
    filtered_data = {
        'mstell': filtered_mstell,
        'm200b': filtered_m200b,
        'pos': filtered_pos,
        'centrals': filtered_centrals
    }
    
    return tau_map, filtered_data


def match_aperture_radii(z_real):
    """Get aperture radii matching the observational data."""
    figure_content = np.load("/home/fb635/fedirfiles/tracing_cosmic_gas/data/Fig2_sim.npz")
    theta_arcmin = figure_content['theta_arcmins']
    cosmo = FlatLambdaCDM(H0=H*100, Om0=OM_M, Tcmb0=2.725)
    _, d_C = compute_distances(cosmo, z_real)
    theta_rad = theta_arcmin * (np.pi / 180) / 60  # radians
    r_comoving_mpc = d_C * theta_rad  # cMpc/h
    return r_comoving_mpc

def compute_theta_arcmins_comoving(r_comoving_mpc_h, z):
    cosmo = FlatLambdaCDM(H0=H * 100.0, Om0=OM_M, Tcmb0=2.725)
    _, d_C_mpc_h = compute_distances(cosmo, z)
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


def prepare_pixel_coordinates_batch(stamp_template, ras, decs, cmbMap):
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
        
        ipos = rotfuncs.recenter(opos[::-1], [0, 0, sourcecoord[0], sourcecoord[1]])[::-1]
        
        # Convert to pixel coordinates
        pix_coords = cmbMap.sky2pix(ipos, safe=False)
        coords_batch[i] = pix_coords
    
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
            stamp_template, ras_batch, decs_batch, tau_xy_map
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
    
    return tau_inner, tau_outer
    

def get_paths_from_config(config):
    """Derive file paths from configuration."""
    gas_type = config['gas_type']
    tau_method = config['tau_method']
    beam_smoothing = config.get('beam_smoothing', True)
    projection_type = config.get('projection_type', 'car')
    selection_regime = config.get('selection_regime', 'mhalo_sel')  # Default to halo mass selection
    
    # Determine variant from gas_type
    if gas_type in ['fiducial_reconstructed', 'fiducial']:
        variant = 'fiducial'
    else:
        variant = 'strongest_AGN'
    
    # Determine tau map filename based on selection regime
    # Don't add regime_suffix for fullFT_tau_reconstruction
    if tau_method == 'fullFT_tau_reconstruction':
        regime_suffix = ''
    else:
        regime_suffix = '_mgal_sel' if selection_regime == 'mgal_sel' else ''
    
    # Handle mass-dependent tau reconstruction
    if tau_method == '2D_FT_massdep_tau_reconstruction':
        if 'A' not in config:
            raise ValueError(f"tau_method '{tau_method}' requires parameter 'A' to be specified in config")
        A = config['A']
        tau_map_path = f"/home/fb635/fedirfiles/tracing_cosmic_gas/data/tau_maps/{tau_method}/tau_map_{gas_type}_A_{A:.2f}{regime_suffix}.npy"
    else:
        # Tau map path for standard methods
        tau_map_path = f"/home/fb635/fedirfiles/tracing_cosmic_gas/data/tau_maps/{tau_method}/tau_map_{gas_type}{regime_suffix}.npy"
    
    # Single HDF5 data source
    halo_galaxy_data_path = f"/home/fb635/rds/hpc-work/tracing_cosmic_gas/FLAMINGO_ext_L1000N1800_HYDRO_{variant.upper()}_snap_77.hdf5"
    
    # Output path
    base_dir = "pixell_CAP_code" if projection_type == "car" else "pixell_cea_CAP_code"
    tau_method_dir = f"{tau_method}_smoothed" if beam_smoothing else tau_method
    
    # Add A parameter to filename if using mass-dependent reconstruction
    A_str = f"_A_{config['A']:.2f}" if tau_method == '2D_FT_massdep_tau_reconstruction' else ""
    
    # Add selection regime suffix only for mgal_sel
    regime_str = "_mgal_sel" if selection_regime == 'mgal_sel' else ""
    
    if config.get('halo_mass_range') is not None:
        subdir = f"tau_apertures_{gas_type}_massbin_{config['halo_mass_range']}_ngrid_{config['n_cell']}{regime_str}_JAX{A_str}"
    else:
        subdir = f"tau_apertures_{gas_type}_ngrid_{config['n_cell']}{regime_str}_JAX_MPI{A_str}"
    
    output_file = f"/home/fb635/fedirfiles/tracing_cosmic_gas/data/{base_dir}/{tau_method_dir}/{subdir}/tau_apertures_z_{config['z_real']}.npz"
    
    return {
        'tau_map_path': tau_map_path,
        'halo_galaxy_data_path': halo_galaxy_data_path,
        'output_file': output_file
    }


def get_AP_pixell(config, z_fict=3.0, cutout_pixel_dim=150, fwhm_beam_arcmin=1.6, batch_size=100):
    """Main processing pipeline with MPI parallelization.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary with keys:
        - gas_type: str (e.g., 'fiducial', 'fiducial_reconstructed', 'strongest_AGN', 'strongest_AGN_reconstructed')
        - tau_method: str (e.g., 'fullFT_tau_reconstruction', 'histmethod_tau_reconstruction')
        - n_cell: int (grid resolution, cells per side)
        - z_real: float (real redshift)
        - n_gal_density: float (galaxy number density in cMpc/h^-3, e.g., 87e-5 or 5e-4) - only used when halo_mass_range is None
        - halo_mass_range: int or None (mass bin index 0 to N-1, where N bins are created with ~5000 halos per bin between 10^13 and max mass)
        - beam_smoothing: bool (whether to apply beam smoothing, default True)
        - projection_type: str ('car' or 'cea', default 'car')
        - selection_regime: str ('mhalo_sel' or 'mgal_sel', default 'mhalo_sel')
    z_fict : float, optional
        Fictitious redshift for projection (default 3.0)
    cutout_pixel_dim : int, optional
        Fixed cutout pixel dimensions (default 150)
    fwhm_beam_arcmin : float, optional
        Beam FWHM in arcminutes (default 1.6)
    batch_size : int, optional
        Batch size for processing (default 100)
    """
    
    # Extract config values with defaults
    gas_type = config['gas_type']
    tau_method = config['tau_method']
    n_cell = config['n_cell']
    z_real = config['z_real']
    n_gal_density = config['n_gal_density']
    halo_mass_range = config.get('halo_mass_range', None)
    beam_smoothing = config.get('beam_smoothing', True)
    proj_cutout = config.get('projection_type', 'car')
    selection_regime = config.get('selection_regime', 'mhalo_sel')  # Default to halo mass selection
    
    # Get paths from config
    paths = get_paths_from_config(config)
    
    if rank == 0:
        print("=" * 70)
        print("Advanced JAX Batched Aperture Photometry with MPI")
        print("=" * 70)
        print(f"Number of MPI processes: {size}")
        print(f"Gas type: {gas_type}")
        print(f"Tau method: {tau_method}")
        print(f"N_cell: {n_cell}")
        print(f"Real redshift: {z_real}")
        print(f"Fictitious redshift: {z_fict}")
        print(f"N_gal_density: {n_gal_density:.3e} cMpc/h^-3")
        print(f"Projection type: {proj_cutout}")
        print(f"Selection regime: {selection_regime}")

    # Load data on all ranks
    if rank == 0:
        print(f"\nLoading data from:")
        print(f"  Tau map: {paths['tau_map_path']}")
        print(f"  Catalog: {paths['halo_galaxy_data_path']}")
    
    tau_map, filtered_data = load_data(
        paths['tau_map_path'],
        paths['halo_galaxy_data_path'],
        selection_regime
    )
    
    # Apply beam smoothing if needed
    if beam_smoothing:
        _, Lbox_deg_real, cell_size_deg_real = translate_grid_params_to_degrees(z_real, n_cell)
        tau_map = get_smooth_density(tau_map, fwhm_beam_arcmin, cell_size_deg_real, Lbox_deg_real, n_cell)
    
    # Extract filtered data
    filtered_mstell = filtered_data['mstell']
    filtered_m200b = filtered_data['m200b']
    filtered_pos = filtered_data['pos']
    filtered_centrals = filtered_data['centrals']
    
    if rank == 0:
        if selection_regime == 'mhalo_sel':
            print(f"Using halo mass selection (mhalo_sel) - centrals only")
            print(f"  {len(filtered_pos)} centrals found")
        else:
            print(f"Using galaxy stellar mass selection (mgal_sel) - all galaxies")
            print(f"  {len(filtered_pos)} galaxies found")
    
    # Select halos/galaxies based on selection criteria
    if selection_regime == 'mgal_sel':
        # For mgal_sel, select by stellar mass
        selection_masses = filtered_mstell
    else:
        # For mhalo_sel, select by m200b (halo mass)
        selection_masses = filtered_m200b
    
    # Select halos/galaxies based on selection criteria
    halo_pos_full, inds_sub = select_halos(
        selection_masses, filtered_pos, 
        n_gal_density=n_gal_density, 
        halo_mass_range=halo_mass_range,
        rank=rank,
        select_positive_mass=(selection_regime == 'mgal_sel'),  # For mgal_sel, select galaxies with mstell > 0
        centrals=filtered_centrals  # Pass centrals array for mgal_sel filtering
    )
    
    if rank == 0:
        if selection_regime == 'mgal_sel':
            print(f"Selected {len(inds_sub)} galaxies based on stellar mass")
        else:
            print(f"Selected {len(inds_sub)} centrals based on halo mass")
    
    # Clean up filtered arrays
    del filtered_data, filtered_mstell, filtered_m200b, filtered_pos, filtered_centrals, selection_masses
    
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
    R_COMOVING_MPC = match_aperture_radii(z_real)
    THETA_ARCMINS = compute_theta_arcmins_comoving(R_COMOVING_MPC, z_fict)
    
    # Get geometry
    Lbox_rad_fict, Lbox_deg_fict, cell_size_deg_fict = translate_grid_params_to_degrees(z_fict, n_cell)
    
    r_ap_max_arcmin = THETA_ARCMINS.max()
    RES_CUTOUT_ARCMIN = 2 * r_ap_max_arcmin / cutout_pixel_dim
    
    # Run batched processing on local subset
    if rank == 0:
        print(f"\nRunning batched processing with batch_size={batch_size}")
    
    tau_inner_local, tau_outer_local = run_batched_aperture_photometry(
        tau_map, halo_pos_local, THETA_ARCMINS, R_COMOVING_MPC,
        Lbox_rad_fict, Lbox_deg_fict, cell_size_deg_fict, n_cell, LBOX,
        r_ap_max_arcmin, RES_CUTOUT_ARCMIN, proj_cutout,
        batch_size=batch_size, rank=rank, size=size
    )
    
    # Gather results from all ranks to rank 0
    if USE_MPI:
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
    config = {
        'gas_type': 'fiducial_reconstructed',
        'tau_method': 'fullFT_tau_reconstruction',
        'n_cell': 2048,
        'z_real': 0.74,
        'n_gal_density': 5e-4,
        'halo_mass_range': None,
        'beam_smoothing': True,
        'projection_type': 'car',
        'selection_regime': 'mhalo_sel'
    }
    get_AP_pixell(config)


if __name__ == "__main__":
    main()