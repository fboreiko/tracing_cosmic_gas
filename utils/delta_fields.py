"""
   This script contains utility functions for computing gas/dm and halo 
   overdensity fields (delta fields) from particle and halo catalogs. 

   -- compute_delta_field_and_mass is called by the HATF reconstruction pipeline 
      in case it hasn't been computed before.
    
   -- compute_selected_halo_delta_2d is called by the HATF reconstruction pipeline
      and loads/computes the halo delta field based on the config parameters. 
"""

import numpy as np
from abacusnbody.analysis.tsc import tsc_parallel
from pathlib import Path
import gc

from .catalog_loaders import load_particle_properties
from .catalog_loaders import load_halo_properties
from .sample_selection import select_halos
from .pipeline_paths import get_halo_file_path, ensure_parents
from .sim_params import get_sim_params


def compute_delta_2d(pos, box, ngrid, weights, nthread=4):
    """Compute 2D overdensity field using direct 2D TSC assignment."""
    dens_2d = tsc_parallel(pos, (ngrid, ngrid), box, weights=weights, nthread=nthread)
    dens_avg = np.sum(dens_2d) / dens_2d.size
    delta_2d = dens_2d / dens_avg - 1
    return delta_2d


def compute_delta_3d_projected(pos, box, ngrid, weights, nthread=4):
    """Compute 3D overdensity field with TSC and project to 2D by averaging over z."""
    dens_3d = tsc_parallel(pos, ngrid, box, weights=weights, nthread=nthread)
    dens_avg = np.sum(dens_3d) / dens_3d.size
    delta_3d = dens_3d / dens_avg - 1
    delta_2d = np.mean(delta_3d, axis=2)
    return delta_2d


def compute_delta_field_and_mass(
    field_type,
    sim_name,
    dm_particles_file,
    gas_particles_file,
    box,
    ngrid,
    nthread=4,
):
    """Compute projected delta field and total mass used for tau prefactor."""
    if field_type == 'dm':

        if sim_name == 'flamingo':

            print("  Loading gas masses for rescaling...")
            gas_particles = load_particle_properties(
                gas_particles_file,
                'gas',
                requested=('mass',),
                sim_name='flamingo',
                Lbox=box,
            )
            if gas_particles is None:
                raise RuntimeError('Failed to load gas particle properties')

            masses_gas = gas_particles['mass']
            total_gas_mass = np.sum(masses_gas)

            del gas_particles, masses_gas
            gc.collect()

            print("  Loading DM particles...")
            dm_particles = load_particle_properties(
                dm_particles_file,
                'dm',
                requested=('mass', 'pos'),
                sim_name='flamingo',
                Lbox=box,
            )
            if dm_particles is None:
                raise RuntimeError('Failed to load DM particle properties')
            masses_dm = dm_particles['mass']
            pos_dm = dm_particles['pos']

            total_dm_mass = np.sum(masses_dm)
            rescale_factor = total_gas_mass / total_dm_mass

            print(f"  Total DM mass (before rescaling): {total_dm_mass:.4e} (1e10 Msun/h)")
            print(f"  Total gas mass: {total_gas_mass:.4e} (1e10 Msun/h)")
            print(f"  Rescale factor: {rescale_factor:.6f}")

            total_mass_for_tau = np.sum(masses_dm * rescale_factor)
            weights = masses_dm / np.mean(masses_dm)

            del dm_particles, masses_dm
            gc.collect()

            delta_2d_field = compute_delta_2d(pos_dm, box, ngrid, weights, nthread=nthread)

        elif sim_name == 'abacus':

            print("  Processing chunks of DM particles from Abacus...")
            base_path = Path(dm_particles_file)
            halo_info_path = base_path / "halo_info"
            chunk_fns = sorted(halo_info_path.glob("halo_info_*.asdf"))

            total_dm_mass = 0.0
            dens_2d = np.zeros((ngrid, ngrid), dtype=np.float64)

            for i_chunk, _ in enumerate(chunk_fns):
                dm_particles = load_particle_properties(
                    dm_particles_file,
                    'dm',
                    requested=('pos'),
                    sim_name='abacus',
                    Lbox=box,
                    chunkid=i_chunk,
                )
                if dm_particles is None:
                    raise RuntimeError(f'Failed to load DM particles for chunk {i_chunk}')

                pos_dm = dm_particles['pos']
                #print(f"    Processing chunk {i_chunk + 1}/{len(chunk_fns)} with {pos_dm.shape[0]} particles...")
                #print(np.min(pos_dm), np.max(pos_dm))

                num_particles = pos_dm.shape[0]
                total_dm_mass += num_particles * 0.21 * 100/3 # 0.21 Msun/h per particle, 100/3 accounts for downsampling

                dens_2d_chunk = tsc_parallel(pos_dm, (ngrid, ngrid), box, weights=None, nthread=nthread)
                dens_2d += dens_2d_chunk

                del dm_particles, pos_dm, dens_2d_chunk
                gc.collect()

            dens_avg = np.sum(dens_2d) / dens_2d.size
            delta_2d_field = dens_2d / dens_avg - 1

            omega_b = get_sim_params(sim_name)['omega_b']
            omega_cdm = get_sim_params(sim_name)['omega_m'] - get_sim_params(sim_name)['omega_b']
            rescale_factor = omega_b / omega_cdm

            total_mass_for_tau = total_dm_mass * rescale_factor

            print(f"  Total DM mass (before rescaling): {total_dm_mass:.4e} (1e10 Msun/h)")
            print(f"  Total gas mass (estimated from cosmology): {total_mass_for_tau:.4e} (1e10 Msun/h)")
            print(f"  Rescale factor: {rescale_factor:.6f}")

        else:
            raise ValueError(f"Unsupported sim_name for dm field: {sim_name}")

    else:
        print("  Loading gas particles...")
        gas_particles = load_particle_properties(
            gas_particles_file,
            'gas',
            requested=('mass', 'pos'),
            sim_name='flamingo',
            Lbox=box,
        )
        if gas_particles is None:
            raise RuntimeError('Failed to load gas particle properties')
        masses_gas = gas_particles['mass']
        pos_gas = gas_particles['pos']
        total_mass_for_tau = np.sum(masses_gas)

        weights = masses_gas / np.mean(masses_gas)

        print("  Computing 2D delta field using direct TSC...")
        delta_2d_field = compute_delta_2d(pos_gas, box, ngrid, weights, nthread=nthread)

    return delta_2d_field, total_mass_for_tau


def save_halo_props_cache(
    *,
    selected_indices: np.ndarray,
    halo_props_raw: dict,
    cache_path,
    sim_for_halos: str,
):
    """
    Save selected halo properties to a companion .npz cache file.

    Parameters
    ----------
    selected_indices : np.ndarray
        Integer indices of selected halos into the full catalog arrays.
    halo_props_raw : dict
        The dict returned by load_halo_properties for the full catalog.
        Must already contain all fields needed (pos/x_L2com, vel, m200b, r200b).
    cache_path : path-like
        Destination .npz file path (from pipeline_paths.halo_props_cache_path).
    sim_for_halos : str
        'flamingo' or 'abacus' — governs which field names to read.
    """
    if sim_for_halos == 'abacus':
        pos  = np.array(halo_props_raw['x_L2com'][selected_indices], dtype=np.float32)
        vel  = np.array(halo_props_raw['v_L2com'][selected_indices], dtype=np.float32)
        m200b = np.array(halo_props_raw['m200b'][selected_indices],   dtype=np.float32)
        r200b = np.array(halo_props_raw['r95_L2com'][selected_indices], dtype=np.float32)
        
        ensure_parents(cache_path)
        np.savez(
            cache_path,
            pos=pos,
            vel=vel,
            m200b=m200b,
            r200b=r200b,
        )
    else:  # flamingo
        pos  = np.array(halo_props_raw['pos'][selected_indices],      dtype=np.float32)
        vel  = np.array(halo_props_raw['hvel_200b'][selected_indices], dtype=np.float32)
        m200b = np.array(halo_props_raw['m200b'][selected_indices],   dtype=np.float32)
        r200b = np.array(halo_props_raw['r200b'][selected_indices],   dtype=np.float32)
        m200c = np.array(halo_props_raw['m200c'][selected_indices],   dtype=np.float32)
        mstell = np.array(halo_props_raw['mstell_50kpc'][selected_indices], dtype=np.float32)
        
        ensure_parents(cache_path)
        np.savez(
            cache_path,
            pos=pos,
            vel=vel,
            m200b=m200b,
            r200b=r200b,
            m200c=m200c,
            mstell=mstell,
        )
    print(f"  Halo props cache saved ({len(selected_indices)} halos) → {cache_path}")


def compute_selected_halo_delta_2d(
    *,
    sim_for_halos,
    gas_type,
    box,
    ngrid,
    selection_config,
    save_path=None,
    halo_indices_save_path=None,
    halo_props_cache_save_path=None,
    len_halos_flamingo=157910,
    nthread=4,
):
    """Compute a 2D overdensity field for a selected halo sample.

    selection_config should contain the selection parameters used by the main
    reconstruction pipeline, including selection_mode, selection_mass_def,
    select_nonzero_masses, upper_mass_cut, upper_radius_cut, sat_frac,
    n_gal_density, halo_mass_range, target_mean_mass, target_mass_tolerance,
    mass_bin_halfwidth, and max_iterations.
    """
    mass_type = selection_config['selection_mass_def']
    cen_sat_mode = selection_config.get('selection_mode', 'cen')
    require_nonzero_mass = selection_config.get('select_nonzero_masses', True)
    upper_mass_cut = selection_config.get('upper_mass_cut', False)
    upper_radius_cut = selection_config.get('upper_radius_cut', False)
    sat_frac = selection_config.get('sat_frac', 0.10)
    n_gal_density = selection_config.get('n_gal_density')
    halo_mass_range = selection_config.get('halo_mass_range')
    target_mean_mass = selection_config.get('target_mean_mass')
    target_mass_tolerance = selection_config.get('target_mass_tolerance', 0.005)
    mass_bin_halfwidth = selection_config.get('mass_bin_halfwidth', 0.02)
    max_iterations = selection_config.get('max_iterations', 1000)

    halo_file_path = selection_config.get('halo_file_path')
    if halo_file_path is None:
        halo_file_path = get_halo_file_path(gas_type, sim_name=sim_for_halos)

    delta_2d_path_val = Path(save_path) if save_path is not None else None
    idx_path = Path(halo_indices_save_path) if halo_indices_save_path is not None else None
    cache_path = Path(halo_props_cache_save_path) if halo_props_cache_save_path is not None else None

    # If we already have saved halo indices, load them and skip selection entirely
    if idx_path is not None and idx_path.exists():
        print(f"\nLoading saved halo indices from {idx_path} (sim='{sim_for_halos}')...")
        selected_indices = np.load(idx_path)
        print(f"  Loaded {len(selected_indices)} indices. Skipping selection.")

        # Fast path: load positions from the props cache (no full catalog open)
        if cache_path is not None and cache_path.exists():
            print(f"  loading pos from halo props cache {cache_path}...")
            cached = np.load(cache_path)
            pos    = cached['pos']        # (N,3) already selected
            masses = cached['m200b']      # (N,)
        else:
            # no cache: open the catalog (this is very slow for Abacus, so we prefer to use the cache)
            if sim_for_halos == 'abacus':
                halo_props = load_halo_properties(halo_file_path, ['x_L2com', 'm200b'], sim_name=sim_for_halos)
                if halo_props is None:
                    raise RuntimeError('Failed to load Abacus halo properties')
                all_pos   = halo_props['x_L2com']
                all_m200b = halo_props['m200b']
            else:
                halo_props = load_halo_properties(halo_file_path, ['pos', 'm200b'], sim_name=sim_for_halos)
                if halo_props is None:
                    raise RuntimeError('Failed to load halo properties')
                all_pos   = halo_props['pos']
                all_m200b = halo_props['m200b']
            pos    = all_pos[selected_indices]
            masses = all_m200b[selected_indices]

        print(f"\n  Total objects selected: {len(selected_indices)} (sim='{sim_for_halos}')")
        print(f"  Log10 mass range: [{np.log10(np.min(masses)):.2f}, {np.log10(np.max(masses)):.2f}]")
        print(f"  Log10 mean mass:  {np.log10(np.mean(masses)):.2f} Msun/h")
        
        # Calculate and print number density
        number_density = len(selected_indices) / (box ** 3)
        print(f"  Number density: {number_density:.3e} (cMpc/h)^-3")

        delta_2d_field = compute_delta_2d(pos, box, ngrid, None, nthread=nthread)
        if delta_2d_path_val is not None:
            ensure_parents(delta_2d_path_val)
            np.save(delta_2d_path_val, delta_2d_field)
            print(f"  Delta field saved to {delta_2d_path_val}")
        return delta_2d_field

    print(f"\nHalos for sim='{sim_for_halos}' not found. Computing...")

    if sim_for_halos == 'abacus': 
        # This will change drastically once we do HOD modeling for Abacus
        requested_props = ['x_L2com', 'm200b', 'r95_L2com']
        if cache_path is not None and 'v_L2com' not in requested_props:
            requested_props.append('v_L2com')
        halo_props = load_halo_properties(halo_file_path, requested_props, sim_name=sim_for_halos)
        if halo_props is None:
            raise RuntimeError('Failed to load Abacus halo properties')

        all_pos = halo_props['x_L2com']
        all_centrals = np.ones(len(all_pos), dtype=int) # this is crude
        all_m200b = halo_props['m200b']
        all_m200c = all_m200b # this is crude
        all_r200b = halo_props['r95_L2com'] # this is crude
        all_haloID = np.arange(len(all_pos), dtype=int) # this is crude
        all_mstell = None
        all_mstell_fof = None
        all_mfof = None
    else:
        requested_props = ['centrals', 'm200b', 'm200c', 'r200b', 'pos', 'haloID']
        if mass_type in ('mstell', 'mstell_fof'):
            requested_props.extend(['mstell_50kpc', 'mstell_fof'])
        # Always load mstell and m200c for Flamingo when cache is requested
        if cache_path is not None:
            if 'mstell_50kpc' not in requested_props:
                requested_props.append('mstell_50kpc')
            if 'hvel_200b' not in requested_props:
                requested_props.append('hvel_200b')
        need_mfof = mass_type in ('mstell_fof', 'mfof')
        if need_mfof:
            requested_props.append('mfof')

        halo_props = load_halo_properties(halo_file_path, requested_props, sim_name=sim_for_halos)
        if halo_props is None:
            raise RuntimeError('Failed to load halo properties')

        all_centrals = halo_props['centrals']
        all_m200b = halo_props['m200b']
        all_m200c = halo_props['m200c']
        all_pos = halo_props['pos']
        all_haloID = halo_props['haloID']
        all_r200b = halo_props['r200b']
        all_mstell = halo_props.get('mstell_50kpc')
        all_mstell_fof = halo_props.get('mstell_fof')
        all_mfof = halo_props.get('mfof')

    mass_options = {
        'm200c': all_m200c,
        'm200b': all_m200b,
        'mstell': all_mstell,
        'mstell_fof': all_mstell_fof,
        'mfof': all_mfof,
    }
    if sim_for_halos == 'abacus':
        mass_options['mfof'] = all_m200b

    if mass_type not in mass_options:
        raise ValueError(
            f"Unknown mass_type: {mass_type}. Must be 'm200c', 'm200b', 'mstell', 'mstell_fof', or 'mfof'."
        )

    selection_masses = mass_options[mass_type]
    if selection_masses is None:
        raise RuntimeError(f"Required mass array for mass_type '{mass_type}' is missing")

    if mass_type in ('mstell_fof', 'mfof') and all_mfof is None and all_m200b is None:
        raise RuntimeError('mfof masses are required for this selection but were not loaded')

    mfof_masses = all_mfof if all_mfof is not None else all_m200b

    if target_mean_mass is not None and halo_mass_range is not None:
        if mass_type in ('mstell', 'mstell_fof'):
            raise ValueError(
                f"Mass bin selection (halo_mass_range) requires HALO mass types. Got mass_type='{mass_type}'."
            )
        if cen_sat_mode == 'mixed':
            raise ValueError(
                "Mass bin selection (halo_mass_range) is incompatible with cen_sat_mode='mixed'."
            )

    print(f"    Using {mass_type} for mass-based selection (sim='{sim_for_halos}')")
    print(f"    Selection mode: {cen_sat_mode}")

    if target_mean_mass is not None:
        print(f"\n    TARGET MASS MODE: Iteratively adjusting to reach log10(<M>) = {target_mean_mass:.2f}")

        if halo_mass_range is not None:
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
            N_pool = len(pool_indices)
            win_size = min(1000, N_pool) if sim_for_halos == 'flamingo' else min(100000, N_pool)
            win_step = 100 if sim_for_halos == 'flamingo' else 1000
            win_size_step = 100 if sim_for_halos == 'flamingo' else 10000

            for _ in range(max_iterations):
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
                window_masses = pool_masses[w_start:w_end]
                window_log = pool_log_masses[w_start:w_end]
                window_mean_log = np.log10(np.mean(window_masses))
                span_lo = window_mean_log - float(np.min(window_log))
                span_hi = float(np.max(window_log)) - window_mean_log

                if span_lo >= mass_bin_halfwidth or span_hi >= mass_bin_halfwidth:
                    print(f"    ✓ Converged: win_start={w_start}, win_size={win_size}, log10(<M>)={window_mean_log:.4f}")
                    break

                win_size = min(N_pool, win_size + win_size_step)
                if win_size >= N_pool:
                    w_start = 0
                    win_size = N_pool
                    print(f"    ⚠ Window reached full pool size: {N_pool}")
                    break
            else:
                print(f"    ⚠ Max iterations reached, using: win_start={w_start}, win_size={win_size}")

            selected_indices = pool_indices[w_start:w_start + win_size]

        elif n_gal_density is not None:
            current_param = n_gal_density
            print(f"    Adjusting n_gal_density (starting from {current_param:.2e})")

            for _ in range(max_iterations):
                selected_indices = select_halos(
                    selection_masses, all_centrals, current_param, None,
                    min_mass=1e9 if (mass_type == 'mstell' or mass_type == 'mstell_fof') else 1e13,
                    select_nonzero_masses=require_nonzero_mass,
                    selection_mode=cen_sat_mode, sat_frac=sat_frac, LBOX=box,
                    upper_mass_cut=upper_mass_cut, m200b=all_m200b,
                    upper_radius_cut=upper_radius_cut, pos=all_pos,
                    haloID=all_haloID, r200=all_r200b,
                    verbose=False,
                )

                if mass_type in ('mstell_fof', 'mfof'):
                    assert mfof_masses is not None
                    masses_iter = mfof_masses[selected_indices]
                else:
                    masses_iter = all_m200b[selected_indices]

                mean_log_mass = np.log10(np.mean(masses_iter))
                mass_error = mean_log_mass - target_mean_mass

                if abs(mass_error) < target_mass_tolerance:
                    print(f"    ✓ Converged: n_gal_density={current_param:.3e}, log10(<M>)={mean_log_mass:.4f}")
                    break

                adjustment_factor = 1.0 + 0.5 * abs(mass_error)
                if mass_error > 0:
                    current_param *= adjustment_factor
                else:
                    current_param /= adjustment_factor
            else:
                print(f"    ⚠ Max iterations reached: n_gal_density={current_param:.3e}, log10(<M>)={mean_log_mass:.4f}")

        else:
            raise ValueError("Either halo_mass_range or n_gal_density must be set for target mass mode")
    else:
        print(f"\n  FIXED SELECTION MODE (sim='{sim_for_halos}')")
        selected_indices = select_halos(
            selection_masses, all_centrals, n_gal_density, halo_mass_range,
            min_mass=1e9 if (mass_type == 'mstell' or mass_type == 'mstell_fof') else 1e13,
            select_nonzero_masses=require_nonzero_mass,
            selection_mode=cen_sat_mode, sat_frac=sat_frac, LBOX=box,
            upper_mass_cut=upper_mass_cut, m200b=all_m200b,
            upper_radius_cut=upper_radius_cut, pos=all_pos,
            haloID=all_haloID, r200=all_r200b,
        )

    # BUTCHERED HERE!!!!!
    if sim_for_halos == 'abacus':
        # random uniformly downsample the window to match the number of halos in Flamingo selection
        np.random.seed(42)
        selected_indices = np.random.choice(selected_indices, size=len_halos_flamingo, replace=False)
        print(f"    Randomly downsampled to {len_halos_flamingo} halos for consistency with Flamingo selection")

    # Save halo indices for downstream stages
    if idx_path is not None:
        ensure_parents(idx_path)
        np.save(idx_path, selected_indices)
        print(f"  Halo indices saved: {len(selected_indices)} → {idx_path}")

    # Save companion halo properties cache whenever a cache path is provided
    if cache_path is not None:
        required_for_cache = {
            'abacus':  ['x_L2com', 'v_L2com', 'm200b', 'r95_L2com'],
            'flamingo': ['pos', 'hvel_200b', 'm200b', 'r200b', 'mstell_50kpc', 'm200c'],
        }[sim_for_halos]
        missing = [k for k in required_for_cache if k not in halo_props]
        if missing:
            print(f"  WARNING: cannot save halo props cache — missing fields: {missing}")
            print("  Ensure required velocity/radius fields are requested before saving cache.")
        else:
            save_halo_props_cache(
                selected_indices=selected_indices,
                halo_props_raw=halo_props,
                cache_path=cache_path,
                sim_for_halos=sim_for_halos,
            )

    pos = all_pos[selected_indices]
    masses = mfof_masses[selected_indices] if mass_type in ('mstell_fof', 'mfof') else all_m200b[selected_indices]

    print(f"\n  Total objects selected: {len(selected_indices)} (sim='{sim_for_halos}')")
    print(f"  Log10 mass range: [{np.log10(np.min(masses)):.2f}, {np.log10(np.max(masses)):.2f}]")
    print(f"  Log10 mean mass: {np.log10(np.mean(masses)):.2f} Msun/h")
    
    # Calculate and print number density
    number_density = len(selected_indices) / (box ** 3)
    print(f"  Number density: {number_density:.3e} (cMpc/h)^-3")

    delta_2d_field = compute_delta_2d(pos, box, ngrid, None, nthread=nthread)

    if delta_2d_path_val is not None:
        ensure_parents(delta_2d_path_val)
        np.save(delta_2d_path_val, delta_2d_field)
        print(f"  Computed and saved to {delta_2d_path_val}")

    return delta_2d_field