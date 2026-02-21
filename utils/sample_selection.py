import numpy as np

def select_halos(selection_masses, is_central_flags, 
                 n_gal_density=None, halo_mass_range=None, 
                 rank=0, min_mass=1e13, select_nonzero_masses=False,
                 selection_mode=None, sat_frac=0.10, LBOX = 681,
                 max_mass=10**13.8, upper_cut_mode=False):
    """
    Select halos/galaxies based on mass criteria.
    
    Args:
        selection_masses: Array of stellar masses (M*) used for rank-ordering.
        is_central_flags: Array (bool/int) where 1=Central, 0=Satellite.
        n_gal_density: Number density for selection (cMpc/h)^-3.
        halo_mass_range: Mass bin index for binned selection.
        rank: MPI rank (for logging).
        min_mass: Minimum mass threshold for mass bin selection.
        select_nonzero_masses: If True, only select objects with selection_masses > 0.
        selection_mode: Selection mode - 'sat' (satellites only), 
                        'cen' (centrals only), or 'mixed' (interleaved).
        sat_frac: The desired fraction of satellites in the final sample.
        max_mass: Maximum mass threshold for upper mass cut.
        upper_cut_mode: If True, apply upper mass cut at the beginning.
    
    Returns:
        selected_pos: Selected positions (N, 2).
        inds_sub: Original indices of selected objects.
    """

    if upper_cut_mode:
        upper_mass_mask = selection_masses <= max_mass
        if rank == 0:
            print(f"  Applied upper mass cut (mass <= {max_mass:.2e}): {np.sum(upper_mass_mask)} objects pass")
    else:
        upper_mass_mask = np.ones(len(selection_masses), dtype=bool)

    if select_nonzero_masses:
        mass_mask = (selection_masses > 0) & upper_mass_mask
    else:
        mass_mask = upper_mass_mask
    
    if selection_mode is not None:
        if selection_mode == 'cen': # Select only centrals
            central_mask = (is_central_flags == 1) & mass_mask
            filtered_indices = np.where(central_mask)[0]
            if rank == 0:
                print(f"  Applied central-only filter: {len(filtered_indices)} objects (mass > 0: {select_nonzero_masses})")
                
        elif selection_mode == 'sat': # Select only satellites
            sat_mask = (is_central_flags != 1) & mass_mask
            filtered_indices = np.where(sat_mask)[0]
            if rank == 0:
                print(f"  Applied satellite-only filter: {len(filtered_indices)} objects (mass > 0: {select_nonzero_masses})")
                
        elif selection_mode == 'mixed': # Interleaved selection with mass filtering
            cen_mask = (is_central_flags == 1) & mass_mask
            sat_mask = (is_central_flags != 1) & mass_mask

            idx_cen_sorted = np.where(cen_mask)[0][np.argsort(selection_masses[cen_mask])[::-1]]
            idx_sat_sorted = np.where(sat_mask)[0][np.argsort(selection_masses[sat_mask])[::-1]]

            n_centrals_per_sat = int((1.0 - sat_frac) / sat_frac)

            interleaved_indices = []
            cen_idx = 0
            sat_idx = 0

            while cen_idx < len(idx_cen_sorted) or sat_idx < len(idx_sat_sorted):
                for _ in range(n_centrals_per_sat):
                    if cen_idx < len(idx_cen_sorted):
                        interleaved_indices.append(idx_cen_sorted[cen_idx])
                        cen_idx += 1
                    else:
                        break
                if sat_idx < len(idx_sat_sorted):
                    interleaved_indices.append(idx_sat_sorted[sat_idx])
                    sat_idx += 1

            filtered_indices = np.array(interleaved_indices)
            
            if rank == 0:
                final_cen = np.sum(is_central_flags[filtered_indices] == 1)
                final_sat = len(filtered_indices) - final_cen
                print(f"  Applied mixed (interleaved) filter: {len(filtered_indices)} objects ({final_cen} Cen, {final_sat} Sat)")
                print(f"    Interleaving ratio: 1 sat per {n_centrals_per_sat} centrals")
        else:
            raise ValueError(f"Unknown selection_mode: {selection_mode}. Use 'cen', 'sat', or 'mixed'")
    else: # No central/satellite filtering
        filtered_indices = np.where(mass_mask)[0]
        if rank == 0 and select_nonzero_masses:
            print(f"  Applied mass > 0 filter: {len(filtered_indices)} objects")
    
    # Apply selection mode on filtered data
    filtered_masses = selection_masses[filtered_indices]
    
    if halo_mass_range is not None:
        mask_above_min = filtered_masses >= min_mass
        n_halos_above_min = np.sum(mask_above_min)
        
        halos_per_bin = 30000 #20000 
        n_bins = (n_halos_above_min + halos_per_bin - 1) // halos_per_bin
        
        sorted_filtered_indices = np.argsort(filtered_masses)[::-1]
        indices_above_min = sorted_filtered_indices[filtered_masses[sorted_filtered_indices] >= min_mass]
        
        if len(indices_above_min) == 0:
            raise ValueError(f"No halos above minimum mass {min_mass}")
        
        if halo_mass_range < 0 or halo_mass_range >= n_bins:
            raise ValueError(f"halo_mass_range must be between 0 and {n_bins-1}")
        
        start_idx = halo_mass_range * halos_per_bin
        end_idx = n_halos_above_min if halo_mass_range == n_bins - 1 else start_idx + halos_per_bin
        selected_filtered_indices = indices_above_min[start_idx:end_idx]
        inds_sub = filtered_indices[selected_filtered_indices]
        
        if rank == 0:
            print(f"  Mass bin {halo_mass_range}: Selected {len(inds_sub)} objects from bin")
        
    elif n_gal_density is not None:
        N_gal = int(n_gal_density * LBOX ** 3)
        N_gal = min(N_gal, len(filtered_indices))  # Can't select more than available
        
        # Sort filtered indices by mass and select top N
        sorted_filtered_indices = np.argsort(filtered_masses)[::-1][:N_gal]
        inds_sub = filtered_indices[sorted_filtered_indices]
        
        if rank == 0:
            print(f"   Number Density Mode: Selected top {len(inds_sub)} objects (requested {N_gal})")
        
    else: # All halos mode
        sorted_filtered_indices = np.argsort(filtered_masses)[::-1]
        inds_sub = filtered_indices[sorted_filtered_indices]
        
        if rank == 0:
            print(f"  All halos mode: Selected all {len(inds_sub)} filtered objects")
    
    return inds_sub