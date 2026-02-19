import numpy as np

def select_halos(selection_masses, halo_pos, n_gal_density=None, 
                 halo_mass_range=None, rank=0, min_mass=1e13, 
                 select_positive_mass=False, is_central_flag=None,
                 sat_frac=0.10, LBOX = 681):
    """
    Select halos/galaxies based on mass criteria.
    
    Args:
        selection_masses: Array of stellar masses (M*) used for rank-ordering.
        halo_pos: Array of positions (N, 3).
        n_gal_density: Number density for selection (cMpc/h)^-3.
        halo_mass_range: Mass bin index for binned selection.
        rank: MPI rank (for logging).
        min_mass: Minimum mass threshold for mass bin selection.
        select_positive_mass: If True, triggers special selection modes.
        is_central_flag: Array (bool/int) where 1=Central, 0=Satellite.
        sat_frac: The desired fraction of satellites in the final sample.
    
    Returns:
        selected_pos: Selected positions (N, 2).
        inds_sub: Original indices of selected objects.
    """
   
    if halo_mass_range is not None:
        # Binned selection logic
        mask_above_min = selection_masses >= min_mass
        n_halos_above_min = np.sum(mask_above_min)
        
        halos_per_bin = 40000 
        n_bins = (n_halos_above_min + halos_per_bin - 1) // halos_per_bin
        
        indices_above_min = np.where(mask_above_min)[0]
        sorted_indices = indices_above_min[np.argsort(selection_masses[indices_above_min])[::-1]]
        
        if halo_mass_range < 0 or halo_mass_range >= n_bins:
            raise ValueError(f"halo_mass_range must be between 0 and {n_bins-1}")
        
        start_idx = halo_mass_range * halos_per_bin
        end_idx = n_halos_above_min if halo_mass_range == n_bins - 1 else start_idx + halos_per_bin
        inds_sub = sorted_indices[start_idx:end_idx]
        
    elif n_gal_density is not None:
        
        N_gal = int(n_gal_density * LBOX ** 3)
        # Select the N most massive objects based on selection_masses
        inds_sub = np.argsort(selection_masses)[::-1][:N_gal]
        
        if rank == 0:
            print(f"Number Density Mode: Selected top {N_gal} objects")
        
    else: # all halos mode or targeted selection mode
        if select_positive_mass: # this is for mgal_sel business
            if is_central_flag is not None: # this is for more specific selection
                cen_mask = (is_central_flag == 1) & (selection_masses > 0)
                sat_mask = (is_central_flag != 1) & (selection_masses > 0)

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

                interleaved_indices = np.array(interleaved_indices)
                
                # Apply number density cutoff (adjust this value as needed)
                n_gal_density_targeted = 5.5e-3 # (cMpc/h)^-3
                N_gal = int(n_gal_density_targeted * LBOX ** 3)
                inds_sub = interleaved_indices[:N_gal]

                final_cen = np.sum(is_central_flag[inds_sub] == 1)
                final_sat = len(inds_sub) - final_cen

                if rank == 0:
                    print(f"--- Targeted Selection (Interleaved) ---")
                    print(f"Target number density: {n_gal_density_targeted:.2e} (cMpc/h)^-3")
                    print(f"Interleaving ratio: 1 sat per {n_centrals_per_sat} centrals")
                    print(f"Result: {len(inds_sub)} galaxies ({final_cen} Cen, {final_sat} Sat)")
                    print(f"Actual Sat Fraction: {final_sat/len(inds_sub)*100:.1f}%")

            else:
                # Simple mass > 0 fallback
                mass_mask = selection_masses > 0
                inds_sub = np.where(mass_mask)[0][np.argsort(selection_masses[mass_mask])[::-1]]
                if rank == 0:
                    print(f"All halos mode (mass > 0): Selected {len(inds_sub)} objects")
        else: # this is for mhalo_sel business - just select all halos regardless of mass
            inds_sub = np.arange(len(selection_masses))
            if rank == 0:
                print(f"All halos mode: Selected all {len(inds_sub)} objects")
    
    # Select positions (first two columns)
    selected_pos = halo_pos[inds_sub, :2]
    return selected_pos, inds_sub