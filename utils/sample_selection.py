import numpy as np


# Simulation constants
LBOX = 681  # cMpc/h


def select_halos(halo_mstar, halo_pos, n_gal_density=None, halo_mass_range=None, rank=0):
   
    if halo_mass_range is not None:
        # Use mass bins with exactly 5000 halos per bin, numbered from highest mass
        # Bin 0 = highest mass, Bin N-1 = lowest mass (with remainder)
        min_mass = 1e13  # Msun/h
        
        # Count halos above minimum mass
        mask_above_min = halo_mstar >= min_mass
        n_halos_above_min = np.sum(mask_above_min)
        
        # Calculate number of bins to have exactly 5000 halos per bin
        halos_per_bin = 16700 #9000, 16700, 40000
        n_bins = (n_halos_above_min + halos_per_bin - 1) // halos_per_bin  # Ceiling division
        remainder = n_halos_above_min % halos_per_bin
        if remainder == 0:
            remainder = halos_per_bin
        
        # Get indices of halos above minimum mass, sorted by mass (descending)
        indices_above_min = np.where(mask_above_min)[0]
        masses_above_min = halo_mstar[indices_above_min]
        sorted_indices = indices_above_min[np.argsort(masses_above_min)[::-1]]  # Descending order
        
        # Select halos in the specified bin (bin 0 = highest mass)
        if halo_mass_range < 0 or halo_mass_range >= n_bins:
            raise ValueError(f"halo_mass_range must be between 0 and {n_bins-1}, got {halo_mass_range}")
        
        # Calculate start and end indices for this bin
        start_idx = halo_mass_range * halos_per_bin
        if halo_mass_range == n_bins - 1:
            # Last bin gets the remainder
            end_idx = n_halos_above_min
        else:
            end_idx = start_idx + halos_per_bin
        
        # Get the indices for this bin
        inds_sub = sorted_indices[start_idx:end_idx]
        
    else:
        # Use number density selection (original behavior)
        if n_gal_density is None:
            raise ValueError("Either n_gal_density or halo_mass_range must be provided")
        
        N_gal = int(n_gal_density * LBOX ** 3)
        inds_sub = np.argsort(halo_mstar)[::-1][:N_gal]
    
    # Select positions without z coordinate
    selected_pos = halo_pos[inds_sub, :2]  # cMpc/h
    return selected_pos, inds_sub