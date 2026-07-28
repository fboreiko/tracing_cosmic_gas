"""
Halo/galaxy sample selection: the regime presets and the selection algorithm.

`SELECTION_REGIMES` is the ONE definition of each named regime. Every entry
point (HATF_reconstruction, get_AP_*) must import `selection_defaults` from
here rather than re-deriving its own regime block; the copies had drifted
once already (upper_mass_cut True in one place vs False in the other), which
silently produced two different selection tags for nominally the same
selection.
"""

import numpy as np

# ----------------------------------------------------------------------
# Selection presets
# ----------------------------------------------------------------------
SELECTION_REGIMES = {
    'mhalo_sel': dict(
        selection_mode='cen',
        selection_mass_def='m200b',
        select_nonzero_masses=True,
        upper_mass_cut=False,
        max_mass=1e14,
        upper_radius_cut=False,
        sat_frac=0.10,
    ),
    'mgal_sel': dict(
        selection_mode='mixed',
        selection_mass_def='mstell',
        select_nonzero_masses=True,
        upper_mass_cut=False,
        max_mass=1e14,
        upper_radius_cut=False,
        sat_frac=0.10,
    ),
}


def selection_defaults(regime: str) -> dict:
    """Return the standard bundle of explicit selection parameters for a regime.

    This bundle is partial; the caller must additionally supply
    convergence_mode, n_gal_density, halo_mass_range, target_mean_mass
    (and identity fields) before the dict is schema-complete.
    """
    if regime not in SELECTION_REGIMES:
        raise ValueError(
            f"Unknown regime {regime!r}. Use one of {sorted(SELECTION_REGIMES)}."
        )
    return dict(SELECTION_REGIMES[regime])


# ----------------------------------------------------------------------
# Selection algorithm
# ----------------------------------------------------------------------
def select_halos(selection_masses, is_central_flags,
                 n_gal_density=None, halo_mass_range=None,
                 rank=0, min_mass=1e13, select_nonzero_masses=False,
                 selection_mode=None, sat_frac=0.10, LBOX=None,
                 max_mass=10**14, upper_mass_cut=False, m200b=None,
                 upper_radius_cut=False, pos=None, haloID=None, r200=None, periodic=True,
                 verbose=True):
    """
    Select halos/galaxies based on mass criteria.

    Args:
        selection_masses: Array used for rank-ordering (e.g. stellar mass).
        is_central_flags: Array (bool/int) where 1=Central, 0=Satellite.
        n_gal_density: Number density for selection (cMpc/h)^-3.
        halo_mass_range: List [win_start, win_size] defining a window into filtered_indices
                         (the pool after all prior filter stages, in the order produced by
                         those stages — including mixed-mode interleaving).
                         win_start is the start index, win_size is the number of objects.
        rank: MPI rank (for logging).
        min_mass: Minimum mass threshold (used in Step 1 filtering only).
        select_nonzero_masses: If True, only select objects with selection_masses > 0.
        selection_mode: Selection mode:
            - 'sat': satellites only
            - 'cen': centrals only
            - 'mixed': interleaved centrals/satellites
            - 'cens_sat': select centrals first, then one satellite per central
            - None: all objects passing other filters
        sat_frac: Desired satellite fraction for 'mixed' mode.
        LBOX: Box size (used for number density; also for periodic distances if periodic=True).
        max_mass: Maximum mass threshold for upper mass cut.
        upper_mass_cut: If True, apply upper mass cut at the beginning (on satellites only).
        m200b: Required if upper_mass_cut=True.
        upper_radius_cut: If True, keep centrals; for satellites, keep only those within r200 of host.
        pos: (N,3) positions array (required if upper_radius_cut=True).
        haloID: (N,) integer array mapping each object to its host/central index
                (required if upper_radius_cut=True for satellites).
                For centrals it can be self-index or anything; centrals are handled explicitly.
        r200: (N,) array of r200 values *for centrals/hosts* (required if upper_radius_cut=True).
              Satellite i uses r200[haloID[i]].
        periodic: If True, use minimum-image convention with box size LBOX.
        verbose: If True, print progress messages. Set to False to suppress output during iterations.

    Returns:
        inds_sub: Original indices of selected objects (sorted by selection_masses descending).
    """
    
    if LBOX is None:
        raise ValueError("LBOX must be provided explicitly from get_sim_params(sim).")

    # VALIDATION: Mass bin selection requires consistent mass types
    if halo_mass_range is not None:
        # Only allow 'cen', 'sat', or 'cens_sat' modes for mass bin selection
        # 'mixed' mode creates interleaved order that doesn't preserve mass sorting
        # 'cens_sat' works like 'cen' during selection, then converts to satellites
        if selection_mode == 'mixed':
            raise ValueError(
                "halo_mass_range (mass bin selection) is incompatible with selection_mode='mixed'. "
                "Use 'cen', 'sat', or 'cens_sat' mode. Mixed mode creates interleaved order that breaks mass-based binning."
            )
        
        if rank == 0 and verbose:
            print(" MASS BIN MODE: Ensure selection_masses is a HALO mass (m200b/m200c/mfof),")
            print(" NOT stellar mass.")
  
    if select_nonzero_masses:
        nonzero_mask = (selection_masses > 0)
        if rank == 0 and verbose:
            n_before = len(selection_masses)
            n_after = np.sum(nonzero_mask)
            if n_before > 0:
                print(f"  Non-zero selection mass filter: {n_before} -> {n_after} ({n_after/n_before*100:.1f}% retained)")
            else:
                print(f"  Non-zero selection mass filter: {n_before} -> {n_after}")
    else:
        nonzero_mask = np.ones(len(selection_masses), dtype=bool)

    mass_mask = nonzero_mask

    if upper_mass_cut:
        if m200b is None:
            raise ValueError("upper_mass_cut=True requires m200b to be provided")
        # Apply cut only on satellites, let all centrals pass
        is_cen = (is_central_flags == 1)
        is_sat = ~is_cen
        upper_mass_mask = is_cen | (m200b <= max_mass)
        mass_mask = mass_mask & upper_mass_mask
        if rank == 0 and verbose:
            n_sat_before = np.sum(is_sat & nonzero_mask)
            n_sat_passing = np.sum(is_sat & mass_mask)
            print(f"  Upper mass cut (satellites only, m200b <= {max_mass:.2e}):")
            if n_sat_before > 0:
                print(f"    Satellites: {n_sat_before} -> {n_sat_passing} ({n_sat_passing/n_sat_before*100:.1f}% retained)")
            else:
                print(f"    Satellites: {n_sat_before} -> {n_sat_passing}")
            print(f"    Total passing: {np.sum(mass_mask)}")

    if upper_radius_cut:
        if pos is None or haloID is None or r200 is None:
            raise ValueError("upper_radius_cut=True requires pos, haloID, and r200 to be provided")

        pos = np.asarray(pos)
        if pos.ndim != 2 or pos.shape[1] < 3:
            raise ValueError("pos must be an (N,3) array (or at least 3 columns)")

        N = len(selection_masses)
        if len(pos) != N or len(haloID) != N or len(r200) != N:
            raise ValueError("pos, haloID, r200, and selection_masses must have the same length")

        haloID = np.asarray(haloID)
        r200 = np.asarray(r200)

        is_cen = (is_central_flags == 1)
        is_sat = ~is_cen

        # haloID maps each object to its host/central index
        # For centrals: haloID[i] == i (self-index)
        # For satellites: haloID[i] points to the host central
        haloID = np.asarray(haloID, dtype=int)

        if np.any(haloID < 0) or np.any(haloID >= N):
            bad = np.where((haloID < 0) | (haloID >= N))[0][:10]
            raise ValueError(f"haloID contains out-of-range indices. Example bad indices: {bad}")

        host_pos = pos[haloID, :3]
        d = pos[:, :3] - host_pos
        if periodic:
            d = d - LBOX * np.rint(d / LBOX)

        dist = np.sqrt(np.sum(d * d, axis=1))
        host_r200 = r200[haloID]

        radius_mask = is_cen | (dist <= host_r200)

        if rank == 0 and verbose:
            # Count satellites before and after radius cut (but with prior filters still applied)
            n_sat_before_radius = np.sum(is_sat & mass_mask)
            n_sat_after_radius = np.sum(is_sat & mass_mask & radius_mask)
            print("  Upper radius cut (satellites only, dist <= r200 of host):")
            if n_sat_before_radius > 0:
                print(f"    Satellites: {n_sat_before_radius} -> {n_sat_after_radius} ({n_sat_after_radius/n_sat_before_radius*100:.1f}% retained)")
            else:
                print(f"    Satellites: {n_sat_before_radius} -> {n_sat_after_radius}")

        mass_mask = mass_mask & radius_mask

        if rank == 0 and verbose:
            print(f"    Total passing: {np.sum(mass_mask)}")

    if selection_mode is not None:
        if selection_mode == 'cen':  # Select only centrals
            central_mask = (is_central_flags == 1) & mass_mask
            filtered_indices = np.where(central_mask)[0]
            if rank == 0 and verbose:
                print(f"  Selection mode 'cen': {len(filtered_indices)} centrals selected")

        elif selection_mode == 'sat':  # Select only satellites
            sat_mask = (is_central_flags != 1) & mass_mask
            filtered_indices = np.where(sat_mask)[0]
            if rank == 0 and verbose:
                print(f"  Selection mode 'sat': {len(filtered_indices)} satellites selected")

        elif selection_mode == 'mixed':  # Interleaved selection with mass filtering
            cen_mask = (is_central_flags == 1) & mass_mask
            sat_mask = (is_central_flags != 1) & mass_mask

            idx_cen_sorted = np.where(cen_mask)[0][np.argsort(selection_masses[cen_mask])[::-1]]
            idx_sat_sorted = np.where(sat_mask)[0][np.argsort(selection_masses[sat_mask])[::-1]]

            n_centrals_per_sat = int((1.0 - sat_frac) / sat_frac)

            interleaved_indices = []
            cen_idx = 0
            sat_idx = 0

            while cen_idx < len(idx_cen_sorted) and sat_idx < len(idx_sat_sorted):
                for _ in range(n_centrals_per_sat):
                    if cen_idx < len(idx_cen_sorted):
                        interleaved_indices.append(idx_cen_sorted[cen_idx])
                        cen_idx += 1
                    else:
                        break
                if sat_idx < len(idx_sat_sorted):
                    interleaved_indices.append(idx_sat_sorted[sat_idx])
                    sat_idx += 1

            filtered_indices = np.array(interleaved_indices, dtype=int)

            if rank == 0 and verbose:
                final_cen = np.sum(is_central_flags[filtered_indices] == 1)
                final_sat = len(filtered_indices) - final_cen
                print(f"  Selection mode 'mixed': {len(filtered_indices)} objects ({final_cen} cen, {final_sat} sat)")
                print(f"    Interleaving ratio: {n_centrals_per_sat} centrals per 1 satellite")

        elif selection_mode == 'cens_sat':  # Select centrals first (will convert to satellites later)
            if haloID is None:
                raise ValueError("selection_mode='cens_sat' requires haloID to map satellites to centrals")
            
            # Select centrals (same as 'cen' mode) - final selection happens later
            central_mask = (is_central_flags == 1) & mass_mask
            filtered_indices = np.where(central_mask)[0]
            if rank == 0 and verbose:
                print(f"  Selection mode 'cens_sat': {len(filtered_indices)} centrals selected (will be converted to satellites after final selection)")

        else:
            raise ValueError(f"Unknown selection_mode: {selection_mode}. Use 'cen', 'sat', 'mixed', or 'cens_sat'")
    else:
        filtered_indices = np.where(mass_mask)[0]
        if rank == 0 and verbose:
            n_cen = np.sum(is_central_flags[filtered_indices] == 1)
            n_sat = len(filtered_indices) - n_cen
            print(f"  No selection mode: {len(filtered_indices)} objects selected ({n_cen} cen, {n_sat} sat)")

    if selection_mode not in ['mixed']:
        # Sort filtered_indices by descending selection_masses
        # Skip for 'mixed' since it has its own ordering logic
        sorted_order = np.argsort(selection_masses[filtered_indices])[::-1]
        filtered_indices = filtered_indices[sorted_order]
        if rank == 0 and verbose:
            print(f"  Re-sorted by descending mass (highest mass at index 0)")

    if halo_mass_range is not None:
        if not isinstance(halo_mass_range, (list, tuple)) or len(halo_mass_range) != 2:
            raise ValueError("halo_mass_range must be a list/tuple [win_start, win_size]")
        win_start, win_size = halo_mass_range
        inds_sub = filtered_indices[win_start : win_start + win_size]

        if rank == 0 and verbose:
            print(f"  Window selection: start={win_start}, size={win_size}")

    elif n_gal_density is not None:
        N_gal = min(int(n_gal_density * LBOX ** 3), len(filtered_indices))
        inds_sub = filtered_indices[:N_gal]

        if rank == 0 and verbose:
            print(f"  Number density selection: {N_gal} objects "
                  f"(target density {n_gal_density:.2e})")

    else:
        inds_sub = filtered_indices

        if rank == 0 and verbose:
            print(f"  No final selection: {len(inds_sub)} objects returned")

    # Post-processing: For 'cens_sat' mode, convert selected centrals to satellites
    if selection_mode == 'cens_sat':
        if haloID is None:
            raise ValueError("selection_mode='cens_sat' requires haloID")
        
        # inds_sub now contains the final selected centrals
        selected_centrals = inds_sub
        
        # Build a mapping from central index to list of satellite indices
        # Use the original mass_mask to get all passing satellites
        sat_mask = (is_central_flags != 1) & mass_mask
        sat_indices = np.where(sat_mask)[0]
        
        haloID_arr = np.asarray(haloID, dtype=int)
        central_to_sats = {}
        for sat_idx in sat_indices:
            host_idx = haloID_arr[sat_idx]
            if host_idx not in central_to_sats:
                central_to_sats[host_idx] = []
            central_to_sats[host_idx].append(sat_idx)
        
        # For each selected central, pick one satellite (most massive)
        selected_satellites = []
        centrals_with_sats = 0
        for cen_idx in selected_centrals:
            if cen_idx in central_to_sats and len(central_to_sats[cen_idx]) > 0:
                sat_list = central_to_sats[cen_idx]
                # Pick the most massive satellite
                sat_masses = selection_masses[sat_list]
                best_sat_idx = sat_list[np.argmax(sat_masses)]
                selected_satellites.append(best_sat_idx)
                centrals_with_sats += 1
        
        inds_sub = np.array(selected_satellites, dtype=int)
        
        if rank == 0 and verbose:
            print(f"  'cens_sat' post-processing: {len(selected_centrals)} centrals -> {len(inds_sub)} satellites")
            print(f"    {centrals_with_sats} centrals had satellites")

    return inds_sub