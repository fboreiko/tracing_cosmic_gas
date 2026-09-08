import numpy as np
import deepdish as dd
from pathlib import Path
import glob
import asdf
import gc
from typing import Any, cast
from .sim_params import get_sim_params


_FLAMINGO_HALOPROPS_PATHS = {
    'mstell': '/galaxies/mstell',
    'mstell_50kpc': '/galaxies/mstell_50kpc',
    'mstell_fof': '/galaxies/mstell_fof',
    'centrals': '/galaxies/centrals',
    'm200b': '/galaxies/m200b',
    'm200c': '/galaxies/m200c',
    'TotalMass_fof': '/galaxies/TotalMass_fof',
    'pos': '/galaxies/pos',
    'haloID': '/galaxies/haloID',
    'r200b': '/galaxies/r200b',
    'hvel_200b': '/galaxies/hvel_200b',
}

_FLAMINGO_PARTICLE_GROUPS = {
    'gas': 'GasParticles',
    'dm': 'DMParticles',
}


def read_abacus(rv_fn, Lbox):
    from abacusnbody.data.bitpacked import unpack_rvint

    # read the halo (L0+L1) matter particles
    rv_data = asdf.open(rv_fn)['data']['rvint'][:]

    # unpack positions
    pos_mat, _ = unpack_rvint(rv_data, Lbox, float_dtype=np.float32, velout=None)
    pos_mat = np.array(pos_mat[:], dtype=np.float32)
    
    del rv_data
    gc.collect()

    return pos_mat



def load_halo_properties(file_path, properties, compute_mfof_if_needed=True, sim_name='flamingo', halo_indices=None):
    # Load only requested halo properties from a corresponding halo catalog.
    # If halo_indices is provided, only load properties for those indices.
    # If halo_indices is None, load properties for all halos.
    requested = set(properties)

    if sim_name == 'flamingo':
        supported = set(_FLAMINGO_HALOPROPS_PATHS.keys()) | {'mfof'}
        unknown = sorted(requested - supported)
        if unknown:
            raise ValueError(f"Unknown halo properties requested: {unknown}")

        out = {}

        direct_fields = [p for p in requested if p in _FLAMINGO_HALOPROPS_PATHS]
        for prop in direct_fields:
            prop_data = dd.io.load(file_path, _FLAMINGO_HALOPROPS_PATHS[prop])
            # Subset to requested indices if provided
            if halo_indices is not None:
                prop_data = prop_data[halo_indices]
            out[prop] = prop_data

        if 'mfof' in requested:
            if not compute_mfof_if_needed:
                raise ValueError("'mfof' was requested but compute_mfof_if_needed=False")

            total_mass_fof = out.get('TotalMass_fof')
            if total_mass_fof is None:
                total_mass_fof = dd.io.load(file_path, _FLAMINGO_HALOPROPS_PATHS['TotalMass_fof'])
                if halo_indices is not None:
                    total_mass_fof = total_mass_fof[halo_indices]

            halo_ids = out.get('haloID')
            if halo_ids is None:
                halo_ids = dd.io.load(file_path, _FLAMINGO_HALOPROPS_PATHS['haloID'])
                if halo_indices is not None:
                    halo_ids = halo_ids[halo_indices]

            _, inverse_indices = np.unique(halo_ids, return_inverse=True)
            mfof_sums = np.bincount(inverse_indices, weights=total_mass_fof)
            out['mfof'] = mfof_sums[inverse_indices]

            if 'TotalMass_fof' not in requested and 'TotalMass_fof' in out:
                del out['TotalMass_fof']

        return out
    
    if sim_name == 'abacus':
        from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog

        supported = {'x_L2com', 'v_L2com', 'm200b', 'r95_L2com'}
        unknown = sorted(requested - supported)
        if unknown:
            raise ValueError(f"Unknown/unsupported Abacus halo properties requested: {unknown}")

        cat = CompaSOHaloCatalog(
            str(file_path),
            cleaned=True,
            fields=cast(Any, ["x_L2com", "v_L2com", "N", "r95_L2com"]),
        )

        halos = cat.halos

        out = {}

        Lbox = get_sim_params(sim_name)['box_size_cMpc_h']

        if 'x_L2com' in requested:
            x_data = np.asarray(halos['x_L2com']) + Lbox / 2.0  # Shift from [-L/2, L/2] to [0, L]
            out['x_L2com'] = x_data[halo_indices] if halo_indices is not None else x_data
        if 'v_L2com' in requested:
            v_data = np.asarray(halos['v_L2com'])
            out['v_L2com'] = v_data[halo_indices] if halo_indices is not None else v_data
        if 'r95_L2com' in requested:
            r_data = np.asarray(halos['r95_L2com'])
            out['r95_L2com'] = r_data[halo_indices] if halo_indices is not None else r_data
        if 'm200b' in requested:
            n_data = np.asarray(halos['N']) * 0.21 * (1e10)  # Msun/h
            out['m200b'] = n_data[halo_indices] if halo_indices is not None else n_data

        return out


def load_particle_properties(file_path, particle_type, requested=('mass', 'pos'), sim_name='flamingo', Lbox=None, chunkid=None):
    #Load only requested particle properties for a given species.
    if sim_name == 'flamingo':
        if particle_type not in _FLAMINGO_PARTICLE_GROUPS:
            raise ValueError(
                f"Unknown particle_type: {particle_type}. Must be one of {sorted(_FLAMINGO_PARTICLE_GROUPS.keys())}."
            )

        group = _FLAMINGO_PARTICLE_GROUPS[particle_type]
        file_data = dd.io.load(file_path)

        out = {}
        if 'mass' in requested:
            out['mass'] = np.array(file_data[group]['mass']) * 100
        if 'pos' in requested:
            out['pos'] = np.array(file_data[group]['pos'])
        return out

    elif sim_name == 'abacus':
        if particle_type != 'dm':
            raise ValueError(
                "Abacus particle loader currently supports particle_type='dm' only. "
                "Use sim='flamingo' for gas target particles."
            )
        
        Lbox = get_sim_params(sim_name)['box_size_cMpc_h']

        base_path = Path(file_path)
        
        halo_rv_fn = base_path / "halo_rv_A" / f"halo_rv_A_{chunkid:03d}.asdf"
        field_rv_fn = base_path / "field_rv_A" / f"field_rv_A_{chunkid:03d}.asdf"

        halo_parts_pos = read_abacus(halo_rv_fn, Lbox)
        field_parts_pos = read_abacus(field_rv_fn, Lbox)
        
        dm_parts_pos = np.concatenate([halo_parts_pos, field_parts_pos], axis=0)
        dm_parts_pos += Lbox / 2.0  # Shift from [-L/2, L/2] to [0, L]

        del halo_parts_pos, field_parts_pos
        gc.collect()
        
        out = {}
        if 'pos' in requested:
            out['pos'] = dm_parts_pos
        return out

