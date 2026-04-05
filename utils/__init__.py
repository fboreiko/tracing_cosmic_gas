# === Power Spectrum Utilities ===
from .power_spectrum_utils import (
    compute_delta,
    #compute_power_spectrum,
    compute_2d_fft,
    compute_k_grid_2d,
    bin_power_spectrum_2d,
    interp_extrapolate_loglog,
    compute_2d_power_spectrum,
)

from .tools import (cutoutGeometry, extractStamp, 
                    build_aperture_templates_jax, 
                    compute_all_apertures_jax, 
                    extractStamp_jax, 
                    bilinear_interpolate_batch)

from .rotfuncs import (moveaxis, rotmatrix, ang2rect, 
                       rect2ang, euler_mat, euler_rot,
                       recenter)

from .sample_selection import select_halos
from .catalog_loaders import load_halo_properties, load_particle_properties
from .delta_fields import (
    compute_delta_2d,
    compute_delta_3d_projected,
    compute_delta_field_and_mass,
    compute_selected_halo_delta_2d,
)
from .tau_prefactor import compute_prefactor

from .pipeline_paths import (get_halo_file_path, get_particle_file_path,
                             selection_defaults,
                             resolve_halo_indices, delta_2d_path, tau_prefactor_path,
                             tau_map_path, ap_output_path, plot_path, ensure_parents,
                             _selection_tag)

__all__ = ['compute_delta', #'compute_power_spectrum',
           'compute_2d_fft', 'compute_k_grid_2d', 'bin_power_spectrum_2d',
           'interp_extrapolate_loglog', 'compute_2d_power_spectrum',
           'cutoutGeometry', 'extractStamp', 'build_aperture_templates_jax', 
           'compute_all_apertures_jax', 'extractStamp_jax',
           'moveaxis', 'rotmatrix', 'ang2rect', 'rect2ang',
           'euler_mat', 'euler_rot', 'recenter', 'bilinear_interpolate_batch',
           'select_halos', 'load_halo_properties', 'load_particle_properties',
           'compute_delta_2d', 'compute_delta_3d_projected', 'compute_delta_field_and_mass', 'compute_selected_halo_delta_2d',
           'compute_prefactor',
           'get_halo_file_path', 'get_particle_file_path',
           'selection_defaults',
           'resolve_halo_indices', 'delta_2d_path', 'tau_prefactor_path',
           'tau_map_path', 'ap_output_path', 'plot_path', 'ensure_parents', '_selection_tag']