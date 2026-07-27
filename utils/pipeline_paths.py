"""
Single source of truth for all file paths in the kSZ pipeline.
"""

import numpy as np
from pathlib import Path

from utils.pipeline_config import PipelineConfig

# Root paths — edit once here if directories change
DATA_ROOT = Path("/Users/fedorboreiko/Documents/Cambridge/project_github/data")
PLOT_ROOT = Path("/Users/fedorboreiko/Documents/Cambridge/project_github/plots")
FLAMINGO_ROOT = Path("/home/fb635/rds/hpc-work/tracing_cosmic_gas/FLAMINGO")
ABACUS_ROOT = Path("/home/fb635/rds/hpc-work/tracing_cosmic_gas/abacus")

SNAPSHOT = 63 # 63, 77 -- different FLAMINGO snapshots

def _sim_data_root(sim_name: str) -> Path:
    return DATA_ROOT / (sim_name)

# Simulation data (read-only)
def get_halo_file_path(
    gas_type: str,
    sim_name: str = "flamingo",
    ) -> Path:
    
    if sim_name == "flamingo":
        gt = gas_type.replace("_reconstructed", "").upper()
        return FLAMINGO_ROOT / f"FLAMINGO_ext_L1000N1800_HYDRO_{gt}_snap_{SNAPSHOT}.hdf5"

    elif sim_name == "abacus":
        return ABACUS_ROOT / f"AbacusSummit_base_c000_ph000/halos/z0.800/halo_info"
    else:
        raise ValueError(f"Unsupported sim_name: {sim_name}")


def get_particle_file_path(
    gas_type: str,
    sim_name: str = "flamingo",
) -> Path:

    if sim_name == "flamingo":
        gt = gas_type.replace("_reconstructed", "").upper()
        return FLAMINGO_ROOT / f"FLAMINGO_particles_L1000N1800_HYDRO_{gt}_snap_{SNAPSHOT}_diluted_100.hdf5"

    elif sim_name == "abacus":
        return ABACUS_ROOT / f"AbacusSummit_base_c000_ph000/halos/z0.800"
    else:
        raise ValueError(f"Unsupported sim_name: {sim_name}")


# Selection parameter initialiser
def selection_defaults(regime: str) -> dict:
    """
    Return the standard bundle of explicit selection parameters for a regime label.

    Parameters returned:
        selection_mode          : 'cen' | 'sat' | 'mixed' | None
        selection_mass_def      : 'mstell' | 'mstell_fof' | 'm200c' | 'm200b'
        select_nonzero_masses   : bool
        upper_mass_cut          : bool
        max_mass                : float  (ceiling applied when upper_mass_cut=True)
        upper_radius_cut        : bool
        sat_frac                : float  (only active when selection_mode='mixed')

    This bundle is partial; the caller must additionally supply
    convergence_mode, n_gal_density, halo_mass_range, target_mean_mass
    (and identity fields) before the dict is schema-complete.
    """
    if regime == 'mhalo_sel':
        return dict(
            selection_mode        = 'cen',
            selection_mass_def    = 'm200c',
            select_nonzero_masses = False,
            upper_mass_cut        = False,
            max_mass              = 1e14,
            upper_radius_cut      = False,
            sat_frac              = 0.10,
        )
    elif regime == 'mgal_sel':
        return dict(
            selection_mode        = 'mixed',
            selection_mass_def    = 'mstell',
            select_nonzero_masses = True,
            upper_mass_cut        = True,
            max_mass              = 1e14,
            upper_radius_cut      = False,
            sat_frac              = 0.10,
        )
    else:
        raise ValueError(f"Unknown regime '{regime}'. Use 'mhalo_sel' or 'mgal_sel'.")


# Selection tag — the single place that encodes selection parameters in filenames
def _selection_tag(config: dict) -> str:
    """
    Build a filename token encoding the selection INTENT — the parameters a
    user sets in a PROFILE_CONFIG — not the converged numeric outcome.

    The implementation now lives in PipelineConfig.selection_tag(); this
    wrapper is kept because external modules import this name.
    """
    return PipelineConfig.coerce(config).selection_tag()


def resolve_halo_indices(config: dict) -> np.ndarray:
    """
    Load the saved halo index array for the given selection config.

    This is the replacement for resolve_converged_param. Instead of parsing
    tau-map filenames to recover converged numeric parameters and then
    re-running select_halos, callers simply load the pre-computed indices
    produced by HATF.

    Args:
        config : selection config dict (as used throughout the pipeline).
                 Must include 'sim_name' and all _selection_tag keys.
                 'target_mean_mass' should be set to the intended target so
                 the tag matches what HATF wrote.

    Returns:
        np.ndarray of integer halo indices into the full catalog.

    Raises:
        FileNotFoundError if the indices file does not exist (run HATF first).
    """
    config = PipelineConfig.coerce(config, require_selection=True)
    path = halo_indices_path(config)
    if not path.exists():
        raise FileNotFoundError(
            f"resolve_halo_indices: halo indices file not found:\n"
            f"  {path}\n"
            f"Run HATF first to generate the halo sample for this selection config."
        )
    return np.load(path)


def _peek_tau_method(config):
    """Read tau_method before validation (tau_map_path needs it to decide
    whether selection fields are required). Accepts a dict or a PipelineConfig."""
    if isinstance(config, PipelineConfig):
        value = config.tau_method
    else:
        value = config.get("tau_method")
    return value if value is not None else "fullFT_tau_reconstruction"


# Stage 1 — HATF / tau-map paths
def delta_2d_path(config: dict, label: str) -> Path:
    """
    Path for a saved 2-D projected δ field (label: 'dm', 'gas', or 'halos').
    
    For 'dm' and 'gas': only gas_type matter (these are pure physics fields)
    For 'halos': includes comprehensive selection tag with ALL parameters for maximum specificity
                 to prevent different selection configs from colliding
    """
    config   = PipelineConfig.coerce(config, require_selection=(label == "halos"))
    sim_name = config.sim_name
    gas_type = config.gas_type.replace("_reconstructed", "")
    sel_tag  = _selection_tag(config) if label == "halos" else ""
    return _sim_data_root(sim_name) / "delta_fields" / f"delta_2d_{label}_{gas_type}{sel_tag}.npy"


def halo_indices_path(config: dict) -> Path:
    """
    Path for the saved halo index array (.npy) produced by HATF and consumed
    by get_AP and TkSZ.

    The file sits next to the halo delta field:
        data/<sim>/halo_indices/halo_indices_<gas_type><sel_tag>.npy

    'gas_type' here is the raw physics label (no _reconstructed suffix), matching
    the convention used by delta_2d_path for the 'halos' label.
    """
    config   = PipelineConfig.coerce(config, require_selection=True)
    sim_name = config.sim_name
    gas_type = config.gas_type.replace("_reconstructed", "")
    sel_tag  = _selection_tag(config)
    return _sim_data_root(sim_name) / "halo_indices" / f"halo_indices_{gas_type}{sel_tag}.npy"

def halo_props_cache_path(config: dict) -> Path:
    """
    Path for the companion halo properties cache (.npz) saved alongside
    halo_indices when convergence_mode is 'massbin'.

    File layout (same directory as halo_indices):
        data/<sim>/halo_indices/halo_props_<gas_type><sel_tag>.npz

    Arrays stored inside the npz:
        pos    : (N, 3) float32  — 3-D positions (cMpc/h)
        vel    : (N, 3) float32  — 3-D velocities (km/s)
        m200b  : (N,)   float32  — halo mass M200b (Msun/h)
        r200b  : (N,)   float32  — halo radius R200b (cMpc/h)

    Notes
    -----
    * For Abacus, 'vel' contains v_L2com and 'r200b' contains r95_L2com
      (the closest available proxy).
    * This file is only written/read when is_massbin_config(config) is True.
    """
    config   = PipelineConfig.coerce(config, require_selection=True)
    sim_name = config.sim_name
    gas_type = config.gas_type.replace("_reconstructed", "")
    sel_tag  = _selection_tag(config)
    return _sim_data_root(sim_name) / "halo_indices" / f"halo_props_{gas_type}{sel_tag}.npz"


def tau_prefactor_path(config: dict) -> Path:
    """
    Path for the τ-map prefactor.
    Depends only on gas physics and grid resolution, not on selection.
    """
    config   = PipelineConfig.coerce(config, require_selection=False)
    sim_name = config.sim_name
    gas_type = config.gas_type.replace("_reconstructed", "")
    return _sim_data_root(sim_name) / "tau_map_prefactors" / f"tau_prefactor_{gas_type}.npy"


def tau_map_path(config: dict) -> Path:
    """
    Path for the reconstructed (or direct) τ map.

    fullFT: legacy filenames (no selection tag) — these maps do not depend
            on which halos were selected.
    2D_FT methods: comprehensive selection tag included so different selection 
                   configs produce different files (prevents collisions).
    """
    tau_method = _peek_tau_method(config)
    config     = PipelineConfig.coerce(
        config, require_selection=(tau_method != "fullFT_tau_reconstruction")
    )
    sim_name   = config.sim_name
    gas_type   = config.gas_type
    field_type = config.field_type
    method_dir = _sim_data_root(sim_name) / "tau_maps" / tau_method
    prefix     = "tau_map_dm_" if field_type == "dm" else "tau_map_"

    if tau_method == "fullFT_tau_reconstruction":
        fname = f"{prefix}{gas_type}.npy"

    elif tau_method == "2D_FT_massdep_tau_reconstruction":
        sel = _selection_tag(config)
        fname = f"{prefix}{gas_type}_A_{config.A:.5f}{sel}.npy"

    else:
        sel   = _selection_tag(config)
        fname = f"{prefix}{gas_type}{sel}.npy"

    return method_dir / fname

# aperture-photometry output
def ap_output_path(config: dict, method: str = "simple") -> Path:
    """
    Unified path for aperture photometry outputs from both simple and pixell methods.

    Directory tree:
        <method>_CAP_code/        ('simple' or 'pixell')
          <gas_type>/
            <tau_method>/
              <field_type>/        ('gas' or 'dm')
                <selection_tag>/   (from _selection_tag, leading '_' stripped)
                  tau_apertures.npz

    For 2D_FT_massdep the tau_method directory gains an _A_<value> suffix.
    
    Args:
        config: Configuration dictionary
        method: 'simple' (get_AP_simple_jax_batched) or 'pixell' (get_AP_pixell_jax_batched)
    """
    config     = PipelineConfig.coerce(config, require_selection=True)
    sim_name   = config.sim_name
    gas_type   = config.gas_type
    tau_method = config.tau_method
    field_type = config.field_type
    sel_tag    = _selection_tag(config).lstrip("_")

    # Base directory depends on method
    base_dir = f"{method}_CAP_code"

    # Optional A-parameter suffix (massdep method only)
    if tau_method == "2D_FT_massdep_tau_reconstruction":
        method_dir = f"{tau_method}_A_{config.A:.5f}"
    else:
        method_dir = tau_method

    return (
        _sim_data_root(sim_name) / base_dir
        / gas_type
        / method_dir
        / field_type
        / sel_tag
        / "tau_apertures.npz"
    )


# plot output
def plot_path(category: str, *subdirs: str, stem: str) -> Path:
    """
    Resolve a plot output path under PLOT_ROOT.

    Args:
        category:  top-level category string, e.g. 'hatf/power_spectra',
                   'hatf/transfer_fn', 'ksz_profiles/cap_ksz',
                   or 'ksz_profiles/cap_ksz_over_dm'.
        *subdirs:  additional subdirectory components (e.g. gas_type, tau_method).
                   Pass each component separately; None values are skipped.
        stem:      filename without extension.

    Returns:
        Path object ending in <stem>.pdf
    """
    parts = [p for p in subdirs if p is not None]
    return PLOT_ROOT / category / Path(*parts) / f"{stem}.pdf"


# Utility
def ensure_parents(*paths):
    """Create parent directories for each path if they do not exist."""
    for p in paths:
        Path(p).parent.mkdir(parents=True, exist_ok=True)