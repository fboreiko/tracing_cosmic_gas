"""
Single source of truth for all file paths in the kSZ pipeline.

Path layout (v2, see NAMING_REFACTOR.md)
-----------------------------------------
    data/<sim>/
      delta_fields/     delta_2d_<label>_<feedback>[_<seltag>].npy
      halo_indices/     halo_indices_<feedback>_<seltag>.npy
                        halo_props_<feedback>_<seltag>.npz
      tau_prefactors/   tau_prefactor_<feedback>.npy
      tau_maps/<tau_source>/<tracer>/
                        tau_<feedback>[_<seltag>].npy
      aperture_photometry/<projection>/<feedback>/<tau_source>/<tracer>/<seltag>/
                        tau_apertures.npz

`tau_source` ('truth' | 'recon') is the ONLY thing that distinguishes a
reconstructed product from a true one. `feedback` never carries a
'_reconstructed' suffix.
"""

import os
from pathlib import Path

import numpy as np

from utils.pipeline_config import PipelineConfig

# ----------------------------------------------------------------------
# Roots. Override per machine with environment variables so the same
# checkout runs on the laptop and on the HPC without editing source.
# ----------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent.parent

DATA_ROOT = Path(os.environ.get("TCG_DATA_ROOT", _HERE / "data"))
PLOT_ROOT = Path(os.environ.get("TCG_PLOT_ROOT", _HERE / "plots"))
FLAMINGO_ROOT = Path(os.environ.get("TCG_FLAMINGO_ROOT", _HERE / "sims" / "FLAMINGO"))
ABACUS_ROOT = Path(os.environ.get("TCG_ABACUS_ROOT", _HERE / "sims" / "abacus"))

SNAPSHOT = int(os.environ.get("TCG_SNAPSHOT", 63))  # 63, 77 -- FLAMINGO snapshots


def _sim_data_root(sim_name: str) -> Path:
    return DATA_ROOT / sim_name


# ----------------------------------------------------------------------
# Simulation data (read-only)
# ----------------------------------------------------------------------
def get_halo_file_path(feedback: str, sim_name: str = "flamingo") -> Path:
    if sim_name == "flamingo":
        return FLAMINGO_ROOT / (
            f"FLAMINGO_ext_L1000N1800_HYDRO_{feedback.upper()}_snap_{SNAPSHOT}.hdf5"
        )
    if sim_name == "abacus":
        return ABACUS_ROOT / "AbacusSummit_base_c000_ph000/halos/z0.800/halo_info"
    raise ValueError(f"Unsupported sim_name: {sim_name}")


def get_particle_file_path(feedback: str, sim_name: str = "flamingo") -> Path:
    if sim_name == "flamingo":
        return FLAMINGO_ROOT / (
            f"FLAMINGO_particles_L1000N1800_HYDRO_{feedback.upper()}"
            f"_snap_{SNAPSHOT}_diluted_100.hdf5"
        )
    if sim_name == "abacus":
        return ABACUS_ROOT / "AbacusSummit_base_c000_ph000/halos/z0.800"
    raise ValueError(f"Unsupported sim_name: {sim_name}")


# ----------------------------------------------------------------------
# Selection presets - the ONE definition of each regime.
#
# These values are authoritative. HATF_reconstruction.py must import them
# rather than re-deriving its own regime block; the two copies had drifted
# (upper_mass_cut True here vs False there), which silently produced two
# different selection tags for nominally the same selection.
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
# Selection tag
# ----------------------------------------------------------------------
def selection_tag(config) -> str:
    """Bare selection token (no leading underscore)."""
    return PipelineConfig.coerce(config).selection_tag()


def _suffix(config, include: bool) -> str:
    """'_<seltag>' when include is True, else ''."""
    return ("_" + selection_tag(config)) if include else ""


def resolve_halo_indices(config) -> np.ndarray:
    """Load the saved halo index array for the given selection config."""
    config = PipelineConfig.coerce(config, require_selection=True)
    path = halo_indices_path(config)
    if not path.exists():
        raise FileNotFoundError(
            f"resolve_halo_indices: halo indices file not found:\n  {path}\n"
            f"Run HATF first to generate the halo sample for this selection config."
        )
    return np.load(path)


# ----------------------------------------------------------------------
# Stage 1 - HATF / tau-map paths
# ----------------------------------------------------------------------
def delta_2d_path(config, label: str) -> Path:
    """Path for a saved 2-D projected delta field.

    label: 'dm' | 'gas' | 'halos'.
    'dm' and 'gas' are pure physics fields and depend only on feedback.
    'halos' additionally carries the full selection tag.
    """
    config = PipelineConfig.coerce(config, require_selection=(label == "halos"))
    return (
        _sim_data_root(config.sim_name)
        / "delta_fields"
        / f"delta_2d_{label}_{config.feedback}{_suffix(config, label == 'halos')}.npy"
    )


def halo_indices_path(config) -> Path:
    """Saved halo index array (.npy) produced by HATF, consumed by get_AP/TkSZ."""
    config = PipelineConfig.coerce(config, require_selection=True)
    return (
        _sim_data_root(config.sim_name)
        / "halo_indices"
        / f"halo_indices_{config.feedback}_{selection_tag(config)}.npy"
    )


def halo_props_cache_path(config) -> Path:
    """Companion halo-properties cache (.npz) written alongside halo_indices.

    Arrays stored inside: pos (N,3), vel (N,3), m200b (N,), r200b (N,).
    For Abacus, 'vel' holds v_L2com and 'r200b' holds r95_L2com.
    """
    config = PipelineConfig.coerce(config, require_selection=True)
    return (
        _sim_data_root(config.sim_name)
        / "halo_indices"
        / f"halo_props_{config.feedback}_{selection_tag(config)}.npz"
    )


def tau_prefactor_path(config) -> Path:
    """Tau-map prefactor. Depends only on gas physics, not on selection."""
    config = PipelineConfig.coerce(config, require_selection=False)
    return (
        _sim_data_root(config.sim_name)
        / "tau_prefactors"
        / f"tau_prefactor_{config.feedback}.npy"
    )


def tau_map_path(config) -> Path:
    """Path for a tau map.

        tau_maps/<tau_source>/<tracer>/tau_<feedback>[_<seltag>].npy

    'truth' maps carry no selection tag - they do not depend on which halos
    were selected. 'recon' maps do, because the transfer function is fitted
    against a specific halo sample.
    """
    # tau_source decides whether selection fields are required, so read it first.
    if isinstance(config, PipelineConfig):
        tau_source = config.tau_source
    else:
        tau_source = config.get("tau_source")
    if tau_source is None:
        raise ValueError("tau_map_path requires 'tau_source' ('truth' or 'recon')")

    config = PipelineConfig.coerce(config, require_selection=(tau_source != "truth"))
    return (
        _sim_data_root(config.sim_name)
        / "tau_maps"
        / config.tau_source
        / config.tracer
        / f"tau_{config.feedback}{_suffix(config, tau_source != 'truth')}.npy"
    )


# ----------------------------------------------------------------------
# Aperture photometry output
# ----------------------------------------------------------------------
def ap_output_path(config, projection: str = "simple") -> Path:
    """Aperture-photometry output, for both the simple and pixell projections.

        aperture_photometry/<projection>/<feedback>/<tau_source>/<tracer>/<seltag>/
            tau_apertures.npz
    """
    if projection not in ("simple", "pixell", "pixell_cea"):
        raise ValueError(f"Unknown projection {projection!r}")
    config = PipelineConfig.coerce(config, require_selection=True)
    return (
        _sim_data_root(config.sim_name)
        / "aperture_photometry"
        / projection
        / config.feedback
        / config.tau_source
        / config.tracer
        / selection_tag(config)
        / "tau_apertures.npz"
    )


# ----------------------------------------------------------------------
# Plot output
# ----------------------------------------------------------------------
def plot_path(category: str, *subdirs: str, stem: str) -> Path:
    """Resolve a plot output path under PLOT_ROOT, ending in <stem>.pdf."""
    parts = [p for p in subdirs if p is not None]
    return PLOT_ROOT / category / Path(*parts) / f"{stem}.pdf"


# ----------------------------------------------------------------------
# Utility
# ----------------------------------------------------------------------
def ensure_parents(*paths):
    """Create parent directories for each path if they do not exist."""
    for p in paths:
        Path(p).parent.mkdir(parents=True, exist_ok=True)
