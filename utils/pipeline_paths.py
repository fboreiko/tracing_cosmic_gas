"""
Single source of truth for all file paths in the kSZ pipeline.
"""

import numpy as np
from pathlib import Path
import re

# Root paths — edit once here if directories change
DATA_ROOT = Path("/home/fb635/fedirfiles/tracing_cosmic_gas/data")
PLOT_ROOT = Path("/home/fb635/fedirfiles/tracing_cosmic_gas/plots")
HPC_ROOT  = Path("/home/fb635/rds/hpc-work/tracing_cosmic_gas")

SNAPSHOT = 63 # 63, 77

# Simulation data (read-only)
def halo_galaxy_hdf5(gas_type: str) -> Path:
    """HDF5 catalogue for halos/galaxies."""
    gt = gas_type.replace("_reconstructed", "").upper()
    return HPC_ROOT / f"FLAMINGO_ext_L1000N1800_HYDRO_{gt}_snap_{SNAPSHOT}.hdf5"


def particles_hdf5(gas_type: str) -> Path:
    """Diluted particle HDF5 file."""
    gt = gas_type.replace("_reconstructed", "").upper()
    return Path(
        f"/rds-d6/user/fb635/hpc-work/tracing_cosmic_gas/"
        f"FLAMINGO_particles_L1000N1800_HYDRO_{gt}_snap_{SNAPSHOT}_diluted_100.hdf5"
    )


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
    Build a comprehensive filename token encoding ALL parameters that affect
    which objects are selected into the sample.
    """
    parts = []

    mode = config.get('selection_mode', 'mixed')
    parts.append(f"mode_{mode}" if mode is not None else "mode_all")

    mdef = config.get('selection_mass_def', 'mstell')
    parts.append(f"mass_{mdef}")

    nonzero = config.get('select_nonzero_masses', True)
    parts.append("nzmass" if nonzero else "allm")

    umc = config.get('upper_mass_cut', True)
    if umc:
        mmax = config.get('max_mass', 1e14)
        exp = int(round(np.log10(mmax)))
        parts.append(f"umc1e{exp:02d}")
    else:
        parts.append("noumc")

    urc = config.get('upper_radius_cut', False)
    parts.append("urc" if urc else "nourc")

    if mode == 'mixed':
        sf = config.get('sat_frac', 0.10)
        parts.append(f"sf{int(round(sf * 100)):02d}")

    # Target mean mass (emit if not None; HATF stage only)
    # Check this BEFORE ngal/massbin to conditionally skip them
    tm = config.get('target_mean_mass')
    ngal = config.get('n_gal_density')
    mbin = config.get('halo_mass_range')
    
    if tm is not None:
        # Target mass mode: include which parameter is being adjusted
        # Either massbin or ngal should be non-None to indicate the adjustment variable
        if mbin is not None:
            parts.append("adjmb")  # adjusting massbin
        elif ngal is not None:
            parts.append("adjngal")  # adjusting ngal
        else:
            raise ValueError("When target_mean_mass is set, either halo_mass_range or n_gal_density must be non-None")
        parts.append(f"tm{tm:.1f}".replace('.', 'p'))
    else:
        # Fixed selection mode: include the actual ngal and massbin values
        if ngal is not None:
            raw = f"{ngal:.2e}".replace("e-0", "em0").replace("e+0", "ep0") \
                               .replace("e-",  "em" ).replace("e+",  "ep" )
            parts.append(f"ngal{raw}")
        else:
            parts.append("ngal_none")

        if mbin is not None:
            if isinstance(mbin, (list, tuple)) and len(mbin) == 2:
                parts.append(f"massbin_start{mbin[0]}_size{mbin[1]}")
            else:
                raise ValueError("halo_mass_range must be a list or tuple of [start, size]")
        else:
            parts.append("massbin_none")

    return "_" + "_".join(parts)


# Converged parameter resolution
def resolve_converged_param(config: dict, mode: str = None) -> dict:
    """
    Scan tau_maps/ for the matching reconstructed tau map file and parse the
    converged n_gal_density (mode='ngal') or halo_mass_range (mode='massbin')
    directly from the filename using regex anchored on the known key strings.

    For non-reconstructed gas types (fullFT, dm field_type) this looks up the
    corresponding _reconstructed run's file so all methods share the same selection.

    Args:
        config : configuration dict with selection parameters.
                 If config['convergence_mode'] is set it is used as the default
                 for `mode`, so callers that stored the mode in the profile dict
                 don't need to pass it explicitly.
        mode   : 'ngal' or 'massbin'.  Falls back to config['convergence_mode']
                 when not supplied.  Raises if neither is available.

    Returns a copy of config with the converged value filled in.
    When target_mean_mass is None, returns config unchanged.
    """
    import re

    if config.get('target_mean_mass') is None:
        return config  # fixed-selection path — caller already has explicit values

    # Resolve mode from argument or from the profile dict
    if mode is None:
        mode = config.get('convergence_mode')
    if mode not in ('massbin', 'ngal'):
        raise ValueError(f"mode must be 'massbin' or 'ngal', got: {mode!r}")

    # Always search the _reconstructed tau files — that is where HATF writes
    # the converged parameters regardless of which downstream method uses them.
    gas_type_raw = config['gas_type']
    if not gas_type_raw.endswith('_reconstructed'):
        gas_type = gas_type_raw + '_reconstructed'
    else:
        gas_type = gas_type_raw

    tau_method_dir = DATA_ROOT / "tau_maps" / "2D_FT_upgrade_tau_reconstruction"

    # Build a prefix anchor: _mass_<def>_<nzmass|allm>_<umc|noumc>_
    # These three tokens are always adjacent in the filename, so we concatenate
    # them without wildcards to uniquely identify the selection (handles e.g.
    # mstell_fof vs mstell, and umc1e14 vs noumc).
    mdef      = config.get('selection_mass_def', 'mstell')
    zm_token  = "nzmass" if config.get('select_nonzero_masses', True) else "allm"
    umc       = config.get('upper_mass_cut', True)
    umc_token = f"umc1e{int(round(np.log10(config.get('max_mass', 1e14)))):02d}" if umc else "noumc"
    prefix_anchor = f"_mass_{mdef}_{zm_token}_{umc_token}_"

    if mode == 'ngal':
        sf = config.get('sat_frac')
        sf_token = f"_sf{int(round(sf * 100)):02d}_" if (sf is not None and config.get('selection_mode') == 'mixed') else "_"
        # e.g. …_mass_mstell_fof_nzmass_umc1e14_…_sf01_ngal1.31em03_massbin_none.npy
        pattern = f"tau_map_{gas_type}_*{prefix_anchor}*{sf_token}ngal[!_]*_massbin_none.npy"
    else:
        # e.g. …_mass_mfof_allm_noumc_…_ngal_none_massbin_start62090_size6500.npy
        pattern = f"tau_map_{gas_type}_*{prefix_anchor}*_ngal_none_massbin_start*.npy"

    matches = sorted(tau_method_dir.glob(pattern))

    if len(matches) == 0:
        raise FileNotFoundError(
            f"resolve_converged_param: no tau map file found for mode='{mode}':\n"
            f"  Pattern : {tau_method_dir / pattern}\n"
            f"Run HATF first to generate the converged tau map."
        )
    if len(matches) > 1:
        raise ValueError(
            f"resolve_converged_param: ambiguous — {len(matches)} files match for mode='{mode}':\n"
            + "\n".join(f"  {m}" for m in matches)
        )

    stem = matches[0].stem  # filename without .npy

    result = dict(config)

    if mode == 'ngal':
        # Extract the encoded float between '_ngal' and '_massbin'
        m = re.search(r'_ngal([^_]+(?:_[^_]+)?)_massbin_none$', stem)
        if not m:
            raise ValueError(
                f"resolve_converged_param: could not find '_ngal..._massbin_none' in:\n  {stem}"
            )
        raw = m.group(1).replace('p', '.').replace('em', 'e-').replace('ep', 'e+')
        result['n_gal_density']   = float(raw)
        result['halo_mass_range'] = None

    else:  # massbin
        m = re.search(r'_massbin_start(\d+)_size(\d+)$', stem)
        if not m:
            raise ValueError(
                f"resolve_converged_param: could not find '_massbin_start..._size...' in:\n  {stem}"
            )
        result['halo_mass_range'] = [int(m.group(1)), int(m.group(2))]
        result['n_gal_density']   = None

    # Clear target_mean_mass so _selection_tag uses the real converged values
    # (not the adjngal/adjmb/tm tokens) when building any downstream file paths.
    result['target_mean_mass'] = None

    return result


# Stage 1 — HATF / tau-map paths
def delta_2d_path(config: dict, label: str) -> Path:
    """
    Path for a saved 2-D projected δ field (label: 'dm', 'gas', or 'halos').
    
    For 'dm' and 'gas': only gas_type matter (these are pure physics fields)
    For 'halos': includes comprehensive selection tag with ALL parameters for maximum specificity
                 to prevent different selection configs from colliding
    """
    gas_type = config["gas_type"].replace("_reconstructed", "")
    sel_tag  = _selection_tag(config) if label == "halos" else ""
    return DATA_ROOT / "delta_fields" / f"delta_2d_{label}_{gas_type}{sel_tag}.npy"


def tau_prefactor_path(config: dict) -> Path:
    """
    Path for the τ-map prefactor.
    Depends only on gas physics and grid resolution, not on selection.
    """
    gas_type = config["gas_type"].replace("_reconstructed", "")
    return DATA_ROOT / "tau_map_prefactors" / f"tau_prefactor_{gas_type}.npy"


def tau_map_path(config: dict) -> Path:
    """
    Path for the reconstructed (or direct) τ map.

    fullFT: legacy filenames (no selection tag) — these maps do not depend
            on which halos were selected.
    2D_FT methods: comprehensive selection tag included so different selection 
                   configs produce different files (prevents collisions).
    """
    gas_type   = config["gas_type"]
    tau_method = config.get("tau_method", "fullFT_tau_reconstruction")
    field_type = config.get("field_type", "gas")
    method_dir = DATA_ROOT / "tau_maps" / tau_method
    prefix     = "tau_map_dm_" if field_type == "dm" else "tau_map_"

    if tau_method == "fullFT_tau_reconstruction":
        fname = f"{prefix}{gas_type}.npy"

    elif tau_method == "2D_FT_massdep_tau_reconstruction":
        sel = _selection_tag(config)
        fname = f"{prefix}{gas_type}_A_{config['A']:.5f}{sel}.npy"

    else:
        sel   = _selection_tag(config)
        fname = f"{prefix}{gas_type}{sel}.npy"

    return method_dir / fname

# aperture-photometry output
def ap_output_path(config: dict) -> Path:
    """
    Path for the .npz written by get_AP_simple_jax_batched.

    Directory tree:
        simple_CAP_code/
          <gas_type>/
            <tau_method>/
              <field_type>/          ('gas' or 'dm')
                <selection_tag>/     (from _selection_tag, leading '_' stripped)
                  tau_apertures.npz

    For 2D_FT_massdep the tau_method directory gains an _A_<value> suffix.
    """
    gas_type   = config["gas_type"]
    tau_method = config.get("tau_method", "fullFT_tau_reconstruction")
    field_type = config.get("field_type", "gas")
    sel_tag    = _selection_tag(config).lstrip("_")   # strip leading underscore for dir name

    # Optional A-parameter suffix (massdep method only)
    if tau_method == "2D_FT_massdep_tau_reconstruction":
        method_dir = f"{tau_method}_A_{config['A']:.5f}"
    else:
        method_dir = tau_method

    return (
        DATA_ROOT / "simple_CAP_code"
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
        Path object ending in <stem>.png
    """
    parts = [p for p in subdirs if p is not None]
    return PLOT_ROOT / category / Path(*parts) / f"{stem}.png"


# Utility
def ensure_parents(*paths):
    """Create parent directories for each path if they do not exist."""
    for p in paths:
        Path(p).parent.mkdir(parents=True, exist_ok=True)