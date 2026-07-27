"""
Single strict schema for the kSZ / HATF pipeline config dict.

`PipelineConfig` is the ONLY thing allowed to interpret a config dict. It
rejects unknown keys and requires every key that affects a filename.
"""

import math
from dataclasses import dataclass, field, fields
from pathlib import Path

# Allowed value sets (empty set == not constrained)
_ALLOWED_SIM_NAME = ('flamingo', 'abacus')
_ALLOWED_GAS_TYPE = (
    'fiducial',
    'strongest_AGN',
    'fiducial_reconstructed',
    'strongest_AGN_reconstructed',
)
_ALLOWED_TAU_METHOD = (
    'fullFT_tau_reconstruction',
    '2D_FT_upgrade_tau_reconstruction',
    '2D_FT_massdep_tau_reconstruction',
)
_ALLOWED_FIELD_TYPE = ('gas', 'dm')
_ALLOWED_SELECTION_MODE = ('cen', 'sat', 'mixed', 'cens_sat', None)
_ALLOWED_SELECTION_MASS_DEF = ('mstell', 'mstell_fof', 'm200b', 'm200c', 'mfof')
_ALLOWED_CONVERGENCE_MODE = ('ngal', 'massbin')
_ALLOWED_PROJECTION_TYPE = ('simple', 'pixell', 'pixell_cea', None)

# Selection fields that must be present when require_selection is True.
# The value None is acceptable only for those listed in _NULLABLE_SELECTION.
_REQUIRED_SELECTION = (
    'selection_mode',
    'selection_mass_def',
    'select_nonzero_masses',
    'upper_mass_cut',
    'upper_radius_cut',
    'convergence_mode',
    'n_gal_density',
    'halo_mass_range',
    'target_mean_mass',
)
_NULLABLE_SELECTION = (
    'selection_mode',
    'n_gal_density',
    'halo_mass_range',
    'target_mean_mass',
)

_MASSDEP_METHOD = '2D_FT_massdep_tau_reconstruction'


@dataclass(frozen=True)
class PipelineConfig:
    """Frozen, validated pipeline configuration. Build with `from_dict`."""

    # Identity
    sim_name: str = None
    gas_type: str = None
    tau_method: str = None
    field_type: str = 'gas'
    z_real: float = None

    # Selection (required whenever a selection tag may be built)
    selection_mode: str = None
    selection_mass_def: str = None
    select_nonzero_masses: bool = None
    upper_mass_cut: bool = None
    upper_radius_cut: bool = None
    convergence_mode: str = None
    n_gal_density: float = None
    halo_mass_range: tuple = None
    target_mean_mass: float = None

    # Conditionally required
    max_mass: float = None
    sat_frac: float = None
    A: float = None

    # Selection-loop knobs (defaults equal the pre-WP3 delta_fields fallbacks)
    target_mass_tolerance: float = 0.005
    mass_bin_halfwidth: float = 0.02
    max_iterations: int = 1000
    halo_file_path: str = None

    # Presentation metadata — never used in tags or paths
    name: str = None
    projection_type: str = None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, d, *, require_selection=True):
        """Validate a plain config dict and return a frozen PipelineConfig.

        Raises ValueError on a legacy 'sim' key, on unknown keys, on missing
        required keys, and on any cross-field rule violation.
        """
        d = dict(d)  # never mutate the caller's dict

        # 1. Legacy-key trap
        if 'sim' in d:
            raise ValueError(
                "Config key 'sim' is not accepted; use 'sim_name'. "
                "(This key was a historical typo that silently selected the "
                "wrong simulation.)"
            )

        # 2. Unknown keys
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(d) - known)
        if unknown:
            raise ValueError(
                "Unknown config keys (typos?): " + ", ".join(repr(k) for k in unknown)
            )

        # 3. Identity requiredness.
        #
        # NOTE (WP3 deviation, flagged in the WP report): the WP requires
        # sim_name, gas_type AND tau_method for every construction. Applied to
        # require_selection=False that contradicts the WP's own hard constraint,
        # because live callers legitimately pass identity-only dicts holding just
        # sim_name and gas_type (HATF_reconstruction.py line 152, and the WP1
        # golden probes simprobe_flamingo_strongest_AGN / simprobe_abacus_fiducial).
        # Section 2.4 itself describes those call sites as reading "only
        # sim_name/gas_type". tau_method is therefore required only when
        # require_selection is True.
        for key in ('sim_name', 'gas_type'):
            if d.get(key) is None:
                raise ValueError(f"Missing required config key: '{key}'")
        if require_selection and d.get('tau_method') is None:
            raise ValueError("Missing required config key: 'tau_method'")

        # 4. Selection requiredness
        if require_selection:
            for key in _REQUIRED_SELECTION:
                if key not in d:
                    raise ValueError(f"Missing required config key: '{key}'")
                if d[key] is None and key not in _NULLABLE_SELECTION:
                    raise ValueError(
                        f"Config key '{key}' must not be None"
                    )

        # 5. Conditional fields
        if d.get('upper_mass_cut') is True and d.get('max_mass') is None:
            raise ValueError(
                "Config key 'max_mass' is required when upper_mass_cut is True"
            )
        if d.get('selection_mode') in ('mixed', 'sat') and d.get('sat_frac') is None:
            raise ValueError(
                "Config key 'sat_frac' is required when selection_mode is "
                f"{d.get('selection_mode')!r}"
            )
        if d.get('tau_method') == _MASSDEP_METHOD:
            if d.get('A') is None:
                raise ValueError(
                    f"Config key 'A' is required when tau_method is {_MASSDEP_METHOD!r}"
                )
        elif d.get('A') is not None:
            raise ValueError(
                f"Config key 'A' is only valid when tau_method is {_MASSDEP_METHOD!r}"
            )

        # 6. Allowed values
        cls._check_allowed(d, 'sim_name', _ALLOWED_SIM_NAME)
        cls._check_allowed(d, 'gas_type', _ALLOWED_GAS_TYPE)
        if d.get('tau_method') is not None:
            cls._check_allowed(d, 'tau_method', _ALLOWED_TAU_METHOD)
        if 'field_type' in d:
            cls._check_allowed(d, 'field_type', _ALLOWED_FIELD_TYPE)
        if 'projection_type' in d:
            cls._check_allowed(d, 'projection_type', _ALLOWED_PROJECTION_TYPE)
        if require_selection:
            cls._check_allowed(d, 'selection_mode', _ALLOWED_SELECTION_MODE)
            cls._check_allowed(d, 'selection_mass_def', _ALLOWED_SELECTION_MASS_DEF)
            cls._check_allowed(d, 'convergence_mode', _ALLOWED_CONVERGENCE_MODE)

        # 7. Cross-field validation
        cls._validate_cross_fields(d, require_selection=require_selection)

        instance = cls(**d)
        object.__setattr__(instance, '_has_selection', bool(require_selection))
        return instance

    @classmethod
    def coerce(cls, obj, *, require_selection=True):
        """Return `obj` unchanged if already a PipelineConfig, else from_dict."""
        if isinstance(obj, cls):
            return obj
        return cls.from_dict(obj, require_selection=require_selection)

    @staticmethod
    def _check_allowed(d, key, allowed):
        if key in d and d[key] not in allowed:
            raise ValueError(
                f"Config key '{key}' has invalid value {d[key]!r}; "
                f"allowed values are {allowed!r}"
            )

    @staticmethod
    def _validate_cross_fields(d, *, require_selection):
        halo_mass_range = d.get('halo_mass_range')
        selection_mode = d.get('selection_mode')
        convergence_mode = d.get('convergence_mode')
        target_mean_mass = d.get('target_mean_mass')
        selection_mass_def = d.get('selection_mass_def')

        # Rule 5 — shape of halo_mass_range
        if halo_mass_range is not None:
            if not isinstance(halo_mass_range, (list, tuple)) or len(halo_mass_range) != 2:
                raise ValueError(
                    "Config key 'halo_mass_range' must be a list/tuple of length 2 "
                    f"or None; got {halo_mass_range!r}"
                )

        # Rule 1 — mass-bin selection is incompatible with mixed cen+sat mode
        if halo_mass_range is not None and selection_mode == 'mixed':
            raise ValueError(
                "halo_mass_range selection is not supported for "
                "selection_mode='mixed'; use n_gal_density instead."
            )

        # Rule 2 — massbin convergence needs a mass range
        if convergence_mode == 'massbin' and halo_mass_range is None:
            raise ValueError(
                "convergence_mode='massbin' requires a non-None halo_mass_range "
                "(use the placeholder [0, 1] when indices are loaded from the "
                "halo_indices file)."
            )

        # Rule 3 — mass-bin targeting requires a halo mass definition
        if (
            target_mean_mass is not None
            and halo_mass_range is not None
            and selection_mass_def in ('mstell', 'mstell_fof')
        ):
            raise ValueError(
                "target_mean_mass with halo_mass_range requires a halo mass "
                f"definition; selection_mass_def={selection_mass_def!r} is a "
                "stellar mass."
            )

        # Rule 4 — Abacus catalog restrictions (selection-validated calls only)
        if require_selection and d.get('sim_name') == 'abacus':
            if selection_mass_def != 'm200b':
                raise ValueError(
                    "For sim_name='abacus', selection_mass_def must be 'm200b' "
                    f"(mass-only catalog). Got: {selection_mass_def!r}"
                )
            if selection_mode not in ('cen', None):
                raise ValueError(
                    f"For sim_name='abacus', selection_mode={selection_mode!r} is "
                    "unsupported (no satellite labels available). Use 'cen' or None."
                )

    # ------------------------------------------------------------------
    # Selection tag
    # ------------------------------------------------------------------

    def selection_tag(self):
        """Build the filename token encoding the selection INTENT.

        Tokens emitted (in order):
            mode_<mode> | mode_all
            mass_<mdef>
            nzmass | allm
            umc1e<exp> | noumc
            urc | nourc
            cmode_<convergence_mode>
            sf<nn>          (mixed / sat mode only)
            tm<v>           (when target_mean_mass is set, e.g. tm13p2)

        Numeric convergence results (ngal, massbin window) are intentionally
        NOT included — those live in the halo_indices file, not in filenames.
        """
        if not getattr(self, '_has_selection', False):
            raise ValueError(
                "selection_tag requested but selection fields were not provided"
            )

        parts = []

        mode = self.selection_mode
        parts.append(f"mode_{mode}" if mode is not None else "mode_all")

        parts.append(f"mass_{self.selection_mass_def}")

        parts.append("nzmass" if self.select_nonzero_masses else "allm")

        if self.upper_mass_cut:
            exp = int(round(math.log10(self.max_mass)))
            parts.append(f"umc1e{exp:02d}")
        else:
            parts.append("noumc")

        parts.append("urc" if self.upper_radius_cut else "nourc")

        parts.append(f"cmode_{self.convergence_mode}")

        if mode == 'mixed' or mode == 'sat':
            parts.append(f"sf{int(round(self.sat_frac * 100)):02d}")

        tm = self.target_mean_mass
        if tm is not None:
            parts.append(f"tm{tm:.1f}".replace('.', 'p'))

        return "_" + "_".join(parts)