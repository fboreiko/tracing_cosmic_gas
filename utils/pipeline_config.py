"""
Single strict schema for the kSZ / HATF pipeline config dict.

`PipelineConfig` is the ONLY thing allowed to interpret a config dict. It
rejects unknown keys and requires every key that affects a filename.

Vocabulary
---------------------------------------
    feedback    : 'fiducial' | 'strongest_AGN'
                  Which FLAMINGO AGN-feedback variant the gas comes from.
                  (was: gas_type, which also encoded truth/recon via a
                  '_reconstructed' suffix)

    tau_source  : 'truth' | 'recon'
                  Where the tau map came from. 'truth' is measured directly
                  from the gas particles; 'recon' is reconstructed from the
                  DM field via the HATF transfer function.

    tracer      : 'gas' | 'dm'
                  Which particle field the map traces.
                  (was: field_type)

These three axes are orthogonal. Do not re-encode one inside another.
"""

import math
from dataclasses import dataclass, fields

# Allowed value sets
_ALLOWED_SIM_NAME = ('flamingo', 'abacus')
_ALLOWED_FEEDBACK = ('fiducial', 'strongest_AGN')
_ALLOWED_TAU_SOURCE = ('truth', 'recon')
_ALLOWED_TRACER = ('gas', 'dm')
_ALLOWED_SELECTION_MODE = ('cen', 'sat', 'mixed', 'cens_sat', None)
_ALLOWED_SELECTION_MASS_DEF = ('mstell', 'mstell_fof', 'm200b', 'm200c', 'mfof')
_ALLOWED_CONVERGENCE_MODE = ('ngal', 'massbin')
_ALLOWED_PROJECTION_TYPE = ('simple', 'pixell', 'pixell_cea', None)

# Legacy names -> new names. Used only to produce a helpful error message.
_LEGACY_KEYS = {
    'sim': 'sim_name',
    'gas_type': 'feedback',
    'tau_method': 'tau_source',
    'field_type': 'tracer',
    'A': None,  # deleted with the massdep method
}

_LEGACY_VALUES = {
    'feedback': {
        'fiducial_reconstructed': "'fiducial' with tau_source='recon'",
        'strongest_AGN_reconstructed': "'strongest_AGN' with tau_source='recon'",
    },
    'tau_source': {
        'fullFT_tau_reconstruction': "'truth'",
        '2D_FT_upgrade_tau_reconstruction': "'recon'",
        '2D_FT_massdep_tau_reconstruction': "nothing - this method was removed",
    },
}

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


@dataclass(frozen=True)
class PipelineConfig:
    """Frozen, validated pipeline configuration. Build with `from_dict`."""

    # Identity
    sim_name: str = None
    feedback: str = None
    tau_source: str = None
    tracer: str = 'gas'
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

    # Selection-loop knobs
    target_mass_tolerance: float = 0.005
    mass_bin_halfwidth: float = 0.02
    mass_bin_halfwidth_tol: float = 0.001
    max_iterations: int = 1000
    halo_file_path: str = None

    # Presentation metadata - never used in tags or paths
    name: str = None
    projection_type: str = None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, d, *, require_selection=True):
        """Validate a plain config dict and return a frozen PipelineConfig.

        Raises ValueError on legacy keys, on legacy values, on unknown keys,
        on missing required keys, and on any cross-field rule violation.
        """
        d = dict(d)  # never mutate the caller's dict

        # 1. Legacy-key trap
        for old, new in _LEGACY_KEYS.items():
            if old in d:
                if new is None:
                    raise ValueError(
                        f"Config key {old!r} was removed along with the "
                        f"mass-dependent tau method. Delete it."
                    )
                raise ValueError(
                    f"Config key {old!r} was renamed to {new!r}. "
                    f"See NAMING_REFACTOR.md."
                )

        # 2. Unknown keys
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(d) - known)
        if unknown:
            raise ValueError(
                "Unknown config keys (typos?): " + ", ".join(repr(k) for k in unknown)
            )

        # 3. Legacy-value trap (catches half-migrated configs)
        for key, mapping in _LEGACY_VALUES.items():
            value = d.get(key)
            if value in mapping:
                raise ValueError(
                    f"Config key {key!r} has legacy value {value!r}; "
                    f"use {mapping[value]}. See NAMING_REFACTOR.md."
                )

        # 4. Identity requiredness.
        # sim_name and feedback are always required; identity-only dicts
        # (e.g. for particle/halo file lookup) legitimately omit tau_source.
        for key in ('sim_name', 'feedback'):
            if d.get(key) is None:
                raise ValueError(f"Missing required config key: '{key}'")
        if require_selection and d.get('tau_source') is None:
            raise ValueError("Missing required config key: 'tau_source'")

        # 5. Selection requiredness
        if require_selection:
            for key in _REQUIRED_SELECTION:
                if key not in d:
                    raise ValueError(f"Missing required config key: '{key}'")
                if d[key] is None and key not in _NULLABLE_SELECTION:
                    raise ValueError(f"Config key '{key}' must not be None")

        # 6. Conditional fields
        if d.get('upper_mass_cut') is True and d.get('max_mass') is None:
            raise ValueError(
                "Config key 'max_mass' is required when upper_mass_cut is True"
            )
        if d.get('selection_mode') in ('mixed', 'sat') and d.get('sat_frac') is None:
            raise ValueError(
                "Config key 'sat_frac' is required when selection_mode is "
                f"{d.get('selection_mode')!r}"
            )

        # 7. Allowed values
        cls._check_allowed(d, 'sim_name', _ALLOWED_SIM_NAME)
        cls._check_allowed(d, 'feedback', _ALLOWED_FEEDBACK)
        if d.get('tau_source') is not None:
            cls._check_allowed(d, 'tau_source', _ALLOWED_TAU_SOURCE)
        if 'tracer' in d:
            cls._check_allowed(d, 'tracer', _ALLOWED_TRACER)
        if 'projection_type' in d:
            cls._check_allowed(d, 'projection_type', _ALLOWED_PROJECTION_TYPE)
        if require_selection:
            cls._check_allowed(d, 'selection_mode', _ALLOWED_SELECTION_MODE)
            cls._check_allowed(d, 'selection_mass_def', _ALLOWED_SELECTION_MASS_DEF)
            cls._check_allowed(d, 'convergence_mode', _ALLOWED_CONVERGENCE_MODE)

        # 8. Cross-field validation
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

    def replace(self, **changes):
        """Return a NEW config with `changes` applied and re-validated.

        Use this instead of mutating a shared config dict. Mutation in place
        was the mechanism behind the old `_config['gas_type'] += '_reconstructed'`
        bug, where a path built before the mutation and one built after
        silently pointed at different files.
        """
        d = {f.name: getattr(self, f.name) for f in fields(self)}
        d.update(changes)
        return type(self).from_dict(
            d, require_selection=getattr(self, '_has_selection', False)
        )

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

        # Rule 5 - shape of halo_mass_range
        if halo_mass_range is not None:
            if not isinstance(halo_mass_range, (list, tuple)) or len(halo_mass_range) != 2:
                raise ValueError(
                    "Config key 'halo_mass_range' must be a list/tuple of length 2 "
                    f"or None; got {halo_mass_range!r}"
                )

        # Rule 1 - mass-bin selection is incompatible with mixed cen+sat mode
        if halo_mass_range is not None and selection_mode == 'mixed':
            raise ValueError(
                "halo_mass_range selection is not supported for "
                "selection_mode='mixed'; use n_gal_density instead."
            )

        # Rule 2 - massbin convergence needs a mass range
        if convergence_mode == 'massbin' and halo_mass_range is None:
            raise ValueError(
                "convergence_mode='massbin' requires a non-None halo_mass_range "
                "(use the placeholder [0, 1] when indices are loaded from the "
                "halo_indices file)."
            )

        # Rule 3 - mass-bin targeting requires a halo mass definition
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

        # Rule 4 - Abacus catalog restrictions (selection-validated calls only)
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

        Returns a BARE token with no leading underscore. Callers that need a
        separator add it themselves. (The old version returned a leading '_'
        and half the call sites immediately did .lstrip('_').)

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
        NOT included - those live in the halo_indices file, not in filenames.
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

        if mode in ('mixed', 'sat'):
            parts.append(f"sf{int(round(self.sat_frac * 100)):02d}")

        tm = self.target_mean_mass
        if tm is not None:
            parts.append(f"tm{tm:.1f}".replace('.', 'p'))

        return "_".join(parts)
