#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The run configuration, and the command line that builds one.

    1. what is being reconstructed   the bundle keys and the integration range
    2. MassRange                     M_r / M_max resolved against a bundle
    3. PmxConfig                     every knob, one frozen dataclass
    4. the command line              argparse groups, assembled by base_parser
"""
import argparse
from dataclasses import dataclass, fields
import numpy as np
from utils.sim_params import get_sim_params

from Pmx_reconstruction.pmxlib.geometry import (GEOMETRIES, NGRID_3D_DEFAULT,
                                                 deconvolve_is_default)
from Pmx_reconstruction.pmxlib.nfw import CONCENTRATION_SOURCES


INT_LOGM_LO = 8.0
INT_NODES = 256


@dataclass(frozen=True)
class MassRange:
    """Which bins R sums over, resolved against a particular bundle.

    Separate from PmxConfig because it is built at a different moment. A
    PmxConfig exists before any file is opened and is pure user intent
    ("M_r around 10^12"); turning that into bin indices, per-bin mass fractions
    and an integration range takes the bundle's counts, M_mean, logM_edges and
    box.
    """
    validate: bool
    r: int                  # M_r bin: first bin in R, and the U template
    t: int                  # M_max bin: last bin in R
    resolved: np.ndarray    # bins in R
    hidden: np.ndarray      # occupied bins below M_r
    above: np.ndarray       # occupied bins above M_max
    logM_r: float           # log10 <M> of bin r
    logM_t: float           # log10 <M> of bin t
    M_ref: float            # <M> of bin r
    M_u_hi: float           # U integral upper limit = lower edge of bin r
    int_logm_lo: float      # U integral lower limit
    f_i: np.ndarray         # catalogue mass fraction per bin
    f_resolved: float
    f_hidden: float
    f_above: float

    @property
    def f_u(self):
        """Catalogue amplitude of U: everything not in R and not above M_max."""
        return 1.0 - self.f_resolved - self.f_above

    @property
    def is_default(self):
        return not (self.hidden.any() or self.above.any())

    def tag(self):
        mode = 'val' if self.validate else 'prod'
        return f'_Mr{self.logM_r:.2f}_Mmax{self.logM_t:.2f}_{mode}'

    def payload(self):
        return dict(validate=self.validate, ref_bin=self.r, top_bin=self.t,
                    logM_r=self.logM_r, logM_t=self.logM_t, M_ref=self.M_ref,
                    M_u_hi=self.M_u_hi, int_logm_min=self.int_logm_lo,
                    resolved=self.resolved, hidden=self.hidden,
                    above=self.above, f_u_cat=self.f_u,
                    f_hidden=self.f_hidden, f_above=self.f_above)

    @classmethod
    def from_config(cls, cfg, data):
        """Resolve a config's M_r / M_max against a loaded bundle.

        logm_r and logm_max_rec are what the user asked for; which BIN that is
        depends on the bundle, so this is the step that needs the data and is
        why MassRange is a separate object from PmxConfig at all. Both are
        snapped to the occupied bin with the nearest <M>.

        R sums bins r..t, i.e. [M_r, M_max]. U models the mass below M_r: its
        template is bin r and its integral stops at the LOWER EDGE of bin r, so
        that the two cover the mass axis once between them and never twice.
        """
        counts = np.asarray(data['counts'], float)
        M = np.asarray(data['M_mean'], float)
        edges = np.asarray(data['logM_edges'], float)
        occ = counts > 0
        if not occ.any():
            raise SystemExit("No occupied mass bins: check "
                             "--bundle-logm-min/max.")

        occ_idx = np.flatnonzero(occ)
        logM = np.log10(M[occ_idx])

        def nearest(target):
            return int(occ_idx[np.argmin(np.abs(logM - target))])

        r = occ_idx[0] if cfg.logm_r is None else nearest(cfg.logm_r)
        t = nearest(cfg.logm_max_rec)
        if t < r:
            raise SystemExit(f"--logm-max {cfg.logm_max_rec} is below "
                             f"--logm-r {cfg.logm_r}.")

        idx = np.arange(occ.size)
        resolved = occ & (idx >= r) & (idx <= t)
        hidden = occ & (idx < r)
        above = occ & (idx > t)
        if cfg.validate and not hidden.any():
            raise SystemExit("--validate needs --logm-r above the lowest "
                             "occupied bin.")

        if cfg.int_logm_min is not None:
            lo = float(cfg.int_logm_min)
        elif cfg.validate:
            lo = float(edges[0])       # the hidden bins ARE the integration range
        else:
            lo = INT_LOGM_LO

        f_i = counts / float(data['box']) ** 3 * M / cfg.rhobar_m
        return cls(
            validate=cfg.validate, r=int(r), t=int(t),
            resolved=resolved, hidden=hidden, above=above,
            logM_r=float(np.log10(M[r])), logM_t=float(np.log10(M[t])),
            M_ref=float(M[r]), M_u_hi=float(10.0 ** edges[r]), int_logm_lo=lo,
            f_i=f_i, f_resolved=float(f_i[resolved].sum()),
            f_hidden=float(f_i[hidden].sum()), f_above=float(f_i[above].sum()))

    def report(self, prefix):
        print(f"{prefix} {'validation' if self.validate else 'production'}: "
              f"R over bins {self.r}..{self.t} "
              f"(log<M> {self.logM_r:.2f}..{self.logM_t:.2f})")
        print(f"{prefix} mass fractions: R {self.f_resolved:.4f}, below M_r "
              f"{self.f_hidden:.4f}, above M_max {self.f_above:.4f}; "
              f"f_u = {self.f_u:.4f}")
        print(f"{prefix} U integral over logM {self.int_logm_lo:.2f}.."
              f"{np.log10(self.M_u_hi):.2f}")
        if self.above.any():
            print(f"{prefix} WARNING: bins above M_max are not modelled; "
                  f"their matter is reported as V*.")


@dataclass(frozen=True)
class PmxConfig:
    """Every knob of a run, in one frozen object."""

    # --- simulation -----------------------------------------------------------
    sim_name: str = 'flamingo'
    feedback: str = 'strongest_AGN'
    nthread: int = None          # None -> sim_params['nthread_default']

    # --- how the spectra are measured ----------------------------------------
    # '2d' projects along z (k up to ~9, but k dk modes); '3d' paints a cubic
    # grid (k^2 dk modes, so sqrt(2k/k_f) less noise, but k_Nyquist = pi n / L
    # puts the reachable k_max far lower). See pmxlib.geometry.
    geometry: str = '2d'
    ngrid: int = None            # None -> 2048 for '2d', 384 for '3d'
    kmax: float = None           # None -> the grid's own largest |k|
    # None -> each geometry's own default: off for '2d' (so results measured
    # before there was a switch keep their meaning), on for '3d'.
    deconvolve_window: bool = None

    # --- halo binning ---------------------------------------------------------
    mass_def: str = 'm200b'
    logm_min: float = 11.0       # log10(M / [Msun/h])
    logm_max: float = 15.0       # (the physics cut is logm_r / logm_max_rec)
    nbins: int = 30              # log-spaced bins between logm_min and logm_max
    nkbins: int = 600            # None -> max(32, ngrid // 10)
    centrals_only: bool = True   # satellites repeat their host's m200b; counting
                                 # them would double-count halo mass in n_i M_i
    # Unit of the catalogue's halo-mass array, in Msun/h. Particle masses in
    # this repo are stored in 1e10 Msun/h; halo masses are read straight out of
    # /galaxies/m200b. If the sanity check in pmxlib.binning fires, set 1e10.
    halo_mass_unit_msun_h: float = 1.0

    # --- halo profile ---------------------------------------------------------
    # c(M,z). colossus is the only source; which published relation it uses is
    # colossus_conc_model, and that is the knob to turn when comparing fits.
    concentration_source: str = 'colossus'
    colossus_conc_model: str = 'diemer19'

    # --- mass range, resolved against a bundle by MassRange.from_config -------
    # R sums the bins in [M_r, M_max]; U models the mass below M_r.
    logm_r: float = None          # None -> lowest occupied bin
    logm_max_rec: float = 15.0
    validate: bool = False        # experiment A: hide the bins below M_r
    int_logm_min: float = None    # override the U integral's lower limit

    # --- how U extrapolates P_halo_gas(k|M) below M_r -------------------------
    extrap: tuple = ('flat', 'bias', 'halomodel')
    hmf: str = 'tinker'            # 'tinker' | 'catalog' (needs validate)
    fu: str = 'catalog'            # 'catalog' (measured deficit) | 'model'
    power_spectrum: str = 'camb'   # 'camb' | 'eisenstein98'

    def __post_init__(self):
        if self.geometry not in GEOMETRIES:
            raise ValueError(f"geometry is {self.geometry!r}; expected one of "
                             f"{GEOMETRIES}")
        if self.mass_def != 'm200b':
            raise ValueError(
                f"mass_def is {self.mass_def!r}, but r200m_of_M assumes a "
                f"mean-background (200m) definition. Either use 'm200b' or "
                f"change r200m_of_M to match.")

    # --- derived from the simulation's own parameter file ---------------------
    @property
    def sim_params(self):
        return get_sim_params(self.sim_name)

    @property
    def box(self):
        return self.sim_params['box_size_cMpc_h']      # 681.0 cMpc/h

    @property
    def z(self):
        return self.sim_params['redshift']             # 0.74

    @property
    def h(self):
        return self.sim_params['h']

    @property
    def omega_m(self):
        return self.sim_params['omega_m']

    @property
    def grid(self):
        if self.ngrid:
            return self.ngrid
        return (NGRID_3D_DEFAULT if self.geometry == '3d'
                else self.sim_params['ngrid_default'])

    @property
    def threads(self):
        return self.nthread if self.nthread else self.sim_params['nthread_default']

    @property
    def k_Nyquist(self):
        return np.pi * self.grid / self.box

    @property
    def geom_tag(self):
        """Filename token for the geometry and its window handling.

        Empty for a default '2d' run, so every cache written before there was
        a choice keeps its name; a non-default window setting always shows,
        because it changes the numbers without changing anything else.
        """
        tag = '' if self.geometry == '2d' else f'_{self.geometry}'
        if not deconvolve_is_default(self):
            tag += '_dcw' if self.deconvolve_window else '_nodcw'
        return tag

    @property
    def rhobar_m(self):
        """Comoving mean matter density in (Msun/h)/(Mpc/h)^3."""
        from Pmx_reconstruction.pmxlib.cosmology import mean_matter_density
        return mean_matter_density(self.sim_params)

    # --- construction ---------------------------------------------------------
    @classmethod
    def from_args(cls, args, **overrides):
        """Build from an argparse Namespace, taking only the fields it carries."""
        names = {f.name for f in fields(cls)}
        kw = {k: v for k, v in vars(args).items() if k in names and v is not None}
        if 'extrap' in kw:
            kw['extrap'] = tuple(kw['extrap'])     # argparse hands back a list
        kw.update(overrides)
        return cls(**kw)


_D = PmxConfig()


def _add_binning_args(ap):
    g = ap.add_argument_group('Halo binning and grid')
    g.add_argument('--nbins', type=int, default=_D.nbins,
                   help="number of log-spaced halo mass bins")
    g.add_argument('--bundle-logm-min', '--logm-min', dest='logm_min',
                   type=float, default=_D.logm_min,
                   help="lowest bin edge of the cached bundle to read; not a "
                        "physics cut (use --logm-r)")
    g.add_argument('--bundle-logm-max', dest='logm_max', type=float,
                   default=_D.logm_max,
                   help="highest bin edge of the cached bundle to read; not a "
                        "physics cut (use --logm-max)")
    g.add_argument('--nkbins', type=int, default=_D.nkbins,
                   help="number of k bins in the measured spectra")
    g.add_argument('--ngrid', type=int, default=None,
                   help=f"FFT grid size per axis (default: the simulation's "
                        f"ngrid_default for --geometry 2d, "
                        f"{NGRID_3D_DEFAULT} for 3d)")
    g.add_argument('--nthread', type=int, default=None,
                   help="threads for painting and FFTs (default: nthread_default)")
    return g


def _add_geometry_args(ap):
    g = ap.add_argument_group('Measurement geometry')
    g.add_argument('--geometry', default=_D.geometry, choices=list(GEOMETRIES),
                   help="'2d' projects the box along z, reaching k ~ 9 h/cMpc "
                        "with k dk modes per bin. '3d' paints a cubic grid: "
                        "k^2 dk modes, so sigma falls by sqrt(2k/k_f) (4.7x at "
                        "k = 0.1), but k_Nyquist = pi ngrid / L_box caps how "
                        "far in k a run can go. They are alternatives; a run "
                        "is one or the other")
    g.add_argument('--deconvolve-window', dest='deconvolve_window',
                   action='store_true', default=None,
                   help="divide out the TSC assignment window. On by default "
                        "for '3d', where it is 1.8%% of P at k = 0.1 on a "
                        "256^3 grid; off by default for '2d', where it is 3e-4 "
                        "at k = 0.1 on 2048^2 but 22%% by k = 3. Turn it on for "
                        "BOTH runs to compare the two geometries above k ~ 1")
    g.add_argument('--no-deconvolve-window', dest='deconvolve_window',
                   action='store_false',
                   help="leave the TSC window in (the '2d' default)")
    g.add_argument('--kmax', type=float, default=None,
                   help="upper edge of the k binning (default: the grid's own "
                        "largest |k|). Give a 2-D and a 3-D run the same "
                        "--kmax and --nkbins and they land on identical bin "
                        "edges, which is what makes them comparable bin by bin")
    return g


def _add_profile_args(ap):
    g = ap.add_argument_group('Halo profile')
    g.add_argument('--concentration', dest='concentration_source',
                   default=_D.concentration_source,
                   choices=list(CONCENTRATION_SOURCES),
                   help="where c(M,z) comes from. Only 'colossus' is "
                        "implemented; the switch is the dispatch point for "
                        "adding a relation that is not a colossus model")
    g.add_argument('--concentration-model', dest='colossus_conc_model',
                   default=_D.colossus_conc_model,
                   help="colossus c(M,z) model name (diemer19, duffy08, "
                        "ishiyama21, ...). This is the knob for comparing fits")
    return g


def _add_mass_range_args(ap, validate=True):
    g = ap.add_argument_group('Mass range')
    g.add_argument('--logm-r', dest='logm_r', type=float, default=None,
                   help="log10 M_r: R sums the bins from here up, U models the "
                        "mass below; snapped to the bin with the nearest <M> "
                        "(default: lowest occupied bin)")
    g.add_argument('--logm-max', dest='logm_max_rec', type=float,
                   default=_D.logm_max_rec,
                   help="log10 M_max: upper end of R, snapped like --logm-r")
    if validate:
        g.add_argument('--validate', action='store_true',
                       help="score U against the exact contribution of the "
                            "bins masked below M_r (integral starts at the "
                            "bundle edge)")
    g.add_argument('--int-logm-min', dest='int_logm_min', type=float,
                   default=None,
                   help=f"override the lower limit of the U integral "
                        f"(default {INT_LOGM_LO})")
    return g


def _add_extrapolation_args(ap):
    g = ap.add_argument_group('Extrapolating P_halo_gas(k|M) below M_r')
    g.add_argument('--extrap', nargs='+', default=list(_D.extrap),
                   choices=['flat', 'simhc', 'bias', 'halomodel'],
                   help="which extrapolations to run")
    g.add_argument('--hmf', default=_D.hmf, choices=['tinker', 'catalog'],
                   help="mass function used in the U integral: 'tinker' or "
                        "'catalog' (the masked bins; needs --validate)")
    g.add_argument('--fu', default=_D.fu, choices=['catalog', 'model'],
                   help="amplitude of U: 'catalog' (measured deficit) or "
                        "'model' (the HMF integral)")
    g.add_argument('--power-spectrum', dest='power_spectrum',
                   default=_D.power_spectrum,
                   choices=['camb', 'eisenstein98'],
                   help="linear P(k) behind the halo model")
    return g


def base_parser(description, validate=True):
    """The parser both experiments start from.

    Everything it adds maps onto a PmxConfig field, so a script's main() is
    `cfg = PmxConfig.from_args(base_parser(...).parse_args())` plus whatever is
    genuinely its own. `validate=False` drops --validate, which only experiment
    A can honour.
    """
    ap = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--recompute', action='store_true',
                    help="ignore any cached spectra bundle and re-measure")
    _add_binning_args(ap)
    _add_geometry_args(ap)
    _add_profile_args(ap)
    _add_mass_range_args(ap, validate=validate)
    _add_extrapolation_args(ap)
    return ap
