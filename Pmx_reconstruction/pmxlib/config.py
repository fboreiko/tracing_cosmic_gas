#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The run configuration, and the command line that builds one.

    1. what is being reconstructed   the bundle keys and the integration range
    2. MassRange                     M_r / M_max resolved against a bundle
    3. PmxConfig                     every knob, one frozen dataclass
    4. the command line              argparse groups, assembled by base_parser
    5. figure stems                  stem_A / stem_B / stem_C, one per script
"""
import argparse
from dataclasses import dataclass, fields
import numpy as np
from utils.power_spectrum_utils import log_k_bin_edges
from utils.sim_params import get_sim_params

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
    ngrid: int = None            # None -> sim_params['ngrid_default']   (2048)
    nthread: int = None          # None -> sim_params['nthread_default']

    # --- large scales, measured in 3-D and stitched under the 2-D run ---------
    # Projecting to k_z = 0 throws away a factor 2k/k_f in modes: 4.7x in sigma
    # at k = 0.1, 10x at k = 0.5. ngrid_3d = None leaves the pipeline purely 2-D
    ngrid_3d: int = None
    k_split: float = 0.5         # 3-D below, 2-D above [h/cMpc]

    # --- halo binning ---------------------------------------------------------
    mass_def: str = 'm200b'
    logm_min: float = 11.0       # log10(M / [Msun/h])
    logm_max: float = 15.0       # (the physics cut is logm_r / logm_max_rec)
    nbins: int = 30              # log-spaced bins between logm_min and logm_max

    # --- k binning ------------------------------------------------------------
    kbins_per_decade: float = 25.0
    kbin_wmin_kf: float = 2.0

    centrals_only: bool = True   # satellites repeat their host's m200b; counting
                                 # them would double-count halo mass in n_i M_i
    # Unit of the catalogue's halo-mass array, in Msun/h. Particle masses in
    # this repo are stored in 1e10 Msun/h; halo masses are read straight out of
    # /galaxies/m200b. If the sanity check in pmxlib.binning fires, set 1e10.
    halo_mass_unit_msun_h: float = 1.0

    # --- halo profile ---------------------------------------------------------
    concentration_source: str = 'colossus'
    colossus_conc_model: str = 'diemer19'
    # Where u_m(k|M) comes from in R and in U's S(k).
    #   'nfw'      the model: u_nfw with c(M,z) from the two knobs above.
    #   'measured' the stacked profile from pmxlib.u_bar, interpolated in
    #              log M, with the model kept only below the measured floor.
    # 'measured' also switches the mass weights from n_i M_i / rhobar_m to the
    # mass actually assigned inside the aperture.
    profile_source: str = 'nfw'

    # --- clumping correction --------------------------------------------------
    # The intra-halo clumping correction of pmxlib.clumping, measured by
    # experiment D: u_m -> u_m [1 + xi(k r200m)].
    clumping: bool = False
    clump_a: float = None          # None -> clumping.CLUMP_A
    clump_alpha: float = None      # None -> clumping.CLUMP_ALPHA
    clump_max: float = None        # None -> clumping.XI_MAX; inf uncaps it

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
        if self.ngrid_3d:
            k_ny3 = np.pi * self.ngrid_3d / self.box
            if self.k_split > 0.6 * k_ny3:
                need = int(np.ceil(self.k_split * self.box / (0.6 * np.pi)))
                raise ValueError(
                    f"--k-split {self.k_split:g} is above 0.6 k_Nyquist "
                    f"({0.6 * k_ny3:.3f}) for --ngrid-3d {self.ngrid_3d}, "
                    f"where TSC aliasing stops being negligible. Use "
                    f"--ngrid-3d {need} or lower --k-split.")
        if self.profile_source not in ('nfw', 'measured'):
            raise ValueError(
                f"profile_source is {self.profile_source!r}; expected 'nfw' "
                f"or 'measured'. Checked here because Profile treats anything "
                f"that is not 'nfw' as a request for the measured cache, so a "
                f"typo would otherwise go looking for one rather than fail.")
        if self.mass_def != 'm200b':
            raise ValueError(
                f"mass_def is {self.mass_def!r}, but r200m_of_M assumes a "
                f"mean-background (200m) definition. Either use 'm200b' or "
                f"change r200m_of_M to match.")
        if self.clumping:
            from Pmx_reconstruction.pmxlib import clumping as cl
            a, alpha, xi_max = self.clump_params
            if a <= 0 or alpha <= 0:
                raise ValueError(
                    f"--clumping is on with clump_a={a:g}, clump_alpha="
                    f"{alpha:g}; both must be positive. a = 0 would leave the "
                    f"correction switched on and doing nothing, which is the "
                    f"one outcome that looks like a result.")
            if xi_max >= 1.0:
                raise ValueError(
                    f"clump_max={xi_max:g} makes the 1/(1 - xi) factor "
                    f"diverge or change sign. Cap below 1; the measured "
                    f"plateau is {cl.XI_MAX:g}.")

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
        return self.ngrid if self.ngrid else self.sim_params['ngrid_default']

    @property
    def threads(self):
        return self.nthread if self.nthread else self.sim_params['nthread_default']

    @property
    def grid_3d(self):
        return self.ngrid_3d

    @property
    def k_Nyquist_3d(self):
        return np.pi * self.ngrid_3d / self.box if self.ngrid_3d else None

    @property
    def k_bins(self):
        """The k bin edges of this run. One definition, used everywhere.

        Built from the PROJECTED grid even when a 3-D measurement is spliced
        in: the 3-D half is binned onto these same edges, so the splice stays
        an assignment rather than an interpolation.
        """
        k_f = 2.0 * np.pi / self.box
        k_max = np.sqrt(2.0) * np.pi * self.grid / self.box   # rfft2 corner
        return log_k_bin_edges(k_f, k_max, self.kbins_per_decade,
                               self.kbin_wmin_kf * k_f)

    @property
    def kbin_tag(self):
        """Filename token for the k binning."""
        w = '' if self.kbin_wmin_kf == 2.0 else f'w{self.kbin_wmin_kf:g}'
        return f'kd{self.kbins_per_decade:g}{w}'

    @property
    def clump_params(self):
        """(a, alpha, xi_max) with pmxlib.clumping's defaults filled in."""
        from Pmx_reconstruction.pmxlib import clumping as cl
        return (cl.CLUMP_A if self.clump_a is None else float(self.clump_a),
                cl.CLUMP_ALPHA if self.clump_alpha is None else float(self.clump_alpha),
                cl.XI_MAX if self.clump_max is None else float(self.clump_max))

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
    g.add_argument('--kbins-per-decade', dest='kbins_per_decade', type=float,
                   default=_D.kbins_per_decade,
                   help="k bins per decade. The grid is log-spaced, with the "
                        "width floored at --kbin-wmin fundamentals")
    g.add_argument('--kbin-wmin', dest='kbin_wmin_kf', type=float,
                   default=_D.kbin_wmin_kf,
                   help="floor on the bin width, in units of k_f = 2 pi / L. "
                        "Below about 2, the mode count per bin stops growing "
                        "smoothly and the low-k points scatter erratically")
    g.add_argument('--ngrid', type=int, default=None,
                   help="FFT grid size (default: the simulation's ngrid_default)")
    g.add_argument('--nthread', type=int, default=None,
                   help="threads for painting and FFTs (default: nthread_default)")
    return g


def _add_largescale_args(ap):
    g = ap.add_argument_group('Large scales in 3-D')
    g.add_argument('--ngrid-3d', dest='ngrid_3d', type=int, nargs='?',
                   const=256, default=None,
                   help="measure k < --k-split on a cubic grid of this size "
                        "and stitch it under the 2-D run. Bare --ngrid-3d "
                        "means 256. Omitted, the pipeline is purely 2-D")
    g.add_argument('--k-split', dest='k_split', type=float, default=_D.k_split,
                   help="where the 3-D measurement hands over to the 2-D one")
    g.add_argument('--recompute-3d', dest='recompute_3d', action='store_true',
                   help="ignore any cached 3-D measurement and re-measure")
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
    g.add_argument('--profile', dest='profile_source',
                   default=_D.profile_source, choices=['nfw', 'measured'],
                   help="u_m(k|M) in R and in U's S(k). 'nfw' is the model; "
                        "'measured' reads the stacked profile cached by "
                        "pmxlib.u_bar and ALSO switches the mass weights to "
                        "the assigned mass the profile is normalised to "
                        "(f_part instead of n_i M_i / rhobar_m), because the "
                        "two only mean anything together. Below the measured "
                        "floor the model is kept")
    g.add_argument('--clumping', action='store_true',
                   help="apply the intra-halo clumping correction measured by "
                        "experiment D: u_m -> u_m / [1 - xi(k r200m)], so R "
                        "carries the matter-gas correlation inside a halo "
                        "that no radial profile can. Composes with --profile "
                        "and with any --extrap mode; --extrap halomodel "
                        "--clumping is 'halomodel plus the correction'")
    g.add_argument('--clump-a', dest='clump_a', type=float, default=None,
                   help="amplitude of xi = a (k r200m)^alpha "
                        "(default: the fitted value in pmxlib.clumping)")
    g.add_argument('--clump-alpha', dest='clump_alpha', type=float,
                   default=None, help="index of the same (default: fitted)")
    g.add_argument('--clump-max', dest='clump_max', type=float, default=None,
                   help="cap on xi, where the power law leaves the range "
                        "experiment D measured. 'inf' uncaps it, which "
                        "extrapolates to a >100%% correction for the most "
                        "massive bins near Nyquist -- read pmxlib.clumping "
                        "before using it")
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
                   choices=['flat', 'simhc', 'bias', 'halomodel', 'gascdm'],
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
    _add_largescale_args(ap)
    _add_profile_args(ap)
    _add_mass_range_args(ap, validate=validate)
    _add_extrapolation_args(ap)
    return ap


def models_tag(cfg, power_spectrum=True, profile=True, grid_3d=True,
               kbins=True):
    """The run-configuration suffix every figure name ends with."""
    out = ''
    if cfg.concentration_source != PmxConfig.concentration_source:
        out += f'_conc-{cfg.concentration_source}'
    if cfg.colossus_conc_model != PmxConfig.colossus_conc_model:
        out += f'_cm-{cfg.colossus_conc_model}'
    if power_spectrum and cfg.power_spectrum != PmxConfig.power_spectrum:
        out += f'_ps-{cfg.power_spectrum}'
    if profile:
        if cfg.profile_source != PmxConfig.profile_source:
            out += f'_prof-{cfg.profile_source}'
        if cfg.clumping:
            from Pmx_reconstruction.pmxlib import clumping as cl
            a, alpha, xi_max = cfg.clump_params
            out += '_clump'
            if (a, alpha) != (cl.CLUMP_A, cl.CLUMP_ALPHA):
                out += f'-a{a:g}p{alpha:g}'
            if xi_max != cl.XI_MAX:
                out += f'-max{xi_max:g}'
    if kbins and (cfg.kbins_per_decade != PmxConfig.kbins_per_decade
                  or cfg.kbin_wmin_kf != PmxConfig.kbin_wmin_kf):
        out += f'_{cfg.kbin_tag}'
    if grid_3d and cfg.ngrid_3d:
        out += f'_3d{cfg.ngrid_3d}k{cfg.k_split:g}'
    return out


def stem_A(cfg, kind, mr):
    """experiment_A's figure name. `kind` is the panel, `mr` its MassRange."""
    return (f'expA_{kind}_gas_{cfg.mass_def}_nb{cfg.nbins}{mr.tag()}'
            f'_hmf{cfg.hmf}_fu{cfg.fu}{models_tag(cfg)}')


def stem_B(cfg, kind, mr, apertures, weights, err_mode):
    """experiment_B's figure name.

    `apertures` is the swept list, `weights` 'aperture' or 'catalogue', and
    `err_mode` the mode the error budget panel is drawn for -- dropped for the
    shell-mass figure, which has no budget panel to name."""
    mode_tag = '' if kind == 'shellmass' or not err_mode else f'_mode{err_mode}'
    return (f'expB_{kind}_gas_{cfg.mass_def}_nb{cfg.nbins}'
            f'_logMmin{cfg.logm_min:.2f}_logMmax{cfg.logm_max:.2f}'
            f'{"" if mr.is_default else mr.tag()}'
            f'_ap{"-".join(f"{x:g}" for x in apertures)}'
            f'{mode_tag}_w{weights}{models_tag(cfg)}')


def stem_C(cfg, nbins_u, logm_lo, logm_hi, n_show, aperture):
    """experiment_C's figure name."""
    ap = '' if float(aperture) == 1.0 else f'_ap{float(aperture):g}'
    models = models_tag(cfg, power_spectrum=False, profile=False,
                        grid_3d=False, kbins=False)
    return (f'expC_profile_{cfg.mass_def}_nbu{int(nbins_u)}'
            f'_logMu{float(logm_lo):g}-{float(logm_hi):g}'
            f'_nshow{n_show}{ap}{models}')
