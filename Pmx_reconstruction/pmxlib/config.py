#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Frozen run configuration for the Pmx reconstruction pipeline.

Replaces the module-global constants that predict_Pmx_from_Phx.py carried and
that measure_u_tilde.py / plot_omega_weights.py mutated from the outside:

    P.NTHREAD = int(args.nthread)                    # measure_u_tilde.py:757
    P.CONCENTRATION_SOURCE = args.concentration
    P.COLOSSUS_CONC_MODEL  = args.concentration_model

A PmxConfig is built once from argv, passed down explicitly, and never
modified. Two objects in one process cannot disagree, and no function's return
value depends on how far through main() the interpreter has got.
"""
import dataclasses
from dataclasses import dataclass, fields

from utils.sim_params import get_sim_params

# --- which spectrum is being reconstructed ------------------------------------
# This is the second field x in P_matter_x = (1/rhobar_m) sum_i n_i M_i u_m P_halo_x.
# The identity is blind to x, so this is purely a choice of what to point it at:
#   'matter_gas'    : x = gas,    reconstruct P(matter x gas)    from P_halo_gas
#   'matter_dm'     : x = dm,     reconstruct P(matter x dm)     from P_halo_dm
#   'matter_matter' : x = matter, reconstruct P(matter x matter) from P_halo_matter
# All three halo cross-spectra live in the same bundle, so switching is free.
#
# NAMING CONVENTION, used everywhere below
#     P_<first field>_<second field>
# with the fields drawn from {halo, matter, dm, gas} and 'x' standing for
# whichever tracer the current --target has selected. So:
#     P_halo_gas, P_halo_dm, P_halo_matter   per-bin, shape (nbins, nk)
#     P_matter_gas, P_matter_dm, P_matter_matter, P_dm_gas, P_dm_dm   truths, (nk,)
#     P_halo_x, P_matter_x_true, P_dm_x      the target-dependent aliases
# Nothing is called P_hg, P_me or P_dme any more: with three modes in play the
# one-letter abbreviations stopped being readable.

TRACER_OF_TARGET = {'matter_gas': 'gas',
                    'matter_dm': 'dm',
                    'matter_matter': 'matter'}

# Everything downstream keys off this table rather than off string surgery on
# the target name, so adding a fourth tracer is a single entry here plus a
# measurement in section 3.
#
#   halo_key     the per-bin halo cross spectrum that drives the reconstruction
#   truth_key    P(matter x x), what the reconstruction is scored against
#   dm_cross_key P(dm x x), the definitional-mismatch curve: the same
#                reconstruction run against a DM-only first field would target
#                this instead, so the gap between the two is the floor on
#                achievable agreement, not an error
#   sym          latex symbol for the tracer in plot labels
TRACER_INFO = {
    'gas': dict(tag='gas', sym=r'\rm gas',
                halo_key='P_halo_gas',
                truth_key='P_matter_gas',
                dm_cross_key='P_dm_gas'),
    'dm': dict(tag='dm', sym=r'\rm dm',
               halo_key='P_halo_dm',
               truth_key='P_matter_dm',
               dm_cross_key='P_dm_dm'),
    # P(dm x matter) is P(matter x dm) by symmetry, hence the shared key.
    'matter': dict(tag='matter', sym=r'\rm m',
                   halo_key='P_halo_matter',
                   truth_key='P_matter_matter',
                   dm_cross_key='P_matter_dm'),
}

# --- what plays the role of "matter" in the truth: not a choice ---------------
# The reconstruction's weights use M200b (TOTAL halo mass) and
# rhobar_m = Omega_m rho_crit (TOTAL matter density), so the right-hand side of
# the halo-model identity targets P(total matter x e) and nothing else:
#
#     delta_m = f_c delta_dm + f_g delta_gas (+ f_star delta_star)
#
# P(dm x e) is measured too and plotted as the definitional-mismatch curve, but
# it is not a competing definition of the truth -- comparing the reconstruction
# against it would just be scoring the identity for a mismatch that is built
# into the question. The mismatch grows at high k, where feedback makes gas
# smoother than DM.
#
# NOTE: star particles are in neither field, so delta_m as built here is still
# missing f_star ~ 1% of the matter, and those are the MOST clustered baryons.
# The residual inconsistency with rhobar_m (which does include stars) is at that
# level. Note also that P(m x x) contains an explicit f_x P_xx term, so unlike a
# pure cross-spectrum it is not entirely free of the tracer's shot noise -- with
# weight f_g ~ 0.16 for gas, f_c ~ 0.84 for DM, and 1 for matter, where the
# target is a genuine auto spectrum and nothing cancels at all. That self-pair
# term is measured and removed in pmxlib.self_pairs; what is left here is the
# genuine f_x P_xx CLUSTERING, which the identity does have to reproduce.


@dataclass(frozen=True)
class PmxConfig:
    """Everything the pipeline used to keep in module globals."""

    # --- simulation -----------------------------------------------------------
    sim_name: str = 'flamingo'
    feedback: str = 'strongest_AGN'
    ngrid: int = None            # None -> sim_params['ngrid_default']   (2048)
    nthread: int = None          # None -> sim_params['nthread_default']

    # --- halo binning ---------------------------------------------------------
    # Halos are binned by their virial mass. In the FLAMINGO catalogue the
    # closest available definition is m200b (M200 w.r.t. the mean background
    # density), which is also what the r200m <-> M conversion in u_nfw assumes.
    # Switching this to 'm200c' would make r200m_of_M inconsistent, hence the
    # guard in __post_init__.
    mass_def: str = 'm200b'
    logm_min: float = 11.0       # log10(M / [Msun/h])
    logm_max: float = 15.0
    nbins: int = 30              # log-spaced bins between logm_min and logm_max
    centrals_only: bool = True   # satellites repeat their host's m200b; counting
                                 # them would double-count halo mass in n_i M_i

    # Unit of the halo-mass array in the catalogue, expressed in Msun/h.
    # Particle masses in this repo are stored in 1e10 Msun/h; halo masses are
    # read straight out of /galaxies/m200b. If the sanity check in
    # pmxlib.binning fires, set this to 1e10.
    halo_mass_unit_msun_h: float = 1.0

    nkbins: int = 600            # None -> bin_power_spectrum_2d's default,
                                 # max(32, ngrid//10)

    # --- concentration --------------------------------------------------------
    concentration_source: str = 'powerlaw'    # 'powerlaw' | 'colossus'
    colossus_conc_model: str = 'diemer19'

    # --- self-pair (shot-noise) subtraction -----------------------------------
    # See pmxlib.self_pairs. The measurement is a Monte-Carlo one: each species
    # is painted at uniformly random positions with its true masses and the auto
    # spectrum of the resulting field is, in expectation, exactly the self-pair
    # term that the real field carries. The seed is fixed so that a cache is
    # reproducible, and it is the same seed measure_u_tilde.py uses.
    subtract_self_pairs: bool = True    # switched off by --no-shot-subtract
    shot_seed: int = 12345
    shot_nreal: int = 1          # random realisations averaged; more = less noise
    shot_chunk: float = 5e7      # particles per painting chunk (caps peak memory)

    # --- output switches ------------------------------------------------------
    # P(delta_dm x delta_gas) is drawn on both panels in every run, whatever the
    # target is. It is the cleanest single picture of baryon-vs-DM clustering in
    # the box, so having it fixed on the axes makes the gas-target and dm-target
    # plots directly comparable rather than each being self-referential. In
    # 'matter_gas' mode it IS the definitional-mismatch curve P(dm x e), so it
    # is drawn once and labelled as both rather than plotted twice.
    show_dmgas_reference: bool = True

    # sum_i n_i M_i < rhobar_m always: mass in halos below 10^logm_min and in
    # the diffuse IGM is not represented. The correction adds f_u * P_halo_x of
    # the LOWEST mass bin, i.e. it assumes the unresolved mass cross-correlates
    # with the tracer like the smallest resolved halos do. That is an
    # approximation and mildly over-corrects (unresolved mass is less biased
    # than 10^11 Msun/h halos), but it needs no linear theory and no bias model.
    # It is reported separately from the raw reconstruction so you can always
    # see how much it moved things.
    apply_missing_mass: bool = True

    target_mode: str = 'matter_gas'

    def __post_init__(self):
        if self.mass_def != 'm200b':
            raise ValueError(
                f"mass_def is {self.mass_def!r}, but r200m_of_M assumes a "
                f"mean-background (200m) definition. Either use 'm200b' or "
                f"change r200m_of_M to match.")
        if self.target_mode not in TRACER_OF_TARGET:
            raise ValueError(f"unknown target_mode {self.target_mode!r}")

    # --- derived, read-only ---------------------------------------------------
    @property
    def sim_params(self):
        return get_sim_params(self.sim_name)

    @property
    def box(self):
        return self.sim_params['box_size_cMpc_h']      # 681.0 cMpc/h

    @property
    def grid(self):
        return self.ngrid if self.ngrid else self.sim_params['ngrid_default']

    @property
    def threads(self):
        return self.nthread if self.nthread else self.sim_params['nthread_default']

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
    def rhobar_m(self):
        """Comoving mean matter density in (Msun/h)/(Mpc/h)^3."""
        from Pmx_reconstruction.pmxlib.cosmology import mean_matter_density
        return mean_matter_density(self.sim_params)

    @property
    def tracer(self):
        return TRACER_OF_TARGET[self.target_mode]

    @property
    def tracer_info(self):
        return TRACER_INFO[self.tracer]

    @classmethod
    def from_args(cls, args, **overrides):
        """Build from an argparse Namespace, taking only the fields it carries."""
        names = {f.name for f in dataclasses.fields(cls)}
        kw = {k: v for k, v in vars(args).items() if k in names and v is not None}
        kw.update(overrides)
        return cls(**kw)


# ==============================================================================
# Shared argparse fragments, so the three CLIs cannot drift
# ==============================================================================
_D = PmxConfig()


def add_target_args(ap):
    ap.add_argument('--target', dest='target_mode', default=_D.target_mode,
                    choices=list(TRACER_OF_TARGET),
                    help="which spectrum to reconstruct: matter_gas (default), "
                         "matter_dm, or matter_matter. All three halo cross "
                         "spectra live in the same bundle, so switching is free.")


def add_binning_args(ap):
    g = ap.add_argument_group('Halo binning')
    g.add_argument('--nbins', type=int, default=_D.nbins,
                   help=f"number of log-spaced halo mass bins (default {_D.nbins})")
    g.add_argument('--logm-min', dest='logm_min', type=float, default=_D.logm_min,
                   help=f"log10(M/[Msun/h]) of the lowest bin edge "
                        f"(default {_D.logm_min})")
    g.add_argument('--logm-max', dest='logm_max', type=float, default=_D.logm_max,
                   help=f"log10(M/[Msun/h]) of the highest bin edge "
                        f"(default {_D.logm_max})")
    g.add_argument('--nkbins', type=int, default=_D.nkbins,
                   help="number of k bins in the measured spectra "
                        "(default 600; None -> max(32, ngrid//10))")
    g.add_argument('--ngrid', type=int, default=None,
                   help="FFT grid size (default: the simulation's ngrid_default)")
    g.add_argument('--nthread', type=int, default=None,
                   help="threads for painting and FFTs (default: nthread_default)")
    return g


def add_concentration_args(ap):
    g = ap.add_argument_group('Concentration-mass relation')
    g.add_argument('--concentration', dest='concentration_source',
                   default=_D.concentration_source,
                   choices=['powerlaw', 'colossus'],
                   help="c(M,z) used in u_m. 'powerlaw' (default) is the "
                        "built-in fit and is kept as the default so that "
                        "upgrading this file does not silently move existing "
                        "plots. 'colossus' swaps in a published relation and is "
                        "the better choice for new work, especially in "
                        "matter_dm / matter_matter modes where the high-k ratio "
                        "is essentially u_model/u_true.")
    g.add_argument('--concentration-model', dest='colossus_conc_model',
                   default=_D.colossus_conc_model,
                   help=f"colossus c(M,z) model name, used only with "
                        f"--concentration colossus (default "
                        f"{_D.colossus_conc_model})")
    return g


def add_selfpair_args(ap):
    g = ap.add_argument_group('Self-pair (shot-noise) subtraction')
    g.add_argument('--no-shot-subtract', dest='subtract_self_pairs',
                   action='store_false', default=_D.subtract_self_pairs,
                   help="do not subtract the measured self-pair term from the "
                        "truths. The raw and corrected variants are both stored "
                        "in the bundle either way; this only picks which one "
                        "the reconstruction is scored against.")
    g.add_argument('--shot-seed', type=int, default=_D.shot_seed,
                   help=f"RNG seed for the uniform-random realisations "
                        f"(default {_D.shot_seed}). It is part of the cache "
                        f"filename, and it is the same seed "
                        f"measure_u_tilde.py uses.")
    g.add_argument('--shot-nreal', type=int, default=_D.shot_nreal,
                   help="random realisations averaged (default 1). The residual "
                        "scatter falls as 1/sqrt(n) but only matters at low k, "
                        "where the self-pair term is negligible anyway.")
    g.add_argument('--shot-chunk', type=float, default=_D.shot_chunk,
                   help=f"particles per painting chunk, caps peak memory "
                        f"(default {_D.shot_chunk:g})")
    return g


# ==============================================================================
# Experiment A options
# ==============================================================================
# Kept here rather than in experiment_A.py so that predict_Pmx_from_Phx.py can
# build the --experiment A argument group without importing experiment_A, which
# would pull in CAMB and the colossus mass-function and bias modules on every
# plain run. run_experiment_A itself is still imported inside the branch.

# Integration range for the unresolved-mass integral, log10(M / [Msun/h]).
INT_LOGM_LO = 8.0             # see the f_u instability note above
INT_NODES = 256


@dataclass(frozen=True)
class ExperimentAOptions:
    """The Experiment A knobs, kept off PmxConfig on purpose.

    --concentration and --concentration-model are deliberately NOT here: they
    live on PmxConfig, because the baseline reconstruction uses u_m too and both
    scripts have to agree on c(M,z).
    """
    extrap: tuple = ('flat', 'bias', 'halomodel')
    split_logm: float = None
    ref_logm: float = None
    hmf: str = 'tinker'            # 'tinker' | 'catalog'
    fu: str = 'catalog'            # 'catalog' | 'model'
    int_logm_min: float = None
    power_spectrum: str = 'camb'   # 'camb' | 'eisenstein98'
    compute_rstar: bool = True

    @classmethod
    def from_args(cls, args, **overrides):
        names = {f.name for f in fields(cls)}
        kw = {k: v for k, v in vars(args).items() if k in names and v is not None}
        kw.update(overrides)
        if 'extrap' in kw:
            kw['extrap'] = tuple(kw['extrap'])
        return cls(**kw)


def add_experiment_a_args(ap):
    """The Experiment A argument group, shared by both entry points."""
    g = ap.add_argument_group('Experiment A')
    g.add_argument('--extrap', nargs='+', default=list(ExperimentAOptions.extrap),
                   choices=['flat', 'simhc', 'bias', 'halomodel'],
                   help="which extrapolations of P_halo_x(k|M) below M_min to "
                        "run (default: flat bias halomodel)")
    g.add_argument('--split-logm', dest='split_logm', type=float, default=None,
                   help="VALIDATION mode: declare bins above this log10(M) "
                        "resolved and hide the rest, so the exact contribution "
                        "of the hidden bins is known and every extrapolation "
                        "can be scored against a truth. Omit for PRODUCTION "
                        "mode, where there is no exact target.")
    g.add_argument('--ref-logm', dest='ref_logm', type=float, default=None,
                   help="log10(M) of the reference bin M_r whose P_halo_x is "
                        "the template (default: the lowest resolved bin)")
    g.add_argument('--hmf', default=ExperimentAOptions.hmf,
                   choices=['tinker', 'catalog'],
                   help="mass function used in the unresolved integral: "
                        "'tinker' (default) or 'catalog', a power-law fit to "
                        "the measured counts")
    g.add_argument('--fu', default=ExperimentAOptions.fu,
                   choices=['catalog', 'model'],
                   help="where the amplitude f_u comes from: 'catalog' "
                        "(default) takes it from the measured deficit "
                        "1 - sum_i f_i and the shape from the model; 'model' "
                        "uses the HMF for both and exposes the amplitude "
                        "instability described above")
    g.add_argument('--no-rstar', dest='compute_rstar', action='store_false',
                   default=True,
                   help="PRODUCTION mode: do not measure R*/U* if it is not "
                        "already cached. The (R+U)/(R*+U*) panel is dropped "
                        "and the two separate figures are written instead. "
                        "Build the cache with "
                        "python -m Pmx_reconstruction.pmxlib.rstar_ustar")
    g.add_argument('--int-logm-min', dest='int_logm_min', type=float, default=None,
                   help=f"lower limit of the unresolved-mass integral "
                        f"(default {INT_LOGM_LO}); see the f_u instability note")
    g.add_argument('--power-spectrum', dest='power_spectrum',
                   default=ExperimentAOptions.power_spectrum,
                   choices=['camb', 'eisenstein98'],
                   help="linear P(k) behind the halo model (default camb)")
    return g