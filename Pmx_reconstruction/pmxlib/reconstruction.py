#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The reconstruction itself: R(k) over the resolved bins, U(k) for the rest.

    P_matter_gas(k) ~ R(k) + U(k)

    R(k) = sum_i f_i u_m(k|M_i) P_halo_gas(k; M_i),     f_i = n_i M_i / rhobar_m
    U(k) = f_u <u_m(k|M) T(k,M)>_w P_halo_gas(k|M_r)
         = f_u S(k) P_halo_gas(k|M_r)

R is the halo-model identity evaluated on the bins the catalogue resolves, with
one modelled ingredient in it -- u_m, from NFW and a c(M,z) relation. Every
P_halo_gas is measured, so the true gas profile is carried whatever it happens
to be. U stands in for everything else: halos below M_r, and matter outside the
aperture of a resolved halo.

WHERE u_m COMES FROM
    Profile decides, once, for both R and U. Under --profile nfw it is the
    model; under --profile measured it is the stacked profile from
    pmxlib.u_bar, with the model kept only below the mass where that stack
    stops being measured.

WHICH WEIGHTS GO WITH IT
    reconstruction_weights pairs them. u_bar is normalised to the mass actually
    assigned inside the aperture, so a measured profile takes f_part and
    f_out; the model profile takes the catalogue's n_i M_i / rhobar_m and its
    deficit. Mixing them is silently wrong rather than loudly wrong -- the
    0.255-versus-0.249 gap would simply sit inside the profile error looking
    like physics -- which is why the pairing lives in one function that both
    experiments call instead of at each of the four sites that needs it.
"""
import numpy as np

from Pmx_reconstruction.pmxlib.nfw import conc_of, u_nfw

__all__ = ['Profile', 'reconstruction_weights', 'compute_R', 'transfer_T',
           'compute_U']


class Profile:
    """u_m(k|M), modelled or measured, with the aperture it belongs to.

    Build it with from_config and pass it around; calling u() is then the only
    thing R, U and the validation truth ever do about profiles.
    """

    def __init__(self, cfg, z, cache=None, aperture=1.0):
        self.cfg, self.z = cfg, z
        self.cache = cache
        self.aperture = float(aperture)
        self.source = 'nfw' if cache is None else 'measured'
        self.range = None
        if cache is not None:
            from Pmx_reconstruction.pmxlib.u_bar import measured_range
            self.range = measured_range(cache)

    @classmethod
    def from_config(cls, cfg, z, aperture=1.0, allow_compute=False):
        """The profile this run asked for. u_bar is imported only if needed."""
        if cfg.profile_source == 'nfw':
            return cls(cfg, z, aperture=aperture)
        from Pmx_reconstruction.pmxlib import u_bar as ub
        cache = ub.load_or_measure_u_bar(cfg, aperture=aperture,
                                         allow_compute=allow_compute)
        if cache is None:
            raise SystemExit(
                f"--profile measured needs a stacked-profile cache at "
                f"aperture {aperture:g}:\n  {ub.cache_path(cfg, aperture)}\n"
                f"Build it with\n      python -m "
                f"Pmx_reconstruction.pmxlib.u_bar --aperture {aperture:g}")
        got = float(cache['aperture'])
        if abs(got - float(aperture)) > 1e-9:
            raise SystemExit(f"the u_bar cache was measured at aperture "
                             f"{got:g}, not {aperture:g}")
        obj = cls(cfg, z, cache=cache, aperture=aperture)
        print(f"[profile] measured u_bar, aperture {aperture:g}, nodes over "
              f"logM {obj.range[0]:.2f}..{obj.range[1]:.2f}; the NFW model "
              f"covers anything outside that")
        return obj

    @property
    def tag(self):
        """Filename/label token. Empty for the default, so old stems survive."""
        return '' if self.source == 'nfw' else '_prof-measured'

    def u(self, k, M, trunc=1.0):
        """u_m(k|M), shape (nM, nk). M may be any array of masses.

        In measured mode `trunc` is not free: u_bar is normalised to the mass
        inside the aperture the cache was measured at, so asking for a
        different truncation is asking for a profile that was never measured.
        """
        k = np.atleast_1d(np.asarray(k, dtype=float))
        M = np.atleast_1d(np.asarray(M, dtype=float))
        if self.source == 'measured' and abs(float(trunc) - self.aperture) > 1e-9:
            raise ValueError(
                f"measured u_bar is normalised to the mass inside "
                f"{self.aperture:g} r200m; trunc={trunc:g} was asked for. "
                f"Measure a u_bar cache at that aperture, or use --profile nfw.")
        u_model = np.array([u_nfw(k, m, conc_of(self.cfg, m, self.z),
                                  self.cfg.rhobar_m, trunc=trunc)
                            if m > 0 else np.ones(k.size) for m in M])
        if self.source == 'nfw':
            return u_model
        from Pmx_reconstruction.pmxlib.u_bar import u_bar_interp
        u_meas, inside = u_bar_interp(self.cache, k, M)
        return np.where(inside[:, None], u_meas, u_model)


def reconstruction_weights(cfg, mr, tot=None):
    """(w_i for R, f_u for U) matched to the run's profile source.

    'nfw'       the catalogue's n_i M_i / rhobar_m, masked to the resolved
                bins, and f_u = 1 - (resolved + above).
    'measured'  the mass actually assigned inside the aperture, f_part, and
                f_out = 1 - sum of it. These are the masses u_bar is
                normalised to, so R + U closes on the mass budget exactly --
                which the f_i pairing does not (0.255 against 0.249).

    `tot` is pmxlib.rstar_ustar.measured_partition's dict, already split at
    M_r, and is required in measured mode: it is where f_part lives on the
    BUNDLE's mass grid. The u_bar cache has its own grid and cannot supply it.
    """
    if cfg.profile_source == 'nfw':
        return np.where(mr.resolved, mr.f_i, 0.0), mr.f_u
    if tot is None:
        raise SystemExit(
            "--profile measured needs the R*/U* cache: the profile is "
            "normalised to the assigned mass, so R must be weighted by "
            "f_part rather than n_i M_i / rhobar_m, and f_part is measured "
            "there. Build it with\n"
            "      python -m Pmx_reconstruction.pmxlib.rstar_ustar")
    return np.asarray(tot['f_part'], dtype=float), float(tot['f_out'])


def compute_R(cfg, k, P_halo_gas, M_i, n_i, z, trunc=1.0, profile=None,
              weight=None):
    """R(k) = sum_i w_i u_m(k|M_i) P_halo_gas(k; M_i).

    Parameters
    ----------
    k          : (nk,)          wavenumbers [h/cMpc]
    P_halo_gas : (nbins, nk)    measured halo-gas cross spectra, one row per bin
    M_i        : (nbins,)       representative halo mass per bin [Msun/h]
    n_i        : (nbins,)       comoving number density per bin [(cMpc/h)^-3];
                                zero it outside [M_r, M_max] to restrict the
                                sum. Ignored when `weight` is given.
    z          : scalar         redshift, for the concentration relation
    trunc      : scalar         outer edge of the profile in units of r200m.
                                Leave at 1 unless the weight moves with it: the
                                default n_i M_i is the catalogue M200b, while
                                u_m is normalised to the mass inside
                                trunc * r200m, so the two are a matched pair.
                                experiment_B moves them together.
    profile    : Profile        where u_m comes from. Default: the NFW model,
                                which is what this was before there was a
                                choice.
    weight     : (nbins,)       w_i, already masked to the bins R sums over.
                                Default n_i M_i / rhobar_m. Pass what
                                reconstruction_weights returns rather than
                                choosing here: a measured profile needs the
                                measured mass, and nothing downstream would
                                notice if it did not get it.

    Returns
    -------
    R : (nk,)
    """
    profile = Profile(cfg, z, aperture=trunc) if profile is None else profile
    u_im = profile.u(k, M_i, trunc=trunc)
    if weight is None:
        weight = np.asarray(n_i, dtype=float) * np.asarray(M_i, float) / cfg.rhobar_m
    weight = np.asarray(weight, dtype=float)[:, None]     # (nbins, 1)
    return np.sum(weight * u_im * P_halo_gas, axis=0)     # sum over mass bins


def transfer_T(k, M_nodes, M_ref, mode, hm=None, T_sim=None):
    """Mass transfer function T(k,M) = P_hc(k|M) / P_hc(k|M_r), shape (nM, nk).

    'flat'      -> 1
    'bias'      -> b(M)/b(M_r), the 2-halo limit
    'halomodel' -> full 1-halo + 2-halo ratio
    'simhc'     -> T_sim, supplied by the caller from measured P_halo_dm
    """
    nM, nk = M_nodes.size, k.size
    if mode == 'flat':
        return np.ones((nM, nk))
    if mode == 'simhc':
        if T_sim is None:
            raise ValueError("mode 'simhc' needs measured P_halo_dm ratios "
                             "(validation mode only)")
        return T_sim
    if hm is None:
        raise ValueError(f"mode {mode!r} needs a HaloModel instance")
    if mode == 'bias':
        return (hm.bias(M_nodes) / float(hm.bias(np.array([M_ref]))[0]))[:, None] \
            * np.ones((1, nk))
    if mode == 'halomodel':
        return hm.P_hc(k, M_nodes) / hm.P_hc(k, np.array([M_ref]))[0][None, :]
    raise ValueError(f"unknown extrapolation mode {mode!r}")


def compute_U(cfg, k, P_halo_gas_ref, M_ref, M_u_hi, mode, int_logm_lo,
              hm=None, z=None, cat_M=None, cat_w=None, T_sim=None,
              n_nodes=None, f_u=None, use_um=True, trunc=1.0, profile=None):
    """U(k) = f_u * S(k) * P_halo_gas(k|M_r),   S(k) = <u_m(k|M) T(k,M)>_w

    Two ways of supplying the mass weight w = M n(M):

    cat_M / cat_w given
        The nodes and weights come from the halo CATALOGUE (w_i = n_i M_i).
        Used in validation mode, where the pseudo-unresolved bins are really
        measured, so that n(M) is exact and only T is under test.

    cat_M / cat_w absent
        Nodes are log-spaced on [10^int_logm_lo, M_u_hi], n_nodes of them
        (INT_NODES if not given), and w = M n(M) comes from Tinker+08.

    f_u given
        Overrides the model's own mass integral, i.e. the amplitude is taken
        from the catalogue deficit and only the SHAPE S(k) from the model.

    trunc
        Multiple of r200m at which u_m is truncated (experiment_B varies it).

    profile
        Where u_m comes from, the same object R was given. Default: the NFW
        model. A measured profile only has nodes down to the catalogue floor,
        so below that this still falls back to the model -- which costs
        little, because u_m -> 1 for those halos at every k the box resolves.

    Returns a dict with S, U, the f_u used, the f_u the integral would have
    given, and the per-node T for inspection.
    """
    from Pmx_reconstruction.pmxlib.config import INT_NODES
    n_nodes = INT_NODES if n_nodes is None else n_nodes

    if mode == 'flat' and cat_M is None and hm is None:
        if f_u is None:
            raise ValueError("mode 'flat' without a HaloModel needs f_u to be "
                             "given: the mass integral is the only thing the "
                             "model was supplying.")
        S = np.ones_like(np.asarray(k, dtype=float))
        return dict(S=S, U=float(f_u) * S * P_halo_gas_ref,
                    f_u_used=float(f_u), f_smallhalo_integral=np.nan,
                    M_nodes=np.array([]), w=np.array([]), T=None, mode=mode)

    if cat_M is not None:
        M_nodes = np.asarray(cat_M, dtype=float)
        w = np.asarray(cat_w, dtype=float)
    else:
        M_nodes = np.logspace(int_logm_lo, np.log10(M_u_hi), n_nodes)
        lnM = np.log(M_nodes)
        w = M_nodes * hm.dndM(M_nodes) * M_nodes * _trapz_weights(lnM)

    keep = w > 0
    M_nodes, w = M_nodes[keep], w[keep]
    if T_sim is not None:
        T_sim = np.asarray(T_sim)[keep]

    T = transfer_T(k, M_nodes, M_ref, mode, hm=hm, T_sim=T_sim)

    if use_um and mode != 'flat':
        profile = Profile(cfg, z, aperture=trunc) if profile is None else profile
        u = profile.u(k, M_nodes, trunc=trunc)
    else:
        u = np.ones((M_nodes.size, k.size))

    S = np.sum(w[:, None] * u * T, axis=0) / np.sum(w)
    f_smallhalo_int = float(np.sum(w) / cfg.rhobar_m)
    f_u_used = f_smallhalo_int if f_u is None else float(f_u)

    return dict(S=S, U=f_u_used * S * P_halo_gas_ref, f_u_used=f_u_used,
                f_smallhalo_integral=f_smallhalo_int, M_nodes=M_nodes, w=w,
                T=T, mode=mode)


def _trapz_weights(x):
    """Weights w such that sum(w * f) == trapezoid(f, x).

    The U integral needs the weights themselves, not just the integral: they
    are the w of <.>_w, and sum(w)/rhobar_m is f_(i).
    """
    w = np.empty_like(x)
    w[1:-1] = 0.5 * (x[2:] - x[:-2])
    w[0] = 0.5 * (x[1] - x[0])
    w[-1] = 0.5 * (x[-1] - x[-2])
    return w
