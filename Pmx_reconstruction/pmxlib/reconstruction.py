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
"""
import numpy as np

from Pmx_reconstruction.pmxlib.nfw import conc_of, u_nfw


def compute_R(cfg, k, P_halo_gas, M_i, n_i, z, trunc=1.0):
    """R(k) = (1/rhobar_m) sum_i n_i M_i u_m(k|M_i) P_halo_gas(k; M_i).

    Parameters
    ----------
    k          : (nk,)          wavenumbers [h/cMpc]
    P_halo_gas : (nbins, nk)    measured halo-gas cross spectra, one row per bin
    M_i        : (nbins,)       representative halo mass per bin [Msun/h]
    n_i        : (nbins,)       comoving number density per bin [(cMpc/h)^-3];
                                zero it outside [M_r, M_max] to restrict the sum
    z          : scalar         redshift, for the concentration relation
    trunc      : scalar         outer edge of the NFW profile in units of r200m.
                                Leave at 1 unless the weight moves with it: the
                                n_i M_i here is the catalogue M200b, while u_m
                                is normalised to the mass inside trunc * r200m,
                                so the two are a matched pair. experiment_B
                                moves them together and builds its own R.

    Returns
    -------
    R : (nk,)
    """
    c_i = conc_of(cfg, M_i, z)
    u_im = np.array([u_nfw(k, M_i[i], c_i[i], cfg.rhobar_m, trunc=trunc)
                     for i in range(M_i.size)])
    weight = (n_i * M_i)[:, None] / cfg.rhobar_m          # (nbins, 1)
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
              n_nodes=None, f_u=None, use_um=True, trunc=1.0):
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
        u = np.array([u_nfw(k, m, conc_of(cfg, m, z), cfg.rhobar_m, trunc=trunc)
                      for m in M_nodes])
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
