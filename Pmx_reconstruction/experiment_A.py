#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""==============================================================================
EXPERIMENT A  --  extrapolating P_halo_gas(k|M) below M_r
==============================================================================
This code computes the unresolved cross-power spectrum U:

U(k) = (1/rhobar_m) int_0^M_r dM M n(M) u_m(k|M) P_halo_gas(k|M),

with the unmeasurable integrand P_halo_gas(k|M) supplied by an extrapolation off a
reference bin h_r -- the M_r bin, the lowest one R sums over:

P_halo_gas(k|M)  ~  [P_h_r_e(k) / P_h_r_c(k)] * P_hc(k|M).

c is cold dark matter, the bracket freezes the baryonic response in at the
reference scale.

Define the mass transfer function
T(k,M) = P_hc(k|M) / P_hc(k|M_r)
so that P_halo_gas(k|M) = T(k,M) * P_halo_gas(k|M_r)

U(k) = f_u * <u_m(k|M) T(k,M)>_w * P_halo_gas(k|M_r)
= f_u * S(k) * P_halo_gas(k|M_r),

THE FOUR WAYS OF SUPPLYING T
'flat'      T = 1 and u_m = 1.
'simhc'     T = P_halo_dm(k;M_i) / P_halo_dm(k;M_r), straight from the measured
            halo-DM cross spectra. Only available in validation mode (for now).
'bias'      T = b(M)/b(M_r) from Tinker+10, u_m kept. The 2-halo limit.
'halomodel' T from the full 1-halo + 2-halo model.

f_(i) from an analytic HMF is badly determined: integrating Tinker+08 from
10^10 rather than 10^6 Msun/h changes it by a factor of two, because the
low-mass end contributes mass logarithmically and never converges. The
SHAPE S(k), by contrast, moves by only a few per cent over the same range,
since it is a ratio of two integrals against the same weight. So the
default (--fu catalog) takes the amplitude from the measured catalogue
deficit f_u = 1 - sum_i f_i and the shape from the model.

Template mass is <M> of bin r; the integral stops at its lower edge.

Note that f_u is NOT f_(i), the integral below M_r (f_smallhalo in code):
f_u also collects mass outside r200b of resolved halos. This code improves
the treatment of component (i) only.
------------------------------------------------------------------------------
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')

from utils.pipeline_paths import ensure_parents, plot_path
from utils.plot_data import save_plot_data

from Pmx_reconstruction.pmxlib.config import (MassRange, PmxConfig,
                                              base_parser)
from Pmx_reconstruction.pmxlib.halo_model import HaloModel
from Pmx_reconstruction.pmxlib import plotting as pl
from Pmx_reconstruction.pmxlib import rstar_ustar as ru
from Pmx_reconstruction.pmxlib.nfw import conc_of, u_nfw
from Pmx_reconstruction.pmxlib.reconstruction import compute_R, compute_U

def _lowk(k, y, kmax=0.08):
    """Mean of y over k < kmax, for one-line reporting."""
    sel = np.asarray(k) < kmax
    if np.count_nonzero(sel) < 3:
        sel = np.zeros_like(k, dtype=bool)
        sel[:5] = True
    y = np.asarray(y, dtype=float)[sel]
    return float(np.nanmean(y[np.isfinite(y)]))


def measured_bias_per_bin(data, kmax_fit=0.08):
    """
    Large-scale halo bias per mass bin, b_i = <P_halo_dm / P_dm,dm> over k < kmax.
    """
    k = data['k_center']
    sel = k < kmax_fit
    if np.count_nonzero(sel) < 3:
        sel = np.zeros_like(k, dtype=bool)
        sel[:5] = True
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = data['P_halo_dm'][:, sel] / data['P_dm_dm'][None, sel]
    b = np.nanmean(np.where(np.isfinite(ratio), ratio, np.nan), axis=1)
    return b, float(k[sel][-1])


def run_experiment_A(cfg, data):
    """
    Run Experiment A. R sums the bins in [M_r, M_max]; U models the mass
    below M_r.

    VALIDATION (--validate): the bins masked below M_r play the unresolved
        population; U is integrated from the bundle edge and scored against
        U_exact = sum_hidden f_i u_m(k|M_i) P_halo_gas(k;M_i).
    PRODUCTION: U is integrated from 10^INT_LOGM_LO and the reconstruction is
        scored against the measured R*/U*, re-split at M_r.
    """
    k = data['k_center']
    P_halo_gas = data['P_halo_gas']
    P_matter_gas_true = data['P_matter_gas']
    P_halo_dm = data['P_halo_dm']
    counts = data['counts']
    M_i = data['M_mean']
    logM_cen = data['logM_cen']
    z = float(data['redshift'])
    k_Ny = float(data['k_Nyquist'])
    n_i = counts / float(data['box']) ** 3

    print("\n" + "=" * 70)
    print("EXPERIMENT A -- extrapolating P_halo_gas(k|M) below M_r")
    print("=" * 70)

    mr = MassRange.from_config(cfg, data)
    mr.report('[A]')
    mode_label = 'validation' if mr.validate else 'production'
    occupied = counts > 0
    hidden = mr.hidden
    ref = mr.r
    P_halo_gas_ref = P_halo_gas[ref]

    # --- the exact target, validation only --------------------------------------
    U_exact = None
    f_u_hidden = None
    if mr.validate:
        idx = np.flatnonzero(hidden)
        u_hidden = np.array([u_nfw(k, M_i[j], conc_of(cfg, M_i[j], z),
                                   cfg.rhobar_m)
                             for j in idx])
        U_exact = np.sum(mr.f_i[idx][:, None] * u_hidden * P_halo_gas[idx], axis=0)
        f_u_hidden = mr.f_hidden
        print(f"[A] exact masked-bin contribution built; "
              f"f_u(hidden) = {f_u_hidden:.4f}")

    # --- halo model, only if a mode needs it -----------------------------------
    modes = list(cfg.extrap)
    if 'simhc' in modes and not mr.validate:
        print("[A] dropping mode 'simhc': it needs --validate.")
        modes.remove('simhc')
    # 'flat' needs it too whenever the mass integral is analytic.
    needs_hm = (any(m in ('bias', 'halomodel') for m in modes)
                or (cfg.hmf == 'tinker' and any(m != 'simhc' for m in modes)))
    hm = None
    if needs_hm:
        print("[A] building the halo model ...")
        hm = HaloModel(cfg, z, power_spectrum=cfg.power_spectrum)

    # --- bias sanity check ------------------------------------------------------
    b_meas, k_fit = measured_bias_per_bin(data)
    if hm is not None:
        print(f"[A] measured vs Tinker+10 bias (from P_halo_dm/P_dm,dm, k < {k_fit:.3f}):")
        for j in np.flatnonzero(occupied)[::max(1, np.count_nonzero(occupied) // 8)]:
            print(f"      logM={logM_cen[j]:5.2f}  b_meas={b_meas[j]:6.3f}  "
                  f"b_T10={float(hm.bias(np.array([M_i[j]]))[0]):6.3f}")

    # --- the unresolved mass fraction ----------------------------------------------------------
    if cfg.fu == 'catalog':
        f_u_amp = f_u_hidden if mr.validate else mr.f_u
    else:
        f_u_amp = None
    print(f"[A] f_u source: --fu {cfg.fu}"
          + (f" -> f_u = {f_u_amp:.4f}" if f_u_amp is not None else
             " -> from the Tinker+08 integral, per mode"))

    # --- catalogue weights and T_sim, validation only ---------------------------
    T_sim = None
    if cfg.hmf == 'catalog' and not mr.validate:
        raise SystemExit("--hmf catalog needs --validate: there is no "
                         "catalogue below M_r otherwise.")
    if mr.validate:
        idx = np.flatnonzero(hidden)
        with np.errstate(divide='ignore', invalid='ignore'):
            T_sim_full = P_halo_dm[idx] / P_halo_dm[ref][None, :]
        T_sim = np.where(np.isfinite(T_sim_full), T_sim_full, 0.0)

    # --- run every requested mode ----------------------------------------------
    results = {}
    for mode in modes:
        use_cat = (cfg.hmf == 'catalog') or (mode == 'simhc')
        if use_cat:
            idx = np.flatnonzero(hidden)
            cm, cw = M_i[idx], n_i[idx] * M_i[idx]
        else:
            cm = cw = None
        res = compute_U(
            cfg, k, P_halo_gas_ref, mr.M_ref, mr.M_u_hi, mode, mr.int_logm_lo,
            hm=hm, z=z, cat_M=cm, cat_w=cw,
            T_sim=T_sim if mode == 'simhc' else None, f_u=f_u_amp,
        )
        results[mode] = res
        S = res['S']
        print(f"[A] mode {mode:10s}: f_(i) = {res['f_smallhalo_integral']:.4f}, "
              f"f_u(used) = {res['f_u_used']:.4f}, "
              f"S(k_min) = {S[0]:.3f}, S(k=1) = {np.interp(1.0, k, S):.3f}, "
              f"S(k_Ny) = {S[-1]:.3g}")
        if U_exact is not None:
            with np.errstate(divide='ignore', invalid='ignore'):
                r = res['U'] / U_exact
            good = np.isfinite(r) & (k < k_Ny)
            print(f"                 U/U_exact: {r[good][0]:.3f} at k_min, "
                  f"{np.interp(1.0, k, r):.3f} at k=1, "
                  f"median |1-r| = {np.nanmedian(np.abs(r[good] - 1.0)):.3f}")

    # --- the bias/halomodel spread ----------------------------------------------
    if 'bias' in results and 'halomodel' in results:
        with np.errstate(divide='ignore', invalid='ignore'):
            spread = results['halomodel']['S'] / results['bias']['S'] - 1.0
        print("[A] spread halomodel/bias - 1 (size of the dropped 1-halo term):")
        for kk in (0.1, 0.3, 1.0, 3.0):
            if kk < k[-1]:
                print(f"      k = {kk:4.1f}: {np.interp(kk, k, spread):+.3f}")
        big = k[np.abs(spread) > 0.5]
        if big.size:
            print(f"      |spread| exceeds 50% above k = {big[0]:.2f}; only "
                  f"--validate can decide there.")

    # --- R and the target -------------------------------------------------------
    n_use = np.where(mr.resolved, n_i, 0.0)
    R = compute_R(cfg, k, P_halo_gas, M_i, n_use, z)

    P_target = (R + U_exact) if mr.validate else P_matter_gas_true

    S_exact = None
    if mr.validate and f_u_hidden:
        with np.errstate(divide='ignore', invalid='ignore'):
            S_exact = U_exact / (f_u_hidden * P_halo_gas_ref)

    # measured partition, re-split at M_r / M_max
    tot, _ = ru.load_totals(cfg, data, mr=mr,
                            allow_compute=not mr.validate)
    if tot is None and not mr.validate:
        raise SystemExit(
            "[A] cannot score the reconstruction against R*/U*: the cache "
            "sits on a different k grid from the bundle. Rebuild it with\n"
            "      python -m Pmx_reconstruction.pmxlib.rstar_ustar --recompute")

    def _save(fig, kind):
        """Save one of this run's figures, under a stem naming the whole run.

        Everything that would make two runs differ has to appear here, or the
        second would overwrite the first. `conc` is empty at the default so
        that adding a profile switch does not rename every existing plot.
        """
        # Every model choice that moves the answer has to be in the name, or
        # two runs overwrite each other: c(M,z) enters u_m, and the linear
        # P(k) enters T(k,M) through the halo model. Empty at the defaults,
        # so turning a knob does not rename every existing plot.
        models = ''
        if cfg.concentration_source != PmxConfig.concentration_source:
            models += f'_conc-{cfg.concentration_source}'
        if cfg.colossus_conc_model != PmxConfig.colossus_conc_model:
            models += f'_cm-{cfg.colossus_conc_model}'
        if cfg.power_spectrum != PmxConfig.power_spectrum:
            models += f'_ps-{cfg.power_spectrum}'
        stem = (f'expA_{kind}_gas_{cfg.mass_def}_nb{cfg.nbins}{mr.tag()}'
                f'_hmf{cfg.hmf}_fu{cfg.fu}{models}')
        path = plot_path('pme_reconstruction', cfg.feedback, stem=stem)
        ensure_parents(path)
        pl.save_figure(fig, path)
        print(f"[A][plot] {path}")
        return path

    d = pl.PanelData(k=k, k_Ny=k_Ny, results=results,
                     R=R, P_target=P_target,
                     P_matter_gas_true=P_matter_gas_true,
                     U_exact=U_exact, S_exact=S_exact,
                     P_dm_gas=data['P_dm_gas'],
                     P_selfpair=data['P_matter_gas_selfpair'])
    if tot is not None:
        d.R_star, d.U_star = tot['R_star'], tot['U_star']
        if mr.above.any():
            d.V_star = tot['V_star']
        with np.errstate(divide='ignore', invalid='ignore'):
            d.S_eff = tot['U_star'] / (tot['f_out'] * P_halo_gas_ref)
        print(f"[A] measured S_eff: {_lowk(k, d.S_eff):.3f} at k -> 0, "
              + ", ".join(f"{kk:g}: {float(np.interp(kk, k, d.S_eff)):.3f}"
                          for kk in (1.0, 3.0) if kk <= k[-1]))

    if mr.validate:
        p_main = _save(pl.validation_panel(d), 'validation_panel')
    else:
        p_main = _save(pl.production_panel(d), 'production_panel')

    # --- save everything --------------------------------------------------------
    payload = {
        'k_center': k, 'mode': mode_label,
        'ref_logM': float(logM_cen[ref]),
        'hmf_source': cfg.hmf, 'fu_source': cfg.fu,
        'concentration_source': cfg.concentration_source,
        'f_u': mr.f_u, 'f_resolved': mr.f_resolved,
        'b_measured': b_meas, 'logM_cen': logM_cen, 'M_mean': M_i, 'n_i': n_i,
        'P_halo_gas_ref': P_halo_gas_ref, 'R': R, 'P_matter_gas_true': P_matter_gas_true,
    }
    payload.update(mr.payload())
    if U_exact is not None:
        payload['U_exact'] = U_exact
        payload['S_exact'] = S_exact
        payload['f_u_hidden'] = f_u_hidden
    if tot is not None:
        payload['R_star'] = d.R_star
        payload['U_star'] = d.U_star
        payload['V_star'] = tot['V_star']
        payload['f_out'] = tot['f_out']
        payload['S_eff'] = d.S_eff
    for mode, res in results.items():
        payload[f'S_{mode}'] = res['S']
        payload[f'U_{mode}'] = res['U']
        payload[f'f_smallhalo_integral_{mode}'] = res['f_smallhalo_integral']
    if hm is not None:
        payload['b_tinker10'] = hm.bias(M_i)
        payload['dndM_tinker08'] = hm.dndM(M_i)
    save_plot_data(p_main, payload,
                   description=('Experiment A: P_halo_gas extrapolated below M_r by '
                                'freezing the baryon response at the M_r bin'))
    return results


def main():
    args = base_parser(
        "Experiment A: extrapolate P_halo_gas(k|M) below M_r and correct the "
        "reconstruction.").parse_args()
    cfg = PmxConfig.from_args(args)

    from Pmx_reconstruction.pmxlib.bundle import load_or_measure
    data = load_or_measure(cfg, cfg.nbins, cfg.logm_min, cfg.logm_max,
                           cfg.nkbins, recompute=args.recompute)
    run_experiment_A(cfg, data)


if __name__ == '__main__':
    main()
