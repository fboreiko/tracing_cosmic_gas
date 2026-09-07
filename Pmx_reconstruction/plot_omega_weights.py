#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
The weight function omega_i(k) of the reconstruction, measured.

    P_matter_x(k) = sum_i f_i u_m(k|M_i) P_halo_x(k; M_i) = sum_i R_i(k),
    omega_i(k) = R_i(k) / sum_j R_j(k),        sum_i omega_i = 1,

i.e. the fraction of the reconstructed spectrum contributed by mass bin i at
wavenumber k. This is Equation (58) of the notes with u~_m -> u_m^NFW, and its
two limits are Equation (60):

    dw/dlnM  ~  M dn/dlnM b(M)              (2-halo branch, eps << 1)
    dw/dlnM  ~  M^2 dn/dlnM u_x(k|M)        (1-halo branch, eps >> 1)

TWO SOURCES FOR THE WEIGHT
    'measured'  omega_i = R_star_i / sum_j R_star_j with R_star_i = <delta_m^(i) delta_x*>
                from the u~ cache written by measure_u_tilde.py. This is Eq. (58)
                exactly: no NFW, no c(M), nothing modelled. Default when the cache
                exists.
    'model'     omega_i = f_i u_m^NFW P_halo_x / R, the weights of the MODELLED
                reconstruction R. The two are related by
                    R_star / R = sum_i omega_i^model (u~_m / u_m^NFW)_i,
                so the model weights are the right ones for asking where the
                profile RATIO is evaluated; the measured ones say where the true
                cross spectrum is built. Both are drawn when both are available.

NOTHING IS MODELLED in the measured source; the model source uses only u_m^NFW,
which the reconstruction uses anyway.
f_i and n_i come from the catalogue histogram and P_halo_x(k; M_i) is measured,
so the answer to "which halo masses does the reconstruction actually weight, and
is u_m ~ 1 there?" needs no HMF, no bias model and no gas model. That is the
point: it is the same question the halo model in the notes answers by assumption.

WHY IT MATTERS FOR THE CDM FAILURE
    The reconstruction has exactly one modelled ingredient, u_m. If omega sits
    where u_m ~ 1 the profile cannot be the problem; if it sits where u_m << 1
    it can. Running this for x = gas and x = dm side by side says whether the
    two tracers weight the same masses -- and therefore whether the cdm failure
    is about WHERE u_m is evaluated or about WHAT u_m is there.

    Read the third panel with the second. If <u_m>_omega is far below 1 for BOTH
    tracers while only the cdm reconstruction fails, then the location of the
    weight is not the difference, and the residual is the mismatch between
    u_m^NFW and the tracer-cross-weighted profile u~_m of Equation (56) -- for
    which see the R_star measurement, not this script.

PROJECTION
    Every spectrum in the bundle is a 2-D projected spectrum, i.e. the k_z = 0
    subset of the 3-D modes, and the halo-model identity holds mode by mode. So
    omega computed here is the projected-field weight, which is the one the
    reconstruction actually uses. It is NOT directly comparable to a 3-D
    halo-model omega; do not overlay the two without redoing the model in 2-D.

USAGE
    Run from the repo root, like predict_Pmx_from_Phx.py itself. The bundle must
    already exist -- this script never measures anything.

        python plot_omega_weights.py
        python plot_omega_weights.py --tracers gas dm matter
        python plot_omega_weights.py --nbins 30 --logm-min 11.0 --logm-max 15.0
        python plot_omega_weights.py --concentration colossus
        python plot_omega_weights.py --k-show 0.1 0.5 1 2 5
--------------------------------------------------------------------------------
"""
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reuse the module's own conventions rather than re-deriving them: rhobar_m,
# u_nfw, c(M,z), the bundle path and the f_c/f_g that define P_halo_matter all
# come from there, so this script cannot drift out of step with the
# reconstruction it is diagnosing.
from utils.pipeline_paths import ensure_parents, plot_path
from utils.plot_data import save_plot_data

from Pmx_reconstruction.pmxlib.bundle import bundle_path, derive_P_halo_matter
from Pmx_reconstruction.pmxlib.config import (PmxConfig, TRACER_INFO,
                                              add_binning_args,
                                              add_concentration_args)
from Pmx_reconstruction.pmxlib.nfw import concentration, u_nfw

TRACER_COLOR = {'gas': 'C1', 'dm': 'C0', 'matter': 'C2'}


# ------------------------------------------------------------------------------
def omega_measured(cfg, data, tracer, z=None):
    """omega_i(k) for one tracer, from the bundle. Returns a dict.

    R_i(k) = f_i u_m^NFW(k|M_i) P_halo_x(k; M_i)   -- the summand of the
    reconstruction, term by term, with exactly the f_i, M_i and c(M,z) that
    reconstruct_P_matter_x uses.
    """
    info = TRACER_INFO[tracer]
    k = data['k_center']
    P_halo_x = data[info['halo_key']]                 # (nbins, nk)
    counts = data['counts']
    M_i = data['M_mean']
    n_i = counts / float(data['box']) ** 3
    f_i = n_i * M_i / cfg.rhobar_m
    z = float(data['redshift']) if z is None else z

    c_i = concentration(M_i, z, source=cfg.concentration_source,
                      colossus_model=cfg.colossus_conc_model,
                      sim_params=cfg.sim_params, sim_name=cfg.sim_name)
    u_im = np.array([u_nfw(k, M_i[i], c_i[i], cfg.rhobar_m) for i in range(M_i.size)])

    R_i = f_i[:, None] * u_im * P_halo_x              # (nbins, nk)
    R = R_i.sum(axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        omega = R_i / R[None, :]

    occupied = counts > 0
    return dict(k=k, omega=omega, R_i=R_i, R=R, u_im=u_im, f_i=f_i, M_i=M_i,
                c_i=c_i, occupied=occupied, logM_cen=data['logM_cen'],
                P_halo_x=P_halo_x)


def omega_from_cache(us, data, tracer):
    """omega_i(k) = R_star_i / R_star from the u~ cache, Eq. (58) exactly.

    R_star_i is the measured per-bin cross spectrum of the particles inside
    r200m of the bin's centrals with the tracer. For 'matter' it is the exact
    linear combination f_c R_star_dm + f_g R_star_gas.
    """
    if tracer == 'matter':
        R_i = float(data['f_c']) * us['R_star_dm'] + float(data['f_g']) * us['R_star_gas']
    else:
        R_i = us[f'R_star_{tracer}']
    counts = data['counts']
    occupied = counts > 0
    R = R_i[occupied].sum(axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        omega = R_i / R[None, :]
    # the measured profile per bin, for <u~_m>_omega
    f_i = us['f_i']
    P_halo_x = data[TRACER_INFO[tracer]['halo_key']]
    with np.errstate(divide='ignore', invalid='ignore'):
        u_tilde = R_i / (f_i[:, None] * P_halo_x)
    return dict(k=data['k_center'], omega=omega, R_i=R_i, R=R, u_im=u_tilde,
                f_i=f_i, M_i=data['M_mean'], occupied=occupied,
                logM_cen=data['logM_cen'], P_halo_x=P_halo_x)


def weighted_mass(logM, omega, occupied, kind='median'):
    """Median or peak log10 M of the weight at each k.

    omega can go negative at high k where a cross spectrum is noise-dominated,
    so the median is taken over |omega| restricted to occupied bins. Where the
    negative weight is a large fraction of the total the answer is meaningless;
    neg_frac is returned so it can be masked on the plot.
    """
    w = np.where(occupied[:, None], omega, 0.0)
    neg_frac = (np.abs(np.clip(w, None, 0.0)).sum(axis=0)
                / np.abs(w).sum(axis=0).clip(1e-300))
    a = np.abs(w)
    if kind == 'peak':
        out = logM[np.argmax(a, axis=0)]
    else:
        cdf = np.cumsum(a, axis=0)
        cdf /= cdf[-1:, :].clip(1e-300)
        out = np.array([np.interp(0.5, cdf[:, j], logM) for j in range(a.shape[1])])
    return out, neg_frac


# ------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Plot the measured reconstruction weight omega_i(k) "
                    "(Eq. 58) for each tracer, from the cached bundle.")
    ap.add_argument('--tracers', nargs='+', default=['gas', 'dm'],
                    choices=sorted(TRACER_INFO),
                    help="which tracers to compare (default: gas dm)")
    add_binning_args(ap)
    add_concentration_args(ap)
    ap.add_argument('--k-show', nargs='+', type=float,
                    default=[0.1, 0.5, 1.0, 2.0, 5.0],
                    help="wavenumbers at which to draw dw/dlnM")
    ap.add_argument('--source', choices=['auto', 'measured', 'model', 'both'],
                    default='auto',
                    help="'measured' uses R_star_i from the u~ cache (Eq. 58 "
                         "exactly); 'model' uses f_i u_m^NFW P_halo_x; 'both' "
                         "overlays them; 'auto' = both if the cache exists, "
                         "else model")
    ap.add_argument('--neg-frac-max', type=float, default=0.2,
                    help="mask the weighted-mass curves where the negative "
                         "part of omega exceeds this fraction (noise guard)")
    args = ap.parse_args()

    # One frozen config; this script used to rebind the main module's globals.
    cfg = PmxConfig.from_args(args)

    path = bundle_path(cfg, args.nbins, args.logm_min, args.logm_max,
                         cfg.grid, args.nkbins)
    if not path.exists():
        raise SystemExit(
            f"No bundle at\n  {path}\nRun predict_Pmx_from_Phx.py first "
            f"(this script never measures anything).")
    print(f"Loading bundle:\n  {path}")
    with np.load(path, allow_pickle=False) as f:
        data = {key: f[key] for key in f.files}
    derive_P_halo_matter(cfg, data)

    k = data['k_center']
    logM = data['logM_cen']
    k_Ny = float(data['k_Nyquist'])
    z = float(data['redshift'])
    print(f"  {logM.size} mass bins, {k.size} k bins, z = {z}, "
          f"k_Nyquist = {k_Ny:.2f} h/cMpc")

    # --- which weights to draw ---------------------------------------------
    us = None
    try:
        from tracing_cosmic_gas.Pmx_reconstruction.measure_u_tilde import ustar_path
        upath = ustar_path(args.nbins, args.logm_min, args.logm_max,
                           cfg.grid, args.nkbins)
        if upath.exists():
            with np.load(upath, allow_pickle=False) as f:
                us = {key: f[key] for key in f.files}
            print(f"u~ cache:\n  {upath}")
    except ImportError:
        pass
    source = args.source
    if source == 'auto':
        source = 'both' if us is not None else 'model'
    if source in ('measured', 'both') and us is None:
        raise SystemExit("--source measured/both needs the u~ cache from "
                         "measure_u_tilde.py, which was not found.")
    print(f"weight source: {source}")

    # res[src][tracer]; 'measured' is drawn solid, 'model' dashed
    res = {}
    if source in ('measured', 'both'):
        res['measured'] = {t: omega_from_cache(us, data, t) for t in args.tracers}
    if source in ('model', 'both'):
        res['model'] = {t: omega_measured(cfg, data, t, z) for t in args.tracers}
    primary = 'measured' if 'measured' in res else 'model'
    SRC_LS = {'measured': '-', 'model': '--'}
    SRC_LABEL = {'measured': r'$R_\star$', 'model': r'$u^{\rm NFW}$'}
    occ = res[primary][args.tracers[0]]['occupied']
    dlnM = np.log(10.0) * float(np.mean(np.diff(logM)))    # for dw/dlnM

    # --- figure ---------------------------------------------------------------
    fig, ax = plt.subplots(2, 2, figsize=(11.5, 8.5))
    fig.suptitle(rf"Reconstruction weight $\omega_i(k)$ (Eq. 58), "
                 rf"{cfg.feedback}, $z={z}$  [solid: measured $R_\star$, dashed: NFW model]",
                 fontsize=11)

    # (1) dw/dlnM vs M at a few k, primary source only (both would be unreadable)
    a = ax[0, 0]
    for t in args.tracers:
        w = res[primary][t]['omega']
        for n, kk in enumerate(args.k_show):
            j = int(np.argmin(np.abs(k - kk)))
            alpha = 0.30 + 0.70 * n / max(1, len(args.k_show) - 1)
            a.plot(logM[occ], (w[occ, j] / dlnM), color=TRACER_COLOR[t],
                   alpha=alpha, lw=1.4,
                   label=(f"{t}, k={kk:g}" if n in (0, len(args.k_show) - 1) else None))
    a.axhline(0.0, color='k', lw=0.6)
    a.set_xlabel(r"$\log_{10} M\ [M_\odot/h]$")
    a.set_ylabel(r"$\mathrm{d}\omega/\mathrm{d}\ln M$")
    a.set_title(f"weight per unit ln M, {primary} (darker = higher k)")
    a.legend(fontsize=8, ncol=2)

    # (2) median mass of the weight vs k, every source
    a = ax[0, 1]
    for src, rs in res.items():
        for t in args.tracers:
            m, negf = weighted_mass(logM, rs[t]['omega'], occ, 'median')
            m = np.where(negf < args.neg_frac_max, m, np.nan)
            a.plot(k, m, SRC_LS[src], color=TRACER_COLOR[t], lw=1.4,
                   label=f"{t}, {SRC_LABEL[src]}")
    a.axvline(k_Ny, color='grey', ls=':', lw=1)
    a.set_xscale('log')
    a.set_xlabel(r"$k\ [h/\mathrm{cMpc}]$")
    a.set_ylabel(r"median $\log_{10} M$ of $\omega(k,\cdot)$")
    a.set_title("mass carrying the weight")
    a.legend(fontsize=8)

    # (3) the profile evaluated where the weight sits:
    #     measured source -> <u~_m>_omega ; model source -> <u_m^NFW>_omega
    a = ax[1, 0]
    for src, rs in res.items():
        for t in args.tracers:
            r = rs[t]
            w = np.where(occ[:, None], np.abs(r['omega']), 0.0)
            w /= w.sum(axis=0).clip(1e-300)[None, :]
            u_eff = np.nansum(w * np.where(np.isfinite(r['u_im']), r['u_im'], 0.0), axis=0)
            lab = r"$\langle\tilde u_m\rangle_\omega$" if src == 'measured' \
                else r"$\langle u_m^{\rm NFW}\rangle_\omega$"
            a.plot(k, u_eff, SRC_LS[src], color=TRACER_COLOR[t], lw=1.5,
                   label=f"{t}: " + lab)
    a.axhline(1.0, color='k', lw=0.6)
    a.axvline(k_Ny, color='grey', ls=':', lw=1)
    a.set_xscale('log')
    a.set_ylim(0.0, 1.08)
    a.set_xlabel(r"$k\ [h/\mathrm{cMpc}]$")
    a.set_ylabel(r"$\langle u_m\rangle_\omega$")
    a.set_title("profile suppression at the weighted mass")
    a.legend(fontsize=8)

    # (4) quartiles, primary source
    a = ax[1, 1]
    for t in args.tracers:
        w = np.where(occ[:, None], np.abs(res[primary][t]['omega']), 0.0)
        w /= w.sum(axis=0).clip(1e-300)[None, :]
        cum = np.cumsum(w, axis=0)
        for lvl, ls in ((0.25, ':'), (0.50, '-'), (0.75, '--')):
            m = np.array([np.interp(lvl, cum[:, j], logM) for j in range(k.size)])
            a.plot(k, m, ls, color=TRACER_COLOR[t], lw=1.2,
                   label=(f"{t}: 25/50/75%" if lvl == 0.50 else None))
    a.axvline(k_Ny, color='grey', ls=':', lw=1)
    a.set_xscale('log')
    a.set_xlabel(r"$k\ [h/\mathrm{cMpc}]$")
    a.set_ylabel(r"$\log_{10} M$")
    a.set_title(f"quartiles of the weight distribution, {primary}")
    a.legend(fontsize=8)

    for a_ in ax.flat:
        a_.grid(alpha=0.3)
    fig.tight_layout()

    stem = (f"omega_weights_{'-'.join(args.tracers)}"
            f"_nb{args.nbins}_logM{args.logm_min:g}-{args.logm_max:g}"
            f"_{cfg.concentration_source}_{source}")
    out = plot_path('pme_reconstruction', cfg.feedback, stem=stem)
    ensure_parents(out)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nSaved:\n  {out}")

    payload = dict(k=k, logM_cen=logM, occupied=occ, k_Nyquist=k_Ny,
                   redshift=z, concentration=cfg.concentration_source, source=source)
    for src, rs in res.items():
        for t in args.tracers:
            r = rs[t]
            payload[f'omega_{src}_{t}'] = r['omega']
            payload[f'R_i_{src}_{t}'] = r['R_i']
            payload[f'u_im_{src}_{t}'] = r['u_im']
            med, negf = weighted_mass(logM, r['omega'], occ, 'median')
            payload[f'median_logM_{src}_{t}'] = med
            payload[f'neg_frac_{src}_{t}'] = negf
    payload['f_i'] = res[primary][args.tracers[0]]['f_i']
    payload['M_i'] = res[primary][args.tracers[0]]['M_i']
    if 'model' in res:
        payload['c_i'] = res['model'][args.tracers[0]]['c_i']
    save_plot_data(out, payload,
                     description=("Measured reconstruction weight omega_i(k) "
                                  "(Eq. 58) per mass bin, per tracer, with the "
                                  "u_m^NFW the reconstruction itself uses"))

    # --- the numbers the argument in the notes turns on -----------------------
    for src, rs in res.items():
        print(f"\n[{src}]  k      " + "".join(f"{t:>10s} med logM  <u_m>_w" for t in args.tracers))
        for kk in (0.1, 0.5, 1.0, 2.0, 3.0, 5.0):
            if kk > k[-1]:
                continue
            j = int(np.argmin(np.abs(k - kk)))
            row = f"  {k[j]:6.3f} "
            for t in args.tracers:
                r = rs[t]
                med, _ = weighted_mass(logM, r['omega'], occ, 'median')
                w = np.where(occ, np.abs(r['omega'][:, j]), 0.0)
                w = w / w.sum().clip(1e-300)
                u = np.where(np.isfinite(r['u_im'][:, j]), r['u_im'][:, j], 0.0)
                row += f"{med[j]:>16.2f} {float((w * u).sum()):>8.3f}"
            print(row)


if __name__ == '__main__':
    main()