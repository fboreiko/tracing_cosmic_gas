#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""==============================================================================
7.  EXPERIMENT B  --  growing the halo aperture beyond r200b
==============================================================================
Experiment A studies the halos below the catalogue's mass limit. This one 
studies the matter that belongs to a resolved halo but lies outside its 
r200b sphere and is therefore counted in f_u along with everything else.

THE ONE-LINE VERSION
    Run the whole machinery at a sequence of apertures x, where the halo's
    sphere is x * r200m instead of r200m, and watch where the error goes.

WHAT MOVES WITH x AND WHAT DOES NOT
    Nothing about the halos changes: the catalogue, the mass bins, the bin
    labels M200b, the halo centres and every measured P_halo_x(k; M_i) are
    identical at every x. The spectra bundle is not re-measured. What changes
    is only the PARTITION of the matter particles between "in bin i" and "not
    in any bin", and therefore

        measured   R*_i(x), U*(x)     from pmxlib.rstar_ustar, one particle
                                      pass per aperture
        weight     f_i(x) = M_i(x)/M_tot, the mass actually inside the sphere
        profile    u_m(k|M) truncated at x * r200m

THE TWO THINGS THAT HAVE TO BE KEPT CONSISTENT

    1. The weight is the mass inside the sphere, NOT M200b. Growing the
       aperture adds mass to the halo, and f_i^part(x) is measured,
       not extrapolated with m(xc)/m(c): at x > 1 the spheres of neighbouring
       halos overlap heavily and the NFW extrapolation would double-count the
       shared matter. The measurement does not, because assign_particles gives
       each particle to exactly one halo (the most massive whose sphere
       contains it).

    2. The profile is truncated at the same radius, via the explicit trunc
       argument of u_nfw. r_s is still r200m/c with r200m and c from the
       CATALOGUE mass -- the aperture does not change the halo's concentration,
       only where the profile is cut off and renormalised. The aperture is
       never written into the config: everything else in the pipeline, and the
       halo model in particular, keeps working at r200m.

USAGE
        # the standard sweep
        python -m Pmx_reconstruction.experiment_B --apertures 1.0 1.2 1.5 2.0

        # a first look, reusing whatever is already cached and refusing to
        # measure anything new
        python -m Pmx_reconstruction.experiment_B --no-measure

        # against the dm tracer, to separate geometry from feedback
        python -m Pmx_reconstruction.experiment_B --target matter_dm
------------------------------------------------------------------------------
"""
import argparse
import inspect

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils.pipeline_paths import ensure_parents, plot_path
from utils.plot_data import save_plot_data

from Pmx_reconstruction.pmxlib.bundle import load_or_measure
from Pmx_reconstruction.pmxlib.config import (ExperimentAOptions, PmxConfig,
                                              TRACER_INFO, add_binning_args,
                                              add_concentration_args,
                                              add_experiment_a_args,
                                              add_selfpair_args,
                                              add_target_args, profile_tag)
from Pmx_reconstruction.pmxlib.halo_model import _conc
from Pmx_reconstruction.pmxlib.nfw import r200m_of_M, u_nfw
from Pmx_reconstruction.pmxlib.self_pairs import use_self_pair_corrected
from Pmx_reconstruction.pmxlib import plotting as pl
from Pmx_reconstruction.pmxlib import rstar_ustar as ru

from Pmx_reconstruction.experiment_A import compute_U

KMAX_LOWK = 0.08
K_TABLE = (0.1, 0.3, 1.0, 3.0, 10.0)
def _lowk_mean(k, y, kmax=KMAX_LOWK):
    sel = k < kmax
    if np.count_nonzero(sel) < 3:
        sel = np.zeros_like(k, dtype=bool)
        sel[:5] = True
    y = np.asarray(y, float)[sel]
    return float(np.nanmean(y[np.isfinite(y)]))


def nfw_mass_ratio(c, x):
    """
    M(< x r200m)/M(< r200m) = m(xc)/m(c).
    """
    c = np.asarray(c, dtype=float)
    m = lambda y: np.log1p(y) - y / (1.0 + y)
    return m(float(x) * c) / m(c)


def _U_of(run, mode):
    """The U this run's `mode` produced, or 0 when --extrap was emptied."""
    if mode is None:
        return 0.0
    return run['errors'][mode]['U']


def measure_aperture(cfg, data, aperture, tracer, weights='aperture',
                     allow_compute=True, recompute=False,
                     chunk_particles=5e7, resolved=None):
    """Everything one aperture contributes, measured and modelled.

    Measured, from the R*/U* cache at this aperture:
        R_star_i, R_star, U_star        the exact partition of P_matter_x
        f_part                          mass fraction inside each bin's spheres
        f_out = 1 - sum f_part          mass outside every sphere

    Modelled, from a profile truncated to match the aperture:
        R = sum_i w_i u_m(k|M_i, trunc=x) P_halo_x(k; M_i)

    The weight w and the truncation are a MATCHED PAIR and cannot be chosen
    separately. u_nfw(trunc=x) is normalised by m(xc), so u -> 1 at k -> 0
    means "all of the mass I am weighting by". Pairing it with the M200b weight
    would hold the halo's mass fixed and merely smear it out to x r200m -- the
    density would fall, which is a test of whether the profile is too
    concentrated, not of whether there is mass outside it. Growing the aperture
    has to grow the weight, and `weights` says how:

      'aperture'  (default) w = f_part(x), the mass actually inside the sphere.
                  Exact, no double counting, but it is measured, so R is a
                  hybrid: the weight knows about particles and only the profile
                  SHAPE is modelled. This is the clean choice for attributing
                  the error, since (R - R*) -> 0 at k -> 0 by construction and
                  the residual is purely a profile-shape error.
      'catalogue' w = f_i m(xc)/m(c), the NFW-extrapolated enclosed mass. Uses
                  no particle information, so R is a genuine prediction and at
                  x = 1 it reduces exactly to the production reconstruction of
                  predict_Pmx_from_Phx -- but it double counts overlapping
                  spheres and leans on NFW where NFW is least trustworthy.

    Both are computed and returned whichever is selected, so the difference
    between them is always available and is reported by run_experiment_B.
    """
    cache = ru.load_or_compute(cfg, data, aperture=aperture,
                               allow_compute=allow_compute,
                               recompute=recompute,
                               chunk_particles=chunk_particles)
    if cache is None:
        raise SystemExit(
            f"No R*/U* cache at aperture {aperture:g} and --no-measure was "
            f"given.\n  Build it with\n"
            f"      python -m Pmx_reconstruction.pmxlib.rstar_ustar "
            f"--aperture {aperture:g}")

    k_c = np.asarray(cache['k_center'], float)
    k = np.asarray(data['k_center'], float)
    if k_c.shape != k.shape or not np.allclose(k_c, k, rtol=1e-6):
        raise SystemExit(f"The aperture {aperture:g} cache is on a different k "
                         f"grid from the bundle ({k_c.size} vs {k.size} bins). "
                         f"Rebuild it with --recompute-apertures.")

    tot = ru.totals(cfg, cache, data, tracer)
    counts = np.asarray(data['counts'], float)
    occ = counts > 0
    if resolved is None:
        resolved = occ
    resolved = np.asarray(resolved, bool) & occ
    hidden = occ & ~resolved

    # A raised M_r is applied to the MEASURED partition as well, not only to the
    # model. R* must contain exactly the bins the reconstruction sums over, so
    # the hidden bins' halo-bound matter is moved back into U*, which is then
    # "matter outside every RESOLVED sphere". This stays an exact partition:
    # assignment is to the most massive host, and every resolved halo outranks
    # every hidden one, so nothing is double counted and nothing is dropped.
    # Because the cache is per bin, the split costs no particle pass.
    f_part = np.where(resolved, np.asarray(cache['f_part'], float), 0.0)
    f_out = 1.0 - float(np.sum(f_part))
    if np.any(hidden):
        R_star_i = np.asarray(tot['R_star_i'], float)
        tot = dict(tot)
        tot['U_star'] = tot['U_star'] + np.sum(R_star_i[hidden], axis=0)
        tot['R_star'] = tot['R_star'] - np.sum(R_star_i[hidden], axis=0)
        tot['R_star_i'] = np.where(resolved[:, None], R_star_i, 0.0)

    # Mean mass inside the aperture per halo of the bin, in Msun/h. Only used
    # for reporting: the model weight is f_part directly.
    M_tot = float(cache['M_tot_particles'])
    unit = cfg.rhobar_m * cfg.box ** 3 / M_tot
    mass_bin = (np.asarray(cache['mass_bin_dm'], float)
                + np.asarray(cache['mass_bin_gas'], float))
    with np.errstate(divide='ignore', invalid='ignore'):
        M_ap = np.where(occ, mass_bin * unit / np.maximum(counts, 1.0), 0.0)

    # The model side. M_mean (the catalogue M200b) fixes r200m and c; the
    # aperture only moves the truncation radius.
    info = TRACER_INFO[tracer]
    P_halo_x = np.asarray(data[info['halo_key']], float)
    M_i = np.asarray(data['M_mean'], float)
    z = float(data['redshift'])
    u_i = np.zeros((M_i.size, k.size))
    conc = np.zeros(M_i.size)
    for j in np.flatnonzero(occ):
        conc[j] = _conc(cfg, M_i[j], z)
        u_i[j] = u_nfw(k, M_i[j], conc[j], cfg.rhobar_m, trunc=aperture)

    # The catalogue-side weight, for comparison and for weights='catalogue'.
    # Both weights are zeroed on the hidden bins so that R sums over exactly the
    # bins R* now contains.
    f_i = np.where(resolved, counts / cfg.box ** 3 * M_i / cfg.rhobar_m, 0.0)
    f_cat = np.where(resolved, f_i * nfw_mass_ratio(np.where(occ, conc, 1.0),
                                                    aperture), 0.0)

    if weights == 'aperture':
        w_model = f_part
    elif weights == 'catalogue':
        w_model = f_cat
    else:
        raise ValueError(f"unknown weights {weights!r}")
    R_model = np.sum(w_model[:, None] * u_i * P_halo_x, axis=0)

    return dict(aperture=float(aperture), cache=cache, k=k, weights=weights,
                R_star_i=tot['R_star_i'], R_star=tot['R_star'],
                U_star=tot['U_star'], total=tot['total'],
                resolved=resolved, hidden=hidden,
                f_part=f_part, f_out=f_out, M_ap=M_ap, u_i=u_i, conc=conc,
                f_i=f_i, f_cat=f_cat, w_model=w_model,
                R_model=R_model,
                assigned_over_m200b=float(np.asarray(
                    cache.get('assigned_over_m200b_median', np.nan))))


def run_experiment_B(cfg, data, apertures, opts, tracer=None,
                     weights='aperture', allow_compute=True, recompute=False,
                     chunk_particles=5e7):
    """Sweep the aperture, score each one, and write the two figures."""
    tracer = cfg.tracer if tracer is None else tracer
    info = TRACER_INFO[tracer]
    tag, x_sym = info['tag'], info['sym']
    k = np.asarray(data['k_center'], float)
    k_Ny = float(data['k_Nyquist'])
    z = float(data['redshift'])
    P_true = np.asarray(data[info['truth_key']], float)
    P_halo_x = np.asarray(data[info['halo_key']], float)
    counts = np.asarray(data['counts'], float)
    logM_cen = np.asarray(data['logM_cen'], float)
    M_i = np.asarray(data['M_mean'], float)
    occ = counts > 0

    # --- choose M_r, exactly as experiment_A does ------------------------------
    if opts.split_logm is not None:
        resolved = occ & (logM_cen >= opts.split_logm)
        hidden = occ & ~resolved
        if not np.any(resolved):
            raise SystemExit(f"--split-logm {opts.split_logm} leaves no "
                             f"occupied bin resolved.")
        if not np.any(hidden):
            raise SystemExit(f"--split-logm {opts.split_logm} hides no occupied "
                             f"bin; omit it to use the whole catalogue.")
    else:
        resolved = occ.copy()
        hidden = np.zeros_like(occ)

    ref = int(np.flatnonzero(resolved)[0]) if opts.ref_logm is None else \
        int(np.argmin(np.abs(logM_cen - opts.ref_logm)))
    if not resolved[ref]:
        raise SystemExit(f"--ref-logm {opts.ref_logm} selects bin {ref}, which "
                         f"is not in the resolved set.")
    M_ref = float(M_i[ref])
    boundary = int(np.flatnonzero(resolved)[0])
    M_min = 10.0 ** float(data['logM_edges'][boundary])

    # x = 1 is the baseline every difference is taken against, so it is never
    # optional. Sorted so the shell differences are monotone in x.
    apertures = sorted({1.0} | {float(x) for x in apertures})

    print("\n" + "=" * 70)
    print("EXPERIMENT B -- growing the halo aperture beyond r200b")
    print("=" * 70)
    print(f"  tracer x = {tracer}, apertures = "
          f"{', '.join(f'{x:g}' for x in apertures)}, weights = {weights}")
    print(f"  reference bin {ref} at logM = {logM_cen[ref]:.2f}, "
          f"r200m = {r200m_of_M(M_ref, cfg.rhobar_m):.3f} cMpc/h")
    if np.any(hidden):
        f_i_all = np.where(occ, counts / cfg.box ** 3 * M_i / cfg.rhobar_m, 0.0)
        print(f"  M_r raised to logM = {opts.split_logm:.2f}: "
              f"{int(np.count_nonzero(hidden))} of "
              f"{int(np.count_nonzero(occ))} occupied bins hidden, "
              f"carrying f = {float(np.sum(f_i_all[hidden])):.4f} of the matter")
        print("  their halo-bound particles are counted in U*, not R*")

    # --- the halo model, only if a U mode needs it -----------------------------
    modes = [m for m in opts.extrap if m != 'simhc']
    if 'simhc' in opts.extrap:
        print("[B] dropping mode 'simhc': it needs hidden bins, which this "
              "experiment does not create.")
    hm = None
    if any(m in ('bias', 'halomodel') for m in modes):
        from Pmx_reconstruction.pmxlib.halo_model import HaloModel
        print("[B] building the halo model ...")
        hm = HaloModel(cfg, z, power_spectrum=opts.power_spectrum)

    # --- one pass per aperture -------------------------------------------------
    runs = {}
    for x in apertures:
        print("\n" + "-" * 70)
        print(f"[B] aperture {x:g} r200m")
        print("-" * 70)
        runs[x] = measure_aperture(cfg, data, x, tracer, weights=weights,
                                   allow_compute=allow_compute,
                                   recompute=recompute,
                                   chunk_particles=chunk_particles,
                                   resolved=resolved)

        runs[x]['U_models'] = {}
        for mode in modes:
            res = compute_U(
                cfg, k, P_halo_x[ref], M_ref, M_min, mode, hm=hm, z=z,
                f_u=runs[x]['f_out'], trunc=float(x))
            runs[x]['U_models'][mode] = res

    # --- the two weights, side by side -----------------------------------------
    print("\n" + "=" * 70)
    print("WEIGHTS      sum_i f_part (measured)  vs  sum_i f_i m(xc)/m(c) (NFW)")
    print("=" * 70)
    print("     x    measured   NFW extrap   ratio   R(measured)/R(NFW) at k=0.1")
    for x in apertures:
        r = runs[x]
        sp, sc = float(np.sum(r['f_part'])), float(np.sum(r['f_cat']))
        R_meas = np.sum(r['f_part'][:, None] * r['u_i'] * P_halo_x, axis=0)
        R_nfw = np.sum(r['f_cat'][:, None] * r['u_i'] * P_halo_x, axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            rr = float(np.interp(0.1, k, R_meas / R_nfw))
        r['R_model_alt'] = R_nfw if weights == 'aperture' else R_meas
        print(f"  {x:4.2f}   {sp:8.4f}   {sc:10.4f}   {sp / sc:6.3f}   "
              f"{rr:22.3f}")
    print("\n  ratio < 1 means the spheres hold LESS than NFW predicts: at "
          "x = 1 that is\n  the stellar mass and the mass definition, and its "
          "fall with x is overlap\n  plus the profile steepening past r200m. "
          "The reconstruction is running on the\n  '"
          + weights + "' column; see --weights.")

    # --- the table -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("APERTURE SWEEP")
    print("=" * 70)
    print("    x    sum f_part   f_out    <M_ap/M200b>   beta_out(low k)   "
          "U*/(R*+U*) at k=1")
    base = runs[1.0]
    for x in apertures:
        r = runs[x]
        with np.errstate(divide='ignore', invalid='ignore'):
            beta_out = _lowk_mean(k, r['U_star'] / (r['f_out'] * P_true))
            frac1 = float(np.interp(1.0, k, r['U_star'] / r['total']))
            m_ratio = np.nanmedian(np.where(occ & (M_i > 0),
                                            r['M_ap'] / M_i, np.nan))
        print(f"  {x:4.2f}   {np.sum(r['f_part']):9.4f}  {r['f_out']:7.4f}   "
              f"{m_ratio:12.3f}   {beta_out:15.3f}   {frac1:16.3f}")
        r['beta_out_lowk'] = beta_out

    print("\n  <M_ap/M200b> is the median over bins of the mass inside the "
          "aperture divided\n  by the catalogue M200b. At x = 1 it is below 1 "
          "by the stellar mass and the\n  overlap rule; its growth with x is "
          "the mass the halo model was not counting.\n")
    print("  beta_out is the effective bias of the matter still outside every "
          "sphere. If\n  the deficit is the outskirts of massive halos, this "
          "falls towards the low-mass\n  plateau as x grows.")

    # --- the error budget, mode by mode and aperture by aperture ---------------
    print("\n" + "=" * 70)
    print("ERROR BUDGET      (profile = (R-R*)/(R*+U*), template = (U-U*)/(R*+U*))")
    print("=" * 70)
    for mode in modes:
        print(f"\n  mode {mode}:")
        print("     x    " + "".join(f"  k={kk:<5g}: prof/tmpl/tot"
                                     for kk in (0.1, 1.0, 3.0)))
        for x in apertures:
            r = runs[x]
            denom = r['total']
            U = r['U_models'][mode]['U']
            with np.errstate(divide='ignore', invalid='ignore'):
                prof = (r['R_model'] - r['R_star']) / denom
                tmpl = (U - r['U_star']) / denom
                totl = (r['R_model'] + U) / denom - 1.0
            r.setdefault('errors', {})[mode] = dict(profile=prof, template=tmpl,
                                                    total=totl, U=U)
            cells = "".join(
                f"   {float(np.interp(kk, k, prof)):+.3f}/"
                f"{float(np.interp(kk, k, tmpl)):+.3f}/"
                f"{float(np.interp(kk, k, totl)):+.3f}"
                for kk in (0.1, 1.0, 3.0))
            print(f"  {x:4.2f}  {cells}")

    print("\n  The signature to look for: the TEMPLATE term at k ~ 1-3 shrinks "
          "with x while\n  the PROFILE term grows (NFW is a poor description "
          "beyond r200m). A sweep that\n  improved both terms at once would "
          "mean something else is going on.")

    # --- the shell spectra -----------------------------------------------------
    # U*(1) - U*(x) is exactly <delta_m^shell delta_x>, the cross spectrum of
    # the matter between r200m and x r200m with the tracer. It is a difference
    # of two measurements, so it carries no model at all.
    print("\n" + "=" * 70)
    print("SHELL SPECTRA      U*(1) - U*(x) = <delta_m^shell delta_x>")
    print("=" * 70)
    ks = [kk for kk in K_TABLE if kk <= k[-1]]
    print("     x    " + "".join(f"    k={kk:<6g}" for kk in ks)
          + f"   (as a fraction of P^m{tag})")
    for x in apertures:
        if x == 1.0:
            continue
        shell = base['U_star'] - runs[x]['U_star']
        runs[x]['shell'] = shell
        runs[x]['shell_per_bin'] = runs[x]['R_star_i'] - base['R_star_i']
        with np.errstate(divide='ignore', invalid='ignore'):
            rel = shell / base['total']
        row = "".join(f"  {float(np.interp(kk, k, rel)):9.4f}" for kk in ks)
        print(f"  {x:4.2f}  {row}")

    # Which host masses the shell matter belongs to, at the largest aperture.
    x_top = apertures[-1]
    if x_top > 1.0:
        print(f"\n  per-bin shell contribution at x = {x_top:g}, "
              f"as a fraction of P^m{tag} at k = 1:")
        shell_i = runs[x_top]['shell_per_bin']
        idx = np.flatnonzero(occ)
        for j in idx[:: max(1, idx.size // 10)]:
            with np.errstate(divide='ignore', invalid='ignore'):
                v = float(np.interp(1.0, k, shell_i[j] / base['total']))
            print(f"    logM={logM_cen[j]:5.2f}  {v:+.4f}")
        print("  This is the direct answer to 'whose outskirts is the missing "
              "power'.")

    # --- figures ---------------------------------------------------------------
    stem = (f'expB_aperture_{tag}_{cfg.mass_def}_nb{cfg.nbins}'
            f'_logMmin{cfg.logm_min:.2f}_logMmax{cfg.logm_max:.2f}'
            f'{"" if opts.split_logm is None else f"_Mr{opts.split_logm:.2f}"}'
            f'{"" if opts.ref_logm is None else f"_ref{logM_cen[ref]:.2f}"}'
            f'_ap{"-".join(f"{x:g}" for x in apertures)}'
            f'_mode{"-".join(modes)}_w{weights}{profile_tag(cfg)}'
            f'{"" if bool(data.get("self_pairs_removed", False)) else "_shotpresent"}')

    def _save(fig, this_stem):
        path = plot_path('pme_reconstruction', cfg.feedback, stem=this_stem)
        ensure_parents(path)
        pl.save_figure(fig, path)
        print(f"[B][plot] {path}")
        return path

    err_mode = modes[-1] if modes else None
    pdata = pl.ApertureData(
        k=k, k_Ny=k_Ny, apertures=apertures, runs=runs, P_true=P_true,
        logM_cen=logM_cen, occupied=occ, x_sym=x_sym, error_mode=err_mode,
        P_halo_ref=P_halo_x[ref],
        r200m=np.array([r200m_of_M(m, cfg.rhobar_m) if m > 0 else np.nan
                        for m in M_i]))

    p_main = _save(pl.aperture_panel(pdata), stem)
    _save(pl.shell_mass_figure(pdata),
          stem.replace('expB_aperture', 'expB_shellmass'))

    # --- save ------------------------------------------------------------------
    payload = dict(
        k_center=k, tracer=tracer, weights=weights,
        apertures=np.array(apertures, float), f_i=np.asarray(
            runs[1.0]['f_i'], float),
        logM_cen=logM_cen, M_mean=M_i, counts=counts, ref_bin=ref,
        M_ref=M_ref, P_true=P_true, P_halo_x_ref=P_halo_x[ref],
        resolved=resolved, M_min=M_min,
        split_logm=(np.nan if opts.split_logm is None
                    else float(opts.split_logm)),
        modes=np.array(modes, dtype=object).astype(str),
        self_pairs_removed=bool(data.get('self_pairs_removed', False)),
    )
    for x in apertures:
        r = runs[x]
        key = f'{x:g}'.replace('.', 'p')
        payload[f'f_part_{key}'] = r['f_part']
        payload[f'f_cat_{key}'] = r['f_cat']
        payload[f'w_model_{key}'] = r['w_model']
        payload[f'f_out_{key}'] = r['f_out']
        payload[f'M_ap_{key}'] = r['M_ap']
        payload[f'R_star_{key}'] = r['R_star']
        payload[f'R_star_i_{key}'] = r['R_star_i']
        payload[f'U_star_{key}'] = r['U_star']
        payload[f'R_model_{key}'] = r['R_model']
        payload[f'beta_out_lowk_{key}'] = r['beta_out_lowk']
        if 'shell' in r:
            payload[f'shell_{key}'] = r['shell']
            payload[f'shell_per_bin_{key}'] = r['shell_per_bin']
        for mode in modes:
            payload[f'U_model_{mode}_{key}'] = r['errors'][mode]['U']
            payload[f'err_profile_{mode}_{key}'] = r['errors'][mode]['profile']
            payload[f'err_template_{mode}_{key}'] = r['errors'][mode]['template']
            payload[f'err_total_{mode}_{key}'] = r['errors'][mode]['total']
    save_plot_data(p_main, payload,
                   description=('Experiment B: R*/U* re-measured with the halo '
                                'membership sphere grown to x r200m, with the '
                                'model weight and the NFW truncation moved to '
                                'match'))
    return runs


# ==============================================================================
# 7c.  Figures
# ==============================================================================
def main():
    ap = argparse.ArgumentParser(
        description="Experiment B: grow the halo membership sphere to x r200m "
                    "and watch where the reconstruction error goes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_target_args(ap)
    add_binning_args(ap)
    add_concentration_args(ap)
    add_selfpair_args(ap)
    add_experiment_a_args(ap)          # --extrap and --power-spectrum reused
    g = ap.add_argument_group('Experiment B')
    g.add_argument('--apertures', nargs='+', type=float,
                   default=[1.0, 1.2, 1.5, 2.0],
                   help="membership radii in units of r200m. 1.0 is always "
                        "included: it is the baseline the shell spectra are "
                        "differenced against.")
    g.add_argument('--weights', default='aperture',
                   choices=('aperture', 'catalogue'),
                   help="the mass weight in R. 'aperture' uses the measured "
                        "mass inside x r200m: exact, but it is a measurement, "
                        "so R becomes a hybrid whose weight knows about "
                        "particles. 'catalogue' uses f_i m(xc)/m(c), which is "
                        "a genuine prediction and reduces at x = 1 to the "
                        "production reconstruction, but double counts "
                        "overlapping spheres and trusts NFW beyond r200m. Run "
                        "both: they bracket the answer.")
    g.add_argument('--no-measure', dest='allow_compute', action='store_false',
                   default=True,
                   help="refuse to start a particle pass; use only apertures "
                        "that are already cached. Useful for a first look and "
                        "for replotting.")
    g.add_argument('--recompute-apertures', dest='recompute_apertures',
                   action='store_true',
                   help="re-measure every aperture even if cached")
    g.add_argument('--chunk-particles', type=float, default=5e7,
                   help="expected member particles per membership chunk. The "
                        "aperture^3 scaling is applied on top of this.")
    ap.add_argument('--recompute', action='store_true',
                    help="ignore any cached spectra bundle and re-measure")
    args = ap.parse_args()

    cfg = PmxConfig.from_args(args)
    opts = ExperimentAOptions.from_args(args)

    data = load_or_measure(cfg, cfg.nbins, cfg.logm_min, cfg.logm_max,
                           cfg.nkbins, recompute=args.recompute,
                           self_pairs=cfg.subtract_self_pairs)
    use_self_pair_corrected(cfg, data, subtract=cfg.subtract_self_pairs)

    run_experiment_B(cfg, data, args.apertures, opts, weights=args.weights,
                     allow_compute=args.allow_compute,
                     recompute=args.recompute_apertures,
                     chunk_particles=args.chunk_particles)

    print("\n" + "=" * 70)
    print("Done.")
    print("=" * 70)


if __name__ == '__main__':
    main()