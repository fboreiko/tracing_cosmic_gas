#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""==============================================================================
EXPERIMENT B  --  growing the halo aperture beyond r200b
==============================================================================
Experiment A studies the halos below the catalogue's mass limit. This one 
studies the matter that belongs to a resolved halo but lies outside its 
r200b sphere and is therefore counted in f_u along with everything else.

WHAT MOVES WITH x AND WHAT DOES NOT
    Nothing about the halos changes: the catalogue, the mass bins, the bin
    labels M200b, the halo centres and every measured P_halo_gas(k; M_i) are
    identical at every x. The spectra bundle is not re-measured. What changes
    is only the PARTITION of the matter particles between "in bin i" and "not
    in any bin", and therefore

        measured   R*_i(x), U*(x)     from pmxlib.rstar_ustar, one particle
                                      pass per aperture
        weight     f_i(x) = M_i(x)/M_tot, the mass actually inside the sphere
        profile    u_m(k|M) truncated at x * r200m

THE TWO THINGS THAT HAVE TO BE KEPT CONSISTENT

    1. The weight is the mass inside the sphere, NOT M200b. Growing the
       aperture adds mass to the halo, and f_i's need to be recomputed: either
       from measured f_i^part(x), or extrapolated from f_i(x=1) with m(xc)/m(c).
    The latter is tricky: at x > 1 the spheres of neighbouring halos overlap
    more and the NFW extrapolation would double-count the shared matter.
    The measurement does not, because assign_particles gives each particle
    to exactly one halo (the most massive whose sphere contains it), however
       it is not availiable to the model side.

    2. The profile is truncated at the same radius, via the explicit trunc
       argument of u_nfw. r_s is still r200m/c with r200m and c from the
       CATALOGUE mass -- the aperture does not change the halo's concentration,
       only where the profile is cut off and renormalised.

USAGE
        # the standard sweep
        python -m Pmx_reconstruction.experiment_B --apertures 1.0 1.2 1.5 2.0

        # a first look, reusing whatever is already cached and refusing to
        # measure anything new
        python -m Pmx_reconstruction.experiment_B --no-measure

        # R over [10^12, M_max]; everything below M_r is U / U*
        python -m Pmx_reconstruction.experiment_B --logm-r 12
------------------------------------------------------------------------------
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')

from utils.pipeline_paths import ensure_parents, plot_path
from utils.plot_data import save_plot_data

from Pmx_reconstruction.pmxlib.bundle import load_or_measure
from Pmx_reconstruction.pmxlib.config import (MassRange, PmxConfig,
                                              base_parser)
from Pmx_reconstruction.pmxlib.nfw import conc_of, r200m_of_M, u_nfw
from Pmx_reconstruction.pmxlib.reconstruction import compute_U
from Pmx_reconstruction.pmxlib import plotting as pl
from Pmx_reconstruction.pmxlib import rstar_ustar as ru

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


def measure_aperture(cfg, data, aperture, weights='aperture',
                     allow_compute=True, recompute=False,
                     chunk_particles=5e7, mr=None):
    """Everything one aperture contributes, measured and modelled.

    Measured, from the R*/U* cache at this aperture:
        R_star_i, R_star, U_star        the exact partition of P_matter_gas
        f_part                          mass fraction inside each bin's spheres
        f_out = 1 - sum f_part          mass outside every sphere

    Modelled, from a profile truncated to match the aperture:
        R = sum_i w_i u_m(k|M_i, trunc=x) P_halo_gas(k; M_i)

    Growing the aperture has to grow the weight, and `weights` says how:

      'aperture'  (default) w = f_part(x), the mass actually inside the sphere.
                  Exact, no double counting, but it is measured, so R is a
                  hybrid: the weight knows about particles and only the profile
                  SHAPE is modelled. This is the clean choice for attributing
                  the error, since (R - R*) -> 0 at k -> 0 by construction and
                  the residual is purely a profile-shape error.
      'catalogue' w = f_i m(xc)/m(c), the NFW-extrapolated enclosed mass. Uses
                  no particle information, so R is a genuine prediction and at
                  x = 1 it reduces exactly to the production reconstruction of
                  experiment A's production run -- but it double counts
                  overlapping spheres and leans on NFW where NFW is least
                  trustworthy.
    """
    cache = ru.load_or_measure_rstar_ustar(cfg, data, aperture=aperture,
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

    if mr is None:
        mr = MassRange.from_config(cfg, data)
    tot = ru.measured_partition(cfg, cache, data, mr=mr)
    counts = np.asarray(data['counts'], float)
    occ = counts > 0
    resolved = mr.resolved
    f_part, f_out = tot['f_part'], tot['f_out']

    # Mean mass inside the aperture per halo of the bin, in Msun/h. Only used
    # for reporting: the model weight is f_part directly.
    M_tot = float(cache['M_tot_particles'])
    unit = cfg.rhobar_m * cfg.box ** 3 / M_tot
    mass_bin = (np.asarray(cache['mass_bin_dm'], float)
                + np.asarray(cache['mass_bin_gas'], float))
    with np.errstate(divide='ignore', invalid='ignore'):
        M_ap = np.where(occ, mass_bin * unit / np.maximum(counts, 1.0), 0.0)

    P_halo_gas = np.asarray(data['P_halo_gas'], float)
    M_i = np.asarray(data['M_mean'], float)
    z = float(data['redshift'])
    u_i = np.zeros((M_i.size, k.size))
    conc = np.zeros(M_i.size)
    for j in np.flatnonzero(occ):
        conc[j] = conc_of(cfg, M_i[j], z)
        u_i[j] = u_nfw(k, M_i[j], conc[j], cfg.rhobar_m, trunc=aperture)

    f_i = np.where(resolved, mr.f_i, 0.0)
    mu = nfw_mass_ratio(np.where(occ, conc, 1.0), aperture)
    f_cat_all = np.where(occ, mr.f_i * mu, 0.0)
    f_cat = np.where(resolved, f_cat_all, 0.0)
    f_u_cat = 1.0 - float(f_cat_all[resolved | mr.above].sum())

    if weights == 'aperture':
        w_model, f_u_model = f_part, f_out
    elif weights == 'catalogue':
        w_model, f_u_model = f_cat, f_u_cat
    else:
        raise ValueError(f"unknown weights {weights!r}")
    R_model = np.sum(w_model[:, None] * u_i * P_halo_gas, axis=0)

    return dict(aperture=float(aperture), cache=cache, k=k, weights=weights,
                R_star_i=tot['R_star_i'], R_star=tot['R_star'],
                U_star=tot['U_star'], V_star=tot['V_star'], total=tot['total'],
                resolved=resolved, hidden=mr.hidden, above=mr.above,
                f_part=f_part, f_out=f_out, f_u_cat=f_u_cat,
                f_u_model=f_u_model,
                M_ap=M_ap, u_i=u_i, conc=conc,
                f_i=f_i, f_cat=f_cat, w_model=w_model,
                R_model=R_model,
                assigned_over_m200b=float(
                    cache['assigned_over_m200b_median']))


def run_experiment_B(cfg, data, apertures,
                     weights='aperture', allow_compute=True, recompute=False,
                     chunk_particles=5e7):
    """Go through the apertures, score each one, and write the two figures."""
    k = np.asarray(data['k_center'], float)
    k_Ny = float(data['k_Nyquist'])
    z = float(data['redshift'])
    P_true = np.asarray(data['P_matter_gas'], float)
    P_halo_gas = np.asarray(data['P_halo_gas'], float)
    counts = np.asarray(data['counts'], float)
    logM_cen = np.asarray(data['logM_cen'], float)
    M_i = np.asarray(data['M_mean'], float)
    occ = counts > 0

    # --- M_r / M_max ---------------------------------------------------------------
    mr = MassRange.from_config(cfg, data)
    ref, M_ref = mr.r, mr.M_ref

    apertures = sorted({1.0} | {float(x) for x in apertures})

    print("\n" + "=" * 70)
    print("EXPERIMENT B -- growing the halo aperture beyond r200b")
    print("=" * 70)
    print(f"  apertures = {', '.join(f'{x:g}' for x in apertures)}, "
          f"weights = {weights}")
    print(f"  reference bin {ref} at logM = {logM_cen[ref]:.2f}, "
          f"r200m = {r200m_of_M(M_ref, cfg.rhobar_m):.3f} cMpc/h")
    mr.report('  [B]')
    if mr.hidden.any():
        print("  bins masked below M_r: their halo-bound particles are counted "
              "in U*, not R*")

    # --- the halo model, only if a U mode needs it -----------------------------
    modes = [m for m in cfg.extrap if m != 'simhc']
    if 'simhc' in cfg.extrap:
        print("[B] dropping mode 'simhc': it needs measured spectra below "
              "M_r, which only experiment A's validation uses.")
    hm = None
    if any(m in ('bias', 'halomodel') for m in modes):
        from Pmx_reconstruction.pmxlib.halo_model import HaloModel
        print("[B] building the halo model ...")
        hm = HaloModel(cfg, z, power_spectrum=cfg.power_spectrum)

    # --- one pass per aperture -------------------------------------------------
    runs = {}
    for x in apertures:
        print("\n" + "-" * 70)
        print(f"[B] aperture {x:g} r200m")
        print("-" * 70)
        runs[x] = measure_aperture(cfg, data, x, weights=weights,
                                   allow_compute=allow_compute,
                                   recompute=recompute,
                                   chunk_particles=chunk_particles,
                                   mr=mr)

        runs[x]['U_models'] = {}
        for mode in modes:
            res = compute_U(
                cfg, k, P_halo_gas[ref], M_ref, mr.M_u_hi, mode,
                mr.int_logm_lo, hm=hm, z=z,
                f_u=runs[x]['f_u_model'], trunc=float(x))
            runs[x]['U_models'][mode] = res

    # --- the two weights, side by side -----------------------------------------
    print("\n" + "=" * 70)
    print("WEIGHTS      sum_i f_part (measured)  vs  sum_i f_i m(xc)/m(c) (NFW)")
    print("=" * 70)
    print("     x    measured   NFW extrap   ratio   R(measured)/R(NFW) at k=0.1"
          "   f_u used")
    for x in apertures:
        r = runs[x]
        sp, sc = float(np.sum(r['f_part'])), float(np.sum(r['f_cat']))
        R_meas = np.sum(r['f_part'][:, None] * r['u_i'] * P_halo_gas, axis=0)
        R_nfw = np.sum(r['f_cat'][:, None] * r['u_i'] * P_halo_gas, axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            rr = float(np.interp(0.1, k, R_meas / R_nfw))
        r['R_model_alt'] = R_nfw if weights == 'aperture' else R_meas
        print(f"  {x:4.2f}   {sp:8.4f}   {sc:10.4f}   {sp / sc:6.3f}   "
              f"{rr:22.3f}   {r['f_u_model']:8.4f}")
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
            err = dict(profile=prof, template=tmpl, total=totl, U=U)
            tail = ""
            if r['V_star'] is not None:
                with np.errstate(divide='ignore', invalid='ignore'):
                    err['above'] = -r['V_star'] / denom
                tail = (f"   -V*/total(k=1) = "
                        f"{float(np.interp(1.0, k, err['above'])):+.3f}")
            r.setdefault('errors', {})[mode] = err
            cells = "".join(
                f"   {float(np.interp(kk, k, prof)):+.3f}/"
                f"{float(np.interp(kk, k, tmpl)):+.3f}/"
                f"{float(np.interp(kk, k, totl)):+.3f}"
                for kk in (0.1, 1.0, 3.0))
            print(f"  {x:4.2f}  {cells}{tail}")

    print("\n  The signature to look for: the TEMPLATE term at k ~ 1-3 shrinks "
          "with x while\n  the PROFILE term grows (NFW is a poor description "
          "beyond r200m). A sweep that\n  improved both terms at once would "
          "mean something else is going on.")

    # --- the shell spectra -----------------------------------------------------
    # U*(1) - U*(x) is exactly <delta_m^shell delta_e>, the cross spectrum of
    # the matter between r200m and x r200m with the gas. It is a difference
    # of two measurements, so it carries no model at all.
    print("\n" + "=" * 70)
    print("SHELL SPECTRA      U*(1) - U*(x) = <delta_m^shell delta_e>")
    if mr.hidden.any():
        print("  (with bins masked below M_r this also holds masked-halo matter "
              "captured by the growing resolved spheres)")
    print("=" * 70)
    ks = [kk for kk in K_TABLE if kk <= k[-1]]
    print("     x    " + "".join(f"    k={kk:<6g}" for kk in ks)
          + "   (as a fraction of P^me)")
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
              f"as a fraction of P^me at k = 1:")
        shell_i = runs[x_top]['shell_per_bin']
        idx = np.flatnonzero(occ)
        for j in idx[:: max(1, idx.size // 10)]:
            with np.errstate(divide='ignore', invalid='ignore'):
                v = float(np.interp(1.0, k, shell_i[j] / base['total']))
            print(f"    logM={logM_cen[j]:5.2f}  {v:+.4f}")
        print("  This is the direct answer to 'whose outskirts is the missing "
              "power'.")

    # --- figures ---------------------------------------------------------------
    def _save(fig, kind):
        """Save one of this run's figures, under a stem naming the whole run."""
        models = ''
        if cfg.concentration_source != PmxConfig.concentration_source:
            models += f'_conc-{cfg.concentration_source}'
        if cfg.colossus_conc_model != PmxConfig.colossus_conc_model:
            models += f'_cm-{cfg.colossus_conc_model}'
        if cfg.power_spectrum != PmxConfig.power_spectrum:
            models += f'_ps-{cfg.power_spectrum}'
        stem = (f'expB_{kind}_gas_{cfg.mass_def}_nb{cfg.nbins}'
                f'_logMmin{cfg.logm_min:.2f}_logMmax{cfg.logm_max:.2f}'
                f'{"" if mr.is_default else mr.tag()}'
                f'_ap{"-".join(f"{x:g}" for x in apertures)}'
                f'_mode{"-".join(modes)}_w{weights}{models}')
        path = plot_path('pme_reconstruction', cfg.feedback, stem=stem)
        ensure_parents(path)
        pl.save_figure(fig, path)
        print(f"[B][plot] {path}")
        return path

    err_mode = modes[-1] if modes else None
    pdata = pl.ApertureData(
        k=k, k_Ny=k_Ny, apertures=apertures, runs=runs, P_true=P_true,
        logM_cen=logM_cen, occupied=occ, error_mode=err_mode,
        P_halo_ref=P_halo_gas[ref],
        r200m=np.array([r200m_of_M(m, cfg.rhobar_m) if m > 0 else np.nan
                        for m in M_i]))

    p_main = _save(pl.aperture_panel(pdata), 'aperture')
    _save(pl.shell_mass_figure(pdata), 'shellmass')

    # --- save ------------------------------------------------------------------
    payload = dict(
        k_center=k, weights=weights,
        apertures=np.array(apertures, float), f_i=np.asarray(
            runs[1.0]['f_i'], float),
        logM_cen=logM_cen, M_mean=M_i, counts=counts, ref_bin=ref,
        P_true=P_true, P_halo_gas_ref=P_halo_gas[ref],
        modes=np.array(modes, dtype=object).astype(str),
    )
    payload.update(mr.payload())
    for x in apertures:
        r = runs[x]
        key = f'{x:g}'.replace('.', 'p')
        payload[f'f_part_{key}'] = r['f_part']
        payload[f'f_cat_{key}'] = r['f_cat']
        payload[f'w_model_{key}'] = r['w_model']
        payload[f'f_out_{key}'] = r['f_out']
        payload[f'f_u_cat_{key}'] = r['f_u_cat']
        payload[f'f_u_model_{key}'] = r['f_u_model']
        payload[f'M_ap_{key}'] = r['M_ap']
        payload[f'R_star_{key}'] = r['R_star']
        payload[f'R_star_i_{key}'] = r['R_star_i']
        payload[f'U_star_{key}'] = r['U_star']
        if r['V_star'] is not None:
            payload[f'V_star_{key}'] = r['V_star']
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


def main():
    # --extrap and --power-spectrum come from base_parser: U is built by the
    # same pmxlib.reconstruction.compute_U at every aperture.
    ap = base_parser(
        "Experiment B: grow the halo membership sphere to x r200m and watch "
        "where the reconstruction error goes.", validate=False)
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
    args = ap.parse_args()

    cfg = PmxConfig.from_args(args)

    data = load_or_measure(cfg, cfg.nbins, cfg.logm_min, cfg.logm_max,
                           cfg.nkbins, recompute=args.recompute)

    run_experiment_B(cfg, data, args.apertures, weights=args.weights,
                     allow_compute=args.allow_compute,
                     recompute=args.recompute_apertures,
                     chunk_particles=args.chunk_particles)

    print("\n" + "=" * 70)
    print("Done.")
    print("=" * 70)


if __name__ == '__main__':
    main()