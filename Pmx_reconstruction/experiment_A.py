#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""==============================================================================
6.  EXPERIMENT A  --  extrapolating P_halo_x(k|M) below M_min
==============================================================================
The baseline missing-mass correction applied in main() is the flat template

Delta_P(k) = f_u * P_halo_x(k | M_min),                                    (*)

i.e. "the unresolved mass cross-correlates with e exactly like the lowest
resolved bin does". Experiment A replaces (*) by the integral it is standing
in for,

Delta_P(k) = (1/rhobar_m) int_0^Mmin dM M n(M) u_m(k|M) P_halo_x(k|M),

with the unmeasurable integrand P_halo_x(k|M) supplied by an extrapolation off a
reference bin h_r -- the lowest bin we do have measurements for:

P_halo_x(k|M)  ~  [P_h_r_e(k) / P_h_r_c(k)] * P_hc(k|M).

c is cold dark matter, which we trust; the bracket freezes the baryonic
response in at the reference scale.

WHAT ACTUALLY GETS COMPUTED
Define the mass transfer function
T(k,M) = P_hc(k|M) / P_hc(k|M_r)                          [dimensionless]
so that P_halo_x(k|M) = T(k,M) * P_halo_x(k|M_r) and the measured P_h_r_c cancels
out of the final expression entirely. Then

Delta_P(k) = f_u * <u_m(k|M) T(k,M)>_w * P_halo_x(k|M_r)
= f_u * S(k) * P_halo_x(k|M_r),

which is (*) multiplied by a single shape factor S(k). The old formula is
the S = 1 special case, and in the limit u_m -> 1 with both 1-halo terms
dropped S -> <b>_w / b(M_r), which is exactly Equation (eq:plateau) of the
notes. Experiment A is thus a strict generalisation, not a competitor.

THE FOUR WAYS OF SUPPLYING T
'flat'      T = 1 and u_m = 1. Reproduces (*) bit for bit. The control.
'simhc'     T = P_halo_dm(k;M_i) / P_halo_dm(k;M_r), straight from the measured
halo-DM cross spectra. This is the literal reading of the
ansatz with "P_hc taken from simulations", and it needs no
cosmology, no mass function and no bias model. Only available
in validation mode, where the "unresolved" bins are really
measured -- which is the point: it tests the ansatz ALONE.
'bias'      T = b(M)/b(M_r) from Tinker+10, u_m kept. The 2-halo limit.
'halomodel' T from the full 1-halo + 2-halo model.

THE MASS INTEGRAL, AND WHY ITS AMPLITUDE IS NOT TAKEN FROM THE MODEL
f_u from an analytic HMF is badly determined: integrating Tinker+08 from
10^10 rather than 10^6 Msun/h changes it by a factor of two, because the
low-mass end contributes mass logarithmically and never converges. The
SHAPE S(k), by contrast, moves by only a few per cent over the same range,
since it is a ratio of two integrals against the same weight. So the
default (--fu catalog) takes the amplitude from the measured catalogue
deficit f_u^cat = 1 - sum_i f_i and the shape from the model. --fu model
uses the HMF for both and is there to expose exactly this instability.

Note that f_u^cat is NOT the f_u of the integral: it also collects mass
outside r200b of resolved halos, halos above the top bin, and the
dm+gas-vs-total bookkeeping residue. Experiment A improves the treatment
of component (i) only, and carries the rest along with the same template.
------------------------------------------------------------------------------
"""
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')   # headless; must precede the pyplot import in pmxlib.plotting

from utils.pipeline_paths import ensure_parents, plot_path
from utils.plot_data import save_plot_data

from Pmx_reconstruction.pmxlib.config import (INT_LOGM_LO, INT_NODES,
                                              ExperimentAOptions, PmxConfig,
                                              TRACER_INFO, add_binning_args,
                                              add_concentration_args,
                                              add_experiment_a_args,
                                              add_selfpair_args, add_target_args,
                                              profile_tag)
from Pmx_reconstruction.pmxlib.halo_model import HaloModel, _conc, _trapz_weights
from Pmx_reconstruction.pmxlib import plotting as pl
from Pmx_reconstruction.pmxlib import rstar_ustar as ru
from Pmx_reconstruction.pmxlib.nfw import u_nfw

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
            raise ValueError("mode 'simhc' needs measured P_halo_dm ratios (validation "
                             "mode only)")
        return T_sim
    if hm is None:
        raise ValueError(f"mode {mode!r} needs a HaloModel instance")
    if mode == 'bias':
        return (hm.bias(M_nodes) / float(hm.bias(np.array([M_ref]))[0]))[:, None] \
            * np.ones((1, nk))
    if mode == 'halomodel':
        return hm.P_hc(k, M_nodes) / hm.P_hc(k, np.array([M_ref]))[0][None, :]
    raise ValueError(f"unknown extrapolation mode {mode!r}")


def delta_P_experimentA(cfg, k, P_halo_x_ref, M_ref, M_min, mode, hm=None, z=None,
                        cat_M=None, cat_w=None, T_sim=None,
                        int_logm_lo=INT_LOGM_LO, n_nodes=INT_NODES,
                        f_u=None, use_um=True):
    """The Experiment A correction.

        Delta_P(k) = f_u * S(k) * P_halo_x(k|M_r),   S(k) = <u_m(k|M) T(k,M)>_w

    Two ways of supplying the mass weight w = M n(M):

    cat_M / cat_w given
        The nodes and weights come from the halo CATALOGUE (w_i = n_i M_i).
        Used in validation mode, where the pseudo-unresolved bins are really
        measured, so that n(M) is exact and only T is under test.

    cat_M / cat_w absent
        Nodes are log-spaced on [10^int_logm_lo, M_min] and w = M n(M) comes
        from Tinker+08.

    f_u given
        Overrides the model's own mass integral, i.e. the amplitude is taken
        from the catalogue deficit and only the SHAPE S(k) from the model.

    Returns a dict with S, Delta_P, f_u used, f_u from the integral, and the
    per-node T for inspection.
    """
    if cat_M is not None:
        M_nodes = np.asarray(cat_M, dtype=float)
        w = np.asarray(cat_w, dtype=float)
    else:
        M_nodes = np.logspace(int_logm_lo, np.log10(M_min), n_nodes)
        lnM = np.log(M_nodes)
        w = M_nodes * hm.dndM(M_nodes) * M_nodes * _trapz_weights(lnM)

    keep = w > 0
    M_nodes, w = M_nodes[keep], w[keep]
    if T_sim is not None:
        T_sim = np.asarray(T_sim)[keep]

    T = transfer_T(k, M_nodes, M_ref, mode, hm=hm, T_sim=T_sim)

    if use_um and mode != 'flat':
        u = np.array([u_nfw(k, m, _conc(cfg, m, z), cfg.rhobar_m,
                            trunc=cfg.nfw_trunc) for m in M_nodes])
    else:
        u = np.ones((M_nodes.size, k.size))

    S = np.sum(w[:, None] * u * T, axis=0) / np.sum(w)
    f_u_int = float(np.sum(w) / cfg.rhobar_m)
    f_u_used = f_u_int if f_u is None else float(f_u)

    return dict(S=S, Delta_P=f_u_used * S * P_halo_x_ref, f_u_used=f_u_used,
                f_u_integral=f_u_int, M_nodes=M_nodes, w=w, T=T, mode=mode)


def measured_bias_per_bin(data, kmax_fit=0.08):
    """Large-scale halo bias per mass bin, b_i = <P_halo_dm / P_dm,dm> over k < kmax.

    Purely a diagnostic: it is what the Tinker+10 curve is checked against
    before b(M) is trusted below M_min. Averaging over the lowest k bins keeps
    it in the linear regime where the ratio is genuinely constant.

    The denominator P_dm_dm is an AUTO spectrum and so carries a self-pair term;
    it arrives here already corrected if section 3b ran, which is the right
    thing -- an uncorrected denominator biases b_i low, mildly at these k and
    badly if kmax_fit is ever pushed up.
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


# ------------------------------------------------------------------------------
# 6b.  The driver
# ------------------------------------------------------------------------------
def run_experiment_A(cfg, data, opts, tracer=None, tag=None, x_sym=None):
    """Run and plot Experiment A.

    Two configurations, chosen by --split-logm:

    VALIDATION (--split-logm given)
        Bins above the split are declared resolved; bins below are hidden and
        play the part of the unresolved population. Their exact contribution
            Delta_P_exact(k) = sum_hidden f_i u_m(k|M_i) P_halo_x(k;M_i)
        is known, so every extrapolation can be scored against a truth rather
        than against the final reconstruction, where a dozen other errors are
        superposed. This is the test the notes ask for.

    PRODUCTION (no split)
        Everything is resolved; the correction is applied below the catalogue
        limit and there is no exact target. Only the effect on the final
        reconstruction can be judged.
    """
    tracer = cfg.tracer if tracer is None else tracer
    info = TRACER_INFO[tracer]
    tag = info['tag'] if tag is None else tag
    x_sym = info['sym'] if x_sym is None else x_sym
    k = data['k_center']
    P_halo_x = data[info['halo_key']]
    P_matter_x_true = data[info['truth_key']]
    # P_halo_dm is the extrapolation's CDM proxy (the P_hc of the ansatz) and is
    # the same array whatever the target is -- it is not the tracer here.
    P_halo_dm = data['P_halo_dm']
    counts = data['counts']
    M_i = data['M_mean']
    logM_cen = data['logM_cen']
    z = float(data['redshift'])
    k_Ny = float(data['k_Nyquist'])
    n_i = counts / float(data['box']) ** 3

    print("\n" + "=" * 70)
    print("EXPERIMENT A -- extrapolating P_halo_x(k|M) below M_min")
    print("=" * 70)

    occupied = counts > 0

    # --- split the catalogue ---------------------------------------------------
    if opts.split_logm is not None:
        hidden = occupied & (logM_cen < opts.split_logm)
        resolved = occupied & (logM_cen >= opts.split_logm)
        if not np.any(hidden):
            raise SystemExit(f"--split-logm {opts.split_logm} hides no occupied "
                             f"bin; nothing to validate against.")
        mode_label = 'validation'
    else:
        hidden = np.zeros_like(occupied)
        resolved = occupied
        mode_label = 'production'

    # The analytic integral must cover the SAME mass range as the exact target,
    # or the two are simply not comparable. In validation mode that range is
    # bounded below by the catalogue, not by 10^INT_LOGM_LO.
    if opts.int_logm_min is not None: # practically never used?
        int_logm_lo = float(opts.int_logm_min)
    elif opts.split_logm is not None: # validation mode
        int_logm_lo = float(data['logM_edges'][0])
        print(f"[A] integral lower limit defaulted to the catalogue edge "
              f"logM = {int_logm_lo:.2f}, to match the exact target's range")
    else: # production mode
        int_logm_lo = INT_LOGM_LO # 8.0 or something like that

    ref = int(np.flatnonzero(resolved)[0]) if opts.ref_logm is None else \
        int(np.argmin(np.abs(logM_cen - opts.ref_logm)))
    if not resolved[ref]:
        raise SystemExit(f"--ref-logm {opts.ref_logm} selects bin {ref}, which is "
                         f"not in the resolved set.")

    M_ref = float(M_i[ref])
    P_halo_x_ref = P_halo_x[ref]
    boundary = int(np.flatnonzero(resolved)[0])
    M_min = 10.0 ** float(data['logM_edges'][boundary])

    f_i = n_i * M_i / cfg.rhobar_m
    f_resolved = float(np.sum(f_i[resolved]))
    f_u_cat = 1.0 - f_resolved

    print(f"[A] {mode_label} mode; reference bin {ref} at logM = "
          f"{logM_cen[ref]:.2f} (<M> = {M_ref:.3e} Msun/h)")
    print(f"[A] resolved mass fraction {f_resolved:.4f}, catalogue deficit "
          f"f_u^cat = {f_u_cat:.4f}")
    if np.any(hidden):
        print(f"[A] hiding {np.count_nonzero(hidden)} bins below logM = "
              f"{opts.split_logm}, carrying f = {float(np.sum(f_i[hidden])):.4f} "
              f"of the mass")

    # --- the exact target, when we have one ------------------------------------
    Delta_P_exact = None
    f_u_hidden = None
    if np.any(hidden):
        idx = np.flatnonzero(hidden)
        u_hidden = np.array([u_nfw(k, M_i[j], _conc(cfg, M_i[j], z), cfg.rhobar_m,
                                   trunc=cfg.nfw_trunc) for j in idx])
        Delta_P_exact = np.sum(f_i[idx][:, None] * u_hidden * P_halo_x[idx], axis=0)
        f_u_hidden = float(np.sum(f_i[idx]))
        print(f"[A] exact hidden contribution built from the measured bins; "
              f"f_u(hidden) = {f_u_hidden:.4f}")

    # --- halo model, only if a mode needs it -----------------------------------
    modes = list(opts.extrap)
    if 'simhc' in modes and not np.any(hidden):
        print("[A] dropping mode 'simhc': it needs measured P_hc in the "
              "unresolved range, which only exists in validation mode.")
        modes.remove('simhc')
    # 'flat' needs it too whenever the mass integral is analytic, since that is
    # where its f_u comes from.
    needs_hm = (any(m in ('bias', 'halomodel') for m in modes)
                or (opts.hmf == 'tinker' and any(m != 'simhc' for m in modes)))
    hm = None
    if needs_hm:
        print("[A] building the halo model ...")
        hm = HaloModel(cfg, z, power_spectrum=opts.power_spectrum)

    # --- bias sanity check ------------------------------------------------------
    b_meas, k_fit = measured_bias_per_bin(data)
    if hm is not None:
        print(f"[A] measured vs Tinker+10 bias (from P_halo_dm/P_dm,dm, k < {k_fit:.3f}):")
        for j in np.flatnonzero(occupied)[::max(1, np.count_nonzero(occupied) // 8)]:
            print(f"      logM={logM_cen[j]:5.2f}  b_meas={b_meas[j]:6.3f}  "
                  f"b_T10={float(hm.bias(np.array([M_i[j]]))[0]):6.3f}")

    # --- the amplitude ----------------------------------------------------------
    if opts.fu == 'catalog':
        f_u_amp = f_u_hidden if f_u_hidden is not None else f_u_cat
    else:
        f_u_amp = None      # let each mode use its own mass integral
    print(f"[A] amplitude source: --fu {opts.fu}"
          + (f" -> f_u = {f_u_amp:.4f}" if f_u_amp is not None else
             " -> from the Tinker+08 integral, per mode"))

    # --- catalogue weights for the integral, in validation mode -----------------
    # The per-mode nodes and weights are picked inside the loop below; all that
    # is needed up here is the guard and the measured T_sim, which 'simhc' reads.
    # Note that 'simhc' only knows T at the measured bin masses, so it is always
    # evaluated on the catalogue nodes even when the other modes integrate over a
    # continuous mass function.
    T_sim = None
    if opts.hmf == 'catalog' and not np.any(hidden):
        raise SystemExit("--hmf catalog needs --split-logm: outside validation "
                         "mode there is no catalogue below M_min.")
    if np.any(hidden):
        idx = np.flatnonzero(hidden)
        with np.errstate(divide='ignore', invalid='ignore'):
            T_sim_full = P_halo_dm[idx] / P_halo_dm[ref][None, :]
        T_sim = np.where(np.isfinite(T_sim_full), T_sim_full, 0.0)

    # --- run every requested mode ----------------------------------------------
    results = {}
    for mode in modes:
        use_cat = (opts.hmf == 'catalog') or (mode == 'simhc')
        if use_cat and not np.any(hidden):
            continue
        if use_cat:
            idx = np.flatnonzero(hidden)
            cm, cw = M_i[idx], n_i[idx] * M_i[idx]
        else:
            cm = cw = None
        res = delta_P_experimentA(
            cfg, k, P_halo_x_ref, M_ref, M_min, mode, hm=hm, z=z,
            cat_M=cm, cat_w=cw,
            T_sim=T_sim if mode == 'simhc' else None,
            int_logm_lo=int_logm_lo, n_nodes=INT_NODES,
            f_u=f_u_amp,
        )
        results[mode] = res
        S = res['S']
        print(f"[A] mode {mode:10s}: f_u(int) = {res['f_u_integral']:.4f}, "
              f"f_u(used) = {res['f_u_used']:.4f}, "
              f"S(k_min) = {S[0]:.3f}, S(k=1) = {np.interp(1.0, k, S):.3f}, "
              f"S(k_Ny) = {S[-1]:.3g}")
        if Delta_P_exact is not None:
            with np.errstate(divide='ignore', invalid='ignore'):
                r = res['Delta_P'] / Delta_P_exact
            good = np.isfinite(r) & (k < k_Ny)
            print(f"                 Delta_P/exact: {r[good][0]:.3f} at k_min, "
                  f"{np.interp(1.0, k, r):.3f} at k=1, "
                  f"median |1-r| = {np.nanmedian(np.abs(r[good] - 1.0)):.3f}")

    # --- the bias/halomodel spread, as an error bar on the discarded term -------
    if 'bias' in results and 'halomodel' in results:
        with np.errstate(divide='ignore', invalid='ignore'):
            spread = results['halomodel']['S'] / results['bias']['S'] - 1.0
        print("[A] spread halomodel/bias - 1 (this is the size of the 1-halo term "
              "the\n    factorisation argument dropped, NOT a ranking of the two):")
        for kk in (0.1, 0.3, 1.0, 3.0):
            if kk < k[-1]:
                print(f"      k = {kk:4.1f}: {np.interp(kk, k, spread):+.3f}")
        big = k[np.abs(spread) > 0.5]
        if big.size:
            print(f"      |spread| exceeds 50% above k = {big[0]:.2f}; past that "
                  f"point neither\n      mode is controlled and only the "
                  f"--split-logm validation can decide.")

    # --- the figures ------------------------------------------------------------
    # One-way script-to-script import, no cycle: predict_Pmx_from_Phx only
    # imports run_experiment_A inside its --experiment A branch.
    from Pmx_reconstruction.predict_Pmx_from_Phx import reconstruct_P_matter_x
    n_use = np.where(resolved, n_i, 0.0)
    P_rec_resolved = reconstruct_P_matter_x(cfg, k, P_halo_x, M_i, n_use, z)

    has_exact = Delta_P_exact is not None
    P_target = (P_rec_resolved + Delta_P_exact) if has_exact else P_matter_x_true

    S_exact = None
    if has_exact and f_u_hidden:
        with np.errstate(divide='ignore', invalid='ignore'):
            S_exact = Delta_P_exact / (f_u_hidden * P_halo_x_ref)

    stem = (f'expA_shape_{tag}_{cfg.mass_def}_nb{cfg.nbins}'
            f'_split{"none" if opts.split_logm is None else f"{opts.split_logm:.2f}"}'
            f'_ref{logM_cen[ref]:.2f}_hmf{opts.hmf}_fu{opts.fu}{profile_tag(cfg)}'
            f'{"" if bool(data.get("self_pairs_removed", False)) else "_shotpresent"}')

    # Each figure below is then three lines: build, save, report. Saving goes
    # through pl.save_figure so the tick labels are rendered under the same
    # rcParams the panels were built with -- see its docstring.
    def _save(fig, this_stem):
        path = plot_path('pme_reconstruction', cfg.feedback, stem=this_stem)
        ensure_parents(path)
        pl.save_figure(fig, path)
        print(f"[A][plot] {path}")
        return path

    d = pl.PanelData(k=k, k_Ny=k_Ny, results=results,
                     P_rec_resolved=P_rec_resolved, P_target=P_target,
                     P_matter_x_true=P_matter_x_true, tag=cfg.tracer_info['sym'], x_sym=x_sym,
                     Delta_P_exact=Delta_P_exact, S_exact=S_exact)

    if has_exact:
        # Validation puts both halves of the split test on one canvas.
        p_main = _save(pl.validation_panel(d),
                       stem.replace('expA_shape', 'expA_validation_panel'))
    else:
        # Production has no hidden bins to score the correction against, so it
        # scores the whole reconstruction against the measured R* + U* instead.
        # Measures and caches on a first run, a pure lookup thereafter. The
        # k-grid check and the shot subtraction both live in that module.
        tot, _ = ru.load_totals(cfg, data, tracer)
        if tot is not None:
            d.R_star, d.U_star = tot['R_star'], tot['U_star']
            p_main = _save(pl.production_panel(d),
                           stem.replace('expA_shape', 'expA_production_panel'))
        else:
            # Only reachable when the cache exists but sits on a different k
            # grid from the bundle -- a missing cache is measured, not skipped.
            raise SystemExit(f"[A] cannot score the reconstruction against R*/U* because "
                             f"the cache is missing or unusable; run "
                             f"predict_Pmx_from_Phx.py --measure-rstar-ustar first")

    # --- the ansatz itself, bin by bin ------------------------------------------
    if np.any(hidden):
        idx = np.flatnonzero(hidden)
        show = idx[:: max(1, idx.size // 6)]
        with np.errstate(divide='ignore', invalid='ignore'):
            curves = [(rf'logM$={logM_cen[j]:.2f}$',
                       P_halo_x[j] / P_halo_x_ref,      # what we want
                       P_halo_dm[j] / P_halo_dm[ref])   # the proxy we use
                      for j in show]
        _save(pl.ansatz_figure(k, k_Ny, curves,
                               title='Experiment A ansatz: does the tracer '
                                     'cancel in the ratio?'),
              stem.replace('expA_shape', 'expA_ansatz'))

    # --- save everything --------------------------------------------------------
    payload = {
        'k_center': k, 'tracer': tracer, 'mode': mode_label,
        'ref_bin': ref, 'ref_logM': float(logM_cen[ref]), 'M_ref': M_ref,
        'M_min': M_min, 'split_logm': (np.nan if opts.split_logm is None
                                       else float(opts.split_logm)),
        'hmf_source': opts.hmf, 'fu_source': opts.fu,
        'nfw_trunc': float(cfg.nfw_trunc),
        'concentration_source': cfg.concentration_source,
        'int_logm_min': float(int_logm_lo),
        'f_u_cat': f_u_cat, 'f_resolved': f_resolved,
        'self_pairs_removed': bool(data.get('self_pairs_removed', False)),
        'b_measured': b_meas, 'logM_cen': logM_cen, 'M_mean': M_i, 'n_i': n_i,
        'P_halo_x_ref': P_halo_x_ref, 'P_rec_resolved': P_rec_resolved, 'P_matter_x_true': P_matter_x_true,
    }
    if Delta_P_exact is not None:
        payload['Delta_P_exact'] = Delta_P_exact
        payload['f_u_hidden'] = f_u_hidden
    for mode, res in results.items():
        payload[f'S_{mode}'] = res['S']
        payload[f'Delta_P_{mode}'] = res['Delta_P']
        payload[f'f_u_integral_{mode}'] = res['f_u_integral']
    if hm is not None:
        payload['b_tinker10'] = hm.bias(M_i)
        payload['dndM_tinker08'] = hm.dndM(M_i)
    save_plot_data(p_main, payload,
                   description=('Experiment A: P_halo_x extrapolated below M_min by '
                                'freezing the baryon response at a reference bin'))
    return results



# ==============================================================================
# Standalone entry point
# ==============================================================================
def main():
    ap = argparse.ArgumentParser(
        description="Experiment A: extrapolate P_halo_x(k|M) below the "
                    "catalogue's mass limit and correct the reconstruction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_target_args(ap)
    add_binning_args(ap)
    add_concentration_args(ap)
    add_selfpair_args(ap)
    add_experiment_a_args(ap)
    ap.add_argument('--recompute', action='store_true',
                    help="ignore any cached spectra bundle and re-measure")
    args = ap.parse_args()

    cfg = PmxConfig.from_args(args)
    opts = ExperimentAOptions.from_args(args)

    # Same bundle, same filename as predict_Pmx_from_Phx.py, so a bundle written
    # there is picked straight up and nothing is re-measured.
    from Pmx_reconstruction.pmxlib.bundle import load_or_measure
    from Pmx_reconstruction.pmxlib.self_pairs import use_self_pair_corrected

    data = load_or_measure(cfg, cfg.nbins, cfg.logm_min, cfg.logm_max,
                           cfg.nkbins, recompute=args.recompute,
                           self_pairs=cfg.subtract_self_pairs)
    use_self_pair_corrected(cfg, data, subtract=cfg.subtract_self_pairs)

    run_experiment_A(cfg, data, opts)


if __name__ == '__main__':
    main()