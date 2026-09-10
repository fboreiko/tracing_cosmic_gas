"""Figure construction for the Pmx reconstruction scripts.

Everything about how the Experiment A figures LOOK lives here; everything about
what goes in them lives in the scripts. The split is deliberate: a plot can be
restyled, re-laid-out or re-lettered without touching a line of physics, and
the driver never has to know about gridspecs.

Three layers, coarsest last:

    constants   MODE_COLOURS / MODE_STYLES / REF_STYLE / PANEL_RC / YLABEL_X_*
                The palette and the type. A mode keeps the same colour AND the
                same dash pattern in every panel of every figure.

    panels      panel_shape, panel_dP_ratio, panel_dP_abs, panel_reconstruction,
                panel_rec_ratio. Each draws into an axes it is handed and sets
                nothing else. Reusable in any layout.

    figures     validation_panel, shape_figure, reconstruction_figure,
                ansatz_figure. Each builds a whole Figure and returns it
                UNSAVED, so the caller owns the path and the file format.

The figure builders take a PanelData, not fifteen positional arrays, so that
adding a curve to the set does not mean editing six signatures.

Imports matplotlib and numpy only -- no CAMB, no colossus, no config -- so
importing it for a quick replot costs nothing.
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt

__all__ = [
    'MODE_COLOURS', 'MODE_STYLES', 'REF_STYLE', 'PANEL_RC', 'USE_TEX',
    'LABEL_REC_RATIO', 'LABEL_DP_RATIO', 'YLABEL_X_LEFT', 'YLABEL_X_RIGHT', 'YLABEL_X_SINGLE',
    'mode_style', 'set_ylabel_x', 'save_figure', 'PanelData',
    'panel_shape', 'panel_dP_ratio', 'panel_dP_abs', 'panel_reconstruction',
    'panel_rec_ratio', 'panel_total_error',
    'validation_panel', 'production_panel', 'shape_figure',
    'reconstruction_figure', 'ansatz_figure',
]


# ==============================================================================
# Palette
# ==============================================================================
# 'flat' is the control and stays grey-dashed; black is reserved for the target,
# and the measured "exact" curve is orange wherever it appears.
MODE_COLOURS = {'flat': '0.35', 'simhc': 'C3', 'bias': 'C0', 'halomodel': 'C2'}
MODE_STYLES = {'flat': '--', 'simhc': '-', 'bias': '--', 'halomodel': '-.'}
MODE_FALLBACK = ('C4', ':')

REF_STYLE = {
    'truth':    dict(color='0.7', ls='-', lw=1.8),
    'target':   dict(color='k',   ls='-', lw=2.8),
    'resolved': dict(color='0.5', ls='-', lw=1.8),
    'exact':    dict(color='C1',  ls=':', lw=2.8),
}

# R is the reconstruction from the resolved (above-split) bins, U the correction
# standing in for the unresolved ones.
LABEL_REC_RATIO = r'$(R + U_{\rm model})\,/\,(R + U_{\rm exact})$'
LABEL_DP_RATIO = r'$U_{\rm model}\,/\,U_{\rm exact}$'
LABEL_TOTAL_ERR = r'$(R + U)\,/\,(R_\star + U_\star) - 1$'
# The two halves of that error, which add to it exactly. Drawn in the mode's
# own colour so the panel reads as one model's budget, separated by dash
# pattern and alpha rather than by hue.
COMPONENT_STYLE = {
    'total':    dict(ls='-',  lw=2.4, alpha=1.0),
    'profile':  dict(ls='--', lw=1.8, alpha=0.65),
    'template': dict(ls=':',  lw=2.2, alpha=0.65),
}
LABEL_PROFILE_ERR = r'profile, $(R-R_\star)/(R_\star+U_\star)$'
LABEL_TEMPLATE_ERR = r'template, $(U-U_\star)/(R_\star+U_\star)$'
# The only fontsize outside PANEL_RC: these three labels carry their formulae,
# and at the panel's legend.fontsize they crowd the curves.
TOTAL_ERR_LEGEND_FONTSIZE = 12
LABEL_K = r'$k$ [h/cMpc]'

# The Eq. (59) total error is a near-zero comparison; band and range are set
# for the accuracy the reconstruction is aiming at, not for the data.
TOTAL_ERR_YLIM = (-0.5, 0.5)
TOTAL_ERR_BAND = 0.05
# Which extrapolation the production panel scores. One mode only -- overlaying
# four makes the panel unreadable at this y range.
ERROR_MODE = 'halomodel'


def mode_style(mode, lw=2.0):
    """Colour + linestyle for one extrapolation mode, shared by all panels."""
    return dict(color=MODE_COLOURS.get(mode, MODE_FALLBACK[0]),
                ls=MODE_STYLES.get(mode, MODE_FALLBACK[1]), lw=lw)


# ==============================================================================
# Type and fonts
# ==============================================================================
# Applied through a rc_context inside each figure builder rather than a global
# rcParams update, so importing this module does not silently restyle every
# other plot in the pipeline -- and so the settings survive imports that DO
# mutate the global rcParams (predict_Pmx_from_Phx.py sets text.usetex at
# import time). Panels set no fontsize of their own: this is the one place.
#
# On fonts: 'Computer Modern' is NOT a family matplotlib can resolve (the face
# it ships is registered as 'cmr10'), and font.serif governs plain text only --
# everything in $...$, including the log tick labels, goes through mathtext.
# So getting CM without LaTeX takes the cmr10 + mathtext.fontset pair below.
# Flip USE_TEX to True to hand the typesetting to LaTeX instead, which is what
# HATF/replot_HATF_from_bundle.py does; it needs latex and dvipng on PATH, so
# it is off by default for the compute nodes (cf. the note in
# utils/power_spectrum_utils.py).
USE_TEX = False

PANEL_RC = {
    'text.usetex': USE_TEX,
    'font.family': 'serif',
    'font.serif': ['cmr10', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',
    'axes.formatter.use_mathtext': True,
    'axes.unicode_minus': False,   # cmr10 has no U+2212; without this: tofu
    'font.size': 15,
    'axes.labelsize': 18,
    'axes.titlesize': 18,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'legend.fontsize': 14,
    'axes.linewidth': 1.1,
    'xtick.major.size': 6.5, 'ytick.major.size': 6.5,
    'xtick.minor.size': 3.5, 'ytick.minor.size': 3.5,
    'xtick.major.width': 1.1, 'ytick.major.width': 1.1,
    'xtick.minor.width': 0.9, 'ytick.minor.width': 0.9,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True,
    'legend.handlelength': 2.6,
}


# ==============================================================================
# Layout knobs
# ==============================================================================
# The 2x2 validation panel. Columns carry different height ratios -- the left
# is a spectrum with a thin residual strip, the right is two comparably
# interesting panels -- which is why it is built from nested gridspecs.
PANEL_FIGSIZE = (19, 10.5)
PANEL_WIDTH_RATIOS = [1.3, 1]
PANEL_WSPACE = 0.19
PANEL_HEIGHT_RATIOS_LEFT = [3, 1]
PANEL_HEIGHT_RATIOS_RIGHT = [1, 1]
PANEL_HSPACE_LEFT = 0.05
PANEL_HSPACE_RIGHT = 0.08

# The standalone production figures.
SINGLE_FIGSIZE = (9.5, 9.5)
ANSATZ_FIGSIZE = (9.5, 6)

# Horizontal position of the ylabels, in axes-fraction coordinates: 0 is the
# spine, more negative is further out. Set by hand rather than measured, so a
# column's labels line up by construction -- but nothing adapts if the tick
# labels change width, so revisit these after changing ylim or tick formatting.
#
# One knob, YLABEL_X_RIGHT, plus a conversion. Axes fractions are fractions of
# each panel's OWN width, and the left column is PANEL_WIDTH_RATIOS wider, so
# the same number would put its label visibly further from the spine. Scaling
# by the width ratio makes the two columns' gaps equal on the page, which is
# what the eye actually compares.
YLABEL_X_RIGHT = -0.140
YLABEL_X_LEFT = YLABEL_X_RIGHT * PANEL_WIDTH_RATIOS[1] / PANEL_WIDTH_RATIOS[0]

# The standalone production figures are single-column and unrelated to the
# ratios above, so they get their own value.
YLABEL_X_SINGLE = -0.067

DPI = 300


def save_figure(fig, path, **kwargs):
    """Write `fig` to `path` with PANEL_RC live, and close it.

    This is not a convenience wrapper -- it is load-bearing. Axis labels and
    legends are Text objects that captured their font when the panel was built
    inside the rc_context, but TICK labels are produced by the formatter at
    DRAW time, i.e. during savefig. Save outside the context and the figure
    comes out with Computer Modern axis labels and DejaVu Sans ticks, plus
    whatever axes.unicode_minus the global rcParams happen to carry. Paths and
    formats stay the caller's business; only the styling is borrowed back.
    """
    kwargs.setdefault('bbox_inches', 'tight')
    kwargs.setdefault('dpi', DPI)
    with plt.rc_context(PANEL_RC):
        fig.savefig(path, **kwargs)
    plt.close(fig)
    return path


def set_ylabel_x(axes, x):
    """Park every ylabel in `axes` at the same axes-fraction x."""
    for ax in axes:
        ax.yaxis.set_label_coords(x, 0.5)


# ==============================================================================
# What the figures are drawn from
# ==============================================================================
@dataclass
class PanelData:
    """Everything the Experiment A figures plot, already computed.

    `results` is the driver's dict of mode -> {'S', 'Delta_P', ...}. The two
    optional fields are what validation mode has and production mode does not:
    `Delta_P_exact` is the measured contribution of the hidden bins and
    `S_exact` the shape factor implied by it. `has_exact` keys every branch in
    this module off their presence, so a caller never passes a mode flag.
    """
    k: np.ndarray
    k_Ny: float
    results: dict
    P_rec_resolved: np.ndarray
    P_target: np.ndarray
    P_matter_x_true: np.ndarray
    tag: str = 'x'
    x_sym: str = 'x'
    Delta_P_exact: Optional[np.ndarray] = None
    S_exact: Optional[np.ndarray] = None
    # Production only, from pmxlib.rstar_ustar: R_star is the summed true
    # contribution of the occupied bins and U_star that of everything not
    # assigned to one. Their sum closes on the measured cross spectrum by
    # construction, so it is the denominator the reconstruction is scored
    # against when there is no split. They are kept apart rather than summed
    # because the error decomposition needs each one separately.
    R_star: Optional[np.ndarray] = None
    U_star: Optional[np.ndarray] = None
    error_mode: str = ERROR_MODE

    @property
    def has_exact(self) -> bool:
        return self.Delta_P_exact is not None

    @property
    def has_ustar(self) -> bool:
        return self.R_star is not None and self.U_star is not None

    @property
    def R_star_plus_U_star(self):
        return self.R_star + self.U_star


# ==============================================================================
# Panels
# ==============================================================================
# Each takes an axes and draws into it. Keeping them separate is what lets the
# validation panel assemble four of them onto one canvas while production mode
# emits the same panels as two smaller figures, with no curve drawn twice from
# two different pieces of code.
def panel_shape(ax, d: PanelData, title=None, legend=True):
    """The shape factor S(k) = <u_m T>_w, one curve per mode."""
    for mode, res in d.results.items():
        ax.loglog(d.k, np.abs(res['S']), label=rf'$S(k)$, {mode}',
                  **mode_style(mode))
    if d.S_exact is not None:
        ax.loglog(d.k, np.abs(d.S_exact), label=r'$S(k)$ exact (measured)',
                  **REF_STYLE['exact'])
    ax.axvline(d.k_Ny, c='grey', ls=':', lw=1.2)
    ax.axhline(1.0, c='grey', lw=0.6)
    ax.set_ylabel(r'$S(k)=\langle u_m T\rangle_w$')
    if legend:
        ax.legend(frameon=False)
    if title:
        ax.set_title(title)


def panel_dP_ratio(ax, d: PanelData, legend=True):
    """U_model / U_exact: the correction scored on its own. Validation only."""
    for mode, res in d.results.items():
        with np.errstate(divide='ignore', invalid='ignore'):
            ax.semilogx(d.k, res['Delta_P'] / d.Delta_P_exact, label=mode,
                        **mode_style(mode))
    ax.axhline(1.0, c='k', lw=0.8)
    ax.fill_between(d.k, 0.9, 1.1, color='0.85', zorder=0)
    ax.axvline(d.k_Ny, c='grey', ls=':', lw=1.2)
    ax.set_ylim(0.0, 2.0)
    ax.set_ylabel(LABEL_DP_RATIO)
    ax.set_xlabel(LABEL_K)
    if legend:
        ax.legend(frameon=False, ncol=2)


def panel_dP_abs(ax, d: PanelData, legend=True):
    """Production stand-in for the panel above: no exact target to divide by."""
    for mode, res in d.results.items():
        ax.loglog(d.k, np.abs(res['Delta_P']), label=mode, **mode_style(mode))
    ax.axvline(d.k_Ny, c='grey', ls=':', lw=1.2)
    ax.set_ylabel(r'$\Delta P(k)\,L_{\rm box}^2$')
    ax.set_xlabel(LABEL_K)
    if legend:
        ax.legend(frameon=False)


def panel_total_error(ax, d: PanelData, legend=True):
    """The Eq. (59) error budget for one mode, split into its two halves.

        (R + U)/(R* + U*) - 1  =  (R - R*)/(R* + U*)   profile error
                               +  (U - U*)/(R* + U*)   template error

    The identity is exact, so the two components add to the total curve at
    every k -- if they visibly do not, the arrays are not on the same k grid.
    The split says WHERE a given mode goes wrong: the profile term is the
    resolved bins being reconstructed with the wrong shape, the template term
    is the correction standing in for the unresolved ones being the wrong size.
    A mode can look fine in total while both terms are large and cancelling,
    which is exactly what the total on its own would hide.

    One mode (d.error_mode) -- three curves per model is already a full panel.
    """
    if not d.has_ustar:
        raise ValueError('panel_total_error needs R_star and U_star; run '
                         'pmxlib.rstar_ustar to produce them')
    res = d.results.get(d.error_mode)
    if res is None:
        raise KeyError(f'mode {d.error_mode!r} not among {list(d.results)}')

    denom = d.R_star + d.U_star
    with np.errstate(divide='ignore', invalid='ignore'):
        profile = (d.P_rec_resolved - d.R_star) / denom
        template = (res['Delta_P'] - d.U_star) / denom
        total = (d.P_rec_resolved + res['Delta_P']) / denom - 1.0

    base = mode_style(d.error_mode)
    for curve, key, label in ((total, 'total', f'{d.error_mode} total'),
                              (profile, 'profile', LABEL_PROFILE_ERR),
                              (template, 'template', LABEL_TEMPLATE_ERR)):
        style = dict(base, **COMPONENT_STYLE[key])   # mode colour, own dashes
        # zorder so the total stays visible where a component sits under it,
        # which happens whenever one term dominates the budget.
        ax.semilogx(d.k, curve, label=label,
                    zorder=3 if key == 'total' else 2, **style)

    ax.axhline(0.0, c='k', lw=0.8)
    ax.fill_between(d.k, -TOTAL_ERR_BAND, TOTAL_ERR_BAND, color='0.85', zorder=0)
    ax.axvline(d.k_Ny, c='grey', ls=':', lw=1.2)
    ax.set_ylim(*TOTAL_ERR_YLIM)
    ax.set_ylabel(LABEL_TOTAL_ERR)
    if legend:
        # The one legend on the right column: without it nothing says which
        # extrapolation this is, or which curve is which term.
        ax.legend(frameon=False, fontsize=TOTAL_ERR_LEGEND_FONTSIZE)


def panel_reconstruction(ax, d: PanelData, title=None, legend=True):
    """The reconstructed cross spectrum, mode by mode."""
    if d.has_exact:
        ax.loglog(d.k, np.abs(d.P_matter_x_true),
                  label=rf'full truth $\delta_m\times\delta_{{{d.x_sym}}}$ '
                        rf'(incl. mass below the catalogue)', **REF_STYLE['truth'])
        ax.loglog(d.k, np.abs(d.P_target), label='target: all occupied bins',
                  **REF_STYLE['target'])
        ax.loglog(d.k, np.abs(d.P_rec_resolved),
                  label='bins above the split only', **REF_STYLE['resolved'])
    else:
        ax.loglog(d.k, np.abs(d.P_matter_x_true),
                  label=rf'truth $\delta_m\times\delta_{{{d.x_sym}}}$',
                  **REF_STYLE['target'])
        ax.loglog(d.k, np.abs(d.P_rec_resolved), label='resolved bins only',
                  **REF_STYLE['resolved'])
    for mode, res in d.results.items():
        ax.loglog(d.k, np.abs(d.P_rec_resolved + res['Delta_P']),
                  label=f'+ {mode}', **mode_style(mode))
    ax.axvline(d.k_Ny, c='grey', ls=':', lw=1.2)
    ax.set_ylabel(rf'$P_{{\rm m,{d.tag}}}(k)\,L_{{\rm box}}^2$')
    if legend:
        ax.legend(frameon=False)
    if title:
        ax.set_title(title)


def panel_rec_ratio(ax, d: PanelData):
    """(R + U_model) / (R + U_exact) in validation mode, rec/truth otherwise."""
    with np.errstate(divide='ignore', invalid='ignore'):
        if d.has_exact:
            ax.semilogx(d.k, d.P_matter_x_true / d.P_target, **REF_STYLE['truth'])
        ax.semilogx(d.k, d.P_rec_resolved / d.P_target, **REF_STYLE['resolved'])
        for mode, res in d.results.items():
            ax.semilogx(d.k, (d.P_rec_resolved + res['Delta_P']) / d.P_target,
                        **mode_style(mode))
    ax.axhline(1.0, color='k', lw=0.8)
    ax.fill_between(d.k, 0.95, 1.05, color='0.85', zorder=0)
    ax.axvline(d.k_Ny, c='grey', ls=':', lw=1.2)
    # Validation is a near-1 comparison, so a tighter range; production has to
    # accommodate a correction that may be large.
    ax.set_ylim(0.6, 1.5) if d.has_exact else ax.set_ylim(0.0, 2.0)
    ax.set_ylabel(LABEL_REC_RATIO if d.has_exact else 'rec / truth')
    ax.set_xlabel(LABEL_K)


# ==============================================================================
# Figures
# ==============================================================================
# Each returns an unsaved Figure. Callers do the savefig, so paths, formats and
# the plot_data bundle stay the driver's business.
def _two_column_axes(fig):
    """The 2x2 skeleton both panel figures use.

    Nested gridspecs because the columns want different height ratios: the
    left is a spectrum with a thin residual strip, the right is two comparably
    interesting panels. Returns (left top, left bottom, right top, right
    bottom), x tick labels already hidden on the two top panels.
    """
    outer = fig.add_gridspec(1, 2, width_ratios=PANEL_WIDTH_RATIOS,
                             wspace=PANEL_WSPACE)
    gs_l = outer[0, 0].subgridspec(2, 1, height_ratios=PANEL_HEIGHT_RATIOS_LEFT,
                                   hspace=PANEL_HSPACE_LEFT)
    gs_r = outer[0, 1].subgridspec(2, 1, height_ratios=PANEL_HEIGHT_RATIOS_RIGHT,
                                   hspace=PANEL_HSPACE_RIGHT)
    ax_lt = fig.add_subplot(gs_l[0])
    ax_lb = fig.add_subplot(gs_l[1], sharex=ax_lt)
    ax_rt = fig.add_subplot(gs_r[0])
    ax_rb = fig.add_subplot(gs_r[1], sharex=ax_rt)
    for ax in (ax_lt, ax_rt):
        plt.setp(ax.get_xticklabels(), visible=False)
    return ax_lt, ax_lb, ax_rt, ax_rb


def _finish_panel(axes):
    """Common post-draw work: ylabel placement, matched across the columns."""
    ax_lt, ax_lb, ax_rt, ax_rb = axes
    set_ylabel_x([ax_lt, ax_lb], YLABEL_X_LEFT)
    set_ylabel_x([ax_rt, ax_rb], YLABEL_X_RIGHT)


def validation_panel(d: PanelData):
    """The 2x2 split-test panel: reconstruction left, shape factor right.

    The two columns answer the two halves of the split test -- what the
    correction does to the reconstruction, and how the correction itself scores
    against the deliberately hidden bins.

    No titles: the run is pinned down by the filename and the npz bundle, and
    two of them across a tight gutter only crowd the figure. No legend on the
    right column either: the left panel already names every mode and the
    colour/dash pairing is shared across all four.
    """
    if not d.has_exact:
        raise ValueError('validation_panel needs Delta_P_exact; use '
                         'production_panel when there is no split')
    with plt.rc_context(PANEL_RC):
        fig = plt.figure(figsize=PANEL_FIGSIZE, dpi=DPI)
        axes = _two_column_axes(fig)
        ax_lt, ax_lb, ax_rt, ax_rb = axes

        panel_reconstruction(ax_lt, d)
        panel_rec_ratio(ax_lb, d)
        panel_shape(ax_rt, d, legend=False)
        panel_dP_ratio(ax_rb, d, legend=False)

        _finish_panel(axes)
    return fig


def production_panel(d: PanelData):
    """The same 2x2 layout with no split, scored against R* + U*.

    Left column is unchanged in kind -- the reconstruction and its residual --
    but the target is the measured truth rather than a reconstructed one.

    The right column is what differs. Production has no hidden bins, so there
    is no U_exact to divide the correction by; the top panel instead scores the
    whole reconstruction against the measured decomposition, Eq. (59), for
    d.error_mode alone. The shape factor moves down to the bottom panel, where
    it still shows all four modes: it is the model-side quantity and needs no
    exact counterpart, so keeping it preserves the diagnostic and keeps this
    figure readable next to the validation one.
    """
    if not d.has_ustar:
        raise ValueError('production_panel needs R_star and U_star from the '
                         'pmxlib.rstar_ustar cache')
    with plt.rc_context(PANEL_RC):
        fig = plt.figure(figsize=PANEL_FIGSIZE, dpi=DPI)
        axes = _two_column_axes(fig)
        ax_lt, ax_lb, ax_rt, ax_rb = axes

        panel_reconstruction(ax_lt, d)
        panel_rec_ratio(ax_lb, d)
        panel_total_error(ax_rt, d)          # keeps its one-entry legend
        panel_shape(ax_rb, d, legend=False)
        ax_rb.set_xlabel(LABEL_K)            # bottom of its column now

        _finish_panel(axes)
    return fig


def shape_figure(d: PanelData, title=None):
    """Standalone S(k) figure: the shape factor over its own consequence."""
    with plt.rc_context(PANEL_RC):
        fig, axes = plt.subplots(2, 1, figsize=SINGLE_FIGSIZE, sharex=True,
                                 dpi=DPI,
                                 gridspec_kw=dict(height_ratios=[1, 1], hspace=0.08))
        panel_shape(axes[0], d, title=title)
        panel_dP_ratio(axes[1], d) if d.has_exact else panel_dP_abs(axes[1], d)
        set_ylabel_x(axes, YLABEL_X_SINGLE)
    return fig


def reconstruction_figure(d: PanelData, title=None):
    """Standalone reconstruction figure: spectrum over residual."""
    with plt.rc_context(PANEL_RC):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=SINGLE_FIGSIZE, sharex=True,
                                       dpi=DPI,
                                       gridspec_kw=dict(height_ratios=[3, 1],
                                                        hspace=0.05))
        panel_reconstruction(ax1, d, title=title)
        panel_rec_ratio(ax2, d)
        set_ylabel_x([ax1, ax2], YLABEL_X_SINGLE)
    return fig


def ansatz_figure(k, k_Ny, curves: Sequence[tuple], title=None):
    """Does the tracer cancel in the ratio? One colour per mass bin.

    `curves` is a sequence of (label, T_e, T_c): what the ansatz wants
    (solid) against the CDM proxy it actually uses (dashed). Sequential
    colours here, deliberately outside the mode palette -- on this figure a
    colour means a mass bin, not an extrapolation.
    """
    with plt.rc_context(PANEL_RC):
        fig, ax = plt.subplots(figsize=ANSATZ_FIGSIZE, dpi=DPI)
        shades = plt.cm.viridis(np.linspace(0.0, 0.85, max(len(curves), 1)))
        for n, (label, T_e, T_c) in enumerate(curves):
            ax.semilogx(k, T_e, color=shades[n], ls='-', lw=1.6, label=label)
            ax.semilogx(k, T_c, color=shades[n], ls='--', lw=1.2)
        ax.axvline(k_Ny, c='grey', ls=':', lw=1.2)
        ax.set_xlabel(LABEL_K)
        ax.set_ylabel(r'$P^{he}(k|M)/P^{h_re}$ (solid) vs '
                      r'$P^{hc}(k|M)/P^{h_rc}$ (dashed)')
        ax.set_ylim(0, 2)
        ax.legend(frameon=False, ncol=2)
        if title:
            ax.set_title(title)
    return fig