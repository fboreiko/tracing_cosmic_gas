"""Figure construction for the Pmx reconstruction scripts.

Three layers, coarsest last:

    constants   MODE_COLOURS / MODE_STYLES / REF_STYLE / PANEL_RC / YLABEL_X_*
                A mode keeps the same colour AND dash pattern in every panel.

    panels      panel_reconstruction, panel_rec_ratio, panel_shape, panel_dP,
                panel_error, panel_shell_mass. Each draws into an axes it is
                handed and sets nothing else. Every panel accepts either a
                PanelData (Experiment A) or an ApertureData (Experiment B)
                and branches on which it got, so the two experiments cannot
                drift apart stylistically.

    figures     panel_figure (the 2x2 skeleton, validation / production /
                aperture), stacked_figure (spectrum-over-residual or
                shape-over-correction), ansatz_figure, shell_mass_figure.
                Each returns an UNSAVED Figure; the caller owns the path.
"""

from dataclasses import dataclass
from functools import partial
from typing import Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt

TRACER_SYM = 'e'

__all__ = [
    'MODE_COLOURS', 'MODE_STYLES', 'REF_STYLE', 'PANEL_RC', 'USE_TEX',
    'TRACER_SYM', 'labels', 'LABEL_K', 'LABEL_DP_RATIO', 'LABEL_SHAPE', 'SHAPE_YLIM',
    'YLABEL_X_LEFT', 'YLABEL_X_RIGHT',
    'YLABEL_X_SINGLE', 'mode_style', 'set_ylabel_x', 'save_figure',
    'PanelData', 'ApertureData', 'APERTURE_CMAP', 'aperture_colours',
    'panel_reconstruction', 'panel_rec_ratio', 'panel_shape', 'panel_dP',
    'panel_error', 'panel_shell_mass',
    'panel_figure', 'stacked_figure', 'ansatz_figure', 'shell_mass_figure',
    # aliases
    'validation_panel', 'production_panel', 'aperture_panel',
    'shape_figure', 'reconstruction_figure',
]

# ==============================================================================
# Palette, type, layout
# ==============================================================================
MODE_COLOURS = {'flat': '0.35', 'simhc': 'C3', 'bias': 'C2', 'halomodel': 'C0'}
MODE_STYLES = {'flat': '--', 'simhc': '-', 'bias': '--', 'halomodel': '-.'}
MODE_FALLBACK = ('C4', ':')

REF_STYLE = {
    'truth':    dict(color='0.7', ls='-', lw=1.8),
    'target':   dict(color='k',   ls='-', lw=2.8),
    'resolved': dict(color='0.5', ls='-', lw=1.8),
    'exact':    dict(color='C1',  ls=':', lw=2.8),
    'seff':     dict(color='red', ls=(0, (6, 2)), lw=2.0, alpha=0.55),
    'mismatch': dict(color='0.55', ls='-', lw=1.0),
    'selfpair': dict(color='C3',   ls=':', lw=1.2),
}
COMPONENT_STYLE = {
    'total':    dict(ls='-', lw=2.4, alpha=0.55),
    'profile':  dict(color='red',    ls='-', lw=1.8, alpha=0.55),
    'template': dict(color='purple', ls='-', lw=2.0, alpha=0.55),
    'above':    dict(color='0.4',    ls=':', lw=1.6, alpha=0.8),
}

LABEL_K = r'$k$ [h/cMpc]'
LABEL_DP_RATIO = r'$U_{\rm model}\,/\,U_{\rm exact}$'
LABEL_SHAPE = r'$S(k)=\langle u_m T\rangle_w$'
LABEL_HOST_BIN = r'$\log_{10} M_{200b}$ of the host bin'
LABEL_PI_OVER_R200M = r'$\pi/r_{200m}$ [h/cMpc]'


def labels() -> dict:
    """Every label that names the measured truth.

    R_star + U_star IS the measured truth, so it is written P^{m x} here
    rather than as its decomposition: the reader has already met P^{m x} on
    the reconstruction panel, and the axis says what the residual is a
    fraction OF without making them recall which two pieces add up to it.
    The decomposition still shows in the numerators, where it is the point:
    R - R_star is the profile error, U - U_star the template error.

    Both experiments read their labels from here, keyed identically, so a
    panel cannot be called one thing in Experiment A and another in
    Experiment B.
    """
    P = rf'P^{{\,m{TRACER_SYM}}}'
    return {
        'spectrum':  rf'${P}(k)\,L_{{\rm box}}^2$',
        'rec_ratio': rf'$(R + U)\,/\,{P}$',
        # Validation divides by the split target, not by the full truth, so
        # this one names a different denominator on purpose.
        'rec_ratio_split': r'$(R + U_{\rm model})\,/\,(R + U_{\rm exact})$',
        'total':     rf'$(R + U)\,/\,{P} - 1$',
        'profile':   rf'$(R - R_\star)\,/\,{P}$',
        'template':  rf'$(U - U_\star)\,/\,{P}$',
        'above':     rf'$-V_\star\,/\,{P}$ (above $M_{{\max}}$, unmodelled)',
        'seff':      rf'$S_{{\rm eff}}=U_\star/(\tilde f_u \, P^{{\,h_r{TRACER_SYM}}})$',
        'shell':     rf'$(R_{{\star i}}(x)-R_{{\star i}}(1))\,/\,{P}$',
    }


TOTAL_ERR_YLIM = (-0.5, 0.5)
TOTAL_ERR_BAND = 0.05
# One range for every S(k) panel, both experiments, so shape panels can be
# compared across figures by eye. Set to None to autoscale each one instead.
SHAPE_YLIM = (1e-2, 1e1)
ERROR_MODE = 'halomodel'
SMALL_LEGEND = 12
BIG_LEGEND = 16

USE_TEX = False
PANEL_RC = {
    'text.usetex': USE_TEX,
    'font.family': 'serif',
    'font.serif': ['cmr10', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',
    'axes.formatter.use_mathtext': True,
    'axes.unicode_minus': False,   # cmr10 has no U+2212; without this: tofu
    'font.size': 15, 'axes.labelsize': 18, 'axes.titlesize': 18,
    'xtick.labelsize': 15, 'ytick.labelsize': 15, 'legend.fontsize': 14,
    'axes.linewidth': 1.1,
    # Default is 4.0, which puts a label against its tick labels at these
    # font sizes. Sets the gap for every x label and for any y label whose
    # position is not pinned by set_ylabel_x below.
    'axes.labelpad': 9.0,
    'xtick.major.size': 6.5, 'ytick.major.size': 6.5,
    'xtick.minor.size': 3.5, 'ytick.minor.size': 3.5,
    'xtick.major.width': 1.1, 'ytick.major.width': 1.1,
    'xtick.minor.width': 0.9, 'ytick.minor.width': 0.9,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True,
    'legend.handlelength': 2.6,
}

PANEL_FIGSIZE = (19, 10.5)
PANEL_WIDTH_RATIOS = [1.3, 1]
PANEL_WSPACE = 0.19
SINGLE_FIGSIZE = (9.5, 9.5)
ANSATZ_FIGSIZE = (9.5, 6)
SHELL_FIGSIZE = (9.5, 6.5)
DPI = 300

# Pinned y label positions, in axes fractions. These OVERRIDE axes.labelpad
# (set_label_coords wins), so the gap for the panel figures is set here --
# more negative is further from the spine.
YLABEL_X_RIGHT = -0.155
YLABEL_X_LEFT = YLABEL_X_RIGHT * PANEL_WIDTH_RATIOS[1] / PANEL_WIDTH_RATIOS[0]
YLABEL_X_SINGLE = -0.082

APERTURE_CMAP = plt.get_cmap('plasma')
APERTURE_REC_RATIO_YLIM = (0.6, 1.4)
SHELL_K_STYLES = ((0.1, ':'), (1.0, '-'), (3.0, '--'))


def mode_style(mode, lw=2.0):
    """Colour + linestyle for one extrapolation mode, shared by all panels."""
    return dict(color=MODE_COLOURS.get(mode, MODE_FALLBACK[0]),
                ls=MODE_STYLES.get(mode, MODE_FALLBACK[1]), lw=lw)


def aperture_colours(apertures: Sequence[float]) -> dict:
    """One colour per aperture, dark to light in x."""
    n = max(len(apertures), 2)
    return {x: APERTURE_CMAP(0.08 + 0.78 * i / (n - 1))
            for i, x in enumerate(apertures)}


def save_figure(fig, path, **kwargs):
    """Write `fig` to `path` with PANEL_RC live, and close it.

    Load-bearing: tick labels are produced by the formatter at DRAW time, so
    saving outside the rc_context gives Computer Modern axis labels with
    DejaVu Sans ticks.
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


_KEEP = object()   # "caller said nothing", as distinct from an explicit None


def _quiet():
    return np.errstate(divide='ignore', invalid='ignore')


def _proxy(label, **kw):
    """A legend-only line."""
    return plt.Line2D([], [], label=label, **kw)


def _guides(ax, d, level=None, band=None, ylim=None, ylabel=None,
            xlabel=None, title=None, legend=None):
    """The decoration every panel shares: Nyquist line, reference level,
    tolerance band, limits, labels. `legend` is None/False or a dict of
    ax.legend kwargs (frameon=False is implied)."""
    ax.axvline(d.k_Ny, c='grey', ls=':', lw=1.2)
    if getattr(d, 'k_split', None):
        ax.axvline(d.k_split, c='grey', ls='--', lw=1.0, alpha=0.6)
    if level is not None:
        ax.axhline(level, c='k', lw=0.8)
        if band:
            ax.fill_between(d.k, level - band, level + band, color='0.85',
                            zorder=0)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    if title:
        ax.set_title(title)
    if legend:
        ax.legend(frameon=False, **legend)


# ==============================================================================
# What the figures are drawn from
# ==============================================================================
@dataclass
class PanelData:
    """Everything the Experiment A figures plot, already computed.

    `results` is the driver's dict of mode -> {'S', 'U', ...}. `U_exact` /
    `S_exact` exist in validation mode only; `R_star` / `U_star` in production
    only. Every branch keys off their presence, never off a mode flag.
    """
    k: np.ndarray
    k_Ny: float
    results: dict
    R: np.ndarray
    P_target: np.ndarray
    P_matter_gas_true: np.ndarray
    U_exact: Optional[np.ndarray] = None
    # Diagnostics, drawn on the spectrum panel when given. P_dm_gas is the
    # definitional-mismatch curve; P_selfpair is the term removed from the
    # truth, on a log axis so it is obvious where it stops being a footnote.
    P_dm_gas: Optional[np.ndarray] = None
    P_selfpair: Optional[np.ndarray] = None
    S_exact: Optional[np.ndarray] = None
    R_star: Optional[np.ndarray] = None
    U_star: Optional[np.ndarray] = None
    S_eff: Optional[np.ndarray] = None
    V_star: Optional[np.ndarray] = None
    error_mode: str = ERROR_MODE
    # Where a 3-D measurement hands over to the projected one, drawn as a
    # dashed guide by _guides. None when the run was purely 2-D.
    k_split: Optional[float] = None

    @property
    def has_exact(self) -> bool:
        return self.U_exact is not None

    @property
    def has_ustar(self) -> bool:
        return self.R_star is not None and self.U_star is not None


@dataclass
class ApertureData:
    """Everything the Experiment B (aperture sweep) figures plot.

    `runs` maps aperture -> the dict measure_aperture returns, augmented by
    the driver with 'errors', 'shell_per_bin', ... `r200m` is per mass bin
    in cMpc/h and `P_halo_ref` is P^{h_r x}(k) of the reference bin; both are
    passed in so this module keeps no dependency on the cosmology helpers.
    """
    k: np.ndarray
    k_Ny: float
    apertures: Sequence[float]
    runs: dict
    P_true: np.ndarray
    logM_cen: np.ndarray
    occupied: np.ndarray
    r200m: np.ndarray
    P_halo_ref: np.ndarray
    error_mode: Optional[str] = None
    colours: Optional[dict] = None
    k_split: Optional[float] = None

    def __post_init__(self):
        self.apertures = sorted(float(x) for x in self.apertures)
        if not self.colours:
            self.colours = aperture_colours(self.apertures)

    @property
    def base(self) -> dict:
        return self.runs[1.0]

    @property
    def x_top(self) -> float:
        return self.apertures[-1]

    def _model(self, x) -> Optional[dict]:
        """The U-model dict at aperture x: the error mode if present, else
        the last one; None if the run carries no models."""
        models = self.runs[x].get('U_models') or {}
        if not models:
            return None
        key = self.error_mode if self.error_mode in models else list(models)[-1]
        return models[key]

    def U(self, x) -> np.ndarray:
        m = self._model(x)
        return m['U'] if m else np.zeros_like(self.k)

    def S(self, x) -> Optional[np.ndarray]:
        m = self._model(x)
        return m.get('S') if m else None

    def S_eff(self, x) -> np.ndarray:
        """The shape U would need to be exact: U_star/(f_u P^{h_r x})."""
        r = self.runs[x]
        with _quiet():
            return r['U_star'] / (r['f_out'] * self.P_halo_ref)

    def total_model(self, x) -> np.ndarray:
        return self.runs[x]['R_model'] + self.U(x)


def _is_aperture(d) -> bool:
    return isinstance(d, ApertureData)


# ==============================================================================
# Panels -- each accepts PanelData or ApertureData
# ==============================================================================
def panel_reconstruction(ax, d, title=None, legend=True):
    """The reconstructed cross spectrum over the truth.

    PanelData: one curve per mode, reference curves depending on whether a
    split (U_exact) exists. ApertureData: R + U at every aperture, R alone
    dotted.
    """
    sym, L = TRACER_SYM, labels()
    if _is_aperture(d):
        ax.loglog(d.k, np.abs(d.P_true),
                  label=rf'truth $\delta_m\times\delta_{{{sym}}}$',
                  **REF_STYLE['target'])
        for x in d.apertures:
            ax.loglog(d.k, np.abs(d.total_model(x)), color=d.colours[x],
                      lw=2.0, label=rf'$R+U$, $x={x:g}$')
            ax.loglog(d.k, np.abs(d.runs[x]['R_model']), color=d.colours[x],
                      ls=':', lw=1.2)
        legend_kw = dict(fontsize=SMALL_LEGEND)
    else:
        if d.has_exact:
            ax.loglog(d.k, np.abs(d.P_matter_gas_true),
                      label=rf'full truth $\delta_m\times\delta_{{{sym}}}$ '
                            rf'(incl. mass below the catalogue)',
                      **REF_STYLE['truth'])
            ax.loglog(d.k, np.abs(d.P_target), label='target: R + U_exact',
                      **REF_STYLE['target'])
            ax.loglog(d.k, np.abs(d.R), label=r'R over $[M_r, M_{\max}]$',
                      **REF_STYLE['resolved'])
        else:
            ax.loglog(d.k, np.abs(d.P_matter_gas_true),
                      label=rf'truth $\delta_m\times\delta_{{{sym}}}$',
                      **REF_STYLE['target'])
            ax.loglog(d.k, np.abs(d.R), label=r'R over $[M_r, M_{\max}]$',
                      **REF_STYLE['resolved'])
        for mode, res in d.results.items():
            ax.loglog(d.k, np.abs(d.R + res['U']), label=f'+ {mode}',
                      **mode_style(mode))
        if d.P_dm_gas is not None:
            ax.loglog(d.k, np.abs(d.P_dm_gas),
                      label=rf'$\delta_{{\rm dm}}\times\delta_{{{sym}}}$ '
                            rf'(definitional mismatch)', **REF_STYLE['mismatch'])
        if d.P_selfpair is not None and np.any(d.P_selfpair > 0):
            ax.loglog(d.k, np.abs(d.P_selfpair), label='self-pair term (removed)',
                      **REF_STYLE['selfpair'])
        legend_kw = {}
    _guides(ax, d, ylabel=L['spectrum'], title=title,
            legend=legend and legend_kw)


def panel_rec_ratio(ax, d):
    """The residual strip under the reconstruction.

    PanelData: (R + U_model)/(R + U_exact) in validation, rec/truth otherwise
    (tighter y range for the near-1 validation comparison). ApertureData:
    (R + U)/P^{m x} at every aperture -- the same quantity as the production
    case above, so it carries the same label.
    """
    L = labels()
    with _quiet():
        if _is_aperture(d):
            for x in d.apertures:
                ax.semilogx(d.k, d.total_model(x) / d.runs[x]['total'],
                            color=d.colours[x], lw=2.0)
            ylim, ylabel = APERTURE_REC_RATIO_YLIM, L['rec_ratio']
        else:
            if d.has_exact:
                ax.semilogx(d.k, d.P_matter_gas_true / d.P_target,
                            **REF_STYLE['truth'])
            ax.semilogx(d.k, d.R / d.P_target, **REF_STYLE['resolved'])
            for mode, res in d.results.items():
                ax.semilogx(d.k, (d.R + res['U']) / d.P_target,
                            **mode_style(mode))
            ylim = (0.6, 1.5) if d.has_exact else (0.0, 2.0)
            ylabel = L['rec_ratio_split'] if d.has_exact else L['rec_ratio']
    _guides(ax, d, level=1.0, band=TOTAL_ERR_BAND, ylim=ylim, ylabel=ylabel,
            xlabel=LABEL_K)


def _scale_below_Ny(ax, d, curves):
    """Set the y range from k <= k_Ny only.

    S_eff is a ratio of measured spectra and both can cross zero beyond the
    Nyquist frequency, where |.| on a log axis produces excursions of several
    decades; left to autoscale they flatten everything below k_Ny into a line.
    Scaling in y rather than clipping in x keeps the full k range on the axis,
    so this panel can share its x axis with the one above it.
    """
    below = np.asarray(d.k) <= d.k_Ny
    vals = [np.abs(np.asarray(c))[below] for c in curves if c is not None]
    finite = np.concatenate([v[np.isfinite(v) & (v > 0)] for v in vals]) \
        if vals else np.array([])
    if finite.size:
        ax.set_ylim(0.5 * finite.min(), 2.0 * finite.max())


def panel_shape(ax, d, title=None, legend=True, ylim=_KEEP):
    """The shape factor S(k) = <u_m T>_w against the S_eff it would need.

    PanelData: one curve per mode, plus S_exact / S_eff when present.
    ApertureData: model solid and S_eff dashed, one colour per aperture --
    the solid curves barely move while the dashed ones do: what the aperture
    changes is the matter the template stands in for, not the template.

    `ylim` defaults to SHAPE_YLIM so that every shape panel, in either
    experiment and in a 2x2 or on its own, is read against the same decades.
    Pass ylim=None to autoscale instead (see _scale_below_Ny). The x range is
    left alone either way.
    """
    L = labels()
    if _is_aperture(d):
        curves = []
        for x in d.apertures:
            S, S_eff = d.S(x), d.S_eff(x)
            if S is not None:
                ax.loglog(d.k, np.abs(S), color=d.colours[x], lw=2.2)
            ax.loglog(d.k, np.abs(S_eff), color=d.colours[x], lw=1.6, ls='--')
            curves += [S, S_eff]
        ax.axhline(1.0, c='k', lw=0.8)
        legend_kw = dict(fontsize=SMALL_LEGEND, handles=[
            _proxy(r'$S(k)$, model', color='0.3', ls='-', lw=2.2),
            _proxy(L['seff'], color='0.3', ls='--', lw=1.6)])
    else:
        for mode, res in d.results.items():
            ax.loglog(d.k, np.abs(res['S']), label=rf'$S(k)$, {mode}',
                      **mode_style(mode))
        if d.S_exact is not None:
            ax.loglog(d.k, np.abs(d.S_exact), label=r'$S(k)$ exact (measured)',
                      **REF_STYLE['exact'])
        if d.S_eff is not None:
            ax.loglog(d.k, np.abs(d.S_eff), label=L['seff'] + ', measured',
                      **REF_STYLE['seff'])
        ax.axhline(1.0, c='grey', lw=0.6)
        curves = [res['S'] for res in d.results.values()] + [d.S_exact, d.S_eff]
        legend_kw = dict(fontsize=SMALL_LEGEND)
    if ylim is _KEEP:
        ylim = SHAPE_YLIM      # read late, so rebinding the constant works
    if ylim is None:
        _scale_below_Ny(ax, d, curves)
    _guides(ax, d, ylim=ylim, ylabel=LABEL_SHAPE, title=title,
            legend=legend and legend_kw)


def panel_dP(ax, d: PanelData, legend=True):
    """The correction on its own: U_model/U_exact when a split exists (the
    correction scored against the hidden bins), |U| otherwise."""
    if d.has_exact:
        with _quiet():
            for mode, res in d.results.items():
                ax.semilogx(d.k, res['U'] / d.U_exact, label=mode,
                            **mode_style(mode))
        _guides(ax, d, level=1.0, band=0.1, ylim=(0.0, 2.0),
                ylabel=LABEL_DP_RATIO, xlabel=LABEL_K,
                legend=legend and dict(ncol=2))
    else:
        for mode, res in d.results.items():
            ax.loglog(d.k, np.abs(res['U']), label=mode, **mode_style(mode))
        _guides(ax, d, ylabel=r'$\Delta P(k)\,L_{\rm box}^2$', xlabel=LABEL_K,
                legend=legend and {})


def panel_error(ax, d, legend=True):
    """The Eq. (59) error budget, split into its two halves. Writing the
    measured truth P = R* + U* as the labels do:

        (R + U)/P - 1  =  (R - R*)/P   profile error
                       +  (U - U*)/P   template error

    The identity is exact, so the components add to the total at every k --
    if they visibly do not, the arrays are not on the same k grid. The split
    says WHERE a mode goes wrong: profile = resolved bins with the wrong
    shape, template = the correction being the wrong size. A mode can look
    fine in total while both terms are large and cancelling.

    PanelData: all three curves for d.error_mode alone. ApertureData: profile
    (dashed) and template (solid) per aperture, the total omitted -- the y
    axis is still the total, so the two experiments name it the same way.
    """
    L = labels()
    if _is_aperture(d):
        if d.error_mode:
            for x in d.apertures:
                e = d.runs[x]['errors'][d.error_mode]
                ax.semilogx(d.k, e['profile'], color=d.colours[x], ls='--',
                            lw=1.8, alpha=0.85)
                ax.semilogx(d.k, e['template'], color=d.colours[x], ls='-',
                            lw=2.2)
                if 'above' in e:
                    ax.semilogx(d.k, e['above'], color=d.colours[x], ls=':',
                                lw=1.4)
        handles = [_proxy(L['template'], color='0.3', ls='-', lw=2.2),
                   _proxy(L['profile'], color='0.3', ls='--', lw=1.8)]
        handles += [_proxy(rf'$x={x:g}$', color=d.colours[x], lw=2.2)
                    for x in d.apertures]
        legend_kw = dict(handles=handles, ncol=2, fontsize=SMALL_LEGEND)
        ylabel, xlabel = L['total'], LABEL_K
    else:
        if not d.has_ustar:
            raise ValueError('panel_error needs R_star and U_star; run '
                             'pmxlib.rstar_ustar to produce them')
        res = d.results.get(d.error_mode)
        if res is None:
            raise KeyError(f'mode {d.error_mode!r} not among {list(d.results)}')
        denom = d.R_star + d.U_star
        if d.V_star is not None:
            denom = denom + d.V_star
        with _quiet():
            curves = {'total': (d.R + res['U']) / denom - 1.0,
                      'profile': (d.R - d.R_star) / denom,
                      'template': (res['U'] - d.U_star) / denom}
            if d.V_star is not None:
                curves['above'] = -d.V_star / denom
        base = mode_style(d.error_mode)
        for key, curve in curves.items():
            # zorder so the total stays visible where one term dominates.
            ax.semilogx(d.k, curve, label=L[key],
                        zorder=3 if key == 'total' else 2,
                        **dict(base, **COMPONENT_STYLE[key]))
        legend_kw = dict(fontsize=SMALL_LEGEND)
        ylabel, xlabel = L['total'], None
    _guides(ax, d, level=0.0, band=TOTAL_ERR_BAND, ylim=TOTAL_ERR_YLIM,
            ylabel=ylabel, xlabel=xlabel, legend=legend and legend_kw)


def panel_shell_mass(ax, d: ApertureData):
    """Which host bins the matter recovered by the aperture belonged to:
    (R_star_i(x) - R_star_i(1))/P^me per bin at three wavenumbers, with an
    upper axis reading the same bins as pi/r200m."""
    x_top = d.x_top
    run = d.runs.get(x_top, {})
    if x_top <= 1.0 or 'shell_per_bin' not in run:
        return
    shell_i, occ = run['shell_per_bin'], d.occupied
    for kk, ls in SHELL_K_STYLES:
        if kk > d.k[-1]:
            continue
        with _quiet():
            vals = np.array([np.interp(kk, d.k, shell_i[j] / d.base['total'])
                             for j in range(shell_i.shape[0])])
        ax.step(d.logM_cen[occ], vals[occ], where='mid', ls=ls, lw=2.0,
                color=d.colours[x_top], label=rf'$k={kk:g}$')
    ax.axhline(0.0, c='k', lw=0.8)
    ax.set_xlabel(LABEL_HOST_BIN)
    ax.set_ylabel(labels()['shell'].replace('(x)', rf'({x_top:g})'))
    ax.legend(frameon=False, fontsize=BIG_LEGEND)

    axt = ax.twiny()
    axt.set_xlim(ax.get_xlim())
    step = max(1, int(np.count_nonzero(occ)) // 5)
    axt.set_xticks(d.logM_cen[occ][::step])
    axt.set_xticklabels([f'{np.pi / r:.1f}' for r in d.r200m[occ][::step]])
    axt.set_xlabel(LABEL_PI_OVER_R200M)
    ax.tick_params(top=False)   # else the rc's top ticks double the twin's


# ==============================================================================
# Figures -- each returns an unsaved Figure
# ==============================================================================
def _two_column_axes(fig):
    """The 2x2 skeleton: a spectrum with a thin residual strip on the left,
    two comparably interesting panels on the right. Both columns share their
    x axis down, so only the bottom row carries ticks and a k label."""
    outer = fig.add_gridspec(1, 2, width_ratios=PANEL_WIDTH_RATIOS,
                             wspace=PANEL_WSPACE)
    gs_l = outer[0, 0].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
    gs_r = outer[0, 1].subgridspec(2, 1, height_ratios=[1, 1], hspace=0.08)
    ax_lt = fig.add_subplot(gs_l[0])
    ax_lb = fig.add_subplot(gs_l[1], sharex=ax_lt)
    ax_rt = fig.add_subplot(gs_r[0])
    ax_rb = fig.add_subplot(gs_r[1], sharex=ax_rt)
    for ax in (ax_lt, ax_rt):
        plt.setp(ax.get_xticklabels(), visible=False)
    set_ylabel_x([ax_lt, ax_lb], YLABEL_X_LEFT)
    set_ylabel_x([ax_rt, ax_rb], YLABEL_X_RIGHT)
    return ax_lt, ax_lb, ax_rt, ax_rb


def panel_figure(d, kind=None):
    """The 2x2 panel. `kind` is 'validation', 'production' or 'aperture';
    inferred from the data when None.

    validation  reconstruction / residual left; S(k) / U_model-U_exact right.
                No titles, no right-hand legends: the left panel names every
                mode and the colour/dash pairing is shared across all four.
    production  same left column scored against the measured truth; right
                column is the Eq. (59) error budget for d.error_mode over
                S(k) with S_eff overlaid (legend on, so the measured curve
                can be told from the modelled ones).
    aperture    the same skeleton and the same right column for the aperture
                sweep: error split over S(k)/S_eff, one colour per aperture.

    Both columns share their x axis down, so the k label appears once per
    column and the four panels read against a single set of k gridlines.
    """
    if kind is None:
        kind = ('aperture' if _is_aperture(d)
                else 'validation' if d.has_exact else 'production')
    if kind == 'validation' and not d.has_exact:
        raise ValueError('validation panel needs U_exact; use production '
                         'when there is no split')
    if kind == 'production' and not d.has_ustar:
        raise ValueError('production panel needs R_star and U_star from the '
                         'pmxlib.rstar_ustar cache')
    with plt.rc_context(PANEL_RC):
        fig = plt.figure(figsize=PANEL_FIGSIZE, dpi=DPI)
        ax_lt, ax_lb, ax_rt, ax_rb = _two_column_axes(fig)
        panel_reconstruction(ax_lt, d)
        panel_rec_ratio(ax_lb, d)
        if kind == 'validation':
            panel_shape(ax_rt, d, legend=False)
            panel_dP(ax_rb, d, legend=False)
        else:
            panel_error(ax_rt, d)
            panel_shape(ax_rb, d)
        # The right column shares one x axis: label the bottom of it, and
        # drop any label a panel set for its own standalone use.
        ax_rt.set_xlabel('')
        ax_rb.set_xlabel(LABEL_K)
    return fig


def stacked_figure(d: PanelData, which='reconstruction', title=None):
    """Standalone two-row figure: 'reconstruction' (spectrum over residual)
    or 'shape' (S(k) over the correction)."""
    ratios, hspace, top, bottom = {
        'reconstruction': ([3, 1], 0.05, panel_reconstruction, panel_rec_ratio),
        'shape': ([1, 1], 0.08, panel_shape, panel_dP),
    }[which]
    with plt.rc_context(PANEL_RC):
        fig, axes = plt.subplots(2, 1, figsize=SINGLE_FIGSIZE, sharex=True,
                                 dpi=DPI, gridspec_kw=dict(height_ratios=ratios,
                                                           hspace=hspace))
        top(axes[0], d, title=title)
        bottom(axes[1], d)
        set_ylabel_x(axes, YLABEL_X_SINGLE)
    return fig


def ansatz_figure(k, k_Ny, curves: Sequence[tuple], title=None):
    """Does the tracer cancel in the ratio? `curves` is (label, T_e, T_c):
    what the ansatz wants (solid) against the CDM proxy it uses (dashed).
    Sequential colours, deliberately outside the mode palette -- here a
    colour means a mass bin, not an extrapolation."""
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


def shell_mass_figure(d: ApertureData):
    """Standalone: where the recovered matter came from, and nothing else."""
    with plt.rc_context(PANEL_RC):
        fig, ax = plt.subplots(figsize=SHELL_FIGSIZE, dpi=DPI)
        panel_shell_mass(ax, d)
        set_ylabel_x([ax], YLABEL_X_SINGLE)
        fig.tight_layout()
    return fig


# Backwards-compatible names.
validation_panel = partial(panel_figure, kind='validation')
production_panel = partial(panel_figure, kind='production')
aperture_panel = partial(panel_figure, kind='aperture')
reconstruction_figure = partial(stacked_figure, which='reconstruction')
shape_figure = partial(stacked_figure, which='shape')