"""Projected density fields, their FFTs, and azimuthally averaged spectra.

Numerics only: this module draws nothing and deliberately imports no plotting.
It used to set rcParams['font.family'] at import, which restyled matplotlib for
every importer -- including the four pmxlib modules that want nothing from here
but the k binning. Each script that draws now owns its own style: HATF sets
rcParams at the top of itself, and pmxlib.plotting applies PANEL_RC through a
scoped plt.rc_context so it cannot leak either.
"""
import numpy as np
from scipy.fft import rfftfreq, fftfreq, rfft2


def compute_2d_fft(delta_2d, ngrid):
    """Compute a normalized 2D FFT of a projected overdensity field."""
    delta_fft = rfft2(delta_2d)
    delta_fft *= (1.0 / ngrid ** 2)
    return delta_fft


def compute_k_grid_2d(ngrid, box):
    """Compute the 2D k-magnitude grid for an rfft2 output."""
    kx = fftfreq(ngrid, d=box / ngrid) * 2 * np.pi
    ky = rfftfreq(ngrid, d=box / ngrid) * 2 * np.pi
    return np.sqrt(kx[:, np.newaxis] ** 2 + ky[np.newaxis, :] ** 2)


def log_k_bin_edges(k_min, k_max, per_decade, w_min):
    """Bin edges stepping k -> k + max(w_min, k * dlnk), dlnk = ln10/per_decade.

    Log-spaced, except that no bin is allowed to be narrower than w_min. A box
    resolves nothing below its fundamental k_f = 2 pi / L, so log spacing taken
    literally to k_min would ask for bins far narrower than the mode spacing and
    leave the first several of them empty. w_min = 2 k_f keeps the mode count
    growing smoothly from the first bin; k_f alone leaves it jumping about
    (5, 12, 8, 13, ... on FLAMINGO's grid) purely from which lattice points
    happen to land where.

    The last edge is clipped to k_max so the grid ends exactly there. Clipping
    leaves a remainder, and if that remainder is under half the width the rule
    asked for it is absorbed into the bin below rather than left as a sliver --
    otherwise the curve ends on a point of a handful of modes and tens of per
    cent of noise, next to one of a hundred thousand.
    """
    k_min, k_max, w_min = float(k_min), float(k_max), float(w_min)
    dlnk = np.log(10.0) / float(per_decade)
    edges = [k_min]
    while edges[-1] < k_max:
        edges.append(edges[-1] + max(w_min, edges[-1] * dlnk))
    edges[-1] = k_max
    if len(edges) > 2 and (edges[-1] - edges[-2]) < 0.5 * (edges[-2] - edges[-3]):
        del edges[-2]
    return np.asarray(edges, dtype=float)


def bin_mode_stats(k_mag, k_bins):
    """(nmodes, k_eff) per bin: how many grid modes it holds and their mean |k|.

    k_eff is the k the bin's value actually refers to. It is not the midpoint
    of the edges: the modes in a bin are not spread uniformly across it, and
    where a bin is wide compared to the k it sits at -- the lowest few, always
    -- the difference reaches several per cent.
    """
    k_bins = np.asarray(k_bins, dtype=float)
    nbins = k_bins.size - 1
    index = np.digitize(k_mag.ravel(), k_bins) - 1
    keep = (index >= 0) & (index < nbins)
    index = index[keep]
    k_keep = k_mag.ravel()[keep]

    nmodes = np.bincount(index, minlength=nbins).astype(float)
    k_sum = np.bincount(index, weights=k_keep, minlength=nbins)
    with np.errstate(divide='ignore', invalid='ignore'):
        k_eff = np.where(nmodes > 0, k_sum / np.maximum(nmodes, 1), np.nan)
    return nmodes, k_eff


def bin_power_spectrum_2d(P_k_2d, k_mag, ngrid, box, nkbins=None, k_min=None,
                          k_max=None, k_bins=None):
    """Bin a 2D power spectrum into a 1D spectrum as a function of |k|.

    Bin i is the half-open interval [k_bins[i], k_bins[i+1]); modes below
    k_min or at/above k_max fall outside every bin and are dropped, and a bin
    with no modes in it comes back as 0.

    Done in one pass with digitize + bincount rather than a Python loop over
    the bins. The loop rebuilt a full boolean mask over the ngrid^2 grid once
    per bin, i.e. O(nkbins * ngrid^2): at ngrid = 2048 and nkbins = 600 that was
    1.4 s per spectrum, and measure_spectra calls this 64 times. The result is
    the same mean over the same modes; only the order in which each bin's modes
    are summed differs, which moves the answer by ~1e-15 relative.

    `k_bins` supplies the edges directly and overrides nkbins/k_min/k_max. The
    Pmx reconstruction passes log-spaced edges that way; HATF passes none and
    gets the linear grid this function has always built.
    """
    if k_bins is None:
        if nkbins is None:
            nkbins = max(32, ngrid // 10)
        if k_min is None:
            k_min = 2 * np.pi / box
        if k_max is None:
            k_max = float(np.max(k_mag))
        k_bins = np.linspace(k_min, k_max, nkbins)
    else:
        k_bins = np.asarray(k_bins, dtype=float)
    k_center = 0.5 * (k_bins[:-1] + k_bins[1:])
    nbins = k_center.size

    # digitize returns i+1 for k_bins[i] <= k < k_bins[i+1], 0 below the first
    # edge and len(k_bins) at or above the last, so after the shift the modes
    # to keep are exactly those with 0 <= index < nbins.
    index = np.digitize(k_mag.ravel(), k_bins) - 1
    keep = (index >= 0) & (index < nbins)
    index = index[keep]

    counts = np.bincount(index, minlength=nbins)
    totals = np.bincount(index, weights=P_k_2d.ravel()[keep], minlength=nbins)

    P_binned = np.zeros(nbins, dtype=np.float64)
    occupied = counts > 0
    P_binned[occupied] = totals[occupied] / counts[occupied]

    return k_bins, k_center, P_binned


def interp_extrapolate_loglog(k_src, y_src, k_tgt):
    """Log-log interpolate/extrapolate a positive spectrum onto a target k grid."""
    positive = (k_src > 0) & (y_src > 0) & np.isfinite(k_src) & np.isfinite(y_src)
    if np.count_nonzero(positive) < 2:
        raise RuntimeError("Need at least two positive finite points for log-log extrapolation.")

    x = np.log(k_src[positive])
    y = np.log(y_src[positive])
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    x_tgt = np.log(np.asarray(k_tgt))
    y_tgt = np.interp(x_tgt, x, y)

    low = x_tgt < x[0]
    high = x_tgt > x[-1]
    slope_lo = (y[1] - y[0]) / (x[1] - x[0])
    slope_hi = (y[-1] - y[-2]) / (x[-1] - x[-2])

    y_tgt[low] = y[0] + slope_lo * (x_tgt[low] - x[0])
    y_tgt[high] = y[-1] + slope_hi * (x_tgt[high] - x[-1])

    return np.exp(y_tgt)
