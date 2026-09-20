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


def bin_power_spectrum_2d(P_k_2d, k_mag, ngrid, box, nkbins=None, k_min=None, k_max=None):
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
    """
    if nkbins is None:
        nkbins = max(32, ngrid // 10)
    if k_min is None:
        k_min = 2 * np.pi / box
    if k_max is None:
        k_max = float(np.max(k_mag))

    k_bins = np.linspace(k_min, k_max, nkbins)
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
