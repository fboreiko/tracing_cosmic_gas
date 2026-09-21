"""Projected density fields, their FFTs, and azimuthally averaged spectra.

Numerics only: this module draws nothing and deliberately imports no plotting.
It used to set rcParams['font.family'] at import, which restyled matplotlib for
every importer -- including the four pmxlib modules that want nothing from here
but the k binning. Each script that draws now owns its own style: HATF sets
rcParams at the top of itself, and pmxlib.plotting applies PANEL_RC through a
scoped plt.rc_context so it cannot leak either.
"""
import numpy as np
from scipy.fft import rfftfreq, fftfreq, rfft2, rfftn


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


def compute_3d_fft(delta_3d, ngrid):
    """Compute a normalized 3D FFT of an overdensity field."""
    delta_fft = rfftn(delta_3d)
    delta_fft *= (1.0 / ngrid ** 3)
    return delta_fft


def compute_k_grid_3d(ngrid, box):
    """Compute the 3D k-magnitude grid for an rfftn output."""
    kx = fftfreq(ngrid, d=box / ngrid) * 2 * np.pi
    kz = rfftfreq(ngrid, d=box / ngrid) * 2 * np.pi
    return np.sqrt(kx[:, None, None] ** 2 + kx[None, :, None] ** 2
                   + kz[None, None, :] ** 2)


def tsc_window_3d(ngrid, box):
    """W(k) = prod_i sinc^3(k_i Delta/2) on the rfftn layout, Delta = L/ngrid.

    One factor of the TSC assignment window. Negligible on the 2048^2
    projected grid, but 1.8% at k = 0.1 and 36% at k = 0.5 on a 256^3 one, so
    the 3-D path deconvolves it and the 2-D path does not.
    """
    def _sinc(k):
        arg = 0.5 * k * (box / ngrid)
        s = np.ones_like(arg)
        nz = arg != 0
        s[nz] = np.sin(arg[nz]) / arg[nz]
        return s

    kx = fftfreq(ngrid, d=box / ngrid) * 2 * np.pi
    kz = rfftfreq(ngrid, d=box / ngrid) * 2 * np.pi
    sx, sz = _sinc(kx), _sinc(kz)
    return (sx[:, None, None] * sx[None, :, None] * sz[None, None, :]) ** 3


class Binner3D:
    """Precomputed spherical binning of an rfftn-layout field.

    Built once per grid and reused for every spectrum measured on it: the
    digitize over ngrid^3/2 modes costs more than the FFT, and a 3-D bundle
    bins about thirty-five spectra.

    Modes at k_z = 0 and k_z = Nyquist carry both halves of a conjugate pair,
    so they get weight 1/2. That leaves the mean unchanged in expectation and
    makes `nmodes` the count of independent modes.

    `k_bins` is passed in rather than derived, so a 3-D run lands on the edges
    the 2-D run already used and stitching is a splice, not an interpolation.

    MATCHING THE 2-D IN-BIN WEIGHTING
    A shell holds modes in proportion to k^2 dk and an annulus in proportion
    to k dk, so over a bin of finite width the two average P(k) with different
    weights. Weighting each mode by 1/|k| makes the two agree in the continuum,
    at a cost in variance of order (bin width / k)^2 -- a part in 10^3 here.

    On this pipeline's grid (681 cMpc/h, 600 linear k bins, 256^3 against
    2048^2) that is worth 3.8% -> 0.6% in the lowest bin and nothing above
    k ~ 0.03, where the two lattices' 0.2% RMS disagreement is which exact
    |k| values happen to fall in a bin, not how the bin is weighted. 0.2% is
    well inside the 2.4% the 3-D measurement has left at k = 0.1, and it
    alternates in sign bin to bin rather than offsetting the curve.
    """

    def __init__(self, ngrid, box, k_bins, deconvolve_tsc=True,
                 match_2d_weighting=True):
        self.ngrid, self.box = int(ngrid), float(box)
        self.k_bins = np.asarray(k_bins, dtype=float)
        self.nbins = self.k_bins.size - 1
        self.k_center = 0.5 * (self.k_bins[:-1] + self.k_bins[1:])
        self.k_Nyquist = np.pi * ngrid / box

        kk = compute_k_grid_3d(ngrid, box)
        ib = np.digitize(kk.ravel(), self.k_bins) - 1
        ib[(ib < 0) | (ib >= self.nbins)] = self.nbins      # dump bin
        self._ib = ib.astype(np.int32)
        del ib

        plane = np.ones((ngrid, ngrid, ngrid // 2 + 1), dtype=np.float64)
        plane[:, :, 0] = 0.5
        if ngrid % 2 == 0:
            plane[:, :, -1] = 0.5
        plane = plane.ravel()
        kk = kk.ravel()

        if match_2d_weighting:
            wk = np.divide(1.0, kk, out=np.zeros_like(kk), where=kk > 0)
        else:
            wk = np.ones_like(kk)
        w = plane * wk

        self.nmodes = self._sum_raw(plane)                  # independent modes
        self._counts = self._sum_raw(w)
        # Variance of a weighted mean: sigma^2 = P^2 sum w^2 / (sum w)^2, over
        # DISTINCT modes -- a conjugate pair is one mode of weight wk, so the
        # sum of squares carries plane * wk^2 rather than (plane * wk)^2.
        sq = self._sum_raw(plane * wk ** 2)
        with np.errstate(divide='ignore', invalid='ignore'):
            self.nmodes_eff = np.where(sq > 0, self._counts ** 2 / sq, 0.0)
            self.k_eff = np.where(self._counts > 0,
                                  self._sum_raw(kk * w) / self._counts, np.nan)
        del kk, wk

        if deconvolve_tsc:
            w = w / tsc_window_3d(ngrid, box).ravel() ** 2
        self._w = w

    def _sum_raw(self, vals):
        return np.bincount(self._ib, weights=vals,
                           minlength=self.nbins + 1)[:self.nbins]

    def __call__(self, field):
        """Window-deconvolved mean of `field` over each k bin."""
        num = self._sum_raw(np.asarray(field).ravel() * self._w)
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(self._counts > 0, num / self._counts, np.nan)


def mode_stats_2d(ngrid, box, k_bins):
    """(nmodes, k_eff) per k bin on the rfft2 layout.

    nmodes drives the sigma ~ 1/sqrt(N) report; k_eff is the mode-weighted
    mean |k| of the bin, which Binner3D's must match once its weighting does.
    """
    k_bins = np.asarray(k_bins, dtype=float)
    nbins = k_bins.size - 1
    kk = compute_k_grid_2d(ngrid, box)
    w = np.ones_like(kk)
    w[:, 0] = 0.5
    if ngrid % 2 == 0:
        w[:, -1] = 0.5
    ib = np.digitize(kk.ravel(), k_bins) - 1
    ib[(ib < 0) | (ib >= nbins)] = nbins
    ib = ib.astype(np.int32)
    n = np.bincount(ib, weights=w.ravel(), minlength=nbins + 1)[:nbins]
    kw = np.bincount(ib, weights=(kk * w).ravel(), minlength=nbins + 1)[:nbins]
    with np.errstate(divide='ignore', invalid='ignore'):
        return n, np.where(n > 0, kw / n, np.nan)
