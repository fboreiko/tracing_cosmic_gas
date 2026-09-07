#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""2-D TSC painting, the assignment window, and azimuthal averaging.

These lived twice: `paint` / `tsc_window_2d` / `azimuthal_mean` in
measure_u_tilde.py and `_paint_uniform_random` / `_mass_moments` in
predict_Pmx_from_Phx.py, with the docstring below copied between them by hand.

paint_2d and paint_uniform_random_2d stay SEPARATE functions on purpose: the
first normalises weights by their own mean, the second by the global mean
particle mass which it restores at the end, specifically so that chunking cannot
change the answer. Do not merge them.
"""
import numpy as np
from abacusnbody.analysis.tsc import tsc_parallel

from utils.power_spectrum_utils import bin_power_spectrum_2d

# Number of z cells used by the painting functions. See the note in paint_2d;
# 3 is the smallest value that is safe against the _rightwrap index range.
_NZ_PAINT = 3


def paint_2d(pos, weights, box, ngrid, nthread):
    """Mass on the 2-D grid (column sum), no normalisation.

    NOT a call to tsc_parallel with a 2-D grid. In abacusutils 2.1.2 the 2-D
    branch of _tsc_scatter is broken: it sets izw = int16(0) but still writes
    density[ix, iy, izw], i.e. three indices into a two-dimensional array, so
    numba fails to type it. (compute_delta_2d in utils/ hits the same bug on
    this version; if the cached delta fields were built with an older
    abacusutils, that is why they exist and this did not.)

    Instead we paint into a genuine 3-D grid with _NZ_PAINT cells along z and
    sum over z. The TSC weights in z sum to unity for every particle, so the
    column sum is EXACTLY the 2-D result and is independent of _NZ_PAINT -- it
    is not an approximation, and it is not a different smoothing. _NZ_PAINT = 1
    would be the obvious choice but is unsafe: iz = round(z/box) can be 1, and
    then izp1 = _rightwrap(2, 1) = 1 indexes past the end of a size-1 axis,
    which numba does not bounds-check.

    Weights are normalised by their mean before painting and the mean restored
    afterwards, so the float32 accumulation happens on numbers of order unity
    rather than on raw particle masses.
    """
    pos = np.ascontiguousarray(pos, dtype=np.float32)   # tsc wraps this IN PLACE
    w = np.asarray(weights, dtype=np.float64)
    w_mean = float(w.mean()) if w.size else 1.0
    if not np.isfinite(w_mean) or w_mean == 0.0:
        w_mean = 1.0
    grid = np.zeros((ngrid, ngrid, _NZ_PAINT), dtype=np.float32)
    tsc_parallel(pos, grid, box,
                 weights=np.ascontiguousarray(w / w_mean, dtype=np.float32),
                 nthread=(nthread or 1))
    return grid.sum(axis=2, dtype=np.float64) * w_mean


def paint_uniform_random_2d(mass, box, ngrid, nthread, rng, chunk=None):
    """Mass of a uniformly random catalogue on the 2-D grid (column sum).

    Same painting trick as paint_2d (see its docstring for why the 3-D grid and
    the column sum are exact).

    The random positions are generated in chunks and painted straight into the
    shared accumulator, so peak memory is chunk * 12 bytes rather than one
    float32 (N, 3) array for the whole species. Weights are divided by the
    GLOBAL mean particle mass (not the chunk's) and the mean restored at the
    end, so the float32 accumulation happens on numbers of order unity and the
    chunking cannot change the answer.
    """
    # Left in its native dtype on purpose: an .astype(float64) of a species'
    # whole mass array is one of the largest allocations this pipeline could
    # make, and nothing here needs it. Only the reductions are done in float64.
    mass = np.asarray(mass)
    m_mean = float(mass.mean(dtype=np.float64)) if mass.size else 1.0
    if not np.isfinite(m_mean) or m_mean == 0.0:
        m_mean = 1.0

    grid = np.zeros((ngrid, ngrid, _NZ_PAINT), dtype=np.float32)
    n_p = mass.size
    step = n_p if chunk is None else max(1, int(chunk))
    for a in range(0, n_p, step):
        b = min(a + step, n_p)
        pos = rng.uniform(0.0, box, size=(b - a, 3)).astype(np.float32)
        # uniform() is half-open in float64, but the float32 cast can round up
        # to exactly box; tsc wants [0, L). It also wraps pos IN PLACE, which is
        # harmless here because pos is a temporary.
        pos[pos >= box] -= np.float32(box)
        tsc_parallel(pos, grid, box,
                     weights=np.ascontiguousarray(mass[a:b] / m_mean,
                                                  dtype=np.float32),
                     nthread=(nthread or 1))
        del pos
    return grid.sum(axis=2, dtype=np.float64) * m_mean


def mass_moments(mass, chunk=None):
    """(sum m, sum m^2) in float64, accumulated in chunks.

    np.sum(mass ** 2) would materialise a second array the size of the whole
    species; at FLAMINGO particle counts that is worth avoiding for a two-line
    loop. Chunking also keeps the float32 case from losing the sum to
    accumulation error, since each partial sum is taken in float64.
    """
    mass = np.asarray(mass)
    n_p = mass.size
    step = n_p if chunk is None else max(1, int(chunk))
    s1 = s2 = 0.0
    for a in range(0, n_p, step):
        m = np.asarray(mass[a:min(a + step, n_p)], dtype=np.float64)
        s1 += float(m.sum())
        s2 += float(np.dot(m, m))
    return s1, s2


def tsc_window_2d(box, ngrid):
    """W(k) = prod_i sinc^3(k_i Delta/2) on the fft2 layout, Delta = L/ngrid.

    One factor of the TSC assignment window. Every spectrum in the pipeline is
    a product of two painted fields and carries W^2, which cancels in the ratio
    u~_m = R_star_i / (f_i P_halo_x). A stacked profile is ONE painted field and
    carries a single W that does not cancel; it must be divided out, or the
    stack reads low by 10% at k = 2 and 30% at k = 5 for ngrid = 2048.
    """
    kf = 2.0 * np.pi * np.fft.fftfreq(ngrid, d=box / ngrid)
    arg = 0.5 * kf * (box / ngrid)
    s = np.ones_like(arg)
    nz = arg != 0
    s[nz] = np.sin(arg[nz]) / arg[nz]
    return (s[:, None] * s[None, :]) ** 3


def azimuthal_mean(field, box, ngrid, k_bins, deconvolve_tsc=False):
    """Azimuthal average of a full fft2 layout field over the bundle's k bins.

    Self-contained (own fftfreq grid), used only for the profile transforms,
    where the absolute normalisation is fixed by dividing by the k=0 mode and
    the repo's binning helper is not needed. With deconvolve_tsc the field is
    divided by one TSC window before averaging.
    """
    if deconvolve_tsc:
        field = field / tsc_window_2d(box, ngrid)
    kf = 2.0 * np.pi * np.fft.fftfreq(ngrid, d=box / ngrid)
    kk = np.sqrt(kf[:, None] ** 2 + kf[None, :] ** 2)
    ib = np.digitize(kk.ravel(), k_bins) - 1
    ok = (ib >= 0) & (ib < k_bins.size - 1)
    num = np.bincount(ib[ok], weights=field.ravel()[ok], minlength=k_bins.size - 1)
    den = np.bincount(ib[ok], minlength=k_bins.size - 1).astype(float)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(den > 0, num / den, np.nan)


def binned_spectrum(cfg, prod, k_grid, nkbins):
    """The one and only k-binning convention in this file.

    Takes the real part of a product of two transforms (or |transform|^2),
    azimuthally averages it over the k bins and applies the L_box^2 factor that
    every spectrum in this repo carries.

    It is a module-level function rather than a closure inside measure_spectra
    because the self-pair spectra of section 3b are measured in a completely
    separate pass, possibly years apart in wall-clock time, and MUST come out in
    the identical convention or the subtraction is meaningless. Sharing the
    function is the cheapest way to guarantee that.
    """
    kb, kc, P = bin_power_spectrum_2d(prod, k_grid, cfg.grid, cfg.box,
                                      nkbins=nkbins, k_min=2.0 * np.pi / cfg.box)
    return np.asarray(kb), np.asarray(kc), np.asarray(P) * cfg.box ** 2