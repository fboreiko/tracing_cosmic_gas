#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""2-D TSC painting, the assignment window, and azimuthal averaging."""
import numpy as np
from abacusnbody.analysis.tsc import tsc_parallel

from utils.power_spectrum_utils import bin_power_spectrum_2d

_NZ_PAINT = 3

def paint_2d(pos, weights, box, ngrid, nthread):
    """Mass on the 2-D grid (column sum), no normalisation."""
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
    """Mass of a uniformly random catalogue on the 2-D grid (column sum)."""
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
        pos[pos >= box] -= np.float32(box)
        tsc_parallel(pos, grid, box,
                     weights=np.ascontiguousarray(mass[a:b] / m_mean,
                                                  dtype=np.float32),
                     nthread=(nthread or 1))
        del pos
    return grid.sum(axis=2, dtype=np.float64) * m_mean


def mass_moments(mass, chunk=None):
    """(sum m, sum m^2) in float64, accumulated in chunks."""
    mass = np.asarray(mass)
    n_p = mass.size
    step = n_p if chunk is None else max(1, int(chunk))
    s1 = s2 = 0.0
    for a in range(0, n_p, step):
        m = np.asarray(mass[a:min(a + step, n_p)], dtype=np.float64)
        s1 += float(m.sum())
        s2 += float(np.dot(m, m))
    return s1, s2

# tsc_window_2d and azimuthal_mean are used for measuring u tilde directly from the 2-D projected fields


def tsc_window_2d(box, ngrid):
    """W(k) = prod_i sinc^3(k_i Delta/2) on the fft2 layout, Delta = L/ngrid.
    One factor of the TSC assignment window."""
    kf = 2.0 * np.pi * np.fft.fftfreq(ngrid, d=box / ngrid)
    arg = 0.5 * kf * (box / ngrid)
    s = np.ones_like(arg)
    nz = arg != 0
    s[nz] = np.sin(arg[nz]) / arg[nz]
    return (s[:, None] * s[None, :]) ** 3


def azimuthal_mean(field, box, ngrid, k_bins, deconvolve_tsc=False):
    """Azimuthal average of a full fft2 layout field over the bundle's k bins."""
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
    """The one and only k-binning convention in this pipeline.

    Takes the real part of a product of two transforms (or |transform|^2),
    azimuthally averages it over the k bins and applies the L_box^2 factor that
    every spectrum in this repo carries.

    It is not a wrapper around bin_power_spectrum_2d for its own sake. That
    function is shared with HATF, which does NOT use the L_box^2 convention, so
    the factor cannot live down there. Everything that ends up in a bundle, in
    the R*/U* cache or in a self-pair subtraction has to carry it and the same
    k binning, and those three are measured in separate passes -- possibly
    years apart in wall-clock time. This is the one place that decides, so they
    agree by construction rather than by three copies happening to match.

    k_min is passed explicitly even though it equals bin_power_spectrum_2d's own
    default, so that a change to the default in shared utils/ cannot silently
    move this pipeline's k grid.
    """
    kb, kc, P = bin_power_spectrum_2d(prod, k_grid, cfg.grid, cfg.box,
                                      nkbins=nkbins, k_min=2.0 * np.pi / cfg.box)
    return np.asarray(kb), np.asarray(kc), np.asarray(P) * cfg.box ** 2
