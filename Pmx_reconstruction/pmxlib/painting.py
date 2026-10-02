#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""2-D and 3-D TSC painting. The transforms and k binning live in kspace."""
import numpy as np
from abacusnbody.analysis.tsc import tsc_parallel

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


# 3-D painting, for the large-scale half of a stitched measurement
def _scaled_weights(weights):
    """(weights / mean, mean), so tsc accumulates in float32 around unity."""
    if weights is None:
        return None, 1.0
    w = np.asarray(weights, dtype=np.float64)
    w_mean = float(w.mean()) if w.size else 1.0
    if not np.isfinite(w_mean) or w_mean == 0.0:
        w_mean = 1.0
    return np.ascontiguousarray(w / w_mean, dtype=np.float32), w_mean


def paint_3d(pos, weights, box, ngrid, nthread, out=None):
    """Mass on the 3-D grid, no normalisation.

    With `out` (a float32 (ngrid,)*3 array) the mass is added to it in place
    and nothing is allocated beyond one grid; rstar_ustar paints thirty bins
    that way.
    """
    pos = np.ascontiguousarray(pos, dtype=np.float32)   # tsc wraps this IN PLACE
    w, w_mean = _scaled_weights(weights)
    grid = np.zeros((ngrid, ngrid, ngrid), dtype=np.float32)
    tsc_parallel(pos, grid, box, weights=w, nthread=(nthread or 1))
    if out is None:
        return grid.astype(np.float64) * w_mean
    grid *= np.float32(w_mean)
    out += grid
    return out


def paint_uniform_random_3d(mass, box, ngrid, nthread, rng, chunk=None):
    """Mass of a uniformly random catalogue on the 3-D grid."""
    mass = np.asarray(mass)
    m_mean = float(mass.mean(dtype=np.float64)) if mass.size else 1.0
    if not np.isfinite(m_mean) or m_mean == 0.0:
        m_mean = 1.0

    grid = np.zeros((ngrid, ngrid, ngrid), dtype=np.float32)
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
    return grid.astype(np.float64) * m_mean


def overdensity_3d(dens, ngrid):
    """rho/rhobar - 1 from a painted mass grid, in place."""
    dens /= (dens.sum(dtype=np.float64) / ngrid ** 3)
    dens -= 1.0
    return dens
