#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fourier transforms and k binning, one convention for the 2-D and 3-D grids.

Every spectrum in the pipeline is measured the same way on either grid:

    fft_a = fft(delta_a)                          # rfftn / N^d
    b     = make_binning(ngrid, box, k_bins, dim) # once per grid
    P     = bin_spectrum(fft_a * conj(fft_b), b)  # per spectrum

`dim` is 2 for the projected grid, 3 for the cubic one. Three per-mode factors
go into the bin average; two of them are applied on the 3-D grid only, which
keeps the 2-D bin average the plain mean over the rfft2 cells that the v1
caches were measured with:

  conjugate pairs   3-D only. rfftn stores the last axis only up to Nyquist,
                    so modes at k_last = 0 and (even N) k_last = Nyquist carry
                    both halves of a conjugate pair; they get weight 1/2. That
                    leaves the mean unchanged in expectation and makes `nmodes`
                    the count of independent modes. On the 2-D grid every cell
                    counts once, so `nmodes` there counts cells.
  in-bin measure    |k|^(2 - d). An annulus holds modes in proportion to k dk
                    and a shell to k^2 dk, so over a bin of finite width the
                    two would average P(k) with different weights; this makes
                    both average with the annulus's k dk, which is what lets a
                    3-D bin be spliced onto a 2-D one. It is 1 in 2-D. On this
                    pipeline's log grid it takes the RMS disagreement between
                    the two lattices below k = 1.5 from 0.89% to 0.36%.
  TSC window        3-D only: 1 / W(k)^2, W = prod_i sinc^3(k_i Delta / 2).
                    Every field is TSC-painted, so every cross of two fields
                    carries W^2; dividing the product is the same as dividing
                    each field by W. On the 384^3 grid <W^2> is 0.83 at
                    k = 0.5 and 0.45 at k = 1, so without it the splice would
                    step. On the 2048^2 grid it is 0.993 at k = 0.5, and it is
                    left in: it cancels in every ratio taken on that grid.

P is returned in the projected convention, P = L^2 <prod> on either grid. In
2-D that is the projected spectrum itself. In 3-D it is P_3d / L: delta_2d is
the column mean of delta_3d, so P_2d(k) = P_3d(k_perp, k_z = 0) / L, and under
statistical isotropy the full shell has the same expectation as that plane.
"""
import numpy as np
from scipy.fft import fftfreq, rfftfreq, rfftn

__all__ = ['fft', 'k_modulus', 'tsc_window', 'log_k_bin_edges',
           'make_binning', 'bin_spectrum']


def fft(delta):
    """Normalised real FFT of a 2-D or 3-D overdensity field: rfftn / N^d."""
    out = rfftn(delta)
    out *= 1.0 / delta.size
    return out


def _axis_k(ngrid, box, dim):
    """Per-axis wavenumbers on the rfftn layout, broadcastable to its shape."""
    k_full = 2.0 * np.pi * fftfreq(ngrid, d=box / ngrid)
    k_last = 2.0 * np.pi * rfftfreq(ngrid, d=box / ngrid)
    axes = [k_full] * (dim - 1) + [k_last]
    shape = [1] * dim
    out = []
    for a, k in enumerate(axes):
        s = list(shape)
        s[a] = k.size
        out.append(k.reshape(s))
    return out


def k_modulus(ngrid, box, dim):
    """|k| on the rfftn layout of an ngrid^dim grid."""
    return np.sqrt(sum(k ** 2 for k in _axis_k(ngrid, box, dim)))


def tsc_window(ngrid, box, dim):
    """W(k) = prod_i sinc^3(k_i Delta / 2), Delta = L / ngrid, on the rfftn
    layout. One factor of the TSC assignment window."""
    W = 1.0
    for k in _axis_k(ngrid, box, dim):
        x = 0.5 * k * (box / ngrid)
        s = np.ones_like(x)
        nz = x != 0
        s[nz] = np.sin(x[nz]) / x[nz]
        W = W * s ** 3
    return W


def log_k_bin_edges(k_min, k_max, per_decade, w_min):
    """Bin edges stepping k -> k + max(w_min, k * dlnk), dlnk = ln10/per_decade.

    Log-spaced, except that no bin is allowed to be narrower than w_min. A box
    resolves nothing below its fundamental k_f = 2 pi / L, so log spacing taken
    literally to k_min would ask for bins far narrower than the mode spacing and
    leave the first several of them empty. w_min = 2 k_f keeps the mode count
    growing smoothly from the first bin.

    The last edge is clipped to k_max. If the remainder that leaves is under
    half the width the rule asked for, it is absorbed into the bin below rather
    than left as a sliver of a handful of modes.
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


def make_binning(ngrid, box, k_bins, dim):
    """Per-mode bin index and weights for an ngrid^dim grid, built once.

    Digitising the ~ngrid^dim / 2 modes costs more than an FFT, and a bundle
    bins some thirty-five spectra on each grid, so this is done once and the
    result handed to bin_spectrum. `k_bins` is passed in rather than derived,
    so both grids land on the same edges and stitching is a splice.

    Returns a dict:
      dim, ngrid, box, k_bins, k_mid, k_Nyquist
      nmodes      independent modes per bin
      nmodes_eff  (sum w)^2 / sum w^2, the count the bin's variance goes as
      k_eff       w-weighted mean |k| of the bin: the k its value refers to
      index, weight, norm   what bin_spectrum needs
    """
    ngrid, box, dim = int(ngrid), float(box), int(dim)
    k_bins = np.asarray(k_bins, dtype=float)
    nbins = k_bins.size - 1

    k_grid = k_modulus(ngrid, box, dim)
    shape = k_grid.shape
    kk = k_grid.ravel()
    del k_grid
    index = np.digitize(kk, k_bins) - 1
    index[(index < 0) | (index >= nbins)] = nbins        # dump bin
    index = index.astype(np.int32)

    half = np.ones(shape[-1])
    if dim == 3:                                         # conjugate pairs
        half[0] = 0.5
        if ngrid % 2 == 0:
            half[-1] = 0.5
    pair = np.broadcast_to(half, shape).ravel()

    # |k|^(2 - d): 1 on the 2-D grid, 1/|k| on the 3-D one. The k = 0 mode
    # sits in the dump bin either way.
    measure = np.divide(1.0, kk ** (dim - 2), out=np.zeros_like(kk),
                        where=kk > 0)
    w = pair * measure

    def per_bin(vals):
        return np.bincount(index, weights=vals, minlength=nbins + 1)[:nbins]

    norm = per_bin(w)
    # Variance of a weighted mean goes as sum w^2 / (sum w)^2 over DISTINCT
    # modes: a conjugate pair is one mode of weight `measure`, so the sum of
    # squares carries pair * measure^2 rather than (pair * measure)^2.
    sq = per_bin(pair * measure ** 2)
    with np.errstate(divide='ignore', invalid='ignore'):
        nmodes_eff = np.where(sq > 0, norm ** 2 / sq, 0.0)
        k_eff = np.where(norm > 0, per_bin(kk * w) / norm, np.nan)

    # TSC deconvolution on the 3-D grid only.
    weight = w / tsc_window(ngrid, box, dim).ravel() ** 2 if dim == 3 else w
    return dict(dim=dim, ngrid=ngrid, box=box, k_bins=k_bins,
                k_mid=0.5 * (k_bins[:-1] + k_bins[1:]),
                k_Nyquist=np.pi * ngrid / box,
                nmodes=per_bin(pair), nmodes_eff=nmodes_eff, k_eff=k_eff,
                index=index, weight=weight, norm=norm)


def bin_spectrum(prod, binning):
    """The spectrum of a mode-by-mode product, in the projected convention.

    `prod` is fft_a * conj(fft_b), or its real part, on the rfftn layout the
    binning was built for. Returns L^2 times the w-weighted (and, in 3-D,
    window-deconvolved) mean of prod over each k bin; see the module docstring.
    """
    b = binning
    nbins = b['k_bins'].size - 1
    num = np.bincount(b['index'], weights=np.asarray(prod).ravel() * b['weight'],
                      minlength=nbins + 1)[:nbins]
    with np.errstate(divide='ignore', invalid='ignore'):
        mean = np.where(b['norm'] > 0, num / b['norm'], np.nan)
    return mean * b['box'] ** 2
