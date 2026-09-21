#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""How a run turns particles into binned spectra: projected, or cubic.

The pipeline measures every spectrum one of two ways, and --geometry picks one
for the whole run. They are alternatives, not halves of one thing.

    2d   particles projected along z onto an ngrid^2 grid. What this pipeline
         has always done. It measures the k_z = 0 plane of the 3-D field, so
         at ngrid = 2048 it reaches k ~ 9 h/cMpc -- but a plane holds k dk
         modes where a shell holds k^2 dk, so it throws away a factor 2k/k_f.

    3d   particles painted onto an ngrid^3 grid. Every mode of the shell:
         sigma falls by sqrt(2k/k_f), which is 4.7x at k = 0.1 and 10x at
         k = 0.5. The cost is k_Nyquist = pi ngrid / L_box with an ngrid^3
         grid to hold, so a 3-D run cannot reach the 2-D run's k_max -- 2048^3
         would be a terabyte of per-bin grids. A 3-D run is a large-scale run.

Both come back in the same units and mean the same thing. delta_2d is the
column MEAN of delta_3d, so its transform is delta_3d(k_perp, k_z = 0) exactly
and P_2d(k) = P_3d(k) / L_box; painting.as_projected applies that factor, so a
2-D and a 3-D bundle can be compared curve to curve. Give them the same --kmax
and --nkbins and they land on identical k edges, bin for bin.

The step that assumes isotropy is the 3-D shell average standing in for the
k_z = 0 plane. True for these real-space positions; not in redshift space.
"""
import numpy as np

from utils.power_spectrum_utils import (Binner3D, bin_power_spectrum_2d,
                                        compute_2d_fft, compute_3d_fft,
                                        compute_k_grid_2d, mode_stats_2d,
                                        tsc_window_2d)

from Pmx_reconstruction.pmxlib.painting import (as_projected, paint_2d, paint_3d,
                                                paint_uniform_random_2d,
                                                paint_uniform_random_3d)

GEOMETRIES = ('2d', '3d')

# Default cubic grid when --geometry 3d is asked for without --ngrid. 384^3
# reaches k_Nyquist = 1.77 h/cMpc on the 681 cMpc/h box and holds its 30-bin
# stack in 6.8 GB.
NGRID_3D_DEFAULT = 384


class Geometry:
    """Base class; see make_geometry."""

    name = None
    grid_ndim = None

    # Whether this geometry deconvolves the TSC assignment window unless told
    # otherwise. False for '2d' so every bundle measured before there was a
    # switch still means what its filename says; True for '3d', where the
    # window is 1.8% at k = 0.1 on a 256^3 grid rather than 3e-5.
    deconvolve_default = False

    def __init__(self, cfg, nkbins, deconvolve_window=None):
        self.cfg = cfg
        self.box = float(cfg.box)
        self.grid = int(cfg.grid)
        self.nthread = cfg.threads
        self.nkbins = nkbins
        self.k_Nyquist = np.pi * self.grid / self.box
        self.deconvolve_window = (self.deconvolve_default
                                  if deconvolve_window is None
                                  else bool(deconvolve_window))

    # --- the k grid ---------------------------------------------------------
    @property
    def shape(self):
        return (self.grid,) * self.grid_ndim

    def report(self):
        print(f"  geometry {self.name}: {self.grid}"
              f"^{self.grid_ndim} grid, cell {self.box / self.grid:.3f} cMpc/h, "
              f"k_Nyquist {self.k_Nyquist:.3f} h/cMpc, "
              f"{self.k_center.size} k bins up to {self.k_bins[-1]:.3f}")
        self._report_window()
        dk = float(self.k_bins[1] - self.k_bins[0])
        k_f = 2.0 * np.pi / self.box
        empty = int(np.count_nonzero(self.nmodes[self.k_center
                                                 < self.k_Nyquist] == 0))
        if empty:
            want = int((self.k_bins[-1] - self.k_bins[0]) / k_f) + 1
            print(f"  WARNING: {empty} k bins below k_Nyquist hold no modes. "
                  f"The bin width {dk:.5f} is under the fundamental "
                  f"{k_f:.5f}; --nkbins {want} or fewer would fill them.")

    def _report_window(self):
        """Say what the TSC window is doing, and how big it is if left in.

        A spectrum measured here is a product of two painted fields, so it
        carries W^2. Deconvolved, that is removed exactly. Left in, it
        suppresses the measured P by the amounts printed -- which is fine at
        k = 0.1 on a 2048^2 grid (3e-4) and is not at k = 3 (22%).
        """
        if self.deconvolve_window:
            print("  TSC window: deconvolved")
            return
        d = self.box / self.grid
        ks = [k for k in (0.1, 0.5, 1.0, 3.0) if k < self.k_Nyquist]
        shown = []
        for k in ks:
            x = 0.5 * k * d
            shown.append(f"{1.0 - (np.sin(x) / x) ** 6:.1%} at k={k:g}")
        print("  TSC window: NOT deconvolved, so P is low by "
              + ", ".join(shown)
              + ". --deconvolve-window removes it (and writes a separate cache).")

    # --- painting and transforms -------------------------------------------
    def zeros_stack(self, n):
        """(n,) + grid, for accumulating one density grid per mass bin."""
        raise NotImplementedError

    def paint_into(self, out, pos, weights):
        raise NotImplementedError

    def delta_of(self, dens, mass_total):
        """rho / rhobar - 1 from a painted mass grid, against a GLOBAL mean."""
        d = np.asarray(dens, dtype=np.float64) / (mass_total / self.ncells)
        d -= 1.0
        return d

    @property
    def ncells(self):
        return self.grid ** self.grid_ndim

    def fft(self, delta):
        raise NotImplementedError

    def binned(self, prod):
        """A product of two transforms, binned in this pipeline's convention."""
        raise NotImplementedError

    def species_delta(self, label):
        """The overdensity field of 'gas' or 'dm'."""
        raise NotImplementedError

    def halo_delta(self, pos):
        """The unweighted overdensity field of a set of halo positions."""
        dens = self.paint(pos, None)
        return self.delta_of(dens, dens.sum(dtype=np.float64))

    def paint(self, pos, weights):
        raise NotImplementedError

    def random_delta(self, mass, rng, chunk=None):
        """The overdensity of a uniformly random catalogue: the self-pair field."""
        raise NotImplementedError


class Geometry2D(Geometry):
    """Projection along z onto an ngrid^2 grid. The pipeline's original path.

    Deliberately unchanged, down to not deconvolving the TSC window: it is
    3e-5 at k = 0.1 on a 2048^2 grid, and leaving it alone keeps every bundle
    already on disk valid.
    """

    name = '2d'
    grid_ndim = 2

    def __init__(self, cfg, nkbins, kmax=None, deconvolve_window=None):
        super().__init__(cfg, nkbins, deconvolve_window=deconvolve_window)
        self.k_grid = compute_k_grid_2d(self.grid, self.box)
        self._W2 = (tsc_window_2d(self.grid, self.box) ** 2
                    if self.deconvolve_window else None)
        self.kmax = float(kmax) if kmax else float(np.max(self.k_grid))
        self.k_bins, self.k_center, _ = bin_power_spectrum_2d(
            np.zeros_like(self.k_grid), self.k_grid, self.grid, self.box,
            nkbins=nkbins, k_min=2.0 * np.pi / self.box, k_max=self.kmax)
        self.nmodes, self.k_eff = mode_stats_2d(self.grid, self.box, self.k_bins)

    def zeros_stack(self, n):
        return np.zeros((n,) + self.shape, dtype=np.float64)

    def paint(self, pos, weights):
        return paint_2d(pos, weights, self.box, self.grid, self.nthread)

    def paint_into(self, out, pos, weights):
        out += self.paint(pos, weights)

    def fft(self, delta):
        return compute_2d_fft(delta, self.grid)

    def binned(self, prod):
        if self._W2 is not None:
            prod = np.asarray(prod) / self._W2
        _, _, P = bin_power_spectrum_2d(prod, self.k_grid, self.grid, self.box,
                                        nkbins=self.nkbins,
                                        k_min=2.0 * np.pi / self.box,
                                        k_max=self.kmax)
        return np.asarray(P) * self.box ** 2

    def species_delta(self, label):
        """Read the cached projected field rather than re-painting particles."""
        from Pmx_reconstruction.pmxlib.bundle import load_or_compute_delta_2d
        return load_or_compute_delta_2d(self.cfg, label)

    def random_delta(self, mass, rng, chunk=None):
        dens = paint_uniform_random_2d(mass, self.box, self.grid, self.nthread,
                                       rng, chunk=chunk)
        return self.delta_of(dens, float(np.sum(mass, dtype=np.float64)))


class Geometry3D(Geometry):
    """Painting onto an ngrid^3 grid.

    Two things differ from the 2-D path beyond the obvious, both handled in
    Binner3D and painting.as_projected:

    the TSC window is deconvolved, because on a coarse grid it is not a
    footnote -- 1.8% at k = 0.1 and 36% at k = 0.5 on 256^3;

    modes are weighted by 1/|k| inside a bin, so that a shell (k^2 dk) averages
    P(k) across a bin the way an annulus (k dk) does. Without it a 3-D and a
    2-D run disagree by 3.8% in the lowest bin for reasons that are pure
    binning. The variance cost is a part in 10^3.
    """

    name = '3d'
    grid_ndim = 3
    deconvolve_default = True

    def __init__(self, cfg, nkbins, kmax=None, deconvolve_window=None):
        super().__init__(cfg, nkbins, deconvolve_window=deconvolve_window)
        k_corner = np.sqrt(3.0) * self.k_Nyquist
        self.kmax = float(kmax) if kmax else k_corner
        k_bins = np.linspace(2.0 * np.pi / self.box, self.kmax, nkbins)
        self.binner = Binner3D(self.grid, self.box, k_bins,
                               deconvolve_tsc=self.deconvolve_window)
        self.k_bins = self.binner.k_bins
        self.k_center = self.binner.k_center
        self.nmodes = self.binner.nmodes
        self.nmodes_eff = self.binner.nmodes_eff
        self.k_eff = self.binner.k_eff

    def zeros_stack(self, n):
        # float32: tsc accumulates in float32 anyway, and the stack is
        # 6.8 GB at 384^3 against 13.6 in float64.
        return np.zeros((n,) + self.shape, dtype=np.float32)

    def paint(self, pos, weights):
        return paint_3d(pos, weights, self.box, self.grid, self.nthread)

    def paint_into(self, out, pos, weights):
        paint_3d(pos, weights, self.box, self.grid, self.nthread, out=out)

    def fft(self, delta):
        return compute_3d_fft(delta, self.grid)

    def binned(self, prod):
        return as_projected(self.binner(prod) * self.box ** 3, self.box)

    def species_delta(self, label):
        """Paint the particles: a projected field cannot be un-projected."""
        import gc

        from utils.catalog_loaders import load_particle_properties
        from utils.pipeline_paths import get_particle_file_path

        path = get_particle_file_path(self.cfg.feedback,
                                      sim_name=self.cfg.sim_name)
        part = load_particle_properties(path, label, requested=('mass', 'pos'),
                                        sim_name=self.cfg.sim_name,
                                        Lbox=self.box)
        pos = np.mod(np.asarray(part['pos'], dtype=np.float32),
                     np.float32(self.box))
        pos[pos >= self.box] -= np.float32(self.box)
        mass = np.asarray(part['mass'], dtype=np.float64)
        del part
        gc.collect()
        print(f"    painting {pos.shape[0]:.3e} {label} particles")
        dens = self.paint(pos, mass)
        m_tot = float(mass.sum())
        del pos, mass
        gc.collect()
        return self.delta_of(dens, m_tot)

    def random_delta(self, mass, rng, chunk=None):
        dens = paint_uniform_random_3d(mass, self.box, self.grid, self.nthread,
                                       rng, chunk=chunk)
        return self.delta_of(dens, float(np.sum(mass, dtype=np.float64)))


def make_geometry(cfg, nkbins, kmax=None, deconvolve_window=None):
    """The Geometry cfg.geometry asks for."""
    if cfg.geometry not in GEOMETRIES:
        raise ValueError(f"unknown geometry {cfg.geometry!r}; "
                         f"expected one of {GEOMETRIES}")
    cls = Geometry2D if cfg.geometry == '2d' else Geometry3D
    if deconvolve_window is None:
        deconvolve_window = cfg.deconvolve_window
    return cls(cfg, nkbins, kmax=kmax, deconvolve_window=deconvolve_window)


def deconvolve_is_default(cfg):
    """Whether cfg's window setting is this geometry's own default."""
    cls = Geometry2D if cfg.geometry == '2d' else Geometry3D
    return (cfg.deconvolve_window is None
            or bool(cfg.deconvolve_window) == cls.deconvolve_default)
