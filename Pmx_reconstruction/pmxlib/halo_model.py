#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Halo-model P_hc(k|M) and its colossus cosmology backend.

Was section 6a of predict_Pmx_from_Phx.py. It lives in pmxlib rather than in
experiment_A.py because it is a general 1-halo + 2-halo model (Plin, dndM,
bias, I2h, P_hc) with obvious use beyond that one experiment; only the
Experiment A ansatz itself (transfer_T, delta_P_experimentA) stays with the
script.
"""
import numpy as np

from colossus.lss import mass_function as colossus_mf
from colossus.lss import bias as colossus_bias
from colossus.lss import peaks as colossus_peaks

from utils.cosmology import colossus_params, ensure_colossus_cosmology
from Pmx_reconstruction.pmxlib.cosmology import DELTA_HALO, camb_linear_power_table
from Pmx_reconstruction.pmxlib.nfw import concentration, u_nfw


# ------------------------------------------------------------------------------
class ColossusBackend:
    """Tinker+08 n(M) and Tinker+10 b(M) via colossus, on a CAMB spectrum."""

    name = 'colossus'

    def __init__(self, cfg, z, power_spectrum='camb'):
        sim_params, sim_name = cfg.sim_params, cfg.sim_name
        self.z = float(z)
        self.mdef = f'{int(DELTA_HALO)}m'

        # Registration goes through utils.cosmology so that HATF and this
        # pipeline cannot install different cosmologies into colossus's
        # process-global slot.
        self.cosmo = ensure_colossus_cosmology(sim_params, sim_name)
        self.params = colossus_params(sim_params)

        if power_spectrum == 'camb':
            self.ps_args = {'model': 'table', 'path': str(camb_linear_power_table(sim_params, sim_name))}
            self.ps_label = 'CAMB'
        elif power_spectrum == 'eisenstein98':
            self.ps_args = {'model': 'eisenstein98'}
            self.ps_label = 'Eisenstein & Hu 1998 (colossus built-in)'
        else:
            raise ValueError(f"unknown power spectrum {power_spectrum!r}")

    def Plin(self, k):
        return self.cosmo.matterPowerSpectrum(np.atleast_1d(np.asarray(k, float)),
                                              self.z, **self.ps_args)

    def dndM(self, M):
        M = np.atleast_1d(np.asarray(M, float))
        dndlnM = colossus_mf.massFunction(
            M, self.z, mdef=self.mdef, model='tinker08',
            q_in='M', q_out='dndlnM', ps_args=self.ps_args)
        return dndlnM / M

    def bias(self, M):
        # haloBias() does not forward ps_args to the fitting function, so the
        # peak height is computed explicitly on the CAMB spectrum and fed to
        # haloBiasFromNu. Going through haloBias() directly would silently use
        # colossus's built-in Eisenstein & Hu sigma(M) instead.
        nu = colossus_peaks.peakHeight(np.atleast_1d(np.asarray(M, float)),
                                       self.z, ps_args=self.ps_args)
        return colossus_bias.haloBiasFromNu(nu, z=self.z, mdef=self.mdef,
                                            model='tinker10')

    def describe(self):
        return (f"colossus: P_lin from {self.ps_label}, Tinker+08 dn/dM and "
                f"Tinker+10 b(M) at Delta = {self.mdef}")


def _conc(cfg, M, z):
    """c(M,z) with the switches taken from the config."""
    return concentration(M, z, source=cfg.concentration_source,
                         colossus_model=cfg.colossus_conc_model,
                         sim_params=cfg.sim_params, sim_name=cfg.sim_name)


class HaloModel:
    """Halo-model P_hc(k|M), on top of an interchangeable cosmology backend.

    Only the dimensionless ratio T(k,M) = P_hc(k|M)/P_hc(k|M_r) is ever used
    downstream, so the absolute normalisation of anything here cancels. That is
    why the neutrino treatment does not matter either, and why the choice of
    linear spectrum moves S(k) far less than it moves P_lin itself.
    """

    def __init__(self, cfg, z, power_spectrum='camb'):
        self.cfg = cfg
        self.z = float(z)
        self.be = ColossusBackend(cfg, z, power_spectrum)
        self._I2h_cache = {}

        print(f"  [halo model] {self.be.describe()}")
        print(f"  [halo model] Omega_m={self.cfg.omega_m:.4f} Omega_b="
              f"{self.cfg.sim_params.get('omega_b'):.4f} h={self.cfg.h:.4f} "
              f"sigma8={self.cfg.sim_params.get('sigma8'):.4f} n_s={self.cfg.sim_params.get('n_s'):.4f}")
        print(f"  [halo model] at z={self.z:g}: b(1e12)="
              f"{float(self.bias(1e12)[0]):.4f}, "
              f"dn/dlnM(1e12)={float(self.dndM(1e12)[0]) * 1e12:.4e}")

    # thin pass-throughs, so callers never touch the backend directly
    def Plin(self, k):
        return self.be.Plin(k)

    def dndM(self, M):
        return self.be.dndM(M)

    def bias(self, M):
        return self.be.bias(M)

    def I2h(self, k):
        """int dM n(M) b(M) W_m(k,M), normalised so that I(k -> 0) = 1.

        The normalisation IS the bias consistency relation of the notes;
        imposing it by hand is what keeps a mass function that does not
        integrate to self.cfg.rhobar_m from leaking into the 2-halo amplitude.
        """
        key = (k.shape, float(k[0]), float(k[-1]))
        if key in self._I2h_cache:
            return self._I2h_cache[key]
        M = np.logspace(6.0, 16.0, 300)
        w = M * self.dndM(M) * self.bias(M) * M / self.cfg.rhobar_m
        u = np.array([u_nfw(k, m, _conc(self.cfg, m, self.z), self.cfg.rhobar_m) for m in M])
        num = np.trapezoid(w[:, None] * u, np.log(M), axis=0)
        out = num / np.trapezoid(w, np.log(M))
        self._I2h_cache[key] = out
        return out

    def P_hc(self, k, M):
        """Halo-CDM cross spectrum, (nM, nk). Units are arbitrary but internally
        consistent: only ratios of this quantity are ever used."""
        M = np.atleast_1d(np.asarray(M, dtype=float))
        u = np.array([u_nfw(k, m, _conc(self.cfg, m, self.z), self.cfg.rhobar_m) for m in M])
        one_halo = (M / self.cfg.rhobar_m)[:, None] * u
        two_halo = self.bias(M)[:, None] * (self.Plin(k) * self.I2h(k))[None, :]
        return one_halo + two_halo


def _trapz_weights(x):
    """Weights w such that sum(w * f) == trapezoid(f, x)."""
    w = np.empty_like(x)
    w[1:-1] = 0.5 * (x[2:] - x[:-2])
    w[0] = 0.5 * (x[1] - x[0])
    w[-1] = 0.5 * (x[-1] - x[-2])
    return w