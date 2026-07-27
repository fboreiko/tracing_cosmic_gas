"""
Helpers shared by the two aperture-photometry entry points.

`get_AP_simple_jax_batched.py` and `get_AP_pixell_jax_batched.py` each carried
byte-identical copies of these five functions. They now live here so a fix to
one cannot silently miss the other.
"""

import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u


def load_halo_data_from_cache(cache_path):
    """
    Load halo properties directly from the massbin props cache.

    Returns a dict with keys: pos, vel, m200b, r200b.
    All arrays are already the selected subset — no index subsetting needed.
    """
    cached = np.load(cache_path)
    return {
        'pos':   cached['pos'],    # (N, 3) float32
        'vel':   cached['vel'],    # (N, 3) float32 — velocities
        'm200b': cached['m200b'],  # (N,)   float32
        'r200b': cached['r200b'],  # (N,)   float32
        'm200c': None,             # not stored in cache
        'mfof':  None,
    }

def compute_distances(cosmo, z, h):
    d_L = cosmo.luminosity_distance(z).to(u.Mpc).value
    d_C = d_L / (1.0 + z)                      # Mpc
    d_A = d_L / (1.0 + z) ** 2                 # Mpc
    d_C *= h                                   # Mpc/h
    d_A *= h                                   # Mpc/h
    return d_A, d_C

def translate_grid_params_to_degrees(z, n_cell, sim_params):
    h = sim_params['h']
    om_m = sim_params['omega_m']
    tcmb0 = sim_params['tcmb0']
    lbox = sim_params['box_size_cMpc_h']

    a = 1.0 / (1.0 + z)
    cosmo = FlatLambdaCDM(H0=h * 100.0, Om0=om_m, Tcmb0=tcmb0)
    d_A, _ = compute_distances(cosmo, z, h)
    Lbox_rad = (a * lbox) / d_A                     # rad
    Lbox_deg = Lbox_rad * 180.0 / np.pi          # degrees
    cell_size_deg = Lbox_deg / n_cell

    return Lbox_rad, Lbox_deg, cell_size_deg

def gauss_beam(ellsq, fwhm):
    """
    Gaussian beam of size fwhm
    """
    tht_fwhm = np.deg2rad(fwhm/60.)
    return np.exp(-(tht_fwhm**2.)*(ellsq)/(2.*8.*np.log(2.)))

def get_smooth_density(D, fwhm, pixsizedeg, Lboxdeg, N_dim):
    """
    Smooth density map D ((0, Lbox] and N_dim^2 cells) with Gaussian beam of FWHM
    """
    kstep = 2*np.pi/(N_dim*np.pi/180*pixsizedeg)
    #karr = np.fft.fftfreq(N_dim, d=Lbox/(2*np.pi*N_dim)) # physical (not correct)
    karr = np.fft.fftfreq(N_dim, d=Lboxdeg*np.pi/180./(2*np.pi*N_dim)) # angular
    #print("kstep = ", kstep, karr[1]-karr[0]) # N_dim/d gives kstep

    # fourier transform the map and apply gaussian beam
    dfour = np.fft.fftn(D)
    dksmo = np.zeros((N_dim, N_dim), dtype=complex)
    ksq = np.zeros((N_dim, N_dim), dtype=complex)
    ksq[:, :] = karr[None, :]**2+karr[:,None]**2
    dksmo[:, :] = gauss_beam(ksq, fwhm)*dfour
    drsmo = np.real(np.fft.ifftn(dksmo))

    return drsmo