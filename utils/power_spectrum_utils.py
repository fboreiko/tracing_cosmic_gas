import numpy as np
import sys
from scipy.fft import rfftn, rfftfreq, fftfreq, rfft2
from abacusnbody.analysis.tsc import tsc_parallel
import matplotlib.pyplot as plt
from matplotlib import rcParams
import jax
import jax.numpy as jnp
from jax import jit
import time

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Computer Modern']
# rcParams['text.usetex'] = True  # Disabled - requires LaTeX on compute nodes


def compute_delta(pos, box, ngrid, weights, nthread=4):
    # Compute density field using TSC assignment
    dens = tsc_parallel(pos, ngrid, box, weights=weights, nthread=nthread)

    # Compute overdensity field delta
    dens_avg = np.sum(dens) / ngrid**3
    delta = dens / dens_avg - 1

    return delta


@jit
def compute_k_mag_slice_jax(kx_val, ky, kz):
    """JIT-compiled function to compute k magnitude for a slice."""
    return jnp.sqrt(kx_val**2 + ky[:, jnp.newaxis]**2 + kz[jnp.newaxis, :]**2)


@jit
def compute_power_slice_jax(delta_fft_slice):
    """JIT-compiled function to compute power for a slice."""
    return (delta_fft_slice * jnp.conj(delta_fft_slice)).real


@jit
def bin_power_slice_jax(k_mag_slice, P_slice, k_bins):
    """JIT-compiled function to bin power spectrum for a single slice."""
    nkbins = len(k_bins) - 1
    power_binned = jnp.zeros(nkbins)
    counts_binned = jnp.zeros(nkbins)
    
    for i in range(nkbins):
        mask = (k_mag_slice >= k_bins[i]) & (k_mag_slice < k_bins[i+1])
        P_bin = jnp.where(mask, P_slice, 0.0)
        power_binned = power_binned.at[i].set(jnp.sum(P_bin))
        counts_binned = counts_binned.at[i].set(jnp.sum(mask))
    
    return power_binned, counts_binned


def compute_power_spectrum(delta, box, ngrid, plotting=False, label='gas', workers=4):

    print("Computing FFT of the density field...")
    sys.stdout.flush()

    # FFT of the overdensity field - parallelized using all available CPUs
    delta_fft = rfftn(delta, workers=workers) # returns unnormalised FT
    delta_fft *= (1.0 / ngrid ** 3) # we normalise here
    
    # Compute k-space grid
    kx = fftfreq(ngrid, d=box/ngrid) * 2 * np.pi
    ky = fftfreq(ngrid, d=box/ngrid) * 2 * np.pi
    kz = rfftfreq(ngrid, d=box/ngrid) * 2 * np.pi
    k_Nyquist = np.pi * ngrid / box

    k_max = np.sqrt(np.max(kx)**2 + np.max(ky)**2 + np.max(kz)**2)
    nkbins = ngrid // 2
    k_bins = np.linspace(0, k_max, nkbins)
    k_center = 0.5 * (k_bins[:-1] + k_bins[1:])

    # Convert numpy arrays to JAX arrays for the k-space grid
    kx_jax = jnp.array(kx)
    ky_jax = jnp.array(ky) 
    kz_jax = jnp.array(kz)
    k_bins_jax = jnp.array(k_bins)

    # Initialize arrays for binning
    Power_spectrum = np.zeros(len(k_bins)-1)
    mode_counts = np.zeros(len(k_bins)-1)

    print("Binning power spectrum with JAX acceleration...")
    sys.stdout.flush()

    start_time = time.time()
    
    # Process all slices in serial
    for ix in range(ngrid):
        # Compute k_mag for current slice using JAX
        k_mag_slice = compute_k_mag_slice_jax(kx_jax[ix], ky_jax, kz_jax)
        
        # Extract slice and compute power using JAX
        delta_fft_slice = jnp.array(delta_fft[ix, :, :])
        P_slice = compute_power_slice_jax(delta_fft_slice)
        
        # Bin power spectrum for this slice using JAX
        power_binned, counts_binned = bin_power_slice_jax(k_mag_slice, P_slice, k_bins_jax)
        
        # Accumulate results
        Power_spectrum += np.array(power_binned)
        mode_counts += np.array(counts_binned)

    end_time = time.time()
    print(f"Power spectrum binning completed in {end_time - start_time:.2f} seconds.")
    sys.stdout.flush()
        
    # Average over modes in each bin
    mask_nonzero = mode_counts > 0
    Power_spectrum[mask_nonzero] /= mode_counts[mask_nonzero]
    Power_spectrum *= box ** 3 # normalise by box volume

    if plotting:
        fig, ax = plt.subplots(dpi=300, figsize=(6, 4))
        ax.plot(k_center, Power_spectrum * k_center, c='black', alpha=0.7, label='Power Spectrum')
        ax.axvline(k_Nyquist, c='blue', linestyle='--', label='$k_{Nyquist}$')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('$k$')
        ax.set_ylabel('$kP(k)$')
        ax.set_title('Power Spectrum')
        ax.legend()
        fig.savefig(f'plots/{label}_power_spectrum.png', bbox_inches='tight')

    return delta_fft, k_center, k_Nyquist, Power_spectrum


def compute_2d_power_spectrum(delta_2d, box, ngrid, nkbins):
    """
    Computes the 2D power spectrum of a projected field.
    """
    # 2D FFT: Returns (ngrid, ngrid//2 + 1)
    delta_fft = rfft2(delta_2d)
    delta_fft *= (1.0 / ngrid ** 2) # Normalise for 2D

    # Compute k-space grid for 2D
    kx = fftfreq(ngrid, d=box/ngrid) * 2 * np.pi
    ky = rfftfreq(ngrid, d=box/ngrid) * 2 * np.pi
    k_Nyquist = np.pi * ngrid / box

    # Create k_magnitude grid
    k_mag = np.sqrt(kx[:, np.newaxis]**2 + ky[np.newaxis, :]**2)
    
    k_max = np.max(k_mag)
    k_bins = np.linspace(0, k_max, nkbins)
    k_center = 0.5 * (k_bins[:-1] + k_bins[1:])

    power_spectrum = np.zeros(len(k_bins)-1)
    mode_counts = np.zeros(len(k_bins)-1)

    # Compute power magnitude
    P_2d_field = (delta_fft * np.conj(delta_fft)).real

    # Binning the 2D modes
    for i in range(len(k_bins)-1):
        mask = (k_mag >= k_bins[i]) & (k_mag < k_bins[i+1])
        P_bin = P_2d_field[mask]
        if P_bin.size > 0:
            power_spectrum[i] = np.mean(P_bin)
    
    # Normalise by "Area" (Box^2) for 2D power spectrum
    power_spectrum *= box ** 2 

    return delta_fft, k_mag, k_center, k_Nyquist, power_spectrum