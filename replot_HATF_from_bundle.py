#!/usr/bin/env python3
"""
Comprehensive replot script for HATF_reconstruction outputs.

Automatically discovers and replots all plot bundles for a given sim_name
(flamingo or abacus), regenerating publication-quality plots from saved NPZ data.

Usage:
    python replot_from_bundle.py flamingo    # Replot all FLAMINGO diagnostics
    python replot_from_bundle.py abacus      # Replot all Abacus diagnostics
    python replot_from_bundle.py all         # Replot everything
    python replot_from_bundle.py list        # List available plots
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import argparse
from typing import Dict, Tuple, List, Optional
from utils.plot_data import load_plot_data
from matplotlib import rcParams
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Computer Modern']
rcParams['text.usetex'] = True
rcParams['axes.labelsize'] = 20
rcParams['xtick.labelsize'] = 18
rcParams['ytick.labelsize'] = 18
rcParams['xtick.major.size'] = 8
rcParams['xtick.major.width'] = 1
rcParams['ytick.major.size'] = 8
rcParams['ytick.major.width'] = 1
rcParams['legend.fontsize'] = 20

TAU_MAP_MAX_SIDE_PX = 1024
TAU_MAP_EXPORT_DPI = 180


def replot_transfer_function(npz_file: Path, output_dir: Optional[Path] = None) -> Path:
    """
    Replot transfer function: FLAMINGO to Abacus extrapolation.
    
    Parameters
    ----------
    npz_file : Path
        Path to the NPZ data bundle
    output_dir : Path, optional
        Directory to save replot. If None, uses npz_file's directory.
    
    Returns
    -------
    Path
        Path to saved replot
    """
    data, metadata = load_plot_data(npz_file)
    print(f"  Replotting: {metadata.get('description', 'Transfer Function')}")
    
    plt.figure(figsize=(6, 4), dpi=300)
    plt.loglog(data['flamingo_k_center'], np.abs(data['flamingo_T_k_1d']), 
               c='tab:orange', linewidth=1.2, label='FLAMINGO $T(k)$')
    plt.loglog(data['k_center'], np.abs(data['T_k_1d']), 
               c='tab:blue', linewidth=1.2, label='Extrapolated to Abacus $k$')
    plt.axvline(data['k_Nyquist_gas'], c='tab:orange', linestyle='--', 
                linewidth=1.0, label='$k_{Nyquist}^{gas}$')
    plt.axvline(data['k_Nyquist_dm'], c='tab:blue', linestyle='--', 
                linewidth=1.0, label='$k_{Nyquist}^{dm}$')
    plt.xlabel('$k$ [h/cMpc]')
    plt.ylabel('$T(k)$')
    plt.title('Transfer Function: FLAMINGO to Abacus extrapolation', fontsize=14)
    plt.legend()
    
    output_dir = output_dir or npz_file.parent
    output_path = output_dir / (npz_file.stem + '_replotted.pdf')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"    ✓ Saved: {output_path.name}")
    plt.close()
    
    return output_path


def extract_sf_label(filename: str) -> str:
    """
    Extract star formation (sf) parameter from transfer function filename.
    
    Parameters
    ----------
    filename : str
        Transfer function filename
    
    Returns
    -------
    str
        Label like "sf=0.0" or "sf=0.10"
    """
    # Check if filename contains sf parameter
    import re
    sf_match = re.search(r'sf(\d+)', filename)
    
    if sf_match:
        # Extract number after 'sf', e.g., 'sf10' -> 10
        sf_num = int(sf_match.group(1))
        # Convert to decimal: sf10 -> 0.10, sf12 -> 0.12, etc.
        sf_value = sf_num / 100.0
        return f"sf={sf_value:.2f}"
    else:
        # No sf parameter means sf=0.0
        return "sf=0.0"


def plot_transfer_function_family(tf_dir: Path, output_path: Optional[Path] = None) -> Path:
    """
    Plot transfer function family for different star formation parameters.
    
    Parameters
    ----------
    tf_dir : Path
        Directory containing transfer function NPZ files
    output_path : Path, optional
        Path to save output plot. If None, saves to tf_dir/transfer_functions.pdf
    
    Returns
    -------
    Path
        Path to saved plot
    """
    tf_dir = Path(tf_dir)
    
    if not tf_dir.exists():
        raise ValueError(f"Directory not found: {tf_dir}")
    
    # Find all transfer function NPZ files
    tf_files = sorted(tf_dir.glob('T_k_*.npz'))
    
    if not tf_files:
        raise ValueError(f"No transfer function files found in {tf_dir}")
    
    print(f"Plotting transfer function family from {tf_dir.name}:")
    print(f"  Found {len(tf_files)} transfer function files")
    
    # Color map for different sf values
    colors = plt.cm.viridis(np.linspace(0, 1, len(tf_files)))
    
    plt.figure(figsize=(10, 7), dpi=300)
    
    # Load and plot each transfer function
    for npz_file, color in zip(tf_files, colors):
        try:
            data, _ = load_plot_data(npz_file)
            label = extract_sf_label(npz_file.name)
            
            # Plot the transfer function
            plt.loglog(data['k_center'], np.abs(data['T_k_1d']), 
                      color=color, linewidth=2.0, label=label, alpha=0.8)
            
            print(f"  ✓ Loaded: {npz_file.name} ({label})")
        except Exception as e:
            print(f"  ✗ Error loading {npz_file.name}: {e}")
    
    plt.xlabel('$k$ [h/cMpc]')
    plt.ylabel('$T(k)$')
    plt.legend(loc='best', ncol=1)
    
    # Set output path
    if output_path is None:
        output_path = tf_dir / 'transfer_functions.pdf'
    else:
        output_path = Path(output_path)
    
    plt.savefig(output_path, bbox_inches='tight', dpi=300, format='pdf')
    print(f"  ✓ Saved: {output_path}\n")
    plt.close()
    
    return output_path


def replot_power_spectra_abacus(npz_file: Path, output_dir: Optional[Path] = None) -> Path:
    """
    Replot power spectra for Abacus branch (1x2 comparison plots).
    
    Parameters
    ----------
    npz_file : Path
        Path to the NPZ data bundle (gas or dm)
    output_dir : Path, optional
        Directory to save replot. If None, uses npz_file's directory.
    
    Returns
    -------
    Path
        Path to saved replot
    """
    data, metadata = load_plot_data(npz_file)
    print(f"  Replotting: {metadata.get('description', 'Power Spectra')}")
    
    k_flamingo = data['flamingo_k_center']
    k_abacus = data['k_center']
    box_gas = data['box_gas']
    box_dm = data['box_dm']
    k_Nyquist_gas = np.pi * data['ngrid_gas'] / box_gas
    k_Nyquist_dm = np.pi * data['ngrid_dm'] / box_dm
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    
    if 'P_gas_halos_target_binned' in data:
        # Gas power spectra plot
        axes[0].loglog(k_flamingo, data['P_gas_halos_target_binned'] * box_gas, 
                       c='blue', alpha=0.7, linewidth=2.5, label=r'FLAMINGO target')
        axes[0].loglog(k_abacus, data['P_gas_halos_reconstructed_binned'] * box_dm, 
                       c='red', alpha=0.7, linewidth=2.5, label=r'HATF reconstruction on Abacus')
        axes[0].axvline(k_Nyquist_gas, c='blue', 
                        linestyle='--', linewidth=1.5, alpha=0.6)
        axes[0].axvline(k_Nyquist_dm, c='red', 
                        linestyle='--', linewidth=1.5, alpha=0.6)
        axes[0].set_xlabel('$k$ [$h$/cMpc]')
        axes[0].set_ylabel(r'$\hat C_{\rm gas,\,halo}(k)\,L_{\rm box}$')
        axes[0].legend(loc='best')
        
        axes[1].loglog(k_flamingo, data['P_gas_gas_target_binned'] * box_gas, 
                       c='blue', alpha=0.7, linewidth=2.5, label=r'FLAMINGO target')
        axes[1].loglog(k_abacus, data['P_gas_gas_reconstructed_binned'] * box_dm, 
                       c='red', alpha=0.7, linewidth=2.5, label=r'HATF reconstruction on Abacus')
        axes[1].axvline(k_Nyquist_gas, c='blue', 
                        linestyle='--', linewidth=1.5, alpha=0.6)
        axes[1].axvline(k_Nyquist_dm, c='red', 
                        linestyle='--', linewidth=1.5, alpha=0.6)
        axes[1].set_xlabel('$k$ [$h$/cMpc]')
        axes[1].set_ylabel(r'$\hat C_{\rm gas,\,gas}(k)\,L_{\rm box}$')
        axes[1].legend(loc='best')
    else:
        # DM power spectra plot (Abacus branch only)
        axes[0].loglog(k_flamingo, data['flamingo_P_dm_halos_binned'] * box_gas, 
                       c='blue', alpha=0.7, linewidth=2.5, label=r'FLAMINGO')
        axes[0].loglog(k_abacus, data['abacus_P_dm_halos_binned'] * box_dm, 
                       c='red', alpha=0.7, linewidth=2.5, label=r'Abacus')
        axes[0].axvline(k_Nyquist_gas, c='blue', 
                        linestyle='--', linewidth=1.5, alpha=0.6)
        axes[0].axvline(k_Nyquist_dm, c='red', 
                        linestyle='--', linewidth=1.5, alpha=0.6)
        axes[0].set_xlabel('$k$ [$h$/cMpc]')
        axes[0].set_ylabel(r'$\hat C_{\rm dm,\,halo}(k)\,L_{\rm box}$')
        axes[0].legend(loc='best')
        
        axes[1].loglog(k_flamingo, data['flamingo_P_dm_dm_binned'] * box_gas, 
                       c='blue', alpha=0.7, linewidth=2.5, label=r'FLAMINGO')
        axes[1].loglog(k_abacus, data['P_dm_dm_foundation_binned'] * box_dm, 
                       c='red', alpha=0.7, linewidth=2.5, label=r'Abacus')
        axes[1].axvline(k_Nyquist_gas, c='blue', 
                        linestyle='--', linewidth=1.5, alpha=0.6)
        axes[1].axvline(k_Nyquist_dm, c='red', 
                        linestyle='--', linewidth=1.5, alpha=0.6)
        axes[1].set_xlabel('$k$ [$h$/cMpc]')
        axes[1].set_ylabel(r'$\hat C_{\rm dm,\,dm}(k)\,L_{\rm box}$')
        axes[1].legend(loc='best')
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3)
    
    output_dir = output_dir or npz_file.parent
    output_path = output_dir / (npz_file.stem + '_replotted.pdf')
    plt.savefig(output_path, bbox_inches='tight', dpi=300, format='pdf')
    print(f"    ✓ Saved: {output_path.name}")
    plt.close()
    
    return output_path


def replot_power_spectra_flamingo(npz_file: Path, output_dir: Optional[Path] = None) -> Path:
    """
    Replot power spectra for FLAMINGO diagnostics branch (1x2 comparison plots).
    
    Parameters
    ----------
    npz_file : Path
        Path to the NPZ data bundle (gas or dm)
    output_dir : Path, optional
        Directory to save replot. If None, uses npz_file's directory.
    
    Returns
    -------
    Path
        Path to saved replot
    """
    data, metadata = load_plot_data(npz_file)
    print(f"  Replotting: {metadata.get('description', 'Power Spectra')}")
    
    k_center = data['k_center']
    k_Nyquist = data['k_Nyquist']
    print(f"    Nyquist frequency: {k_Nyquist:.3f} h/cMpc")
    box_dm = data['box_dm']
    box_gas = data.get('box_gas', box_dm)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    
    # Left plot: Gas-halo power spectrum
    axes[0].loglog(k_center, data['P_gas_halos_target_binned'] * box_dm, 
                   c='blue', alpha=0.7, linewidth=2.5, label=r'Target')
    axes[0].loglog(k_center, data['P_gas_halos_reconstructed_binned'] * box_dm, 
                   c='red', alpha=0.7, linewidth=2.5, label=r'Reconstructed')
    axes[0].axvline(k_Nyquist, c='darkgrey', linestyle='--', linewidth=2, alpha=0.6)
    axes[0].set_xlabel('$k$ [$h$/cMpc]')
    axes[0].set_ylabel(r'$\hat C_{\rm gas,\,halo}(k)\,L_{\rm box}$')
    axes[0].legend(loc='best')
    
    # Right plot: Gas-gas power spectrum
    axes[1].loglog(k_center, data['P_gas_gas_target_binned'] * box_gas, 
                   c='blue', alpha=0.7, linewidth=2.5, label=r'Target')
    axes[1].loglog(k_center, data['P_gas_gas_reconstructed_binned'] * box_dm, 
                   c='red', alpha=0.7, linewidth=2.5, label=r'Reconstructed')
    axes[1].axvline(k_Nyquist, c='darkgrey', linestyle='--', linewidth=2, alpha=0.6)
    axes[1].set_xlabel('$k$ [$h$/cMpc]')
    axes[1].set_ylabel(r'$\hat C_{\rm gas,\,gas}(k)\,L_{\rm box}$')
    axes[1].legend(loc='best')
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3)
    
    output_dir = output_dir or npz_file.parent
    output_path = output_dir / (npz_file.stem + '_replotted.pdf')
    plt.savefig(output_path, bbox_inches='tight', dpi=300, format='pdf')
    print(f"    ✓ Saved: {output_path.name}")
    plt.close()
    
    return output_path


def replot_correlation(npz_file: Path, output_dir: Optional[Path] = None) -> Path:
    """
    Replot cross-correlation coefficient between reconstructed and true fields.
    
    Parameters
    ----------
    npz_file : Path
        Path to the NPZ data bundle
    output_dir : Path, optional
        Directory to save replot. If None, uses npz_file's directory.
    
    Returns
    -------
    Path
        Path to saved replot
    """
    data, metadata = load_plot_data(npz_file)
    print(f"  Replotting: {metadata.get('description', 'Cross-correlation')}")
    
    plt.figure(figsize=(7, 5), dpi=300)
    plt.semilogx(data['k_center'], data['r_reconstructed_vs_target'], 
                 c='red', alpha=0.8, linewidth=2.5)
    plt.axhline(1.0, c='grey', linestyle='--', linewidth=2, label='Perfect Correlation')
    plt.axvline(data['k_Nyquist'], c='blue', linestyle='--', linewidth=1.5, label='$k_{Nyquist}$')
    plt.xlabel('$k$ [h/cMpc]')
    plt.ylabel('$r(k)$')
    plt.ylim([0, 1.1])
    plt.legend()
    
    output_dir = output_dir or npz_file.parent
    output_path = output_dir / (npz_file.stem + '_replotted.pdf')
    plt.savefig(output_path, bbox_inches='tight', dpi=300, format='pdf')
    print(f"    ✓ Saved: {output_path.name}")
    plt.close()
    
    return output_path


def replot_tau_maps(npz_file: Path, output_dir: Optional[Path] = None) -> Path:
    """
    Replot tau maps for both branches.

    Abacus bundles contain:
      - tau_map_reconstructed

    FLAMINGO bundles contain:
      - tau_map_target
      - tau_map_reconstructed
    """
    data, metadata = load_plot_data(npz_file)
    print(f"  Replotting: {metadata.get('description', 'Tau maps')}")

    if 'tau_map_reconstructed' not in data:
        raise KeyError("Missing 'tau_map_reconstructed' in tau map bundle")

    is_flamingo_bundle = 'tau_map_target' in data

    if is_flamingo_bundle:
        tau_target = data['tau_map_target']
        tau_recon = data['tau_map_reconstructed']
        tau_delta = tau_recon - tau_target
        box_target = float(data.get('box_gas', data.get('box_dm', 1.0)))
        box_recon = float(data.get('box_dm', box_target))

        # Independent color scaling for reconstructed map and delta map.
        finite_recon = tau_recon[np.isfinite(tau_recon)].ravel()
        if finite_recon.size == 0:
            vmin_recon, vmax_recon = -1.0, 1.0
            maxabs_recon = np.nan
        else:
            vmin_recon, vmax_recon = np.percentile(finite_recon, [0, 100])
            maxabs_recon = np.max(np.abs(finite_recon))
        print(f"    True max |tau_recon|: {maxabs_recon:.6e}")

        finite_delta = tau_delta[np.isfinite(tau_delta)].ravel()
        if finite_delta.size == 0:
            vmax_delta = 1.0
            maxabs_delta = np.nan
        else:
            vmax_delta = np.percentile(np.abs(finite_delta), 99)
            maxabs_delta = np.max(np.abs(finite_delta))
            if vmax_delta <= 0:
                vmax_delta = 1.0
        vmin_delta = -vmax_delta
        print(f"    True max |tau_delta|: {maxabs_delta:.6e}")

        fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=300, sharey=True)

        im0 = axes[0].imshow(
            tau_recon,
            origin='lower',
            cmap='viridis',
            extent=[0, box_recon, 0, box_recon],
            vmin=vmin_recon,
            vmax=vmax_recon,
            aspect='equal',
            rasterized=True,
        )
        axes[0].set_xlabel('$x$ [cMpc/$h$]')
        axes[0].set_ylabel('$y$ [cMpc/$h$]')

        im1 = axes[1].imshow(
            tau_delta,
            origin='lower',
            cmap='RdBu_r',
            extent=[0, box_recon, 0, box_recon],
            vmin=vmin_delta,
            vmax=vmax_delta,
            aspect='equal',
            rasterized=True,
        )
        axes[1].set_xlabel('$x$ [cMpc/$h$]')
        axes[1].set_ylabel('$y$ [cMpc/$h$]')
        axes[1].tick_params(labelleft=True)

        plt.subplots_adjust(left=0.09, right=0.98, bottom=0.12, top=0.98, wspace=0.24)

        # Colorbar for reconstructed map (default right side of left panel).
        div0 = make_axes_locatable(axes[0])
        cax0 = div0.append_axes('right', size='4%', pad=0.06)
        cbar0 = fig.colorbar(im0, cax=cax0)
        fmt0 = ScalarFormatter(useMathText=True)
        fmt0.set_scientific(True)
        fmt0.set_powerlimits((0, 0))
        cbar0.formatter = fmt0
        cbar0.update_ticks()
        cbar0.set_label(r'$\tilde\tau(\mathbf{x})$', rotation=0, labelpad=30)
        cbar0.ax.yaxis.set_label_position('right')
        cbar0.ax.tick_params(labelsize=rcParams['ytick.labelsize'])
        cbar0.ax.yaxis.set_offset_position('right')
        cbar0.ax.yaxis.offsetText.set_horizontalalignment('left')
        cbar0.ax.yaxis.offsetText.set_x(1.0)
        cbar0.ax.yaxis.offsetText.set_fontsize(rcParams['ytick.labelsize'])

        # Colorbar for delta map (default right side of right panel).
        div1 = make_axes_locatable(axes[1])
        cax1 = div1.append_axes('right', size='4%', pad=0.06)
        cbar1 = fig.colorbar(im1, cax=cax1)
        fmt1 = ScalarFormatter(useMathText=True)
        fmt1.set_scientific(True)
        fmt1.set_powerlimits((0, 0))
        cbar1.formatter = fmt1
        cbar1.update_ticks()
        cbar1.set_label(r'$\Delta\tilde\tau(\mathbf{x})$', rotation=0, labelpad=30)
        cbar1.ax.yaxis.set_label_position('right')
        cbar1.ax.tick_params(labelsize=rcParams['ytick.labelsize'])
        cbar1.ax.yaxis.set_offset_position('right')
        cbar1.ax.yaxis.offsetText.set_horizontalalignment('left')
        cbar1.ax.yaxis.offsetText.set_x(1.0)
        cbar1.ax.yaxis.offsetText.set_fontsize(rcParams['ytick.labelsize'])
    else:
        tau_recon = data['tau_map_reconstructed']
        box_recon = float(data.get('box_dm', 1.0))

        finite_vals = tau_recon[np.isfinite(tau_recon)].ravel()
        if finite_vals.size == 0:
            vmin, vmax = -1.0, 1.0
            maxabs_recon = np.nan
        else:
            vmin, vmax = np.percentile(finite_vals, [1, 99])
            maxabs_recon = np.max(np.abs(finite_vals))
        print(f"    True max |tau_recon|: {maxabs_recon:.6e}")

        fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
        im = ax.imshow(
            tau_recon,
            origin='lower',
            cmap='viridis',
            extent=[0, box_recon, 0, box_recon],
            vmin=vmin,
            vmax=vmax,
            aspect='equal',
            rasterized=True,
        )
        ax.set_xlabel('$x$ [cMpc/$h$]')
        ax.set_ylabel('$y$ [cMpc/$h$]')
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fmt = ScalarFormatter(useMathText=True)
        fmt.set_scientific(True)
        fmt.set_powerlimits((0, 0))
        cbar.formatter = fmt
        cbar.update_ticks()
        cbar.ax.set_title(r'$\tilde\tau(\mathbf{x})$', pad=8)
        plt.tight_layout()

    output_dir = output_dir or npz_file.parent
    output_path = output_dir / (npz_file.stem + '_replotted.pdf')
    plt.savefig(output_path, bbox_inches='tight', dpi=TAU_MAP_EXPORT_DPI, format='pdf')
    print(f"    ✓ Saved: {output_path.name}")
    plt.close()

    return output_path


def discover_plots_by_sim(sim_name: str, plots_dir: Optional[Path] = None) -> Dict[str, List[Path]]:
    """
    Discover available NPZ plot bundles organized by sim branch.
    
    Parameters
    ----------
    sim_name : str
        'abacus', 'flamingo', or 'all'
    plots_dir : Path, optional
        Root plots directory. Defaults to 'plots/'
    
    Returns
    -------
    dict
        Dictionary mapping sim names to list of NPZ file paths
    """
    if plots_dir is None:
        plots_dir = Path('plots')
    
    if not plots_dir.exists():
        print(f"Error: plots directory not found at {plots_dir}")
        return {}
    
    discovered = {'abacus': [], 'flamingo': []}
    
    for npz_file in sorted(plots_dir.rglob('*.npz')):
        name = npz_file.name
        
        # Classify by sim branch based on filename pattern
        if name.endswith('_abacus.npz') or 'transfer_flamingo_to_abacus' in name:
            discovered['abacus'].append(npz_file)
        elif 'tau_map_' in name and 'tau_maps_' not in name:
            discovered['abacus'].append(npz_file)
        elif 'tau_maps_' in name:
            discovered['flamingo'].append(npz_file)
        elif 'P_gas_combined' in name or 'P_dm_combined' in name or 'r_recon' in name:
            discovered['flamingo'].append(npz_file)
    
    if sim_name == 'all':
        return discovered
    elif sim_name in discovered:
        return {sim_name: discovered[sim_name]}
    else:
        print(f"Error: Unknown sim_name '{sim_name}'. Use 'abacus', 'flamingo', or 'all'")
        return {}


def list_available_plots(plots_dir: Optional[Path] = None):
    """List all available plot bundles organized by sim."""
    if plots_dir is None:
        plots_dir = Path('plots')
    
    plots = discover_plots_by_sim('all', plots_dir)
    
    print("\nAvailable plot bundles:\n")
    
    for sim_name, npz_files in plots.items():
        if not npz_files:
            continue
        print(f"{'='*70}")
        print(f"{sim_name.upper()} Branch ({len(npz_files)} plots)")
        print(f"{'='*70}")
        
        for npz_file in npz_files:
            try:
                data, metadata = load_plot_data(npz_file)
                desc = metadata.get('description', 'N/A')
                n_keys = len(data)
                rel_path = npz_file.relative_to(plots_dir)
                print(f"  • {rel_path}")
                print(f"    Description: {desc}")
                print(f"    Data keys: {n_keys}")
            except Exception as e:
                print(f"  • {npz_file.name} (Error: {e})")
        print()


def replot_all_for_sim(sim_name: str, plots_dir: Optional[Path] = None, output_dir: Optional[Path] = None):
    """
    Replot all bundles for a given sim branch.
    
    Parameters
    ----------
    sim_name : str
        'abacus' or 'flamingo'
    plots_dir : Path, optional
        Root plots directory
    output_dir : Path, optional
        Directory to save replots. If None, saves in npz_file's directory.
    """
    if plots_dir is None:
        plots_dir = Path('plots')
    
    plots = discover_plots_by_sim(sim_name, plots_dir)
    
    if sim_name not in plots or not plots[sim_name]:
        print(f"No plots found for sim='{sim_name}'")
        return
    
    npz_files = plots[sim_name]
    print(f"\nReplotting {len(npz_files)} plots for {sim_name.upper()} branch:")
    print(f"{'='*70}\n")
    
    replot_count = 0
    for npz_file in sorted(npz_files):
        try:
            # Dispatch to appropriate replotting function based on file type
            if 'transfer_flamingo_to_abacus' in npz_file.name:
                replot_transfer_function(npz_file, output_dir)
                replot_count += 1
            elif 'P_gas_combined' in npz_file.name:
                if sim_name == 'abacus':
                    replot_power_spectra_abacus(npz_file, output_dir)
                else:
                    replot_power_spectra_flamingo(npz_file, output_dir)
                replot_count += 1
            elif 'P_dm_combined' in npz_file.name:
                # DM plots only exist for Abacus branch
                if sim_name == 'abacus':
                    replot_power_spectra_abacus(npz_file, output_dir)
                    replot_count += 1
            elif 'r_recon' in npz_file.name:
                # Cross-correlation exists for both branches
                replot_correlation(npz_file, output_dir)
                replot_count += 1
            elif 'tau_map_' in npz_file.name or 'tau_maps_' in npz_file.name:
                # Tau maps exist in both branches, but with different bundle keys.
                replot_tau_maps(npz_file, output_dir)
                replot_count += 1
        except Exception as e:
            print(f"  ✗ Error replotting {npz_file.name}: {e}", file=sys.stderr)
    
    print(f"\n{'='*70}")
    print(f"Successfully replotted {replot_count}/{len(npz_files)} plots\n")


def main():
    parser = argparse.ArgumentParser(
        description='Replot HATF reconstruction outputs from NPZ data bundles',
        epilog='Examples:\n'
               '  python replot_from_bundle.py flamingo\n'
               '  python replot_from_bundle.py abacus\n'
               '  python replot_from_bundle.py all\n'
               '  python replot_from_bundle.py list\n'
               '  python replot_from_bundle.py transfer-fn plots/hatf/transfer_fn/strongest_AGN',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('sim_name', nargs='?', default='list',
                       help="'abacus', 'flamingo', 'all', 'list', or 'transfer-fn' (default: list)")
    parser.add_argument('tf_dir', nargs='?', default=None,
                       help='Transfer function directory (required when sim_name is transfer-fn)')
    parser.add_argument('--plots-dir', type=Path, default=Path('plots'),
                       help='Root directory containing plot bundles (default: plots/)')
    parser.add_argument('--output-dir', type=Path, default=None,
                       help='Directory to save replots (default: same as input)')
    parser.add_argument('--output', type=Path, default=None,
                       help='Output file path for transfer function plot (used with transfer-fn)')
    
    args = parser.parse_args()
    
    if args.sim_name == 'transfer-fn':
        if not args.tf_dir:
            print("Error: transfer_fn_dir required when sim_name is 'transfer-fn'", file=sys.stderr)
            sys.exit(1)
        try:
            plot_transfer_function_family(Path(args.tf_dir), args.output)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    elif args.sim_name == 'list':
        list_available_plots(args.plots_dir)
    elif args.sim_name in ('abacus', 'flamingo', 'all'):
        if args.sim_name == 'all':
            for sim in ['abacus', 'flamingo']:
                replot_all_for_sim(sim, args.plots_dir, args.output_dir)
        else:
            replot_all_for_sim(args.sim_name, args.plots_dir, args.output_dir)
    else:
        print(f"Error: Unknown sim_name '{args.sim_name}'", file=sys.stderr)
        print("Use 'abacus', 'flamingo', 'all', 'list', or 'transfer-fn'", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()