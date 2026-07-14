"""Utility functions for saving and loading plot data bundles."""

import numpy as np
from pathlib import Path


def save_plot_data(plot_path, data_dict, description=None):
    """
    Save plot data to an NPZ file alongside the plot.
    
    Parameters
    ----------
    plot_path : str or Path
        Path to the plot file (e.g., 'path/to/plot.pdf')
    data_dict : dict
        Dictionary containing all plot data (arrays, scalars, strings)
    description : str, optional
        Human-readable description of the plot data
    
    Returns
    -------
    Path
        Path to the saved NPZ file
    
    Examples
    --------
    >>> save_plot_data('power_spectrum.pdf', {
    ...     'k_center': k,
    ...     'P_gas': P_gas,
    ...     'P_dm': P_dm,
    ...     'box_size': 681.0,
    ...     'title': 'Power Spectrum Comparison'
    ... })
    """
    plot_path = Path(plot_path)
    npz_path = plot_path.with_suffix('.npz')
    
    # Add metadata
    save_dict = data_dict.copy()
    if description:
        save_dict['_description'] = description
    save_dict['_plot_file'] = str(plot_path.name)
    
    np.savez_compressed(npz_path, **save_dict)
    print(f"Saved plot data: {npz_path}")
    return npz_path


def load_plot_data(npz_path):
    """
    Load plot data from an NPZ bundle.
    
    Parameters
    ----------
    npz_path : str or Path
        Path to the NPZ file
    
    Returns
    -------
    dict
        Dictionary containing all plot data
    
    Examples
    --------
    >>> data = load_plot_data('power_spectrum.npz')
    >>> k = data['k_center']
    >>> P_gas = data['P_gas']
    """
    npz_path = Path(npz_path)
    with np.load(npz_path, allow_pickle=True) as data:
        result = {key: data[key] for key in data.files}
    
    # Extract metadata if present
    metadata = {}
    if '_description' in result:
        metadata['description'] = str(result.pop('_description'))
    if '_plot_file' in result:
        metadata['plot_file'] = str(result.pop('_plot_file'))
    
    return result, metadata
