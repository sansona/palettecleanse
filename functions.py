"""
Collection of functions
"""
import numpy as np
from classes import CMap

def convert_to_cmap(image_fname: str, cmap_type: str = "sequential", n_colors: int = 5, library: str ='matplotlib'):
    """
    Wrapper function for converting image to colormap in single line

    Args:
        image_fname (str): filename to image
        cmap_type (str): type of colormap. Options are:
            ['sequential, 'diverging', 'cyclic', 'qualitative']
            Default to 'sequential'
        n_colors (int): number of colors to cluster in clustering algorithm.  Default to 5
        library (str): plotting library. Options are: ['matplotlib', 'seaborn', 'plotly']

    Returns:
        (CMap object)
    """
    cmap = CMap(image_fname, cmap_type=cmap_type, n_colors=n_colors)

    if library == 'matplotlib':
        return cmap.cmap
    elif library == 'seaborn':
        # seaborn color palettes are effectively raw lists
        return list(cmap.colors)
    elif library == 'plotly':
        # convert to list of hex values for plotly
        return convert_rgb_cmap_to_hex(list(cmap.colors))

    return "library not in options: ['matplotlib', 'seaborn', 'plotly']"

def convert_rgb_to_hex(rgb: np.array) -> str:
    """
    Converts a len 3 np.array of rgb values to a hex string. Note that
    function rounds any floats, so colors will not be absolutely identical
    between rgb & hex

    https://stackoverflow.com/questions/3380726/converting-an-rgb-color-tuple-to-a-hexidecimal-string
        
    Args:
        rgb (np.array): rgb values in len 3 np.array
    Returns:
        (str): hex value
    """
    rgb = rgb*100
    return f'#{int(round(rgb[0]))}{int(round(rgb[1]))}{int(round(rgb[2]))}'

def convert_rgb_cmap_to_hex(rgb_cmap: list) -> list:
    """
    Converts a list of np.arrays containing an entire colormap of rgb values to a list of hex values

    Args:
        rgb_cmap (list): List of np.arrays, each formatted as a len 3 np.array with rgb values

    Returns:
        (list): list containing converted hex colormap
    
    """
    hex_cmap = []
    for c in rgb_cmap:
        hex_cmap.append(convert_rgb_to_hex(c))
    
    return hex_cmap