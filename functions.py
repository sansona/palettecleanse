"""
Collection of functions
"""

import numpy as np
from classes import CMap


def convert_to_cmap(
    image_fname: str,
    cmap_type: str = "sequential",
    n_colors: int = 5,
    library: str = "matplotlib",
) -> CMap:
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

    if library == "matplotlib":
        return cmap.cmap
    elif library == "seaborn":
        # seaborn color palettes are effectively raw lists
        return sort_rgb_by_hsv(list(cmap.colors))
    elif library == "plotly":
        # convert to list of hex values for plotly
        return convert_rgb_cmap_to_hex(list(cmap.colors))

    return "library not in options: ['matplotlib', 'seaborn', 'plotly']"

