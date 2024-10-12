"""
Collection of functions
"""

from colorsys import rgb_to_hsv

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
        return list(cmap.colors)
    elif library == "plotly":
        # convert to list of hex values for plotly
        return convert_rgb_cmap_to_hex(list(cmap.colors))

    return "library not in options: ['matplotlib', 'seaborn', 'plotly']"


def convert_rgb_to_hex(rgb: np.array) -> str:
    """
    Converts a len 3 np.array of rgb values to a hex string. Note that
    function rounds any floats, so colors will not be absolutely identical
    between rgb & hex

    rgb values start off normalized between [0, 1]

    https://stackoverflow.com/questions/3380726/converting-an-rgb-color-tuple-to-a-hexidecimal-string

    Args:
        rgb (np.array): rgb values in len 3 np.array
    Returns:
        str: hex value
    """
    rgb = rgb * 255  # unnormalize
    rgb = rgb.astype(int)
    return "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])


def sort_rgb_by_hsv(rgb_cmap: list) -> list:
    """
    Sorts a list of RGB colors based off HSV values

    Args:
        rgb_cmap (list): List of np.arrays, each formatted as a len 3 np.array with rgb values

    Returns:
        list:sorted rgb cmap
    """
    return sorted(rgb_cmap, key=lambda x: rgb_to_hsv(x[0], x[1], x[2]))


def convert_rgb_cmap_to_hex(rgb_cmap: list) -> list:
    """
    Converts a list of np.arrays containing an entire colormap of rgb values to a list of hex values

    Args:
        rgb_cmap (list): List of np.arrays, each formatted as a len 3 np.array with rgb values

    Returns:
        list: list containing converted hex colormap

    """
    hex_cmap = []

    # first, sort rgb colors based off hsv
    rgb_cmap_sorted = sorted(rgb_cmap, key=lambda x: rgb_to_hsv(x[0], x[1], x[2]))

    # iterate through sorted cmap - particularly important for plotly since
    # plotly doesn't apply any logic onto colormapss
    for c in rgb_cmap_sorted:
        hex_cmap.append(convert_rgb_to_hex(c))

    return hex_cmap