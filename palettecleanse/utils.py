"""
Collection of utils for `palettecleanse`
"""
from colorsys import rgb_to_hsv

import numpy as np
from PIL import Image

COMPRESSION_SIZE = (500, 500)


def compress_image_inplace(image_path: str) -> None:
    """
    Compresses image in place to reduce package storage.
    Used for packaging `palettecleanse`

    Args:
        image_path (str) -> path to image
    Returns:
        (None)
    """

    # tiffs and other rarer format compression not supported
    if image_path.endswith((".jpg", ".jpeg", ".png", ".bmp", ".gip")):
        img = Image.open(image_path)
        if img.size <= COMPRESSION_SIZE:
            print(
                f"{image_path} size ({img.size}) less than `compression_limit` ({COMPRESSION_SIZE}) - no compression applied"
            )
        else:
            initial_size = img.size
            img = img.resize(COMPRESSION_SIZE)
            # note this saves inplace & overwrites the original
            img.save(image_path, optimize=True, quality=85)
            print(
                f"{image_path} compressed - ({initial_size[0]-COMPRESSION_SIZE[0], initial_size[1]-COMPRESSION_SIZE[1]}) space saved."
            )
    else:
        print(f"{image_path} file extension not supported.")

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


def sort_rgb_by_hsv(rgb_palette: list) -> list:
    """
    Sorts a list of RGB colors based off HSV values

    Args:
        rgb_palette (list): List of np.arrays, each formatted as a len 3 np.array with rgb values

    Returns:
        list:sorted rgb palette
    """
    return sorted(rgb_palette, key=lambda x: rgb_to_hsv(x[0], x[1], x[2]))


def convert_rgb_palette_to_hex(rgb_palette: list) -> list:
    """
    Converts a list of np.arrays containing an entire palette of rgb values to a list of hex values

    Args:
        rgb_palette (list): List of np.arrays, each formatted as a len 3 np.array with rgb values

    Returns:
        list: list containing converted hex palette

    """
    hex_palette = []

    # first, sort rgb colors based off hsv
    rgb_palette_sorted = sorted(rgb_palette, key=lambda x: rgb_to_hsv(x[0], x[1], x[2]))

    # iterate through sorted palette - particularly important for plotly since
    # plotly doesn't apply any logic onto palettes
    for c in rgb_palette_sorted:
        hex_palette.append(convert_rgb_to_hex(c))

    return hex_palette



