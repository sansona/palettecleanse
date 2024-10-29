"""Test suite for palette class & functions in palettecleanse"""

from pathlib import Path

import numpy as np
import PIL
import pytest
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

from ..palettecleanse.palette import *
from ..palettecleanse.utils import convert_rgb_palette_to_hex, convert_rgb_to_hex

im_fpath = Path("palettecleanse/images/vangogh.jpg")


@pytest.fixture
def palette():
    """
    Sets up base palette for following tests

    Note that .jpg used as test fixture - further tests
    validate jpg & other extension parity

    Returns:
        (Palette)
    """
    return Palette(im_fpath)


def test_palette_load_image(palette):
    """Tests that palette properly loads & formats image upon initialization"""
    assert type(palette.image) == PIL.Image.Image


def test_palette_extract_colors(palette):
    """Tests that palette properly extracts colors out of image based off
    initialization parameters"""
    assert type(palette.colors) == np.ndarray
    assert palette.colors.shape == (5, 3)  # 5 for default clusters, 3 for RGB channels


def test_palette_attributes(palette):
    """Tests that palette has properly initialized starting attributes that
    don't change after initialization"""
    # these are the base settings that the palette fixture was initialized with
    n_colors = 10
    more_n_palette = Palette(im_fpath, n_colors=n_colors)
    assert len(more_n_palette.qualitative) == n_colors


def test_palette_n_colors_attribute(palette):
    """Tests that palette has properly initialized with `n_colors` length qualitative
    palettes"""
    # these are the base settings that the palette fixture was initialized with
    assert palette.image_fpath.name == "vangogh.jpg"
    assert palette.n_colors == 5


def test_segmented_palette_generation(palette):
    """Tests that palette generate_palette function properly generates segmented (default) palette"""
    assert type(palette.sequential) == LinearSegmentedColormap
    assert type(palette.diverging) == LinearSegmentedColormap
    assert type(palette.cyclic) == LinearSegmentedColormap
    assert type(palette.qualitative) == np.ndarray
    assert type(palette.qualitative_palette) == ListedColormap
    assert type(palette.plotly) == list


def test_palette_display(palette):
    """Tests that palette display functions display palettes"""
    palette.display_all_palettes()


def test_palette_example_plots(palette):
    """Tests that palette able to make example plots"""
    palette.display_example_plots()
    palette.display_plotly_examples()


def test_palette_image_extension_sensitivity():
    """Tests that palette can load other file extensions. Most
    errors will be caught upon initialization"""
    Palette("palettecleanse/images/vangogh.bmp")
    Palette("palettecleanse/images/vangogh.gif")
    Palette("palettecleanse/images/vangogh.png")
    Palette("palettecleanse/images/vangogh.tiff")


def test_convert_rgb_to_hex():
    "Tests that RGB values are properly converted to hex values"
    white = np.array([0, 0, 0]) / 255.0  # normalize between [0,1]
    black = np.array([255, 255, 255]) / 255.0
    gray = np.array([127, 127, 127]) / 255.0

    assert convert_rgb_to_hex(white) == "#000000"
    assert convert_rgb_to_hex(black) == "#ffffff"
    assert convert_rgb_to_hex(gray) == "#7f7f7f"


def test_convert_rgb_palette_to_hex_palette():
    "Tests that RGB palette is properly converted to hex palette"
    # this is an example palette created to test a variety of colors
    rgb_palette = [
        np.array([0, 0, 0]),
        np.array([255, 255, 255]),
        np.array([127, 127, 127]),
        np.array([65, 65, 65]),
        np.array([199, 250, 0]),
        np.array([0, 77, 20]),
        np.array([210, 0, 7]),
    ]
    # normalize to [0, 1]
    rgb_palette = [x / 255.0 for x in rgb_palette]

    # note that a shuffling step happens so the rgb_palette -> hex_palette
    # ordering is not perfectly 1:1
    hex_palette = convert_rgb_palette_to_hex(rgb_palette)

    assert len(hex_palette) == len(rgb_palette)
    assert hex_palette == [
        "#000000",
        "#414141",
        "#7f7f7f",
        "#ffffff",
        "#c7fa00",
        "#004d14",
        "#d20007",
    ]
