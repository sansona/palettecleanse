"""Test suite for cmap class & functions in palettecleanser"""

import sys

import numpy as np
import PIL
import pytest
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

sys.path.append("..")
from cmap import *

im_fname = "../images/vangogh.jpg"


@pytest.fixture
def cmap():
    """
    Sets up base cmap for following tests

    Note that .jpg used as test fixture - further tests
    validate jpg & other extension parity

    Returns:
        (CMap)
    """
    return CMap(im_fname)


def test_cmap_load_image(cmap):
    """Tests that cmap properly loads & formats image upon initialization"""
    assert type(cmap.image) == PIL.Image.Image


def test_cmap_extract_colors(cmap):
    """Tests that cmap properly extracts colors out of image based off
    initialization parameters"""
    assert type(cmap.colors) == np.ndarray
    assert cmap.colors.shape == (5, 3)  # 5 for default clusters, 3 for RGB channels


def test_cmap_attributes(cmap):
    """Tests that cmap has properly initialized starting attributes that
    don't change after initialization"""
    # these are the base settings that the cmap fixture was initialized with
    n_colors = 10
    more_n_cmap = CMap(im_fname, n_colors=n_colors)
    assert len(more_n_cmap.qualitative) == n_colors


def test_cmap_n_colors_attribute(cmap):
    """Tests that cmap has properly initialized with `n_colors` length qualitative
    cmaps"""
    # these are the base settings that the cmap fixture was initialized with
    assert cmap.image_fname.endswith("vangogh.jpg")
    assert cmap.n_colors == 5


def test_segmented_cmap_generation(cmap):
    """Tests that cmap generate_cmap function properly generates segmented (default) cmap"""
    assert type(cmap.sequential) == LinearSegmentedColormap
    assert type(cmap.diverging) == LinearSegmentedColormap
    assert type(cmap.cyclic) == LinearSegmentedColormap
    assert type(cmap.qualitative) == np.ndarray
    assert type(cmap.qualitative_cmap) == ListedColormap
    assert type(cmap.plotly) == list


def test_cmap_display(cmap):
    """Tests that cmap display functions display cmaps"""
    cmap.display_all_cmaps()


def test_cmap_example_plots(cmap):
    """Tests that cmap able to make example plots"""
    cmap.display_example_plots()


def test_cmap_image_extension_sensitivity():
    """Tests that cmap can load other file extensions. Most
    errors will be caught upon initialization"""
    CMap("../images/vangogh.bmp")
    CMap("../images/vangogh.gif")
    CMap("../images/vangogh.png")
    CMap("../images/vangogh.tiff")


def test_convert_rgb_to_hex():
    "Tests that RGB values are properly converted to hex values"
    white = np.array([0, 0, 0]) / 255.0  # normalize between [0,1]
    black = np.array([255, 255, 255]) / 255.0
    gray = np.array([127, 127, 127]) / 255.0

    assert convert_rgb_to_hex(white) == "#000000"
    assert convert_rgb_to_hex(black) == "#ffffff"
    assert convert_rgb_to_hex(gray) == "#7f7f7f"


def test_convert_rgb_cmap_to_hex_cmap():
    "Tests that RGB cmap is properly converted to hex cmap"
    # this is an example cmap created to test a variety of colors
    rgb_cmap = [
        np.array([0, 0, 0]),
        np.array([255, 255, 255]),
        np.array([127, 127, 127]),
        np.array([65, 65, 65]),
        np.array([199, 250, 0]),
        np.array([0, 77, 20]) ,
        np.array([210, 0, 7]),
    ]
    # normalize to [0, 1]
    rgb_cmap = [x / 255.0 for x in rgb_cmap]

    # note that a shuffling step happens so the rgb_cmap -> hex_cmap
    # ordering is not perfectly 1:1
    hex_cmap = convert_rgb_cmap_to_hex(rgb_cmap)

    assert len(hex_cmap) == len(rgb_cmap)
    assert hex_cmap == [
        "#000000",
        "#414141",
        "#7f7f7f",
        "#ffffff",
        "#c7fa00",
        "#004d14",
        "#d20007",
    ]
