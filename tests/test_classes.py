"""Test suite for classes in snaptocmap"""

import sys

import numpy as np
import PIL
import pytest
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

sys.path.append(".")
from classes import *

im_fname = "images/vangogh.jpg"


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
    assert cmap.image_fname.endswith("vangogh.jpg")
    assert cmap.cmap_type == "sequential"
    assert cmap.n_colors == 5


def test_segmented_cmap_generation(cmap):
    """Tests that cmap generate_cmap function properly generates segmented (default) cmap"""
    assert type(cmap.cmap) == LinearSegmentedColormap


def test_diverging_cmap_generation(cmap):
    """Tests that cmap generate_cmap function properly generates diverging cmap"""
    diverging_cmap = CMap(im_fname, cmap_type="diverging")
    assert type(diverging_cmap.cmap) == LinearSegmentedColormap


def test_cyclic_cmap_generation(cmap):
    """Tests that cmap generate_cmap function properly generates cyclic cmap"""
    cyclic_cmap = CMap(im_fname, cmap_type="cyclic")
    assert type(cyclic_cmap.cmap) == LinearSegmentedColormap


def test_qualitative_cmap_generation(cmap):
    """Tests that cmap generate_cmap function properly generates qualitative cmap"""
    qualitative_cmap = CMap(im_fname, cmap_type="qualitative")
    assert type(qualitative_cmap.cmap) == ListedColormap


def test_cmap_display(cmap):
    """Tests that cmap display functions display cmaps"""
    cmap.display_cmap()
    cmap.display_all_cmaps()


def test_cmap_image_extension_sensitivity():
    """Tests that cmap can load other file extensions. Most
    errors will be caught upon initialization"""
    CMap("images/vangogh.bmp")
    CMap("images/vangogh.gif")
    CMap("images/vangogh.png")
    CMap("images/vangogh.tiff")
