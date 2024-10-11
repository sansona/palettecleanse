"""Test suite for functions in palettecleanser"""

import sys

import numpy as np

sys.path.append(".")
from functions import *

im_fname = "images/vangogh.jpg"


def test_convert_to_cmap_matplotlib():
    "Tests that matplotlib compatible cmap is generated"
    convert_to_cmap(im_fname, library="matplotlib")


def test_convert_to_cmap_seaborn():
    "Tests that seaborn compatible cmap is generated"
    convert_to_cmap(im_fname, library="seaborn")


def test_convert_to_cmap_plotly():
    "Tests that plotly compatible cmap is generated"
    convert_to_cmap(im_fname, library="plotly")


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
    # This is an example cmap created to test a variety of colors
    rgb_cmap = [
        np.array([0, 0, 0]) / 255.0,
        np.array([255, 255, 255]) / 255.0,
        np.array([127, 127, 127]) / 255.0,
        np.array([65, 65, 65]) / 255.0,
        np.array([199, 250, 0]) / 255.0,
        np.array([0, 77, 20]) / 255.0,
        np.array([210, 0, 7]) / 255.0,
    ]

    hex_cmap = convert_rgb_cmap_to_hex(rgb_cmap)

    assert len(hex_cmap) == len(rgb_cmap)
    assert hex_cmap == [
        "#000000",
        "#ffffff",
        "#7f7f7f",
        "#414141",
        "#c7fa00",
        "#004d14",
        "#d20007",
    ]
