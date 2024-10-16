"""Test suite for pre-defined colormaps"""

import sys

sys.path.append("..")
from colormaps import *


def test_all_colormaps() -> None:
    "Tests that all existing colormaps are loaded properly"
    preexisting_cmaps = [Vangogh, GreatWave, PinkRoses, RedRose, TwilightSunset]
    for cmap in preexisting_cmaps:
        # if this function returns no errors, then colormaps
        # all successfully converted
        cmap.display_all_cmaps()
