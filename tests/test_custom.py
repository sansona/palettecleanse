"""Test suite for pre-defined palettes"""

import sys
import os

sys.path.append("..")
os.chdir('..')

from custom import *

def test_all_palettes() -> None:
    "Tests that all existing palettes are loaded properly"
    preexisting_palettes = [Vangogh, GreatWave, PinkRoses, RedRose, TwilightSunset]
    for palette in preexisting_palettes:
        # if this function returns no errors, then palettes
        # all successfully converted
        palette.display_all_palettes()
