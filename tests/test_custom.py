"""Test suite for pre-defined palettes"""

import sys
import os

sys.path.append("..")
os.chdir('..')

from custom import *

def test_all_palettes() -> None:
    """Tests that all existing palettes are loaded properly.
    Note that this is a slow test given the number of displays,
    and may be a better idea to explicitly call the function in
    a notebook to confirm visually"""
    display_all_custom_palettes('sequential')
    display_all_custom_palettes('diverging')
    display_all_custom_palettes('cyclic')
    display_all_custom_palettes('qualitative')
