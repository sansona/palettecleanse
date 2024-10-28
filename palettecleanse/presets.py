"""
Collection of predefined color palettes. See `images` folder
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

starting_dir = os.getcwd()

# for pytest relative import issue
try:
    from palettecleanse.palette import Palette
    from palettecleanse.utils import PaletteTypes

    os.chdir(os.path.join(os.path.dirname(__file__), "../images/"))
    fpath = os.getcwd()
except ImportError:
    from .palette import Palette
    from .utils import PaletteTypes

    fpath = "palettecleanse/images"

Vangogh = Palette(f"{fpath}/vangogh.jpg")
GreatWave = Palette(f"{fpath}/great_wave.jpg")
PinkRoses = Palette(f"{fpath}/pink_roses.jpg")
RedRose = Palette(f"{fpath}/red_roses.jpg")
TwilightSunset = Palette(f"{fpath}/sunset.jpg")
BladerunnerOlive = Palette(f"{fpath}/bladerunner_olive.jpg")
Water = Palette(f"{fpath}/water.jpg")
Candles = Palette(f"{fpath}/candles.jpg")
NeighborhoodSucculents = Palette(f"{fpath}/neighborhood_succulents.jpg")
Dance = Palette(f"{fpath}/dance.jpg")

all_customs = {
    Vangogh: "Vangogh",
    GreatWave: "GreatWave",
    PinkRoses: "PinkRoses",
    RedRose: "RedRose",
    TwilightSunset: "TwilightSunset",
    BladerunnerOlive: "BladerunnerOlive",
    Water: "Water",
    Candles: "Candles",
    NeighborhoodSucculents: "NeighborhoodSucculents",
    Dance: "Dance",
}

os.chdir(starting_dir)


def display_all_custom_palettes(palette_type) -> None:
    """
    Displays all custom palette options in a single plot

    Args:
        palette_type (str): see utils.PaletteTypes
    Returns:
        (None)
    """
    available_types = [x.name.lower() for x in PaletteTypes]
    if palette_type not in available_types:
        return f"{palette_type} not in [available_types]"

    # get the corresponding colormap for the type of `palette_type`
    n_customs = len(all_customs.keys())
    all_palettes = [getattr(x, palette_type) for x in all_customs.keys()]

    # generate the gradient for palette display
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))

    fig, axes = plt.subplots(n_customs, 1, figsize=(10, n_customs // 1.25))
    # iterate over each custom & display
    for ax, palette, name in tqdm(
        zip(axes, all_palettes, all_customs.values()),
        desc=f"Generating {palette_type} displays...",
        total=n_customs,
    ):
        ax.imshow(gradient, aspect="auto", cmap=palette)
        ax.set_title(name)
        ax.axis("off")
    fig.suptitle(f"All {palette_type} palettes")
    plt.tight_layout()
