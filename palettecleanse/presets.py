"""
Collection of predefined color palettes. See `images` folder
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

starting_dir = Path.cwd()

# for pytest relative import issue
try:
    from palettecleanse.palette import Palette
    from palettecleanse.utils import PaletteTypes

    os.chdir(os.path.join(os.path.dirname(__file__), "../images/"))
    fpath = Path.cwd()
except ImportError:
    from .palette import Palette
    from .utils import PaletteTypes

    fpath = Path("palettecleanse/images")

Vangogh = Palette(fpath / "vangogh.jpg")
GreatWave = Palette(fpath / "great_wave.jpg")
PinkRoses = Palette(fpath / "pink_roses.jpg")
RedRose = Palette(fpath / "red_roses.jpg")
TwilightSunset = Palette(fpath / "sunset.jpg")
BladerunnerOlive = Palette(fpath / "bladerunner_olive.jpg")
Water = Palette(fpath / "water.jpg")
Candles = Palette(fpath / "candles.jpg")
NeighborhoodSucculents = Palette(fpath / "neighborhood_succulents.jpg")
Dance = Palette(fpath / "dance.jpg")

all_presets = {
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


def display_all_preset_palettes(palette_type) -> None:
    """
    Displays all preset palette options in a single plot

    Args:
        palette_type (str): see utils.PaletteTypes
    Returns:
        (None)
    """
    available_types = [x.name.lower() for x in PaletteTypes]
    if palette_type not in available_types:
        return f"{palette_type} not in [available_types]"

    # get the corresponding colormap for the type of `palette_type`
    n_presets = len(all_presets.keys())
    all_palettes = [getattr(x, palette_type) for x in all_presets.keys()]

    # generate the gradient for palette display
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))

    fig, axes = plt.subplots(n_presets, 1, figsize=(10, n_presets // 1.25))
    # iterate over each preset & display
    for ax, palette, name in tqdm(
        zip(axes, all_palettes, all_presets.values()),
        desc=f"Generating {palette_type} displays...",
        total=n_presets,
    ):
        ax.imshow(gradient, aspect="auto", cmap=palette)
        ax.set_title(name)
        ax.axis("off")
    fig.suptitle(f"All {palette_type} palettes")
    plt.tight_layout()
