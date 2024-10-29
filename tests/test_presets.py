"""Test suite for preset palettes"""

from pathlib import Path

from PIL import Image
from tqdm import tqdm

from ..palettecleanse.presets import *
from ..palettecleanse.utils import PaletteTypes

COMPRESSION_SIZE = (500, 500)
IMAGE_PATH = Path("palettecleanse/images")

# high resolution images that are kept for demo purposes
OMIT = [
    IMAGE_PATH / "vangogh.jpg",
    IMAGE_PATH / "pink_roses_full_res.jpg",
    IMAGE_PATH / "vangogh.tiff",
]


def test_image_size() -> None:
    """Tests that all images in image directory are below compression size"""
    ims = [
        x
        for x in IMAGE_PATH.glob("*")
        if x.suffix in [".jpg", ".jpeg", ".png", ".bmp", ".gif"]
    ]

    ims = [x for x in ims if x not in OMIT]
    for im in ims:
        assert Image.open(im).size <= COMPRESSION_SIZE


def test_all_palettes() -> None:
    """Tests that all existing palettes are loaded properly.
    Note that this is a slow test given the number of displays,
    and may be a better idea to explicitly call the function in
    a notebook to confirm visually"""
    palette_types = [x.name.lower() for x in PaletteTypes]

    for pal in tqdm(
        palette_types, desc="Generating test displays...", total=len(palette_types)
    ):
        display_all_preset_palettes(pal)
