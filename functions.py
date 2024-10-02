from classes import CMap
"""
Collection of functions
"""
def convert_to_cmap(image_fname: str, cmap_type: str = "sequential", n_colors: int = 5):
    """
    Wrapper function for converting image to colormap in single line

    Args:
        image_fname (str): filename to image
        cmap_type (str): type of colormap. Options are:
            ['sequential, 'diverging', 'cyclic', 'qualitative']
            Default to 'sequential'
        n_colors (int): number of colors to cluster in clustering algorithm. Default to 5

    Returns:
        (CMap object)
    """
    cmap = CMap(image_fname, cmap_type=cmap_type, n_colors=n_colors)
    return cmap.cmap