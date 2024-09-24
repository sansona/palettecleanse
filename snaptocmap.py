"""
Collection of CMap objects
"""
from PIL import Image
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


class CMap:
    """
    Base colormap (CMap) class
    """

    def __init__(
        self, image_fname: str, cmap_type: str = "sequential", n_colors: int = 5
    ):
        """
        Initialize the CMap object with an image, colormap type, and number of colors

        Args:
            image_fname (str): filename to image
            cmap_type (str): type of colormap. Options are:
                ['sequential, 'diverging', 'cyclic', 'converging', qualitative']
                Default to 'sequential'
            n_colors (int): number of colors to cluster in clustering algorithm. Default to 5

        Returns:
            (None)
        """
        self.image_fname = image_fname
        self.cmap_type = cmap_type
        self.n_colors = n_colors
        self.image = self.load_image()
        self.colors = self.extract_colors()
        self.cmap = self.generate_cmap()

    def load_image(self) -> Image:
        """
        Load the image using PIL and convert to RGB.

        Args:
            (None)
        Returns:
            (Image)
        """
        return Image.open(self.image_fname).convert("RGB")

    def extract_colors(self) -> list:
        """
        Extract the `n_colors` dominant colors using KMeans clustering

        Args:
            (None)
        Returns:
            (np.array): array of normalized extracted colors
        """
        image_array = np.array(self.image)
        # Reshape image array to 2D (n_pixels, 3) where 3 is for RGB
        pixels = image_array.reshape((-1, 3))

        # Use KMeans clustering to find dominant colors
        kmeans = KMeans(n_clusters=self.n_colors)
        kmeans.fit(pixels)

        # Get the cluster centers (dominant colors)
        colors = kmeans.cluster_centers_
        # Normalize to [0, 1] range for compatibility with matplotlib colormaps
        colors = colors / 255.0

        return colors

    def generate_cmap(self):
        """
        Generates a colormap based on the extracted colors from clustering and specified cmap_type

        Args:
            (None)
        Returns:
            (mcolors.ColorMap)
        """

        if self.cmap_type == "sequential":
            return mcolors.LinearSegmentedColormap.from_list(
                "sequential", self.colors, N=256
            )

        elif self.cmap_type == "diverging":
            if len(self.colors) >= 2:
                midpoint = len(self.colors) // 2
                diverging_colors = np.vstack(
                    (self.colors[0], self.colors[midpoint:], self.colors[-1])
                )
                return mcolors.LinearSegmentedColormap.from_list(
                    "diverging", diverging_colors, N=256
                )
            else:
                raise ValueError(
                    "Image must contain at least two colors for diverging colormap"
                )

        elif self.cmap_type == "cyclic":
            cyclic_colors = np.vstack(
                (self.colors, self.colors[0])
            )  # Repeat the first color at the end
            return mcolors.LinearSegmentedColormap.from_list(
                "cyclic", cyclic_colors, N=256
            )

        elif self.cmap_type == "converging":
            # Assuming converging goes towards a specific central color
            if len(self.colors) >= 2:
                midpoint = len(self.colors) // 2
                converging_colors = np.vstack(
                    (
                        self.colors[:midpoint],
                        np.mean(self.colors, axis=0),
                        self.colors[midpoint:],
                    )
                )
                return mcolors.LinearSegmentedColormap.from_list(
                    "converging", converging_colors, N=256
                )
            else:
                raise ValueError(
                    "Image must contain at least two colors for converging colormap"
                )

        elif self.cmap_type == "qualitative":
            return mcolors.ListedColormap(self.colors, name="qualitative")

        else:
            raise ValueError(f"Colormap type '{self.cmap_type}' not implemented")

    def display_cmap(self):
        """
        Displays a colorbar of the cmap. Note that cmap will be shaded according to a gradient

        Args:
            (None)
        Returns:
            (None)
        """
        gradient = np.linspace(0, 1, 256)
        gradient = np.vstack((gradient, gradient))
        plt.figure(figsize=(12, 2))
        plt.imshow(gradient, aspect="auto", cmap=self.cmap)
        plt.axis("off")
        plt.show()

    def display_all_cmaps(self):
        """
        Displays all possible colormap options alongside original image

        Args:
            (None)
        Returns:
            (None)
        """
        cmap_types = ["sequential", "diverging", "cyclic", "converging", "qualitative"]
        n_cmaps = len(cmap_types)

        # Create subplots: one for the image, others for colormaps
        _, axes = plt.subplots(1, n_cmaps + 1, figsize=(24, 2))

        # Display the original image in the first subplot
        axes[0].imshow(self.image)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        # Generate the gradient for colormap display
        gradient = np.linspace(0, 1, 256)
        gradient = np.vstack((gradient, gradient))

        # Iterate over each colormap type and display it
        for ax, cmap_type in zip(axes[1:], cmap_types):
            self.cmap_type = cmap_type
            self.cmap = self.generate_cmap()  # Generate the corresponding colormap

            ax.imshow(gradient, aspect="auto", cmap=self.cmap)
            ax.set_title(f"{cmap_type.capitalize()}")
            ax.axis("off")

        plt.tight_layout()
        plt.show()

    def get_cmap(self):
        """
        Return the generated colormap for use in plotting

        Args:
            (None)
        Returns:
            (mcolors.ColorMap)
        """
        return self.cmap
