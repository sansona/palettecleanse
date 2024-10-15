"""
Collection of CMap objects
"""

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from lifelines import KaplanMeierFitter
from PIL import Image
from sklearn.cluster import KMeans


class CMap:
    """
    Base colormap (CMap) class
    """

    def __init__(
        self, image_fname: str, cmap_type: str = "sequential", n_colors: int = 5
    ) -> None:
        """
        Initialize the CMap object with an image, colormap type, and number of colors

        Args:
            image_fname (str): filename to image
            cmap_type (str): type of colormap. Options are:
                ['sequential, 'diverging', 'cyclic', 'qualitative']
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
        self.sequential = self.generate_sequential()
        self.diverging = self.generate_diverging()
        self.cyclic = self.generate_cyclic()
        self.qualitative = self.generate_qualitative_raw()
        self.qualitative_cmap = self.generate_qualitative_cmap()

    def load_image(self) -> Image:
        """
        Load the image, convert to RGB, and compress image.
        Compression primarily helps with clustering later on - large image
            files will still have a delay

        Args:
            (None)
        Returns:
            (Image)
        """
        img = Image.open(self.image_fname).convert("RGB")
        img = img.resize((100, 100))
        return img

    def extract_colors(self) -> np.ndarray:
        """
        Extract the `n_colors` dominant colors using KMeans clustering

        Args:
            (None)
        Returns:
            (np.ndarray): array of normalized extracted colors
        """
        image_array = np.array(self.image)
        # reshape image array to 2D (n_pixels, 3) where 3 == RGB channels
        pixels = image_array.reshape((-1, 3))

        # kmeans clustering to find dominant colors
        kmeans = KMeans(n_clusters=self.n_colors, n_init=10)
        kmeans.fit(pixels)

        colors = kmeans.cluster_centers_
        # normalize to [0, 1] range since most plotting libraries use that
        colors = colors / 255.0

        return colors

    def generate_sequential(self) -> mcolors.LinearSegmentedColormap:
        """
        Generates a sequential colormap

        Args:
            (None)
        Returns:
            (mcolors.LinearSegmentedColormap)
        """
        # sort RGB array by sorting on least significant to most significant
        # https://stackoverflow.com/questions/2828059/sorting-arrays-in-numpy-by-column/38194077#38194077
        colors = self.colors[self.colors[:, 2].argsort()]
        colors = colors[colors[:, 1].argsort(kind="mergesort")]
        colors = colors[colors[:, 0].argsort(kind="mergesort")]

        return mcolors.LinearSegmentedColormap.from_list(
            "sequential", colors, N=256
        )

    def generate_diverging(self) -> mcolors.LinearSegmentedColormap:
        """
        Generates a diverging colormap

        Args:
            (None)
        Returns:
            (mcolors.LinearSegmentedColorMap)
        """
        # find midpoint of colors and split cmap into less or greater than midpoint
        if len(self.colors) >= 2:
            midpoint = len(self.colors) // 2
            diverging_colors = np.vstack((self.colors[0], self.colors[midpoint:], self.colors[-1]))
            return mcolors.LinearSegmentedColormap.from_list(
                "diverging", diverging_colors, N=256
            )
        else:
            raise ValueError(
                "Image must contain at least two colors for diverging colormap"
            )

    def generate_cyclic(self) -> mcolors.LinearSegmentedColormap:
        """
        Generates a cyclic colormap

        Args:
            (None)
        Returns:
            (mcolors.LinearSegmentedColormap)
        """
        # repeat the first color at the end
        cyclic_colors = np.vstack(
            (self.colors, self.colors[0])
        )
        return mcolors.LinearSegmentedColormap.from_list(
            "cyclic", cyclic_colors, N=256
        )

    

    def generate_qualitative_raw(self) -> np.ndarray:
        """
        Generates a raw array of colors corresponding to qualitative
        colormap.

        Some plotting libraries utilize a raw array of colors as opposed
        to a mcolors object for qualitative colormaps

        Args:
            (None)
        Returns:
            (np.ndarray): a shuffled array of colors spaced within colormap
        """
        np.random.seed(42)
        colors = self.cmap(np.linspace(0, 1, self.n_colors))
        np.random.shuffle(colors)  # this modifies in line
        return colors

    def generate_qualitative_cmap(self) -> mcolors.ListedColormap:
        """
        Generates a qualitative colormap

        Note - the `generate_qualitative_raw` method is likely preferred
        for certain plotting libraries

        Args:
            (None)
        Returns:
            (mcolors.ListedColormap): a shuffled array of colors spaced within colormap
        """
        return mcolors.ListedColormap(self.colors, name="qualitative")

    def generate_cmap(self) -> None:
        """
        Generates a colormap based on the extracted colors from clustering and specified cmap_type

        Args:
            (None)
        Returns:
            (mcolors.ColorMap)
        """
        # since each cmap type modifies, instantiate within function itself
        colors = self.colors

        if self.cmap_type == "sequential":
            # sort RGB array by sorting on least significant to most significant
            # https://stackoverflow.com/questions/2828059/sorting-arrays-in-numpy-by-column/38194077#38194077
            colors = colors[colors[:, 2].argsort()]
            colors = colors[colors[:, 1].argsort(kind="mergesort")]
            colors = colors[colors[:, 0].argsort(kind="mergesort")]

            return mcolors.LinearSegmentedColormap.from_list(
                "sequential", colors, N=256
            )

        elif self.cmap_type == "diverging":
            if len(self.colors) >= 2:
                midpoint = len(colors) // 2
                diverging_colors = np.vstack((colors[0], colors[midpoint:], colors[-1]))
                return mcolors.LinearSegmentedColormap.from_list(
                    "diverging", diverging_colors, N=256
                )
            else:
                raise ValueError(
                    "Image must contain at least two colors for diverging colormap"
                )

        elif self.cmap_type == "cyclic":
            cyclic_colors = np.vstack(
                (colors, colors[0])
            )  # Repeat the first color at the end
            return mcolors.LinearSegmentedColormap.from_list(
                "cyclic", cyclic_colors, N=256
            )

        elif self.cmap_type == "qualitative":
            return mcolors.ListedColormap(self.colors, name="qualitative")

        else:
            raise ValueError(f"Colormap type '{self.cmap_type}' not implemented")

    def display_cmap(self) -> None:
        """
        Displays a colorbar of the cmap. Note that cmap will be shaded according to a gradient

        Args:
            (None)
        Returns:
            (None)
        """
        # create empty template
        gradient = np.linspace(0, 1, 256)
        gradient = np.vstack((gradient, gradient))

        plt.figure(figsize=(12, 2))
        plt.imshow(gradient, aspect="auto", cmap=self.cmap)
        plt.axis("off")

    def display_all_cmaps(self) -> None:
        """
        Displays all possible colormap options

        Args:
            (None)
        Returns:
            (None)
        """
        cmap_types = ["sequential", "diverging", "cyclic", "qualitative"]
        n_cmaps = len(cmap_types)

        # remember what initiated with in order to reset after
        # iterating through
        init_cmap_type = self.cmap_type
        init_cmap = self.cmap
        init_colors = self.colors

        # create subplots: one for the image, others for colormaps
        _, axes = plt.subplots(n_cmaps, 1, figsize=(6, 3))

        # generate the gradient for colormap display
        gradient = np.linspace(0, 1, 256)
        gradient = np.vstack((gradient, gradient))

        # iterate over each colormap type and display it
        for ax, cmap_type in zip(axes, cmap_types):
            self.cmap_type = cmap_type
            self.cmap = self.generate_cmap()

            ax.imshow(gradient, aspect="auto", cmap=self.cmap)
            ax.set_title(f"{cmap_type.capitalize()}")
            ax.axis("off")

        plt.tight_layout()

        # reset cmaps to original. This is to avoid the scenario
        # in which displaying all cmaps changes the cmaps
        self.cmap_type = init_cmap_type
        self.cmap = init_cmap
        self.colors = init_colors

    def display_all_cmaps_with_image(self) -> None:
        """
        Displays all possible colormap options alongside original image

        Args:
            (None)
        Returns:
            (None)
        """
        cmap_types = ["sequential", "diverging", "cyclic", "qualitative"]
        n_cmaps = len(cmap_types)

        # Create subplots: one for the image, others for colormaps
        _, axes = plt.subplots(1, n_cmaps + 1, figsize=(24, 2))

        # display the original image in the first subplot
        axes[0].imshow(self.image)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        # generate the gradient for colormap display
        gradient = np.linspace(0, 1, 256)
        gradient = np.vstack((gradient, gradient))

        # iterate over each colormap type and display it
        for ax, cmap_type in zip(axes[1:], cmap_types):
            self.cmap_type = cmap_type
            self.cmap = self.generate_cmap()

            ax.imshow(gradient, aspect="auto", cmap=self.cmap)
            ax.set_title(f"{cmap_type.capitalize()}")
            ax.axis("off")

        plt.tight_layout()

    def display_example_plots(self) -> None:
        """
        Applies colormap to selection of preprogrammed plots for ease of data
        visualization. Note only matplotlib & seaborn plots generated due to
        difference in plotly subplot interactions

        This function follows bad code practice but given size of
        library, opted to keep all as single function instead of dispersing
        Args:
            (None)
        Returns:
            (None)
        """
        fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 6))

        axes[0, 0].imshow(plt.imread(self.image_fname))
        axes[0, 0].set_title("Image")

        # generate scatter plot - sequential colormap
        s = axes[0, 1].scatter(
            np.random.rand(100),
            np.random.rand(100),
            c=range(10, 1010, 10),
            cmap=self.cmap,
        )
        axes[0, 1].set_title("Scatter Plot")
        fig.colorbar(s, ax=axes[0, 1])

        # generate bar plot - discrete colormap
        axes[0, 2].barh(
            ["cat", "dog", "fish", "owl", "whale"],
            [15, 30, 45, 60, 20],
            color=self.qualitative,
        )
        axes[0, 2].set_title("Pareto Plot")

        # generate stackplot - discrete colormap
        x = list(range(10))
        values = [sorted(np.random.rand(10)) for i in range(5)]
        y = dict(zip(x, values))
        axes[0, 3].stackplot(x, y.values(), alpha=0.8, colors=self.qualitative)
        axes[0, 3].set_title("Stack Plot")

        # generate Kaplan-Meier plot - discrete colormap
        n = 100
        populations = 5
        T = [np.random.exponential(20 * i, n) for i in range(populations)]
        E = [np.random.binomial(1, 0.15, n) for _ in range(populations)]
        kmf = KaplanMeierFitter()

        for i in range(populations):
            kmf.fit(T[i], E[i])
            kmf.plot_survival_function(
                ax=axes[1, 0], color=self.qualitative[i], alpha=0.8
            )
        axes[1, 0].legend().remove()
        axes[1, 0].set_title("Survival Plot")

        # generate violin plot - discrete colormap
        violin_data = [
            np.random.normal(5, 1.5, 100),
            np.random.normal(0, 1, 100),
            np.random.normal(10, 2, 100),
            np.random.normal(3, 5, 100),
        ]
        p = axes[1, 1].violinplot(violin_data, showmedians=False, showmeans=False)

        for i, pc in enumerate(p["bodies"]):
            pc.set_facecolor(self.qualitative[i])
            pc.set_edgecolor(self.qualitative[0])
            pc.set_alpha(0.8)
            # set extrema bars to be last indexed color in cmap
            for partname in ("cbars", "cmins", "cmaxes"):
                p[partname].set_color(self.qualitative[-1])
        axes[1, 1].set_title("Violin Plot")

        # generate kde charts - discrete colormap
        kde_data = [
            np.random.normal(size=100, loc=10, scale=2),
            np.random.normal(size=50, loc=70, scale=4),
            np.random.normal(size=200, loc=20, scale=6),
            np.random.normal(size=70, loc=0, scale=3),
        ]
        for i in range(len(kde_data)):
            axes[1, 2].hist(
                kde_data[i], density=True, color=self.qualitative[i], bins=10
            )
        axes[1, 2].set_title("Histogram Plot")

        # generate heat map
        axes[1, 3] = sns.heatmap(
            sns.load_dataset("glue").pivot(
                index="Model", columns="Task", values="Score"
            ),
            linewidth=0.5,
            annot=True,
            cmap=self.cmap,
        )
        axes[1, 3].set_title("Heat Map")

        # turn off all labels
        for i, ax in enumerate(axes.flatten()):
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.grid(False)
