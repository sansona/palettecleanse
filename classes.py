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
            (np.array): array of normalized extracted colors
        """
        image_array = np.array(self.image)
        # Reshape image array to 2D (n_pixels, 3) where 3 == RGB channels
        pixels = image_array.reshape((-1, 3))

        # Use KMeans clustering to find dominant colors
        kmeans = KMeans(n_clusters=self.n_colors, n_init=10)
        kmeans.fit(pixels)

        # Get the cluster centers (dominant colors)
        colors = kmeans.cluster_centers_
        # Normalize to [0, 1] range
        colors = colors / 255.0

        return colors

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

        # Create subplots: one for the image, others for colormaps
        _, axes = plt.subplots(n_cmaps, 1, figsize=(6, 3))

        # Generate the gradient for colormap display
        gradient = np.linspace(0, 1, 256)
        gradient = np.vstack((gradient, gradient))

        # Iterate over each colormap type and display it
        for ax, cmap_type in zip(axes, cmap_types):
            self.cmap_type = cmap_type
            self.cmap = self.generate_cmap()

            ax.imshow(gradient, aspect="auto", cmap=self.cmap)
            ax.set_title(f"{cmap_type.capitalize()}")
            ax.axis("off")

        plt.tight_layout()

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

        Args:
            (None)
        Returns:
            (None)
        """
        # generate plot dimensions, load image
        fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20,6))
        colors = self.cmap(np.linspace(0, 1, 5))

        axes[0, 0].imshow(plt.imread(self.image_fname))
        axes[0, 0].set_title('Image')

        # generate scatter plot - sequential colormap
        s = axes[0, 1].scatter(np.random.rand(100),
                                np.random.rand(100),
                                c=range(10,1010,10),
                                cmap=self.cmap)
        axes[0, 1].set_title('Scatter Plot')
        fig.colorbar(s, ax=axes[0, 1])

        # generate bar plot - discete colormap
        axes[0, 2].barh(['cat', 'dog', 'fish', 'owl', 'whale'],
                        [15, 30, 45, 60, 20],
                        color=colors)
        axes[0, 2].set_title(f'Pareto Plot')

        # generate stackplot - discrete colormap
        x = list(range(10))
        values = [sorted(np.random.rand(10)) for i in range(5)]
        y = dict(zip(x, values))
        axes[0, 3].stackplot(x, y.values(), alpha=0.8, colors=colors)
        axes[0, 3].set_title('Stack Plot')

        # generate Kaplan-Meier plot - discrete colormap
        n = 100
        populations = 5
        T = [np.random.exponential(20*i, n) for i in range(populations)]
        E = [np.random.binomial(1, 0.15, n) for _ in range(populations)]
        kmf = KaplanMeierFitter()

        for i in range(populations):
            kmf.fit(T[i], E[i])
            kmf.plot_survival_function(ax=axes[1, 0], color=colors[i], alpha=0.8)
        axes[1, 0].legend().remove()
        axes[1, 0].set_title('Survival Plot')

        # generate violin plot - discrete colormap
        violin_data = [np.random.normal(5, 1.5, 100),
                np.random.normal(0, 1, 100),
                np.random.normal(10, 2, 100),
                np.random.normal(3, 5, 100)]
        p = axes[1, 1].violinplot(violin_data, showmedians=False, showmeans=False)

        for i, pc in enumerate(p['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_edgecolor(colors[0])
            pc.set_alpha(0.8)
            # set extrema bars to be last indexed color in cmap
            for partname in ('cbars', 'cmins', 'cmaxes'):
                p[partname].set_color(colors[-1])
        axes[1, 1].set_title('Violin Plot')

        # generate kde charts - discrete colormap
        kde_data = [np.random.normal(size=100, loc=10, scale=2),
                    np.random.normal(size=50, loc=70, scale=4),
                    np.random.normal(size=200, loc=20, scale=8),
                    np.random.normal(size=70, loc=0, scale=3)]
        for i in range(len(kde_data)):
            axes[1, 2].hist(kde_data[i], density=True, color=colors[i], bins=50)
        axes[1, 2].set_title('Histogram Plot')

        axes[1, 3] = sns.heatmap(sns.load_dataset("glue").pivot(index="Model", columns="Task", values="Score"), linewidth=0.5,annot=True, cmap=self.cmap)
        axes[1, 3].set_title('Heat Map')

        # Turn off all labels
        for i, ax in enumerate(axes.flatten()):
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.grid(False)
