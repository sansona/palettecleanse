"""
Collection of Palette objects
"""

from colorsys import rgb_to_hsv

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.subplots as subplots
import seaborn as sns
from lifelines import KaplanMeierFitter
from PIL import Image
from sklearn.cluster import KMeans

COMPRESSION_SIZE = (500, 500)
np.random.seed(42)  # to keep generated palette consistent


class Palette:
    """
    Base palette class
    """

    def __init__(self, image_fname: str, n_colors: int = 5) -> None:
        """
        Initialize the Palette object with an image, palette type, and number of colors

        Args:
            image_fname (str): filename to image
            n_colors (int): number of colors to cluster in clustering algorithm. Default to 5

        Returns:
            (None)
        """
        self.image_fname = image_fname
        self.n_colors = n_colors
        self.image = self.load_image()
        self.colors = self.extract_colors()
        self.sequential = self.generate_sequential()
        self.diverging = self.generate_diverging()
        self.cyclic = self.generate_cyclic()
        self.qualitative = self.generate_qualitative_raw()
        self.qualitative_palette = self.generate_qualitative_palette()
        self.plotly = self.plotly_palette()

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
        Generates a sequential palette

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

        return mcolors.LinearSegmentedColormap.from_list("sequential", colors, N=256)

    def generate_diverging(self) -> mcolors.LinearSegmentedColormap:
        """
        Generates a diverging palette

        Args:
            (None)
        Returns:
            (mcolors.LinearSegmentedColorMap)
        """
        # find midpoint of colors and split palette into less or greater than midpoint
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
                "Image must contain at least two colors for diverging palette"
            )

    def generate_cyclic(self) -> mcolors.LinearSegmentedColormap:
        """
        Generates a cyclic palette

        Args:
            (None)
        Returns:
            (mcolors.LinearSegmentedColormap)
        """
        # repeat the first color at the end
        cyclic_colors = np.vstack((self.colors, self.colors[0]))
        return mcolors.LinearSegmentedColormap.from_list("cyclic", cyclic_colors, N=256)

    def generate_qualitative_raw(self) -> np.ndarray:
        """
        Generates a raw array of colors (self.qualitative) corresponding to qualitative
        palette.

        Some plotting libraries utilize a raw array of colors as opposed
        to a mcolors object for qualitative palettes

        Args:
            (None)
        Returns:
            (np.ndarray): a shuffled array of colors spaced within palette
        """
        colors = self.sequential(np.linspace(0, 1, self.n_colors))
        np.random.shuffle(colors)  # this modifies in line
        return colors

    def generate_qualitative_palette(self) -> mcolors.ListedColormap:
        """
        Generates a qualitative palette (self.qualitative_palette)

        Note - the `generate_qualitative_raw` method is likely preferred
        for certain plotting libraries

        Args:
            (None)
        Returns:
            (mcolors.ListedColormap): a shuffled array of colors spaced within palette
        """
        return mcolors.ListedColormap(self.colors, name="qualitative")

    def plotly_palette(self) -> list:
        """
        Converts colors array to a hex sorted list for plotting in `plotly` library

        Args:
            (None)

        Returns:
            list: list containing converted hex palette
        """

        return convert_rgb_palette_to_hex(list(self.colors))

    def display_all_palettes(self) -> None:
        """
        Displays all possible palette options

        Args:
            (None)
        Returns:
            (None)
        """
        palette_names = ["sequential", "diverging", "cyclic", "qualitative"]
        palette_types = [
            self.sequential,
            self.diverging,
            self.cyclic,
            self.qualitative_palette,
        ]
        n_palettes = len(palette_types)

        # remember what initiated with in order to reset after
        # iterating through
        init_colors = self.colors

        _, axes = plt.subplots(n_palettes, 1, figsize=(6, 3))

        # generate the gradient for palette display
        gradient = np.linspace(0, 1, 256)
        gradient = np.vstack((gradient, gradient))

        # iterate over each palette type and display it
        for ax, palette, name in zip(axes, palette_types, palette_names):
            ax.imshow(gradient, aspect="auto", cmap=palette)
            ax.set_title(name)
            ax.axis("off")

        plt.tight_layout()

        # reset colors to original. This is to avoid the scenario
        # in which displaying all palettes changes the colors var
        self.colors = init_colors

    def display_example_plots(self) -> None:
        """
        Applies palette to selection of preprogrammed plots for ease of data
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

        # scatter plot
        s = axes[0, 1].scatter(
            np.random.rand(100),
            np.random.rand(100),
            c=range(10, 1010, 10),
            cmap=self.sequential,
        )
        axes[0, 1].set_title("Scatter Plot")
        fig.colorbar(s, ax=axes[0, 1])

        # bar plot
        axes[0, 2].barh(
            ["cat", "dog", "fish", "owl", "whale"],
            [15, 30, 45, 60, 20],
            color=self.qualitative,
        )
        axes[0, 2].set_title("Pareto Plot")

        # stackplot
        x = list(range(10))
        values = [sorted(np.random.rand(10)) for _ in range(5)]
        y = dict(zip(x, values))
        axes[0, 3].stackplot(x, y.values(), alpha=0.8, colors=self.qualitative)
        axes[0, 3].set_title("Stack Plot")

        # Kaplan-Meier plot
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

        # violin plot
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
            # set extrema bars to be last indexed color in palette
            for partname in ("cbars", "cmins", "cmaxes"):
                p[partname].set_color(self.qualitative[-1])
        axes[1, 1].set_title("Violin Plot")

        # kde
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

        # heat map
        axes[1, 3] = sns.heatmap(
            sns.load_dataset("glue").pivot(
                index="Model", columns="Task", values="Score"
            ),
            linewidth=0.5,
            annot=True,
            cmap=self.sequential,
        )
        axes[1, 3].set_title("Heat Map")

        # turn off all labels
        for i, ax in enumerate(axes.flatten()):
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.grid(False)

    def display_plotly_examples(self) -> None:
        """
        Applies palette to selection of preprogrammed plots for ease of data
        visualization. Generates plots using Plotly instead of matplotlib/seaborn

        Plotly handles things like ordering and the raw values differently than matplotlib/seaborn do, so plots will not come out perfectly identical

        Args:
            (None)
        Returns:
            (None)
        """
        fig = subplots.make_subplots(
            rows=2,
            cols=4,
            subplot_titles=(
                "Image",
                "Scatter Plot",
                "Pareto Plot",
                "Stack Plot",
                "Survival Plot",
                "Violin Plot",
                "Histogram Plot",
                "Heat Map",
            ),
        )

        # image
        img = go.Image(z=plt.imread(self.image_fname))
        fig.add_trace(img, row=1, col=1)

        # scatter plot
        scatter = go.Scatter(
            x=np.random.rand(100),
            y=np.random.rand(100),
            mode="markers",
            marker=dict(
                color=np.random.rand(100) * 100,
                colorscale=self.plotly,
                cmin=0,
                cmax=100,
                size=7,
                opacity=0.8,
                showscale=False,
            ),
        )
        fig.add_trace(scatter, row=1, col=2)

        # pareto plot
        categories = ["cat", "dog", "fish", "owl", "whale"]
        values = [15, 30, 45, 60, 20]
        pareto = go.Bar(
            x=values,
            y=categories,
            orientation="h",
            marker=dict(color=self.plotly),
        )
        fig.add_trace(pareto, row=1, col=3)

        # stackplot
        x = list(range(10))
        values = [sorted(np.random.rand(10)) for _ in range(5)]
        y = np.array(values).T

        for i in range(len(values)):
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y[i],
                    mode="lines",
                    stackgroup="one",
                    fill="tozeroy",
                    line=dict(color=self.plotly[i]),
                ),
                row=1,
                col=4,
            )

        # Kaplan-Meier plot
        n = 100
        populations = 5
        T = [np.random.exponential(20 * (i + 1), n) for i in range(populations)]
        E = [np.random.binomial(1, 0.15, n) for _ in range(populations)]

        for i in range(populations):
            kmf = KaplanMeierFitter()
            kmf.fit(T[i], E[i])

            fig.add_trace(
                go.Scatter(
                    x=kmf.survival_function_.index,
                    y=kmf.survival_function_["KM_estimate"],
                    mode="lines",
                    line=dict(color=self.plotly[i], width=2),
                ),
                row=2,
                col=1,
            )

        # violin plot
        violin_data = [
            np.random.normal(5, 1.5, 100),
            np.random.normal(0, 1, 100),
            np.random.normal(10, 2, 100),
            np.random.normal(3, 5, 100),
        ]
        for i, data in enumerate(violin_data):
            fig.add_trace(
                go.Violin(y=data, box_visible=True, line_color=self.plotly[i]),
                row=2,
                col=2,
            )

        # histogram plot
        kde_data = [
            np.random.normal(size=300, loc=10, scale=2),
            np.random.normal(size=150, loc=70, scale=4),
            np.random.normal(size=400, loc=20, scale=6),
            np.random.normal(size=200, loc=0, scale=3),
        ]
        for i, data in enumerate(kde_data):
            fig.add_trace(
                go.Histogram(
                    x=data,
                    histnorm="probability density",
                    opacity=0.8,
                    marker=dict(color=self.plotly[i]),
                    nbinsx=300,
                ),
                row=2,
                col=3,
            )

        # heat map
        heatmap_data = sns.load_dataset("glue").pivot(
            index="Model", columns="Task", values="Score"
        )
        fig.add_trace(
            go.Heatmap(
                z=heatmap_data.values,
                x=heatmap_data.columns,
                y=heatmap_data.index,
                colorscale=self.plotly,
                text=heatmap_data.values,
                texttemplate="%{text:.0f}",
                showscale=False,
            ),
            row=2,
            col=4,
        )

        # remove labels
        for i in range(1, 3):
            for j in range(1, 5):
                fig.update_xaxes(showticklabels=False, row=i, col=j)
                fig.update_yaxes(showticklabels=False, row=i, col=j)

        fig.update_layout(
            height=600,
            width=1000,
            title_text="Example Plots in Plotly",
            coloraxis_showscale=False,
            showlegend=False,
        )
        fig.show()


def convert_rgb_to_hex(rgb: np.array) -> str:
    """
    Converts a len 3 np.array of rgb values to a hex string. Note that
    function rounds any floats, so colors will not be absolutely identical
    between rgb & hex

    rgb values start off normalized between [0, 1]

    https://stackoverflow.com/questions/3380726/converting-an-rgb-color-tuple-to-a-hexidecimal-string

    Args:
        rgb (np.array): rgb values in len 3 np.array
    Returns:
        str: hex value
    """
    rgb = rgb * 255  # unnormalize
    rgb = rgb.astype(int)
    return "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])


def sort_rgb_by_hsv(rgb_palette: list) -> list:
    """
    Sorts a list of RGB colors based off HSV values

    Args:
        rgb_palette (list): List of np.arrays, each formatted as a len 3 np.array with rgb values

    Returns:
        list:sorted rgb palette
    """
    return sorted(rgb_palette, key=lambda x: rgb_to_hsv(x[0], x[1], x[2]))


def convert_rgb_palette_to_hex(rgb_palette: list) -> list:
    """
    Converts a list of np.arrays containing an entire palette of rgb values to a list of hex values

    Args:
        rgb_palette (list): List of np.arrays, each formatted as a len 3 np.array with rgb values

    Returns:
        list: list containing converted hex palette

    """
    hex_palette = []

    # first, sort rgb colors based off hsv
    rgb_palette_sorted = sorted(rgb_palette, key=lambda x: rgb_to_hsv(x[0], x[1], x[2]))

    # iterate through sorted palette - particularly important for plotly since
    # plotly doesn't apply any logic onto palettes
    for c in rgb_palette_sorted:
        hex_palette.append(convert_rgb_to_hex(c))

    return hex_palette


def compress_image_inplace(image_path: str) -> None:
    """
    Compresses image in place to reduce package storage.
    Used for packaging `palettecleanse`

    Args:
        image_path (str) -> path to image
    Returns:
        (None)
    """

    # tiffs and other rarer format compression not supported
    if image_path.endswith((".jpg", ".jpeg", ".png", ".bmp", ".gip")):
        img = Image.open(image_path)
        if img.size <= COMPRESSION_SIZE:
            print(
                f"{image_path} size ({img.size}) less than `compression_limit` ({COMPRESSION_SIZE}) - no compression applied"
            )
        else:
            initial_size = img.size
            img = img.resize(COMPRESSION_SIZE)
            # note this saves inplace & overwrites the original
            img.save(image_path, optimize=True, quality=85)
            print(
                f"{image_path} compressed - ({initial_size[0]-COMPRESSION_SIZE[0], initial_size[1]-COMPRESSION_SIZE[1]}) space saved."
            )
    else:
        print(f"{image_path} file extension not supported.")
