# palettecleanser
`palettecleanser` is a python library for quick conversions of images to custom color maps

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
---
## Installation
(TODO)

## Quickstart
To convert an image to a custom color map, simply select an image and load it into `palettecleanser` as a `CMap` object, where desired attributes such as the number of colors (`n_colors`) can be specified as part of the class initialization. All available colormaps for the image can be displayed via. the `display_all_cmaps` method.
```py
from palettecleanser.cmap import CMap
vangogh = CMap('images/vangogh.jpg')
vangogh.display_all_cmaps()
```

![vangogh_image](images/vangogh_small.png?raw=true "Starry Night")

![vangogh_cmap](images/examples/vangogh_cmaps.png "Vangogh CMap Examples")

Specific colormaps ((sequential, qualitative, etc) are stored as attributes for this object and are compatible with `matplotlib`, `seaborn`, and `plotly`.
```py
# sequential colormap in matplotlib
plt.scatter(x, y, c=colors, cmap=vangogh.sequential)

# qualitative colormap in matplotlib
plt.bar(categories, values, color=vangogh.qualitative)

# qualitative colormap in seaborn
sns.swarmplot(df, x="x", y="y", hue="z", palette=vangogh.qualitative)

# generic colormap in plotly
px.scatter(df, x="x", y="y", color="z", color_continuous_scale=vangogh.plotly)
```
To get a sense for how well your colormap works, use the `display_example_plots` method
```py
# this creates some misc plots using your generated colormaps
vangogh.display_example_plots()
```
![vangogh_example](images/examples/vangogh_output.png)

`palettecleanser` also comes prepackaged with some custom colormaps:
```py
from palettecleanser import colormaps

colormaps.TwilightSunset.display_all_cmaps()
```
![TwilightSunset cmap](images/examples/sunset_cmaps.png)

See `usage.ipynb` for more examples.

## Examples
### Hokusai - The Great Wave off Kanagawa
![great_wave_example](images/examples/great_wave_output.png)

### Red Rose
![red_roses_example](images/examples/red_roses_output.png)

### Sunset
![sunset_example](images/examples/sunset_output.png)

More examples available in `usage.ipynb`.

## Contributing
Contributions at all levels are welcome! I'm happy to discuss with anyone the potential for contributions. Please see `CONTRIBUTING.md` for some general guidelines and message me with any questions!

## Meta
Jiaming Chen –  jiaming.justin.chen@gmail.com

Distributed under the GPL 3 (or any later version) license. See ``LICENSE`` for more information.

[https://github.com/sansona/palettecleanser](https://github.com/sansona/)