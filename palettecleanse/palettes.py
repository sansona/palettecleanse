import warnings

warnings.simplefilter("once", DeprecationWarning)

warnings.warn(
    "`palettecleanse.palettes` is deprecated and will be removed in a future release. Please use `palettecleanse.palette`.",
    DeprecationWarning,
    stacklevel=2,
)

from palettecleanse.palette import *
