import warnings

warnings.simplefilter('once', DeprecationWarning)

warnings.warn(
    "`palettecleanse.custom` is deprecated and will be removed in a future release. Please use `palettecleanse.presets`.",
    DeprecationWarning,
    stacklevel=2
)

from palettecleanse.presets import *
