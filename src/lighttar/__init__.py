"""LightTAR: target-aware tabular features from frozen encoder layers, in closed form."""

from ._version import __version__
from .core import LightTARProjection
from .encoder import LayerEncoder, PrecomputedEncoder
from .image import ImageLayerEncoder, detect_image_columns
from .table import GatedLightTAR, LightTAR, detect_text_columns

__all__ = [
    "LightTAR",
    "GatedLightTAR",
    "LightTARProjection",
    "LayerEncoder",
    "PrecomputedEncoder",
    "ImageLayerEncoder",
    "detect_text_columns",
    "detect_image_columns",
    "__version__",
]
