"""Face swap model module."""

from .base import FaceSwapper
from .simswap import SimSwapModel

try:
    from .inswapper import InSwapperModel
except ImportError:
    InSwapperModel = None

try:
    from .hyperswap import HyperSwapModel
except ImportError:
    HyperSwapModel = None

__all__ = [
    "FaceSwapper",
    "SimSwapModel",
    "InSwapperModel",
    "HyperSwapModel",
]
