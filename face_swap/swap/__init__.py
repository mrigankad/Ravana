"""Face swap model module."""

from .base import FaceSwapper
from .simswap import SimSwapModel

try:
    from .inswapper import InSwapperModel
except ImportError:
    InSwapperModel = None

__all__ = [
    "FaceSwapper",
    "SimSwapModel",
    "InSwapperModel",
]
