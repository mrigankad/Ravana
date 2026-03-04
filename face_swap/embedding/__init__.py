"""Identity embedding module using ArcFace or similar models."""

from .arcface import ArcFaceEmbedder
from .base import IdentityEmbedder

__all__ = [
    "IdentityEmbedder",
    "ArcFaceEmbedder",
]
