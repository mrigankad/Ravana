"""Identity embedding module using ArcFace or similar models."""

from .base import IdentityEmbedder
from .arcface import ArcFaceEmbedder

__all__ = [
    "IdentityEmbedder",
    "ArcFaceEmbedder",
]
