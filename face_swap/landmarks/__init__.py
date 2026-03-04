"""Landmark detection module supporting multiple backends."""

from .base import LandmarkDetector
from .mediapipe_lm import MediaPipeLandmarkDetector

try:
    from .insightface_lm import InsightFaceLandmarkDetector
except ImportError:
    InsightFaceLandmarkDetector = None

__all__ = [
    "LandmarkDetector",
    "MediaPipeLandmarkDetector",
    "InsightFaceLandmarkDetector",
]
