"""Landmark detection module supporting multiple backends."""

from .base import LandmarkDetector
from .insightface_lm import InsightFaceLandmarkDetector
from .mediapipe_lm import MediaPipeLandmarkDetector

__all__ = [
    "LandmarkDetector",
    "MediaPipeLandmarkDetector",
    "InsightFaceLandmarkDetector",
]
