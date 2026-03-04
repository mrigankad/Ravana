"""Face detection module supporting multiple backends."""

from .async_detector import AsyncFaceDetector
from .base import FaceDetector
from .retinaface import RetinaFaceDetector

try:
    from .insightface_detector import InsightFaceDetector
except ImportError:
    InsightFaceDetector = None

__all__ = [
    "FaceDetector",
    "RetinaFaceDetector",
    "InsightFaceDetector",
    "AsyncFaceDetector",
]
