"""Face detection module supporting multiple backends."""

from .base import FaceDetector
from .retinaface import RetinaFaceDetector
from .async_detector import AsyncFaceDetector

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
