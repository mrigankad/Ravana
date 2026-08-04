"""
InsightFace landmark detector using the buffalo_l model pack.

Preferred default on modern Python where MediaPipe no longer ships
the legacy ``mp.solutions`` Face Mesh API.
"""

from typing import List

import cv2
import numpy as np

from ..core.types import FaceBBox, Frame, Landmarks
from .base import LandmarkDetector


class InsightFaceLandmarkDetector(LandmarkDetector):
    """
    Landmark detector backed by InsightFace FaceAnalysis.

    Returns 5-point keypoints (eyes, nose, mouth corners) from ``face.kps``,
    which is sufficient for affine face alignment used by the swap pipeline.
    """

    def __init__(
        self,
        device: str = "cuda",
        model_name: str = "buffalo_l",
        det_size: tuple = (640, 640),
    ):
        super().__init__(device)
        self.model_name = model_name
        self.det_size = det_size
        self._face_analysis = None

    def load_model(self) -> None:
        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            raise ImportError(
                "insightface is required. Install with: pip install insightface"
            ) from exc

        providers = (
            ["CUDAExecutionProvider"]
            if self.device == "cuda"
            else ["CPUExecutionProvider"]
        )
        self._face_analysis = FaceAnalysis(
            name=self.model_name, root="./models", providers=providers
        )
        self._face_analysis.prepare(
            ctx_id=0 if self.device == "cuda" else -1, det_size=self.det_size
        )
        self._model = self._face_analysis

    def detect(self, frame: Frame, bbox: FaceBBox) -> Landmarks:
        if self._face_analysis is None:
            self.load_model()

        face = self._match_face(frame, bbox)
        if face is None:
            return Landmarks(points=np.zeros((5, 2), dtype=np.float32))

        if hasattr(face, "kps") and face.kps is not None:
            return Landmarks(
                points=np.asarray(face.kps, dtype=np.float32),
                confidence=bbox.confidence,
            )

        if hasattr(face, "landmark_2d_106") and face.landmark_2d_106 is not None:
            return Landmarks(
                points=np.asarray(face.landmark_2d_106, dtype=np.float32),
                confidence=bbox.confidence,
            )

        return Landmarks(points=np.zeros((5, 2), dtype=np.float32))

    def detect_multi(self, frame: Frame, bboxes: List[FaceBBox]) -> List[Landmarks]:
        return [self.detect(frame, bbox) for bbox in bboxes]

    def _match_face(self, frame: Frame, bbox: FaceBBox):
        frame_rgb = (
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if frame.shape[2] == 3 else frame
        )
        faces = self._face_analysis.get(frame_rgb)
        if not faces:
            return None

        target = np.array([bbox.center.x, bbox.center.y], dtype=np.float32)
        best = None
        best_dist = float("inf")
        for face in faces:
            fb = face.bbox.astype(np.float32)
            center = np.array([(fb[0] + fb[2]) / 2.0, (fb[1] + fb[3]) / 2.0])
            dist = float(np.linalg.norm(center - target))
            if dist < best_dist:
                best_dist = dist
                best = face
        return best
