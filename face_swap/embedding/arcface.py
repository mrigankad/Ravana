"""
ArcFace embedding extractor using InsightFace.

As per PRD Section 5.5, this uses ArcFace-style encoder for identity embeddings.
"""

from typing import Optional

import cv2
import numpy as np

from ..core.types import AlignedFace, Embedding
from .base import IdentityEmbedder


class ArcFaceEmbedder(IdentityEmbedder):
    """
    ArcFace identity embedding extractor.

    Extracts 512-dimensional identity embeddings that are
    relatively invariant to expression and lighting.
    """

    def __init__(
        self,
        device: str = "cuda",
        model_name: str = "buffalo_l",  # InsightFace model name
        embedding_dim: int = 512,
    ):
        """
        Initialize ArcFace embedder.

        Args:
            device: Device to run inference on
            model_name: InsightFace model name
            embedding_dim: Output embedding dimension (512 for ArcFace)
        """
        super().__init__(device, embedding_dim)
        self.model_name = model_name
        self._face_analysis = None

    def load_model(self) -> None:
        """Load the ArcFace model using InsightFace."""
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
            ctx_id=0 if self.device == "cuda" else -1,
            det_size=(640, 640),
        )

    def _recognition(self):
        return self._face_analysis.models["recognition"]

    def extract(self, aligned_face: AlignedFace) -> Embedding:
        """
        Extract identity embedding from an aligned face.

        Args:
            aligned_face: Cropped and aligned face image

        Returns:
            Embedding vector
        """
        if self._face_analysis is None:
            self.load_model()

        face_img = aligned_face.image
        input_size = int(self._recognition().input_size[0])

        # ArcFace expects input_size x input_size (usually 112)
        if face_img.shape[:2] != (input_size, input_size):
            face_img = cv2.resize(face_img, (input_size, input_size))

        # get_feat expects BGR and handles RGB conversion via swapRB=True
        embedding = self._recognition().get_feat(face_img).flatten()

        return Embedding(
            vector=embedding, model_name=f"arcface_{self.model_name}", normalized=True
        )

    def extract_from_image(
        self, image: np.ndarray, bbox: Optional[tuple] = None
    ) -> Embedding:
        """
        Extract embedding directly from an image.

        Args:
            image: Input image (BGR format)
            bbox: Optional bounding box (x1, y1, x2, y2)

        Returns:
            Embedding vector
        """
        if self._face_analysis is None:
            self.load_model()

        faces = self._face_analysis.get(image)
        if not faces:
            raise ValueError("No face detected for embedding extraction")

        face = faces[0]
        if bbox is not None:
            x1, y1, x2, y2 = map(float, bbox)
            target = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)
            best = face
            best_dist = float("inf")
            for candidate in faces:
                fb = candidate.bbox.astype(np.float32)
                center = np.array([(fb[0] + fb[2]) / 2.0, (fb[1] + fb[3]) / 2.0])
                dist = float(np.linalg.norm(center - target))
                if dist < best_dist:
                    best_dist = dist
                    best = candidate
            face = best

        embedding = np.asarray(face.embedding, dtype=np.float32).flatten()
        return Embedding(
            vector=embedding, model_name=f"arcface_{self.model_name}", normalized=True
        )
