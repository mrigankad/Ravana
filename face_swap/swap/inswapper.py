"""
InSwapper (InsightFace) face swap model.

Implements the official InsightFace INSwapper path:
  1. ArcFace ``norm_crop2`` alignment to 128x128 using 5-point kps
  2. Latent = L2(normed_embedding @ emap)  (required — raw ArcFace is wrong)
  3. Paste-back via inverse affine + eroded/blurred mask

Reference:
  https://github.com/deepinsight/insightface/blob/master/python-package/insightface/model_zoo/inswapper.py
"""

from __future__ import annotations

import os
from typing import Any, Optional

import cv2
import numpy as np

from ..core.types import AlignedFace, Embedding, Frame, SwapResult
from .base import FaceSwapper


class InSwapperModel(FaceSwapper):
    """
    Production InSwapper wrapper around InsightFace's ONNX model.

    Prefer ``swap_face(img, target_face, source_face)`` which uses the
    official paste-back. The modular ``swap(AlignedFace, Embedding)`` path
    is kept for plugin compatibility but is lower quality.
    """

    DEFAULT_MODEL_URL = (
        "https://github.com/deepinsight/insightface/releases/"
        "download/v0.7/inswapper_128.onnx"
    )

    def __init__(
        self,
        device: str = "cuda",
        model_path: str = "./models/inswapper_128.onnx",
        resolution: int = 128,
    ):
        super().__init__(device, resolution, use_enhancer=False)
        self.model_path = model_path
        self._swapper = None  # insightface INSwapper instance

    def load_model(self, model_path: Optional[str] = None) -> None:
        path = model_path or self.model_path
        if not os.path.exists(path):
            self._download_model(path)

        try:
            import insightface
            from insightface.model_zoo.inswapper import INSwapper
        except ImportError as exc:
            raise ImportError(
                "insightface is required. Install with: pip install insightface"
            ) from exc

        providers = None
        try:
            from face_swap.core.providers import resolve_ort_providers

            providers = resolve_ort_providers(self.device)
        except Exception:
            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if self.device == "cuda"
                else ["CPUExecutionProvider"]
            )

        try:
            import onnxruntime as ort

            from face_swap.core.fast_video import configure_ort_session_options

            opts = configure_ort_session_options(
                intra_threads=4 if self.device == "cpu" else 0,
                inter_threads=1,
            )
            session = ort.InferenceSession(path, sess_options=opts, providers=providers)
            self._swapper = INSwapper(model_file=path, session=session)
        except Exception:
            # Fallback: let insightface create the session
            self._swapper = insightface.model_zoo.get_model(path)

        self._model = self._swapper

    def _download_model(self, path: str) -> None:
        from face_swap.core.model_manager import ensure_downloaded

        ensure_downloaded(
            path,
            urls=[self.DEFAULT_MODEL_URL],
            min_bytes=50_000_000,
            label="InSwapper 128 ONNX (~550 MB)",
        )

    def ensure_loaded(self) -> None:
        if self._swapper is None:
            self.load_model()

    def swap_face(
        self,
        frame: Frame,
        target_face: Any,
        source_face: Any,
        paste_back: bool = True,
    ) -> Frame:
        """
        Official InsightFace swap with correct alignment + emap + paste-back.

        Args:
            frame: Full BGR target frame
            target_face: InsightFace Face object (needs ``kps``)
            source_face: InsightFace Face object (needs ``normed_embedding``)
            paste_back: If True, return full composited frame

        Returns:
            Composited BGR frame (or 128x128 crop if paste_back=False)
        """
        self.ensure_loaded()
        return self._swapper.get(frame, target_face, source_face, paste_back=paste_back)

    def swap(
        self,
        target_aligned: AlignedFace,
        source_embedding: Embedding,
    ) -> SwapResult:
        """
        Modular swap API (lower quality than ``swap_face``).

        Uses emap transform and 128-aligned crop when possible. Paste-back
        still depends on ``transformation_matrix`` being the ArcFace M matrix.
        """
        self.ensure_loaded()

        target_img = target_aligned.image
        if target_img.shape[:2] != (128, 128):
            target_img = cv2.resize(target_img, (128, 128))

        # Official blob: BGR -> RGB via swapRB, scale 1/255
        blob = cv2.dnn.blobFromImage(
            target_img,
            1.0 / 255.0,
            (128, 128),
            (0.0, 0.0, 0.0),
            swapRB=True,
        )

        latent = source_embedding.vector.astype(np.float32).reshape(1, -1)
        if latent.shape[1] != 512:
            fixed = np.zeros((1, 512), dtype=np.float32)
            n = min(512, latent.shape[1])
            fixed[0, :n] = latent[0, :n]
            latent = fixed

        # Critical: apply model emap then L2-normalize
        emap = getattr(self._swapper, "emap", None)
        if emap is not None:
            latent = np.dot(latent, emap)
        norm = np.linalg.norm(latent)
        if norm > 0:
            latent = latent / norm

        pred = self._swapper.session.run(
            self._swapper.output_names,
            {
                self._swapper.input_names[0]: blob,
                self._swapper.input_names[1]: latent,
            },
        )[0]

        img_fake = pred.transpose((0, 2, 3, 1))[0]
        bgr_fake = np.clip(255 * img_fake, 0, 255).astype(np.uint8)[:, :, ::-1]

        mask = self.get_mask(bgr_fake)

        return SwapResult(
            swapped_face=bgr_fake,
            mask=mask,
            source_embedding=source_embedding,
            target_aligned=target_aligned,
            quality_score=0.9,
        )

    def paste_back(
        self,
        frame: Frame,
        bgr_fake: np.ndarray,
        aimg: np.ndarray,
        M: np.ndarray,
    ) -> Frame:
        """Official InsightFace paste-back (erode + Gaussian soft mask)."""
        fake_diff = np.abs(bgr_fake.astype(np.float32) - aimg.astype(np.float32)).mean(
            axis=2
        )
        fake_diff[:2, :] = 0
        fake_diff[-2:, :] = 0
        fake_diff[:, :2] = 0
        fake_diff[:, -2:] = 0

        IM = cv2.invertAffineTransform(M)
        h, w = frame.shape[:2]
        img_white = np.full((aimg.shape[0], aimg.shape[1]), 255, dtype=np.float32)
        warped = cv2.warpAffine(bgr_fake, IM, (w, h), borderValue=0.0)
        img_white = cv2.warpAffine(img_white, IM, (w, h), borderValue=0.0)

        img_white[img_white > 20] = 255
        img_mask = img_white
        mask_h_inds, mask_w_inds = np.where(img_mask == 255)
        if len(mask_h_inds) == 0:
            return frame

        mask_h = int(np.max(mask_h_inds) - np.min(mask_h_inds))
        mask_w = int(np.max(mask_w_inds) - np.min(mask_w_inds))
        mask_size = int(np.sqrt(mask_h * mask_w))
        k = max(mask_size // 10, 10)
        img_mask = cv2.erode(img_mask, np.ones((k, k), np.uint8), iterations=1)
        k = max(mask_size // 20, 5)
        blur_size = tuple(2 * i + 1 for i in (k, k))
        img_mask = cv2.GaussianBlur(img_mask, blur_size, 0)
        img_mask = (img_mask / 255.0).astype(np.float32)
        img_mask = img_mask[:, :, None]

        merged = img_mask * warped.astype(np.float32) + (1.0 - img_mask) * frame.astype(
            np.float32
        )
        return merged.astype(np.uint8)

    def get_mask(self, swapped_face: np.ndarray) -> np.ndarray:
        h, w = swapped_face.shape[:2]
        mask = np.full((h, w), 255, dtype=np.float32)
        k = max(h // 10, 10)
        mask = cv2.erode(mask, np.ones((k, k), np.uint8), iterations=1)
        k = max(h // 20, 5)
        blur = tuple(2 * i + 1 for i in (k, k))
        mask = cv2.GaussianBlur(mask, blur, 0)
        return (mask / 255.0).astype(np.float32)
