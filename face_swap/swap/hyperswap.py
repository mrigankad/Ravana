"""
HyperSwap 256 ONNX face swapper (FaceFusion models-3.3.0).

Uses ArcFace-128 warp at 256×256 + L2-normalized source embedding
(no InSwapper ``emap``). Paste-back follows InsightFace's erode + Gaussian path.

Model license: ResearchRAIL (FaceFusion HyperSwap). Auto-downloaded on first use.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import cv2
import numpy as np

from ..core.model_manager import ensure_downloaded
from ..core.types import AlignedFace, Embedding, Frame, SwapResult
from .base import FaceSwapper

logger = logging.getLogger("face_swap.hyperswap")

HYPERSWAP_URLS = (
    "https://github.com/facefusion/facefusion-assets/releases/download/models-3.3.0/hyperswap_1a_256.onnx",
    "https://huggingface.co/facefusion/models-3.3.0/resolve/main/hyperswap_1a_256.onnx",
)
HYPERSWAP_DEFAULT = os.path.join("models", "hyperswap_1a_256.onnx")
HYPERSWAP_SHA256 = "c0e98a8a03a238f461ed3d2570e426b49f46745ee400854a60dceeb70c246add"
HYPERSWAP_SIZE = 256


def _paste_back_insightface(
    frame: np.ndarray,
    bgr_fake: np.ndarray,
    aimg: np.ndarray,
    M: np.ndarray,
) -> np.ndarray:
    """InsightFace-style paste-back, restricted to the (padded) face ROI.

    Same erode + Gaussian soft-mask semantics as the full-frame version, but the
    warp, erosion, blur and blend run only on the bounding box of the warped
    crop (padded by the erode/blur support), so cost scales with face area.
    Pixels outside the ROI are returned untouched.
    """
    IM = cv2.invertAffineTransform(M).astype(np.float64)
    h, w = frame.shape[:2]
    ch, cw = aimg.shape[:2]

    corners = np.array(
        [[0, 0, 1], [cw, 0, 1], [cw, ch, 1], [0, ch, 1]], dtype=np.float64
    )
    pts = corners @ IM.T
    bx0, by0 = np.floor(pts.min(axis=0)).astype(int)
    bx1, by1 = np.ceil(pts.max(axis=0)).astype(int)
    bx0, by0 = max(bx0, 0), max(by0, 0)
    bx1, by1 = min(bx1, w), min(by1, h)
    if bx1 - bx0 < 2 or by1 - by0 < 2:
        return frame

    mask_size = int(np.sqrt(max(1, (by1 - by0) * (bx1 - bx0))))
    k_erode = max(mask_size // 10, 10)
    k_blur = max(mask_size // 20, 5)
    pad = k_erode + 2 * k_blur + 4
    x0, y0 = max(bx0 - pad, 0), max(by0 - pad, 0)
    x1, y1 = min(bx1 + pad, w), min(by1 + pad, h)
    rw, rh = x1 - x0, y1 - y0

    IM_roi = IM.copy()
    IM_roi[0, 2] -= x0
    IM_roi[1, 2] -= y0
    bgr_w = cv2.warpAffine(bgr_fake, IM_roi, (rw, rh), borderValue=0.0)
    white = np.full((ch, cw), 255, dtype=np.float32)
    white = cv2.warpAffine(white, IM_roi, (rw, rh), borderValue=0.0)
    white[white > 20] = 255
    if not (white == 255).any():
        return frame

    img_mask = cv2.erode(white, np.ones((k_erode, k_erode), np.uint8), iterations=1)
    img_mask = cv2.GaussianBlur(img_mask, (2 * k_blur + 1, 2 * k_blur + 1), 0)
    img_mask = (img_mask / 255.0).astype(np.float32)[:, :, None]

    out = frame.copy()
    roi = frame[y0:y1, x0:x1].astype(np.float32)
    merged = img_mask * bgr_w.astype(np.float32) + (1.0 - img_mask) * roi
    out[y0:y1, x0:x1] = merged.astype(np.uint8)
    return out


class HyperSwapModel(FaceSwapper):
    """
    FaceFusion HyperSwap-1A 256 via ONNX Runtime (CPU / CUDA / DirectML).

    Inputs (typical): ``source`` (1, 512) L2-norm embedding, ``target`` (1, 3, 256, 256)
    RGB normalized with mean/std 0.5.
    """

    def __init__(
        self,
        device: str = "auto",
        model_path: Optional[str] = None,
        resolution: int = HYPERSWAP_SIZE,
        swap_weight: float = 0.5,
    ):
        super().__init__(device, resolution, use_enhancer=False)
        self.model_path = model_path or HYPERSWAP_DEFAULT
        self.swap_weight = float(np.clip(swap_weight, 0.0, 1.0))
        self._session = None
        self._source_name = "source"
        self._target_name = "target"
        self._output_name = None

    def load_model(self, model_path: Optional[str] = None) -> None:
        import onnxruntime as ort

        from face_swap.core.providers import resolve_ort_providers

        path = model_path or self.model_path
        path = self._ensure_model(path)
        providers = resolve_ort_providers(self.device)
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(
            path, sess_options=opts, providers=providers
        )

        inputs = self._session.get_inputs()
        by_name = {i.name: i for i in inputs}
        if "source" in by_name and "target" in by_name:
            self._source_name = "source"
            self._target_name = "target"
        else:
            # Fallback: embedding is rank-2, image is rank-4
            self._source_name = inputs[0].name
            self._target_name = inputs[-1].name
            for inp in inputs:
                dims = list(inp.shape or [])
                rank = len(dims)
                if rank == 2:
                    self._source_name = inp.name
                elif rank == 4:
                    self._target_name = inp.name

        outs = self._session.get_outputs()
        self._output_name = outs[0].name if outs else None
        self._model = self._session
        logger.info(
            "HyperSwap ONNX loaded (%s) providers=%s inputs=%s",
            path,
            self._session.get_providers(),
            [i.name for i in inputs],
        )

    def ensure_loaded(self) -> None:
        if self._session is None:
            self.load_model()

    def _ensure_model(self, path: str) -> str:
        return ensure_downloaded(
            path,
            urls=HYPERSWAP_URLS,
            min_bytes=50_000_000,
            sha256=HYPERSWAP_SHA256,
            label="HyperSwap 256 ONNX (~400 MB)",
        )

    @staticmethod
    def _source_latent(
        source_face: Any, target_face: Any, swap_weight: float
    ) -> np.ndarray:
        """
        Build (1, 512) embedding.

        FaceFusion HyperSwap uses L2-normalized ArcFace. InsightFace exposes
        that as ``normed_embedding`` (their ``embedding_norm`` is a scalar).
        """
        emb = getattr(source_face, "normed_embedding", None)
        if emb is None:
            raw = getattr(source_face, "embedding", None)
            if raw is None:
                raise ValueError("source_face missing embedding")
            raw = np.asarray(raw, dtype=np.float32).reshape(-1)
            n = float(np.linalg.norm(raw) + 1e-8)
            emb = raw / n
        source = np.asarray(emb, dtype=np.float32).reshape(1, -1)

        # FaceFusion weight: 0.5 → 0 mix (pure source); map [0,1] → [0.35,-0.35]
        w = float(np.interp(swap_weight, [0.0, 1.0], [0.35, -0.35]))
        if abs(w) > 1e-6:
            t_raw = getattr(target_face, "embedding", None)
            if t_raw is not None:
                t = np.asarray(t_raw, dtype=np.float32).reshape(1, -1)
                t = t / (np.linalg.norm(t) + 1e-8)
                source = source * (1.0 - w) + t * w
                source = source / (np.linalg.norm(source) + 1e-8)
        return source.astype(np.float32)

    @staticmethod
    def _prepare_target(crop_bgr: np.ndarray) -> np.ndarray:
        """BGR uint8 → NCHW float RGB, mean/std 0.5 (FaceFusion HyperSwap)."""
        rgb = crop_bgr[:, :, ::-1].astype(np.float32) / 255.0
        rgb = (rgb - 0.5) / 0.5
        return np.expand_dims(rgb.transpose(2, 0, 1), axis=0).astype(np.float32)

    @staticmethod
    def _normalize_output(pred: np.ndarray) -> np.ndarray:
        """Model output → BGR uint8."""
        x = np.asarray(pred)
        if x.ndim == 4:
            x = x[0]
        if x.shape[0] == 3:  # CHW
            x = x.transpose(1, 2, 0)
        x = x * 0.5 + 0.5
        x = np.clip(x, 0.0, 1.0)
        bgr = (x[:, :, ::-1] * 255.0).astype(np.uint8)
        return bgr

    def swap_face(
        self,
        frame: Frame,
        target_face: Any,
        source_face: Any,
        paste_back: bool = True,
    ) -> Frame:
        """Warp → HyperSwap → InsightFace paste-back."""
        self.ensure_loaded()
        from insightface.utils import face_align

        kps = getattr(target_face, "kps", None)
        if kps is None:
            raise ValueError("target_face missing kps")

        aimg, M = face_align.norm_crop2(frame, kps, HYPERSWAP_SIZE)
        target_blob = self._prepare_target(aimg)
        source_latent = self._source_latent(source_face, target_face, self.swap_weight)

        feeds = {
            self._source_name: source_latent,
            self._target_name: target_blob,
        }
        # If names were mis-detected, try both orderings
        try:
            pred = self._session.run(None, feeds)[0]
        except Exception:
            names = [i.name for i in self._session.get_inputs()]
            feeds = {names[0]: source_latent, names[1]: target_blob}
            try:
                pred = self._session.run(None, feeds)[0]
            except Exception:
                feeds = {names[0]: target_blob, names[1]: source_latent}
                pred = self._session.run(None, feeds)[0]

        bgr_fake = self._normalize_output(pred)
        if bgr_fake.shape[:2] != (HYPERSWAP_SIZE, HYPERSWAP_SIZE):
            bgr_fake = cv2.resize(bgr_fake, (HYPERSWAP_SIZE, HYPERSWAP_SIZE))

        if not paste_back:
            return bgr_fake
        return _paste_back_insightface(frame, bgr_fake, aimg, M)

    def swap(
        self,
        target_aligned: AlignedFace,
        source_embedding: Embedding,
    ) -> SwapResult:
        self.ensure_loaded()
        crop = target_aligned.image
        if crop.shape[:2] != (HYPERSWAP_SIZE, HYPERSWAP_SIZE):
            crop = cv2.resize(crop, (HYPERSWAP_SIZE, HYPERSWAP_SIZE))

        latent = source_embedding.vector.astype(np.float32).reshape(1, -1)
        n = float(np.linalg.norm(latent) + 1e-8)
        latent = latent / n
        if latent.shape[1] != 512:
            fixed = np.zeros((1, 512), dtype=np.float32)
            ncopy = min(512, latent.shape[1])
            fixed[0, :ncopy] = latent[0, :ncopy]
            latent = fixed

        pred = self._session.run(
            None,
            {
                self._source_name: latent,
                self._target_name: self._prepare_target(crop),
            },
        )[0]
        bgr = self._normalize_output(pred)
        return SwapResult(
            swapped_face=bgr,
            mask=self.get_mask(bgr),
            source_embedding=source_embedding,
            target_aligned=target_aligned,
            quality_score=0.9,
        )

    def get_mask(self, swapped_face: np.ndarray) -> np.ndarray:
        h, w = swapped_face.shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.ellipse(
            mask,
            (w // 2, h // 2),
            (int(w * 0.42), int(h * 0.48)),
            0,
            0,
            360,
            1.0,
            -1,
        )
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
        return mask
