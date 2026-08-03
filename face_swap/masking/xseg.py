"""
XSeg ONNX face occluder (FaceFusion / DeepFaceLab style).

Predicts a soft face-skin mask so hands, hair, glasses, and other occluders
can be kept from the original frame after an InSwapper paste.
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from face_swap.core.model_manager import ensure_downloaded

logger = logging.getLogger("face_swap.masking.xseg")

XSEG_ONNX_URLS = (
    "https://github.com/facefusion/facefusion-assets/releases/download/models-3.1.0/xseg_1.onnx",
    "https://huggingface.co/facefusion/models-3.1.0/resolve/main/xseg_1.onnx",
)
XSEG_ONNX_DEFAULT = os.path.join("models", "xseg_1.onnx")
XSEG_ONNX_SHA256 = "c4d1498b8a03b5fe2a3a5d2ef2a0402ab03bd51edaf5b2d8d5fb764702a97dd3"


def _pad_bbox(
    bbox: Sequence[float],
    image_shape: Tuple[int, ...],
    pad_frac: float = 0.08,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = image_shape[:2]
    pad = int(pad_frac * max(x2 - x1, y2 - y1))
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
    return x1, y1, x2, y2


def apply_occlusion_blend(
    original: np.ndarray,
    swapped: np.ndarray,
    bbox: Sequence[float],
    soft_mask: np.ndarray,
    pad_frac: float = 0.08,
) -> np.ndarray:
    """
    Blend swapped face with original using an occlusion soft-mask.

    ``soft_mask`` is float HxW in [0, 1] where **1 = keep swapped**,
    **0 = keep original** (occlusion). Mask size should match the padded
    bbox crop; it will be resized if needed.
    """
    x1, y1, x2, y2 = _pad_bbox(bbox, swapped.shape, pad_frac=pad_frac)
    if x2 <= x1 + 4 or y2 <= y1 + 4:
        return swapped

    region_s = swapped[y1:y2, x1:x2]
    region_o = original[y1:y2, x1:x2]
    rh, rw = region_s.shape[:2]

    m = np.asarray(soft_mask, dtype=np.float32)
    if m.ndim == 3:
        m = m[:, :, 0]
    if m.shape[:2] != (rh, rw):
        m = cv2.resize(m, (rw, rh), interpolation=cv2.INTER_LINEAR)
    m = np.clip(m, 0.0, 1.0)
    # Soften edges slightly
    k = max(3, (min(rh, rw) // 20) | 1)
    m = cv2.GaussianBlur(m, (k, k), 0)

    mask3 = m[:, :, None]
    blended = (
        mask3 * region_s.astype(np.float32)
        + (1.0 - mask3) * region_o.astype(np.float32)
    ).astype(np.uint8)

    out = swapped.copy()
    out[y1:y2, x1:x2] = blended
    return out


class XSegOccluder:
    """
    FaceFusion ``xseg_1`` occluder via ONNX Runtime.

    Input: BGR face crop (bbox). Output: soft mask HxW in [0, 1]
    (1 = face skin / keep swap, 0 = occluder / keep original).
    """

    def __init__(
        self,
        device: str = "auto",
        model_path: Optional[str] = None,
        blur_sigma: float = 1.0,
    ):
        self.device = device
        self.model_path = model_path or XSEG_ONNX_DEFAULT
        self.blur_sigma = blur_sigma
        self._session = None
        self._input_name = None
        self._input_size = (256, 256)  # (W, H)
        self._nhwc = True

    def load_model(self) -> None:
        import onnxruntime as ort

        from face_swap.core.providers import resolve_ort_providers

        path = self._ensure_model(self.model_path)
        providers = resolve_ort_providers(self.device)
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(path, sess_options=opts, providers=providers)
        inp = self._session.get_inputs()[0]
        self._input_name = inp.name
        shape = inp.shape
        # Detect NHWC vs NCHW
        if len(shape) == 4:
            # Dynamic dims may be str/'?'
            d1, d2, d3 = shape[1], shape[2], shape[3]
            if d3 == 3 or (isinstance(d3, int) and d3 == 3):
                self._nhwc = True
                h = d1 if isinstance(d1, int) else 256
                w = d2 if isinstance(d2, int) else 256
                self._input_size = (w, h)
            elif d1 == 3 or (isinstance(d1, int) and d1 == 3):
                self._nhwc = False
                h = d2 if isinstance(d2, int) else 256
                w = d3 if isinstance(d3, int) else 256
                self._input_size = (w, h)
            else:
                self._nhwc = True
                self._input_size = (256, 256)
        logger.info(
            "XSeg ONNX loaded (%s) size=%s nhwc=%s providers=%s",
            path,
            self._input_size,
            self._nhwc,
            self._session.get_providers(),
        )

    def _ensure_model(self, path: str) -> str:
        return ensure_downloaded(
            path,
            urls=XSEG_ONNX_URLS,
            min_bytes=1_000_000,
            sha256=XSEG_ONNX_SHA256,
            label="XSeg occluder (~70 MB)",
        )

    def face_mask(self, crop_bgr: np.ndarray) -> np.ndarray:
        """
        Run XSeg on a BGR face crop.

        Returns float mask same HxW as input, values in [0, 1].
        On failure / unloaded session returns ones (no occlusion gating).
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return np.ones((1, 1), dtype=np.float32)

        h0, w0 = crop_bgr.shape[:2]
        try:
            if self._session is None:
                self.load_model()
        except Exception as e:
            logger.warning("XSeg unavailable (%s); using full-face mask.", e)
            return np.ones((h0, w0), dtype=np.float32)

        try:
            resized = cv2.resize(
                crop_bgr, self._input_size, interpolation=cv2.INTER_LINEAR
            )
            img = resized.astype(np.float32) / 255.0
            if self._nhwc:
                blob = img[np.newaxis, ...]
            else:
                # BGR→RGB for NCHW models that expect RGB
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                blob = rgb.transpose(2, 0, 1)[np.newaxis, ...]

            out = self._session.run(None, {self._input_name: blob})[0]
            mask = np.asarray(out).squeeze().astype(np.float32)
            if mask.ndim == 3:
                # Take first channel / mean if multi-channel
                mask = mask[0] if mask.shape[0] <= 4 else mask.mean(axis=-1)
            mask = np.clip(mask, 0.0, 1.0)
            mask = cv2.resize(mask, (w0, h0), interpolation=cv2.INTER_LINEAR)
            if self.blur_sigma and self.blur_sigma > 0:
                mask = cv2.GaussianBlur(mask, (0, 0), self.blur_sigma)
            return mask
        except Exception as e:
            logger.warning("XSeg inference failed (%s); using full-face mask.", e)
            return np.ones((h0, w0), dtype=np.float32)

    def mask_for_bbox(
        self,
        frame: np.ndarray,
        bbox: Sequence[float],
        pad_frac: float = 0.08,
    ) -> np.ndarray:
        """Crop padded bbox from frame and return XSeg soft mask for that crop."""
        x1, y1, x2, y2 = _pad_bbox(bbox, frame.shape, pad_frac=pad_frac)
        if x2 <= x1 + 4 or y2 <= y1 + 4:
            return np.ones((1, 1), dtype=np.float32)
        crop = frame[y1:y2, x1:x2]
        return self.face_mask(crop)
