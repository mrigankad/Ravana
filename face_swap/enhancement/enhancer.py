"""
GAN-based face enhancement and super-resolution.

As per PRD Section 5.6:
  - Provide hooks to integrate GAN-based refiners or enhancers
    (e.g., super-resolution, texture refinement) as optional post-processing.

This module provides:
  - A base `FaceEnhancer` interface for pluggable enhancers.
  - GFPGAN-style blind face restoration.
  - Real-ESRGAN-based super-resolution for upscaling swapped faces.
  - CodeFormer restoration for maximum quality.
"""

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from face_swap.core.model_manager import ensure_downloaded

logger = logging.getLogger("face_swap.enhancement")

# FaceFusion / community GFPGAN 1.4 ONNX (~340 MB)
GFPGAN_ONNX_URLS = (
    "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/gfpgan_1.4.onnx",
    "https://huggingface.co/facefusion/models-3.0.0/resolve/main/gfpgan_1.4.onnx",
)
GFPGAN_ONNX_DEFAULT = os.path.join("models", "gfpgan_1.4.onnx")
GFPGAN_ONNX_SHA256 = "accc4757b26bdb89b32b4d3500d4f79c9dff97c1dd7c7104bf9dcb95e3311385"

# FaceFusion CodeFormer ONNX (~377 MB)
CODEFORMER_ONNX_URLS = (
    "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/codeformer.onnx",
    "https://huggingface.co/facefusion/models-3.0.0/resolve/main/codeformer.onnx",
)
CODEFORMER_ONNX_DEFAULT = os.path.join("models", "codeformer.onnx")
CODEFORMER_ONNX_SHA256 = "21710e7ab61c82683576c428e9c1b6fe1ed419586b7b39e394c3449c294b550f"

# FaceFusion GPEN-BFR 512 ONNX (~284 MB)
GPEN_ONNX_URLS = (
    "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/gpen_bfr_512.onnx",
    "https://huggingface.co/facefusion/models-3.0.0/resolve/main/gpen_bfr_512.onnx",
)
GPEN_ONNX_DEFAULT = os.path.join("models", "gpen_bfr_512.onnx")
GPEN_ONNX_SHA256 = "d5f066b9068a8b74217f9712e28e875a6144629b108a6f7355acbdb3a2832c54"


@dataclass
class EnhancementConfig:
    """Configuration for face enhancement.

    Attributes:
        enabled:         Whether enhancement is active.
        method:          Enhancement method: ``gfpgan``, ``gpen``, ``codeformer``, ``realesrgan``, ``opencv``.
        upscale:         Upscale factor (1 = no upscale, 2 = 2×, 4 = 4×).
        quality:         Quality weight for CodeFormer (0 = quality, 1 = fidelity).
        bg_upsampler:    Whether to also upscale the background.
        device:          ``cuda``, ``cpu``, ``dml``, or ``auto``.
        blend_weight:    Mix of restored vs original crop (FaceFusion ~0.8–0.9).
        target_face_px:  Pixel-boost side length before restore (0 = disabled).
        model_path:      Optional ONNX / weight path override.
    """

    enabled: bool = False
    method: str = "gfpgan"
    upscale: int = 1
    quality: float = 0.5
    bg_upsampler: bool = False
    device: str = "cuda"
    blend_weight: float = 1.0
    target_face_px: int = 0
    model_path: Optional[str] = None


class FaceEnhancer(ABC):
    """Abstract base class for face enhancers (PRD §5.6)."""

    @abstractmethod
    def enhance(
        self,
        face: np.ndarray,
        upscale: int = 1,
    ) -> np.ndarray:
        """
        Enhance a face image.

        Args:
            face: BGR uint8 face crop (H, W, 3).
            upscale: Upscale factor.

        Returns:
            Enhanced face image.
        """
        ...

    @abstractmethod
    def load_model(self) -> None:
        """Load model weights."""
        ...


class GFPGANOnnxEnhancer(FaceEnhancer):
    """
    GFPGAN 1.4 via ONNX Runtime (no basicsr / gfpgan pip package).

    Works with CPU, CUDA, and DirectML (AMD). Expects a 512×512 RGB face
    crop normalized to [-1, 1] — same convention as FaceFusion / Tencent GFPGAN.
    """

    FACE_SIZE = 512

    def __init__(self, config: Optional[EnhancementConfig] = None):
        self.config = config or EnhancementConfig(method="gfpgan")
        self._session = None
        self._input_name = None

    def load_model(self) -> None:
        import onnxruntime as ort

        from face_swap.core.providers import resolve_ort_providers

        path = self.config.model_path or GFPGAN_ONNX_DEFAULT
        path = self._ensure_model(path)
        providers = resolve_ort_providers(self.config.device)
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(path, sess_options=opts, providers=providers)
        self._input_name = self._session.get_inputs()[0].name
        logger.info(
            "GFPGAN ONNX loaded (%s) providers=%s",
            path,
            self._session.get_providers(),
        )

    def _ensure_model(self, path: str) -> str:
        return ensure_downloaded(
            path,
            urls=GFPGAN_ONNX_URLS,
            min_bytes=1_000_000,
            sha256=GFPGAN_ONNX_SHA256,
            label="GFPGAN 1.4 ONNX (~340 MB)",
        )

    def enhance(self, face: np.ndarray, upscale: int = 1) -> np.ndarray:
        if self._session is None:
            self.load_model()
        if face is None or face.size == 0:
            return face

        h0, w0 = face.shape[:2]
        resized = cv2.resize(
            face, (self.FACE_SIZE, self.FACE_SIZE), interpolation=cv2.INTER_LANCZOS4
        )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = (rgb * 2.0 - 1.0).transpose(2, 0, 1)[np.newaxis, ...]

        out = self._session.run(None, {self._input_name: blob})[0]
        img = out[0].transpose(1, 2, 0)
        img = np.clip((img + 1.0) * 0.5, 0.0, 1.0)
        bgr = cv2.cvtColor((img * 255.0).round().astype(np.uint8), cv2.COLOR_RGB2BGR)

        if (h0, w0) != (self.FACE_SIZE, self.FACE_SIZE):
            bgr = cv2.resize(bgr, (w0, h0), interpolation=cv2.INTER_AREA)
        return bgr


class GFPGANEnhancer(FaceEnhancer):
    """
    GFPGAN blind face restoration enhancer (PyTorch / basicsr package).

    Prefer ``GFPGANOnnxEnhancer`` when the pip package is unavailable
    (e.g. Python 3.13).
    """

    def __init__(self, config: Optional[EnhancementConfig] = None):
        self.config = config or EnhancementConfig(method="gfpgan")
        self._restorer = None

    def load_model(self) -> None:
        try:
            from gfpgan import GFPGANer
        except ImportError:
            raise ImportError("GFPGAN is required. Install with: pip install gfpgan")

        self._restorer = GFPGANer(
            model_path="GFPGANv1.4.pth",
            upscale=self.config.upscale,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=self._get_bg_upsampler() if self.config.bg_upsampler else None,
        )
        logger.info("GFPGAN model loaded.")

    def enhance(self, face: np.ndarray, upscale: int = 1) -> np.ndarray:
        if self._restorer is None:
            self.load_model()

        _, _, output = self._restorer.enhance(
            face,
            has_aligned=True,
            only_center_face=True,
            paste_back=False,
        )
        return output

    def _get_bg_upsampler(self):
        """Create a RealESRGAN background upsampler."""
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer

            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=2,
            )
            return RealESRGANer(
                scale=2,
                model_path="RealESRGAN_x2plus.pth",
                model=model,
                tile=400,
                tile_pad=10,
                pre_pad=0,
                half=True,
            )
        except ImportError:
            logger.warning("RealESRGAN not available for background upsampling.")
            return None


class RealESRGANEnhancer(FaceEnhancer):
    """
    Real-ESRGAN super-resolution enhancer.

    Upscales swapped faces for higher output resolution,
    particularly useful when the swap model outputs at 128×128 or 256×256.
    """

    def __init__(self, config: Optional[EnhancementConfig] = None):
        self.config = config or EnhancementConfig(method="realesrgan")
        self._upsampler = None

    def load_model(self) -> None:
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
        except ImportError:
            raise ImportError(
                "RealESRGAN / basicsr required. "
                "Install with: pip install realesrgan basicsr"
            )

        scale = max(self.config.upscale, 2)
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=scale,
        )
        model_name = f"RealESRGAN_x{scale}plus.pth"
        half = self.config.device == "cuda"

        self._upsampler = RealESRGANer(
            scale=scale,
            model_path=model_name,
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=half,
        )
        logger.info("RealESRGAN model loaded (scale=%d).", scale)

    def enhance(self, face: np.ndarray, upscale: int = 2) -> np.ndarray:
        if self._upsampler is None:
            self.load_model()

        output, _ = self._upsampler.enhance(face, outscale=upscale)
        return output


class CodeFormerOnnxEnhancer(FaceEnhancer):
    """
    CodeFormer via ONNX Runtime (FaceFusion ``codeformer.onnx``).

    No PyTorch / basicsr required. Works with CPU, CUDA, and DirectML.
    Fidelity vs quality controlled by ``EnhancementConfig.quality``
    (0 = more restore / quality, 1 = more identity fidelity) — mapped to
    FaceFusion's ``weight`` input.
    """

    FACE_SIZE = 512

    def __init__(self, config: Optional[EnhancementConfig] = None):
        self.config = config or EnhancementConfig(method="codeformer")
        self._session = None
        self._input_name = "input"
        self._weight_name: Optional[str] = "weight"
        self._has_weight = True

    def load_model(self) -> None:
        import onnxruntime as ort

        from face_swap.core.providers import resolve_ort_providers

        path = self.config.model_path or CODEFORMER_ONNX_DEFAULT
        path = self._ensure_model(path)
        providers = resolve_ort_providers(self.config.device)
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(path, sess_options=opts, providers=providers)

        self._has_weight = False
        self._weight_name = None
        for inp in self._session.get_inputs():
            if inp.name == "input":
                self._input_name = "input"
            elif inp.name == "weight":
                self._weight_name = "weight"
                self._has_weight = True
            elif self._input_name is None or inp.name.lower() in ("input", "x"):
                self._input_name = inp.name

        # Ensure input name if model uses a different first tensor
        if not any(i.name == self._input_name for i in self._session.get_inputs()):
            self._input_name = self._session.get_inputs()[0].name

        logger.info(
            "CodeFormer ONNX loaded (%s) providers=%s weight_input=%s",
            path,
            self._session.get_providers(),
            self._has_weight,
        )

    def _ensure_model(self, path: str) -> str:
        return ensure_downloaded(
            path,
            urls=CODEFORMER_ONNX_URLS,
            min_bytes=1_000_000,
            sha256=CODEFORMER_ONNX_SHA256,
            label="CodeFormer ONNX (~377 MB)",
        )

    def enhance(self, face: np.ndarray, upscale: int = 1) -> np.ndarray:
        if self._session is None:
            self.load_model()
        if face is None or face.size == 0:
            return face

        h0, w0 = face.shape[:2]
        resized = cv2.resize(
            face, (self.FACE_SIZE, self.FACE_SIZE), interpolation=cv2.INTER_LANCZOS4
        )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = ((rgb - 0.5) / 0.5).transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)

        feeds = {self._input_name: blob}
        if self._has_weight and self._weight_name:
            # FaceFusion: weight as float64 length-1; config.quality is fidelity
            w = float(np.clip(self.config.quality, 0.0, 1.0))
            feeds[self._weight_name] = np.array([w], dtype=np.float64)

        out = self._session.run(None, feeds)[0]
        img = np.asarray(out)
        if img.ndim == 4:
            img = img[0]
        if img.shape[0] == 3:
            img = img.transpose(1, 2, 0)
        img = np.clip(img, -1.0, 1.0)
        img = (img + 1.0) * 0.5
        bgr = cv2.cvtColor((img * 255.0).round().astype(np.uint8), cv2.COLOR_RGB2BGR)

        if (h0, w0) != (self.FACE_SIZE, self.FACE_SIZE):
            bgr = cv2.resize(bgr, (w0, h0), interpolation=cv2.INTER_AREA)
        return bgr


# Back-compat alias — previous stub was PyTorch-only placeholder
CodeFormerEnhancer = CodeFormerOnnxEnhancer


class GPENOnnxEnhancer(FaceEnhancer):
    """
    GPEN-BFR 512 via ONNX Runtime (FaceFusion ``gpen_bfr_512.onnx``).

    Strong detail restore; often sharper skin texture than GFPGAN.
    Same [-1, 1] RGB NCHW convention as GFPGAN. Works on CPU / CUDA / DirectML.
    """

    FACE_SIZE = 512

    def __init__(self, config: Optional[EnhancementConfig] = None):
        self.config = config or EnhancementConfig(method="gpen")
        self._session = None
        self._input_name = None

    def load_model(self) -> None:
        import onnxruntime as ort

        from face_swap.core.providers import resolve_ort_providers

        path = self.config.model_path or GPEN_ONNX_DEFAULT
        path = self._ensure_model(path)
        providers = resolve_ort_providers(self.config.device)
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(path, sess_options=opts, providers=providers)
        self._input_name = self._session.get_inputs()[0].name
        logger.info(
            "GPEN ONNX loaded (%s) providers=%s",
            path,
            self._session.get_providers(),
        )

    def _ensure_model(self, path: str) -> str:
        return ensure_downloaded(
            path,
            urls=GPEN_ONNX_URLS,
            min_bytes=50_000_000,
            sha256=GPEN_ONNX_SHA256,
            label="GPEN-BFR 512 ONNX (~284 MB)",
        )

    def enhance(self, face: np.ndarray, upscale: int = 1) -> np.ndarray:
        if self._session is None:
            self.load_model()
        if face is None or face.size == 0:
            return face

        h0, w0 = face.shape[:2]
        resized = cv2.resize(
            face, (self.FACE_SIZE, self.FACE_SIZE), interpolation=cv2.INTER_LANCZOS4
        )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = (rgb * 2.0 - 1.0).transpose(2, 0, 1)[np.newaxis, ...]

        out = self._session.run(None, {self._input_name: blob})[0]
        img = np.asarray(out)
        if img.ndim == 4:
            img = img[0]
        if img.shape[0] == 3:
            img = img.transpose(1, 2, 0)
        img = np.clip((img + 1.0) * 0.5, 0.0, 1.0)
        bgr = cv2.cvtColor((img * 255.0).round().astype(np.uint8), cv2.COLOR_RGB2BGR)

        if (h0, w0) != (self.FACE_SIZE, self.FACE_SIZE):
            bgr = cv2.resize(bgr, (w0, h0), interpolation=cv2.INTER_AREA)
        return bgr


class OpenCVEnhancer(FaceEnhancer):
    """
    Dependency-free face sharpening / detail restore.

    Always available. Uses bilateral denoise + unsharp mask + mild CLAHE
    on the luminance channel — good post-InSwapper polish when GFPGAN
    is not installed.
    """

    def __init__(self, config: Optional[EnhancementConfig] = None):
        self.config = config or EnhancementConfig(method="opencv", enabled=True)
        self._loaded = False

    def load_model(self) -> None:
        self._loaded = True

    def enhance(self, face: np.ndarray, upscale: int = 1) -> np.ndarray:
        if face is None or face.size == 0:
            return face

        img = face
        if upscale > 1:
            img = cv2.resize(
                img,
                (img.shape[1] * upscale, img.shape[0] * upscale),
                interpolation=cv2.INTER_CUBIC,
            )

        # Mild path by default — aggressive CLAHE bleaches skin vs neck
        mild = getattr(self.config, "blend_weight", 1.0) < 0.95

        den = cv2.bilateralFilter(img, d=5, sigmaColor=35 if mild else 40, sigmaSpace=35)
        blur = cv2.GaussianBlur(den, (0, 0), sigmaX=1.0 if mild else 1.2)
        sharp = cv2.addWeighted(den, 1.25 if mild else 1.45, blur, -0.25 if mild else -0.45, 0)

        lab = cv2.cvtColor(sharp, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clip = 1.2 if mild else 1.8
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(4, 4))
        l = clahe.apply(l)
        out = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        return out


def _tile_starts(length: int, tile: int, overlap: int) -> list:
    """Start indices so tiles of ``tile`` cover ``length`` with ``overlap``."""
    if length <= tile:
        return [0]
    step = max(1, tile - overlap)
    starts = list(range(0, length - tile + 1, step))
    last = length - tile
    if starts[-1] != last:
        starts.append(last)
    return starts


def _tile_weight(h: int, w: int, overlap: int) -> np.ndarray:
    """Soft edge weights so overlapping tiles blend without seams."""
    wy = np.ones(h, dtype=np.float32)
    wx = np.ones(w, dtype=np.float32)
    ov = max(0, min(overlap, h // 2, w // 2))
    if ov > 0:
        ramp = np.linspace(0.0, 1.0, ov, dtype=np.float32)
        wy[:ov] = ramp
        wy[-ov:] = ramp[::-1]
        wx[:ov] = ramp
        wx[-ov:] = ramp[::-1]
    return np.outer(wy, wx)


def enhance_with_tiles(
    image: np.ndarray,
    enhancer: FaceEnhancer,
    tile_size: int = 512,
    overlap: int = 64,
) -> np.ndarray:
    """
    Run ``enhancer`` on ``image``, tiling when larger than ``tile_size``.

    FaceFusion-style pixel boost: upscale the face crop, restore in overlapping
    512×512 tiles, then the caller resizes back to the original crop.
    """
    if image is None or image.size == 0:
        return image

    h, w = image.shape[:2]
    native = int(getattr(enhancer, "FACE_SIZE", tile_size) or tile_size)
    tile = max(64, native)
    if h <= tile and w <= tile:
        return enhancer.enhance(image, upscale=1)

    ov = int(np.clip(overlap, 0, tile // 2))
    out = np.zeros((h, w, 3), dtype=np.float32)
    acc = np.zeros((h, w), dtype=np.float32)
    weight = _tile_weight(tile, tile, ov)

    for y0 in _tile_starts(h, tile, ov):
        for x0 in _tile_starts(w, tile, ov):
            y1, x1 = y0 + tile, x0 + tile
            patch = image[y0:y1, x0:x1]
            # Edge tiles may be smaller than tile — pad to square for ONNX models
            ph, pw = patch.shape[:2]
            if ph != tile or pw != tile:
                padded = np.zeros((tile, tile, 3), dtype=image.dtype)
                padded[:ph, :pw] = patch
                restored = enhancer.enhance(padded, upscale=1)[:ph, :pw]
                wmask = weight[:ph, :pw]
            else:
                restored = enhancer.enhance(patch, upscale=1)
                wmask = weight
            if restored.shape[:2] != (ph, pw):
                restored = cv2.resize(restored, (pw, ph), interpolation=cv2.INTER_LINEAR)
            out[y0:y1, x0:x1] += restored.astype(np.float32) * wmask[:, :, None]
            acc[y0:y1, x0:x1] += wmask

    acc = np.maximum(acc, 1e-6)
    return np.clip(out / acc[:, :, None], 0, 255).astype(np.uint8)


def enhance_face_region(
    frame: np.ndarray,
    bbox,
    enhancer: FaceEnhancer,
    feather: int = 15,
    blend_weight: float = 1.0,
    target_face_px: int = 0,
    region_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Run enhancer on a face bbox and feather it back into the frame.

    Args:
        frame: Full BGR frame.
        bbox: Face bounding box (x1, y1, x2, y2).
        enhancer: Face enhancer instance.
        feather: Gaussian blur kernel for the ellipse mask.
        blend_weight: 0 = keep swapped crop, 1 = full restore.
        target_face_px: If > 0 and crop is smaller, upscale to this side before
            enhance (pixel boost), then resize back. Values above the enhancer
            native size (usually 512) use overlapping tiles.
        region_mask: Optional float mask (H, W) in [0, 1] for the padded crop
            region; multiplied into the ellipse (occlusion / hull gate).
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = frame.shape[:2]
    pad = int(0.08 * max(x2 - x1, y2 - y1))
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
    if x2 <= x1 + 8 or y2 <= y1 + 8:
        return frame

    crop = frame[y1:y2, x1:x2]
    rh, rw = crop.shape[:2]
    work = crop
    if target_face_px and target_face_px > 0:
        side = max(rh, rw)
        if side < target_face_px:
            scale = target_face_px / float(side)
            work = cv2.resize(
                crop,
                (max(1, int(round(rw * scale))), max(1, int(round(rh * scale)))),
                interpolation=cv2.INTER_LANCZOS4,
            )

    enhanced = enhance_with_tiles(work, enhancer)
    if enhanced.shape[:2] != (rh, rw):
        enhanced = cv2.resize(enhanced, (rw, rh), interpolation=cv2.INTER_AREA)

    w_blend = float(np.clip(blend_weight, 0.0, 1.0))
    restored = (
        w_blend * enhanced.astype(np.float32) + (1.0 - w_blend) * crop.astype(np.float32)
    )

    mask = np.zeros((rh, rw), dtype=np.float32)
    cv2.ellipse(
        mask,
        (rw // 2, int(rh * 0.45)),
        (max(1, rw // 2 - 2), max(1, int(rh * 0.48))),
        0,
        0,
        360,
        1.0,
        -1,
    )
    k = feather if feather % 2 == 1 else feather + 1
    mask = cv2.GaussianBlur(mask, (k, k), 0)

    if region_mask is not None:
        rm = np.asarray(region_mask, dtype=np.float32)
        if rm.shape[:2] != (rh, rw):
            rm = cv2.resize(rm, (rw, rh), interpolation=cv2.INTER_LINEAR)
        mask = mask * np.clip(rm, 0.0, 1.0)

    mask3 = mask[:, :, None]
    blended = (mask3 * restored + (1.0 - mask3) * crop.astype(np.float32)).astype(
        np.uint8
    )

    out = frame.copy()
    out[y1:y2, x1:x2] = blended
    return out


# ── Factory function ────────────────────────────────────────────────────


def create_enhancer(config: EnhancementConfig) -> FaceEnhancer:
    """
    Factory: create an enhancer based on configuration.

    For ``gfpgan`` / ``gpen`` / ``codeformer``, prefers the ONNX Runtime path
    (no basicsr), then OpenCV on failure.
    """
    method = config.method.lower()
    try:
        if method == "gfpgan":
            # ONNX path works on Py 3.13 + DirectML without basicsr
            return GFPGANOnnxEnhancer(config)
        if method == "gfpgan_torch":
            return GFPGANEnhancer(config)
        if method in ("gpen", "gpen_bfr", "gpen_bfr_512"):
            return GPENOnnxEnhancer(config)
        if method == "realesrgan":
            return RealESRGANEnhancer(config)
        if method == "codeformer":
            return CodeFormerOnnxEnhancer(config)
        if method == "opencv":
            return OpenCVEnhancer(config)
        raise ValueError(f"Unknown enhancement method: {method}")
    except ImportError as e:
        logger.warning("%s unavailable (%s); using OpenCV enhancer.", method, e)
        return OpenCVEnhancer(config)
