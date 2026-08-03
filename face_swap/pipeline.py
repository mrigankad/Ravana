"""
Main face swap pipeline orchestrator.

For InSwapper (default), uses the official InsightFace path:
  FaceAnalysis (buffalo_l) → INSwapper.get(..., paste_back=True)

This avoids the broken custom align(256)→resize(128)→inverse-warp path that
produced mis-scaled ghost faces. Modular stages remain available for plugins
and non-InSwapper models.
"""

import logging
from dataclasses import dataclass
from typing import Any, List, Optional, Union

import cv2
import numpy as np

from .alignment import FaceAligner
from .blending import FaceBlender
from .core.adaptive import (
    AdaptiveConfig,
    adaptive_mask_kernels,
    assess_face,
    detect_faces_adaptive,
    detect_lower_face_hair,
    forehead_taper_mask,
    identity_preserved,
    landmark_hull_mask,
    lower_face_preserve_mask,
    match_chrominance_to_target,
    match_face_to_target_skin,
    match_grain_to_target,
    match_lighting_to_target,
    neck_ring_color_reference,
    reinhard_color_match,
    select_faces,
    upscale_frame_for_tiny_faces,
    upscale_if_small_face,
)
from .core.fast_video import (
    FastVideoConfig,
    maybe_downscale_for_detect,
    scale_faces_from_detect_space,
)
from .core.profiler import PipelineProfiler
from .core.providers import insightface_ctx_id, resolve_ort_providers
from .core.quality import QualityValidator
from .core.types import (
    Embedding,
    Frame,
    PipelineResult,
)
from .detection import AsyncFaceDetector, FaceDetector, RetinaFaceDetector
from .embedding import ArcFaceEmbedder, IdentityEmbedder
from .enhancement import EnhancementConfig, create_enhancer, enhance_face_region
from .landmarks import InsightFaceLandmarkDetector, LandmarkDetector
from .masking import apply_occlusion_blend
from .swap import FaceSwapper, InSwapperModel
from .swap.hyperswap import HyperSwapModel
from .temporal import OpticalFlowConfig, OpticalFlowSmoother, TemporalSmoother
from .temporal.smoother import ema_smooth_insight_faces
from .watermark import InvisibleWatermarker, WatermarkConfig

logger = logging.getLogger("face_swap.pipeline")


@dataclass
class PipelineConfig:
    """Configuration for the face swap pipeline."""

    device: str = "auto"
    detection_model: str = "retinaface"
    det_confidence_threshold: float = 0.5
    crop_size: int = 128
    swap_model: str = "inswapper"
    swap_model_path: Optional[str] = None
    blend_mode: str = "alpha"
    color_correction: bool = True
    enable_temporal: bool = True
    temporal_smooth_factor: float = 0.7
    # Video: re-detect every N frames (1 = every frame). EMA bridges gaps.
    video_detect_every_n: int = 1
    # EMA weight on *current* bbox/kps when enable_temporal (0.35 = mild)
    video_face_ema_alpha: float = 0.35
    # Optical-flow blend of previous output into current (0 = off)
    video_flow_blend: float = 0.28
    # Target face selection: all | largest | first | index | pose
    face_select: str = "all"
    face_index: int = 0
    max_faces: int = 0  # 0 = unlimited (fast_video still caps)
    batch_size: int = 1
    async_detection: bool = False
    enable_quality_gate: bool = False  # official path is self-masking
    min_quality_score: float = 0.3
    watermark_config: Optional[WatermarkConfig] = None
    enable_profiling: bool = False
    # When True (default for inswapper), use InsightFace FaceAnalysis + paste_back
    use_native_inswapper: bool = True
    # Adaptive detection / quality / color match (multi-resolution robustness)
    adaptive: bool = True
    adaptive_config: Optional[AdaptiveConfig] = None
    enable_color_match: bool = True
    enable_id_check: bool = False  # optional post-swap ArcFace reject
    min_id_similarity: float = 0.25
    # Post-swap face restore (opencv always available; gfpgan if installed)
    enable_enhance: bool = True
    enhance_method: str = "opencv"  # opencv | gfpgan | gpen | codeformer | realesrgan
    enhance_blend: float = 1.0  # 0..1 mix of restored vs swapped crop
    enhance_target_px: int = 0  # 0 = no boost; seamless uses 1024 (tiled)
    enhance_fidelity: float = 0.5  # CodeFormer weight: 0=quality, 1=fidelity
    use_occlusion_mask: bool = False  # landmark hull gate on color/enhance
    color_match_strength: float = 1.0  # 0..1 skin transfer strength
    # auto | on | off — blend original lower face back (beard / jaw)
    preserve_lower_face: str = "off"
    enable_grain_match: bool = False
    # Low-frequency shading transfer from target onto swapped face
    enable_lighting_match: bool = False
    lighting_match_strength: float = 0.75
    # FaceFusion-style XSeg occlusion (hands / hair / glasses)
    use_xseg_occlusion: bool = False
    xseg_model_path: Optional[str] = None
    # CPU-fast video / realtime path
    fast_video: Optional[FastVideoConfig] = None


class FaceSwapPipeline:
    """Main face swap pipeline."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()

        self.detector: Optional[FaceDetector] = None
        self.landmark_detector: Optional[LandmarkDetector] = None
        self.embedder: Optional[IdentityEmbedder] = None
        self.aligner: Optional[FaceAligner] = None
        self.swapper: Optional[FaceSwapper] = None
        self.blender: Optional[FaceBlender] = None
        self.temporal_smoother: Optional[TemporalSmoother] = None
        self._flow_smoother: Optional[OpticalFlowSmoother] = None
        self._smooth_faces_prev: list = []

        self.quality_validator: Optional[QualityValidator] = None
        self.profiler: Optional[PipelineProfiler] = None
        self.watermarker: Optional[InvisibleWatermarker] = None
        self._async_detector: Optional[AsyncFaceDetector] = None
        self._enhancer = None
        self._xseg = None
        self._cached_faces: list = []
        self._cache_frame_idx: int = -999
        self._fast: Optional[FastVideoConfig] = None

        self._face_app = None
        self._source_face: Any = None
        self._adaptive_cfg = AdaptiveConfig()
        self._initialized = False

    def _providers(self) -> list:
        providers = resolve_ort_providers(self.config.device)
        logger.info("ORT providers for device=%s: %s", self.config.device, providers)
        return providers

    def _ensure_face_app(self):
        if self._face_app is not None:
            return self._face_app
        from insightface.app import FaceAnalysis

        fast = self._fast if (self._fast and self._fast.enabled) else None
        # Fast CPU: skip 106/3D landmarks + genderage (InSwapper only needs
        # detection kps + recognition embedding). Deep-Live-Cam / Kilat style.
        kwargs = dict(name="buffalo_l", root="./models", providers=self._providers())
        if fast is not None:
            kwargs["allowed_modules"] = ["detection", "recognition"]

        app = FaceAnalysis(**kwargs)
        ctx_id = insightface_ctx_id(self.config.device)
        det = fast.det_size if fast is not None else (640, 640)
        app.prepare(ctx_id=ctx_id, det_size=det)
        app.ctx_id = ctx_id
        app._ravana_det_size = det
        self._face_app = app
        return app

    def initialize(self) -> None:
        """Initialize all pipeline components."""
        if self._initialized:
            return

        cfg = self.config
        # Must set before _ensure_face_app() so allowed_modules / det_size apply
        self._fast = cfg.fast_video

        raw_detector = RetinaFaceDetector(
            confidence_threshold=cfg.det_confidence_threshold, device=cfg.device
        )
        if cfg.async_detection:
            self._async_detector = AsyncFaceDetector(raw_detector)
            self._async_detector.start()
        self.detector = raw_detector

        self.landmark_detector = InsightFaceLandmarkDetector(device=cfg.device)
        self.aligner = FaceAligner(crop_size=(cfg.crop_size, cfg.crop_size))
        self.embedder = ArcFaceEmbedder(device=cfg.device)

        if cfg.swap_model == "inswapper":
            self.swapper = InSwapperModel(
                device=cfg.device,
                model_path=cfg.swap_model_path or "./models/inswapper_128.onnx",
            )
            if cfg.use_native_inswapper:
                self._ensure_face_app()
                self.swapper.ensure_loaded()
        elif cfg.swap_model in ("hyperswap", "hyperswap_1a_256"):
            try:
                self.swapper = HyperSwapModel(
                    device=cfg.device,
                    model_path=cfg.swap_model_path,
                )
                self._ensure_face_app()
                self.swapper.ensure_loaded()
            except Exception as e:
                logger.warning(
                    "HyperSwap load failed (%s); falling back to InSwapper.", e
                )
                self.swapper = InSwapperModel(
                    device=cfg.device,
                    model_path="./models/inswapper_128.onnx",
                )
                self._ensure_face_app()
                self.swapper.ensure_loaded()
        else:
            from .swap import SimSwapModel

            self.swapper = SimSwapModel(
                device=cfg.device,
                resolution=cfg.crop_size,
                model_path=cfg.swap_model_path,
            )

        self.blender = FaceBlender(
            blend_mode=cfg.blend_mode, color_correction=cfg.color_correction
        )

        if cfg.enable_temporal:
            self.temporal_smoother = TemporalSmoother(
                smooth_factor=cfg.temporal_smooth_factor
            )
            if cfg.video_flow_blend > 1e-3:
                self._flow_smoother = OpticalFlowSmoother(
                    OpticalFlowConfig(
                        method="farneback",
                        warp_blend=float(cfg.video_flow_blend),
                        flow_smooth_factor=0.55,
                        max_flow_magnitude=40.0,
                        latent_smoothing=False,
                    )
                )
            else:
                self._flow_smoother = None
        else:
            self.temporal_smoother = None
            self._flow_smoother = None
        self._smooth_faces_prev = []

        if cfg.enable_quality_gate:
            self.quality_validator = QualityValidator(
                min_quality_score=cfg.min_quality_score,
            )

        self.profiler = PipelineProfiler()
        self.profiler.enabled = cfg.enable_profiling

        wm_cfg = cfg.watermark_config or WatermarkConfig(enabled=False)
        self.watermarker = InvisibleWatermarker(wm_cfg)

        self._adaptive_cfg = cfg.adaptive_config or AdaptiveConfig(
            enable_color_match=cfg.enable_color_match,
            id_similarity_min=cfg.min_id_similarity,
        )

        if cfg.enable_enhance and not (
            self._fast and self._fast.enabled and self._fast.skip_enhance
        ):
            enh_cfg = EnhancementConfig(
                enabled=True,
                method=cfg.enhance_method,
                device=cfg.device,
                upscale=1,
                quality=cfg.enhance_fidelity,
                blend_weight=cfg.enhance_blend,
                target_face_px=cfg.enhance_target_px,
            )
            self._enhancer = create_enhancer(enh_cfg)
            try:
                self._enhancer.load_model()
            except Exception as e:
                logger.warning(
                    "Enhancer load failed (%s); falling back to OpenCV.", e
                )
                self._enhancer = create_enhancer(
                    EnhancementConfig(
                        enabled=True,
                        method="opencv",
                        device=cfg.device,
                        upscale=1,
                        blend_weight=cfg.enhance_blend,
                        target_face_px=cfg.enhance_target_px,
                    )
                )
                self._enhancer.load_model()
        else:
            self._enhancer = None

        self._xseg = None
        if cfg.use_xseg_occlusion:
            try:
                from .masking import XSegOccluder

                self._xseg = XSegOccluder(
                    device=cfg.device,
                    model_path=cfg.xseg_model_path,
                )
                self._xseg.load_model()
            except Exception as e:
                logger.warning("XSeg occluder load failed (%s); continuing without it.", e)
                self._xseg = None

        self._initialized = True

    def cleanup(self) -> None:
        if self._async_detector is not None:
            self._async_detector.stop()
            self._async_detector = None
        if self._flow_smoother is not None:
            self._flow_smoother.reset()
        self._smooth_faces_prev = []
        self._cached_faces = []
        self._cache_frame_idx = -999
        if self.temporal_smoother is not None:
            self.temporal_smoother.clear_cache()

    def _native_inswapper_enabled(self) -> bool:
        return (
            self.config.use_native_inswapper
            and hasattr(self.swapper, "swap_face")
            and (
                isinstance(self.swapper, InSwapperModel)
                or isinstance(self.swapper, HyperSwapModel)
                or self.config.swap_model
                in ("inswapper", "hyperswap", "hyperswap_1a_256")
            )
        )

    def _detect_faces(self, frame: Frame) -> list:
        """Detect + quality-filter faces (no user selection yet)."""
        app = self._ensure_face_app()
        fast = self._fast if (self._fast and self._fast.enabled) else None

        detect_frame = frame
        scale = 1.0
        if fast is not None:
            detect_frame, scale = maybe_downscale_for_detect(
                frame, fast.detect_max_side
            )
            try:
                if getattr(app, "_ravana_det_size", None) != fast.det_size:
                    app.prepare(
                        ctx_id=-1 if self.config.device != "cuda" else 0,
                        det_size=fast.det_size,
                    )
                    app._ravana_det_size = fast.det_size
            except Exception:
                pass
            faces = app.get(detect_frame) or []
            faces = scale_faces_from_detect_space(faces, scale)
        elif self.config.adaptive:
            faces = detect_faces_adaptive(app, frame, self._adaptive_cfg)
        else:
            faces = app.get(frame) or []

        faces = [
            f
            for f in faces
            if float(getattr(f, "det_score", 1.0))
            >= self.config.det_confidence_threshold
        ]
        if self.config.adaptive and not (fast and fast.enabled):
            kept = []
            for f in faces:
                q = assess_face(f, frame, self._adaptive_cfg)
                if q.ok or q.det_score >= 0.7:
                    kept.append(f)
                else:
                    logger.debug("Skipping face: %s", ", ".join(q.reasons))
            faces = kept or faces[:1]

        faces.sort(key=lambda f: float(f.det_score), reverse=True)
        if fast is not None:
            faces = faces[: max(1, fast.max_faces)]
        return faces

    def _get_faces(self, frame: Frame) -> list:
        """Detect faces and apply ``face_select`` / ``max_faces``."""
        faces = self._detect_faces(frame)
        return select_faces(
            faces,
            mode=getattr(self.config, "face_select", "all"),
            index=int(getattr(self.config, "face_index", 0)),
            max_faces=int(getattr(self.config, "max_faces", 0) or 0),
            source_face=self._source_face,
        )

    def _get_faces_video(self, frame: Frame, frame_number: int) -> list:
        """Detect with optional every-N-frames cache (fast_video or video_detect_every_n)."""
        fast = self._fast if (self._fast and self._fast.enabled) else None
        n = 1
        if fast is not None and fast.detect_every_n > 1:
            n = int(fast.detect_every_n)
        elif int(getattr(self.config, "video_detect_every_n", 1) or 1) > 1:
            n = int(self.config.video_detect_every_n)

        if n <= 1:
            return self._get_faces(frame)

        if (
            frame_number - self._cache_frame_idx >= n
            or not self._cached_faces
            or frame_number == 0
        ):
            self._cached_faces = self._get_faces(frame)
            self._cache_frame_idx = frame_number
        return list(self._cached_faces)

    def _apply_watermark(self, frame: Frame) -> Frame:
        if self.watermarker is None or not self.watermarker.config.enabled:
            return frame
        provenance = self.watermarker.create_provenance(
            model_name=self.config.swap_model,
        )
        return self.watermarker.embed(frame, provenance)

    def process_frame(
        self,
        frame: Frame,
        source_embedding: Embedding,
        return_intermediate: bool = False,
    ) -> Union[Frame, PipelineResult]:
        if not self._initialized:
            self.initialize()

        self.profiler.begin_frame()

        if self._native_inswapper_enabled():
            output = self._process_frame_native(frame)
            timings = self.profiler.end_frame()
            if return_intermediate:
                return PipelineResult(
                    output_frame=output,
                    swap_results=[],
                    processing_time_ms=timings.total_ms,
                )
            return output

        return self._process_frame_modular(frame, source_embedding, return_intermediate)

    def process_video_frame(
        self,
        frame: Frame,
        source_embedding: Embedding,
        frame_number: int = 0,
    ) -> Frame:
        if not self._initialized:
            self.initialize()

        self.profiler.begin_frame()

        if self._native_inswapper_enabled():
            faces = self._get_faces_video(frame, frame_number)
            if self.config.enable_temporal and faces:
                faces = ema_smooth_insight_faces(
                    faces,
                    self._smooth_faces_prev,
                    alpha=float(self.config.video_face_ema_alpha),
                )
                self._smooth_faces_prev = list(faces)
            else:
                self._smooth_faces_prev = list(faces) if faces else []

            output = self._process_frame_native(frame, faces=faces)

            if self._flow_smoother is not None:
                try:
                    output = self._flow_smoother.smooth_frame(output, frame)
                except Exception as e:
                    logger.debug("Optical flow temporal smooth skipped: %s", e)

            self.profiler.end_frame()
            return output

        # Modular video path
        result = self._process_frame_modular(frame, source_embedding, False)
        if self._flow_smoother is not None and isinstance(result, np.ndarray):
            try:
                result = self._flow_smoother.smooth_frame(result, frame)
            except Exception as e:
                logger.debug("Optical flow temporal smooth skipped: %s", e)
        return result

    def _process_frame_native(
        self, frame: Frame, faces: Optional[list] = None
    ) -> Frame:
        """Official InsightFace InSwapper path (correct geometry + blending)."""
        if self._source_face is None:
            raise ValueError(
                "Source face not set. Call extract_source_embedding() first."
            )

        with self.profiler.stage("detection"):
            if faces is None:
                faces = self._get_faces(frame)
        self.profiler.set_num_faces(len(faces))

        if not faces:
            return frame.copy()

        # Tiny faces (e.g. full-body Messi) — boost, swap, then scale back
        work, faces, boost = upscale_frame_for_tiny_faces(
            frame,
            faces,
            ideal_face_px=max(96, int(self._adaptive_cfg.ideal_face_px)),
            max_upscale=float(self._adaptive_cfg.max_upscale),
            min_side_trigger=64,
        )
        if boost > 1.05:
            # Re-detect on upscaled frame for accurate kps at new resolution
            faces = self._get_faces(work) or faces
            if not faces:
                return frame.copy()

        output = work.copy()
        swapped_faces = []
        xseg_masks = {}  # id(face) -> soft mask for padded bbox crop
        fast = self._fast if (self._fast and self._fast.enabled) else None
        do_color = self.config.enable_color_match and self._adaptive_cfg.enable_color_match
        if fast is not None and fast.skip_color_match:
            do_color = False

        with self.profiler.stage("swap"):
            for face in faces:
                before = output
                output = self.swapper.swap_face(output, face, self._source_face)

                if self._xseg is not None:
                    try:
                        xm = self._xseg.mask_for_bbox(before, face.bbox, pad_frac=0.08)
                        output = apply_occlusion_blend(
                            before, output, face.bbox, xm, pad_frac=0.08
                        )
                        xseg_masks[id(face)] = xm
                    except Exception as e:
                        logger.debug("XSeg gate skipped: %s", e)

                if do_color:
                    output = self._color_match_face_region(before, output, face)

                output = self._maybe_preserve_lower_face(before, output, face)

                if self.config.enable_id_check:
                    ok, sim = self._check_id(output, face)
                    if not ok:
                        logger.debug(
                            "ID check failed (sim=%.3f); keeping previous frame region",
                            sim,
                        )
                        output = before
                        continue
                swapped_faces.append(face)

        if self._enhancer is not None and swapped_faces:
            with self.profiler.stage("blend"):
                for face in swapped_faces:
                    region_mask = None
                    if self.config.use_occlusion_mask:
                        region_mask = self._bbox_occlusion_mask(
                            output.shape[:2], face, pad_frac=0.08, tight=True
                        )
                    xm = xseg_masks.get(id(face))
                    if xm is not None:
                        if region_mask is None:
                            region_mask = xm
                        else:
                            # Align shapes then multiply hull * xseg
                            if region_mask.shape[:2] != xm.shape[:2]:
                                xm = cv2.resize(
                                    xm,
                                    (region_mask.shape[1], region_mask.shape[0]),
                                    interpolation=cv2.INTER_LINEAR,
                                )
                            region_mask = region_mask * xm
                    output = enhance_face_region(
                        output,
                        face.bbox,
                        self._enhancer,
                        feather=25,
                        blend_weight=self.config.enhance_blend,
                        target_face_px=self.config.enhance_target_px,
                        region_mask=region_mask,
                    )
                    # GFPGAN shifts hue — re-match chrominance, keep restored L
                    if do_color:
                        output = self._color_match_face_region(
                            work, output, face, chrominance_only=True
                        )
                    if self.config.enable_grain_match:
                        output = self._grain_match_face_region(work, output, face)
                    # Re-apply lower-face preserve after enhance (GFPGAN can wipe beard)
                    output = self._maybe_preserve_lower_face(work, output, face)

        with self.profiler.stage("watermark"):
            output = self._apply_watermark(output)

        if boost > 1.05 and output.shape[:2] != frame.shape[:2]:
            output = cv2.resize(
                output,
                (frame.shape[1], frame.shape[0]),
                interpolation=cv2.INTER_AREA,
            )
        return output

    def _face_landmarks_xy(self, face: Any) -> Optional[np.ndarray]:
        """Extract 2D landmarks from an InsightFace Face (kps or landmark_2d_106)."""
        pts = getattr(face, "landmark_2d_106", None)
        if pts is None:
            pts = getattr(face, "kps", None)
        if pts is None:
            return None
        arr = np.asarray(pts, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] < 3 or arr.shape[1] < 2:
            return None
        return arr[:, :2]

    def _bbox_occlusion_mask(
        self,
        image_shape: tuple,
        face: Any,
        pad_frac: float = 0.08,
        tight: bool = False,
    ) -> Optional[np.ndarray]:
        """
        Soft landmark-hull mask cropped to the padded face bbox used by enhance.

        Returns float mask (rh, rw) in [0, 1], or None if landmarks missing.
        ``tight=True`` erodes more and fades the forehead to suppress hairline bleed.
        """
        pts = self._face_landmarks_xy(face)
        if pts is None:
            return None

        x1, y1, x2, y2 = [int(v) for v in face.bbox]
        h, w = image_shape[:2]
        pad = int(pad_frac * max(x2 - x1, y2 - y1))
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
        if x2 <= x1 + 8 or y2 <= y1 + 8:
            return None

        fw, fh = x2 - x1, y2 - y1
        erode_k, blur_k = adaptive_mask_kernels(fw, fh)
        if tight:
            erode = max(11, erode_k // 2)
            blur = max(25, blur_k * 2)
        else:
            erode = max(5, erode_k // 3)
            blur = max(15, blur_k)
        full = landmark_hull_mask((h, w), pts, erode=erode, blur=blur)
        crop = full[y1:y2, x1:x2].copy()
        if tight:
            crop *= forehead_taper_mask(crop.shape[0], crop.shape[1], top_frac=0.28)
        return crop

    def _color_match_face_region(
        self,
        original: Frame,
        swapped: Frame,
        face: Any,
        chrominance_only: bool = False,
    ) -> Frame:
        """Apply skin-aware color match inside the target face bbox."""
        x1, y1, x2, y2 = [int(v) for v in face.bbox]
        h, w = swapped.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 + 4 or y2 <= y1 + 4:
            return swapped

        # Expand slightly for blending context
        pad = int(0.1 * max(x2 - x1, y2 - y1))
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(w, x2 + pad), min(h, y2 + pad)

        region = swapped[y1:y2, x1:x2]
        ref = original[y1:y2, x1:x2]
        rh, rw = region.shape[:2]

        # Soft elliptical mask — center slightly lower to avoid hat/hairline
        mask = np.zeros((rh, rw), dtype=np.float32)
        cv2.ellipse(
            mask,
            (rw // 2, int(rh * 0.52)),
            (max(1, int(rw * 0.40)), max(1, int(rh * 0.44))),
            0,
            0,
            360,
            1.0,
            -1,
        )
        mask = cv2.GaussianBlur(mask, (25, 25), 11)
        mask *= forehead_taper_mask(rh, rw, top_frac=0.20)

        if self.config.use_occlusion_mask:
            pts = self._face_landmarks_xy(face)
            if pts is not None:
                erode_k, blur_k = adaptive_mask_kernels(rw, rh)
                full = landmark_hull_mask(
                    (h, w),
                    pts,
                    erode=max(9, erode_k // 2),
                    blur=max(25, blur_k * 2),
                )
                mask = mask * full[y1:y2, x1:x2]

        strength = float(getattr(self.config, "color_match_strength", 1.0))
        stats_mask = neck_ring_color_reference(ref, mask, dilate=max(15, rw // 8))
        if chrominance_only:
            matched = match_chrominance_to_target(
                region,
                ref,
                face_mask=mask,
                strength=strength,
                stats_mask=stats_mask,
            )
        else:
            matched = match_face_to_target_skin(
                region, ref, face_mask=mask, strength=strength
            )

        if self.config.enable_lighting_match:
            matched = match_lighting_to_target(
                matched,
                ref,
                face_mask=mask,
                strength=float(self.config.lighting_match_strength),
            )

        mask3 = mask[:, :, None]
        blended = (
            mask3 * matched.astype(np.float32)
            + (1 - mask3) * region.astype(np.float32)
        ).astype(np.uint8)

        out = swapped.copy()
        out[y1:y2, x1:x2] = blended
        return out

    def _grain_match_face_region(
        self, original: Frame, swapped: Frame, face: Any
    ) -> Frame:
        """Transfer mild film grain from original target into the face ROI."""
        x1, y1, x2, y2 = [int(v) for v in face.bbox]
        h, w = swapped.shape[:2]
        pad = int(0.08 * max(x2 - x1, y2 - y1))
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
        if x2 <= x1 + 8 or y2 <= y1 + 8:
            return swapped

        region = swapped[y1:y2, x1:x2]
        ref = original[y1:y2, x1:x2]
        rh, rw = region.shape[:2]
        mask = np.zeros((rh, rw), dtype=np.float32)
        cv2.ellipse(
            mask,
            (rw // 2, int(rh * 0.52)),
            (max(1, int(rw * 0.40)), max(1, int(rh * 0.44))),
            0,
            0,
            360,
            1.0,
            -1,
        )
        mask = cv2.GaussianBlur(mask, (21, 21), 9)
        matched = match_grain_to_target(region, ref, face_mask=mask, strength=0.35)
        mask3 = mask[:, :, None]
        blended = (
            mask3 * matched.astype(np.float32)
            + (1.0 - mask3) * region.astype(np.float32)
        ).astype(np.uint8)
        out = swapped.copy()
        out[y1:y2, x1:x2] = blended
        return out

    def _maybe_preserve_lower_face(
        self, original: Frame, swapped: Frame, face: Any
    ) -> Frame:
        """
        Softly blend original lower face back (beard / jaw).

        Mode ``auto`` only triggers when lower-face hair is detected on the
        original target; ``on`` always; ``off`` never.
        """
        mode = (getattr(self.config, "preserve_lower_face", "off") or "off").lower()
        if mode in ("off", "false", "0", "no"):
            return swapped

        x1, y1, x2, y2 = [int(v) for v in face.bbox]
        h, w = swapped.shape[:2]
        pad = int(0.06 * max(x2 - x1, y2 - y1))
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
        if x2 <= x1 + 8 or y2 <= y1 + 8:
            return swapped

        region_s = swapped[y1:y2, x1:x2]
        region_o = original[y1:y2, x1:x2]
        rh, rw = region_s.shape[:2]

        if mode in ("auto", "detect"):
            if not detect_lower_face_hair(region_o):
                return swapped

        lower = lower_face_preserve_mask(rh, rw, start_frac=0.55, feather=0.14)
        # Keep lower preserve inside a soft ellipse so ears/bg stay swapped
        ellipse = np.zeros((rh, rw), dtype=np.float32)
        cv2.ellipse(
            ellipse,
            (rw // 2, int(rh * 0.55)),
            (max(1, int(rw * 0.42)), max(1, int(rh * 0.45))),
            0,
            0,
            360,
            1.0,
            -1,
        )
        ellipse = cv2.GaussianBlur(ellipse, (15, 15), 0)
        lower = lower * ellipse
        # Cap strength so identity still comes through on mouth
        lower = np.clip(lower * 0.75, 0.0, 1.0)

        mask3 = lower[:, :, None]
        blended = (
            mask3 * region_o.astype(np.float32)
            + (1.0 - mask3) * region_s.astype(np.float32)
        ).astype(np.uint8)
        out = swapped.copy()
        out[y1:y2, x1:x2] = blended
        return out

    def _check_id(self, frame: Frame, face: Any) -> tuple:
        """Re-detect swapped face and compare ArcFace cosine to source."""
        faces = self._get_faces(frame)
        if not faces or self._source_face is None:
            return True, 1.0
        # Pick face closest to original bbox center
        tx = (float(face.bbox[0]) + float(face.bbox[2])) / 2.0
        ty = (float(face.bbox[1]) + float(face.bbox[3])) / 2.0
        best = min(
            faces,
            key=lambda f: (float(f.bbox[0] + f.bbox[2]) / 2 - tx) ** 2
            + (float(f.bbox[1] + f.bbox[3]) / 2 - ty) ** 2,
        )
        src_emb = np.asarray(self._source_face.normed_embedding, dtype=np.float32)
        out_emb = np.asarray(best.normed_embedding, dtype=np.float32)
        return identity_preserved(src_emb, out_emb, self.config.min_id_similarity)

    def _process_frame_modular(
        self,
        frame: Frame,
        source_embedding: Embedding,
        return_intermediate: bool = False,
    ) -> Union[Frame, PipelineResult]:
        """Legacy modular path for non-InSwapper / plugin models."""
        with self.profiler.stage("detection"):
            if self._async_detector is not None and self._async_detector.is_running:
                bboxes = self._async_detector.detect(frame)
            else:
                bboxes = self.detector.detect(frame)
        self.profiler.set_num_faces(len(bboxes))

        if not bboxes:
            processing_time = self.profiler.end_frame().total_ms
            if return_intermediate:
                return PipelineResult(
                    output_frame=frame.copy(),
                    swap_results=[],
                    processing_time_ms=processing_time,
                )
            return frame.copy()

        if self.quality_validator is not None:
            bboxes = [
                b
                for b in bboxes
                if self.quality_validator.validate_detection(b, frame.shape[:2]).passed
            ]
        if not bboxes:
            processing_time = self.profiler.end_frame().total_ms
            if return_intermediate:
                return PipelineResult(
                    output_frame=frame.copy(),
                    swap_results=[],
                    processing_time_ms=processing_time,
                )
            return frame.copy()

        with self.profiler.stage("landmarks"):
            landmarks_list = [
                self.landmark_detector.detect(frame, bbox) for bbox in bboxes
            ]

        with self.profiler.stage("alignment"):
            aligned_faces = []
            for bbox, landmarks in zip(bboxes, landmarks_list):
                if landmarks.num_points > 0:
                    aligned_faces.append(self.aligner.align(frame, landmarks, bbox))
                else:
                    aligned_faces.append(self.aligner.align_simple(frame, bbox))

        with self.profiler.stage("swap"):
            swap_results = [
                self.swapper.swap(aligned, source_embedding) for aligned in aligned_faces
            ]

        if self.quality_validator is not None:
            swap_results = [
                r
                for r in swap_results
                if not self.quality_validator.should_fallback(
                    self.quality_validator.validate_swap(r)
                )
            ]

        with self.profiler.stage("blend"):
            output_frame = frame.copy()
            for result in swap_results:
                output_frame = self.blender.blend(output_frame, result)

        with self.profiler.stage("watermark"):
            output_frame = self._apply_watermark(output_frame)

        timings = self.profiler.end_frame()
        if return_intermediate:
            return PipelineResult(
                output_frame=output_frame,
                swap_results=swap_results,
                processing_time_ms=timings.total_ms,
            )
        return output_frame

    def extract_source_embedding(self, source_image: Frame) -> Embedding:
        """
        Extract identity embedding from a source image.

        For InSwapper, also caches the InsightFace Face object (required for
        ``normed_embedding`` + official paste-back).
        """
        if not self._initialized:
            self.initialize()

        if self._native_inswapper_enabled():
            # Source identity: prefer largest face (ignore target face_select)
            faces = self._detect_faces(source_image)
            if not faces:
                raise ValueError("No face detected in source image")
            chosen = select_faces(faces, mode="largest")
            face = chosen[0]
            q = assess_face(face, source_image, self._adaptive_cfg)
            if not q.ok:
                logger.warning("Source face quality warnings: %s", ", ".join(q.reasons))

            # Upscale tiny source portraits so ArcFace/InSwapper get enough pixels
            if self.config.adaptive:
                source_image, face, _ = upscale_if_small_face(
                    source_image, face, self._adaptive_cfg
                )
                # Re-detect on upscaled image for accurate kps/embedding
                if face is not None:
                    up_faces = self._detect_faces(source_image)
                    if up_faces:
                        face = select_faces(up_faces, mode="largest")[0]

            self._source_face = face
            emb = np.asarray(self._source_face.normed_embedding, dtype=np.float32)
            return Embedding(
                vector=emb, model_name="arcface_buffalo_l", normalized=True
            )

        bbox = self.detector.detect_single(source_image)
        if bbox is None:
            raise ValueError("No face detected in source image")

        landmarks = self.landmark_detector.detect(source_image, bbox)
        aligned = self.aligner.align(source_image, landmarks, bbox)
        with self.profiler.stage("embedding"):
            embedding = self.embedder.extract(aligned)
        return embedding

    def extract_source_embedding_multi(self, source_images: List[Frame]) -> Embedding:
        if not self._initialized:
            self.initialize()

        if self._native_inswapper_enabled():
            embeddings = []
            best_face = None
            best_score = -1.0
            for img in source_images:
                faces = self._detect_faces(img)
                if not faces:
                    continue
                face = select_faces(faces, mode="largest")[0]
                embeddings.append(
                    np.asarray(face.normed_embedding, dtype=np.float32)
                )
                score = float(face.det_score)
                if score > best_score:
                    best_score = score
                    best_face = face
            if not embeddings or best_face is None:
                raise ValueError("No faces detected in source images")
            avg = np.mean(np.stack(embeddings, axis=0), axis=0)
            avg = avg / (np.linalg.norm(avg) + 1e-8)
            # Keep a real Face object for kps-less latent path; embedding averaged
            best_face.embedding = avg
            best_face.normed_embedding = avg
            self._source_face = best_face
            return Embedding(
                vector=avg.astype(np.float32),
                model_name="arcface_buffalo_l",
                normalized=True,
            )

        aligned_faces = []
        for img in source_images:
            bbox = self.detector.detect_single(img)
            if bbox is None:
                continue
            landmarks = self.landmark_detector.detect(img, bbox)
            aligned_faces.append(self.aligner.align(img, landmarks, bbox))
        if not aligned_faces:
            raise ValueError("No faces detected in source images")
        return self.embedder.extract_average(aligned_faces)

    def get_benchmark_report(self):
        return self.profiler.report()
