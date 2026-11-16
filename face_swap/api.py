"""
High-level SDK API for face swapping.

As per PRD Section 9.1, this provides simple high-level functions:
- swap_image(source_img, target_img, config) -> output_img
- swap_video(source_img, input_video, config) -> output_video
- start_realtime_swap(source_img, camera_id, callback, config)
"""

import contextlib
import dataclasses
import logging
import os
import shutil
import subprocess
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, List, Optional, Union

import cv2

from .core.io import imread_unicode, imwrite_unicode
from .core.fast_video import FastVideoConfig
from .core.types import AlignedFace, Embedding, FaceBBox, Frame, Landmarks, SwapResult
from .pipeline import FaceSwapPipeline, PipelineConfig


logger = logging.getLogger("face_swap.api")

# ---------------------------------------------------------------------------
# Pipeline cache: building a pipeline loads ~1 GB of weights, so reuse it.
# ---------------------------------------------------------------------------
_CACHE_SIZE = 2
_cache: "OrderedDict[tuple, tuple]" = OrderedDict()
_cache_lock = threading.Lock()


def _config_fingerprint(cfg: "FaceSwapConfig", mode: str) -> tuple:
    """Hashable fingerprint of a config (+ image/video mode)."""
    return (mode,) + tuple(
        (f.name, getattr(cfg, f.name)) for f in dataclasses.fields(cfg)
    )


@contextlib.contextmanager
def _pooled_pipeline(cfg: "FaceSwapConfig", pipeline_config, mode: str):
    """Yield a cached, initialised pipeline; exclusive use via a per-entry lock."""
    key = _config_fingerprint(cfg, mode)
    with _cache_lock:
        entry = _cache.get(key)
        if entry is None:
            entry = (FaceSwapPipeline(pipeline_config), threading.Lock())
            _cache[key] = entry
            while len(_cache) > _CACHE_SIZE:
                _cache.popitem(last=False)
        else:
            _cache.move_to_end(key)
    pipeline, lock = entry
    with lock:
        if hasattr(pipeline, "reset_temporal_state"):
            pipeline.reset_temporal_state()
        yield pipeline


def clear_cache() -> None:
    """Drop all cached pipelines (frees model memory)."""
    with _cache_lock:
        entries = list(_cache.values())
        _cache.clear()
    for pipeline, _ in entries:
        try:
            pipeline.cleanup()
        except Exception:  # pragma: no cover - best effort
            pass


def _load_image(img: Union[str, "os.PathLike", Frame], role: str) -> Frame:
    """Load an image path (unicode-safe) or pass an array through."""
    if isinstance(img, (str, os.PathLike)):
        path = os.fspath(img)
        loaded = imread_unicode(path)
        if loaded is None:
            raise ValueError(f"Could not load {role} image: {path}")
        return loaded
    return img


@dataclass
class FaceSwapConfig:
    """
    User-friendly configuration for face swapping.

    Simplified interface over PipelineConfig for ease of use.
    """

    # Quality/Speed tradeoff
    quality: str = "high"  # "low" | "fast_cpu" | "medium" | "high" | "seamless"

    # Device: "cuda", "cpu", "dml"/"directml"/"amd" (Windows DirectML), "auto"
    device: str = "auto"

    # Processing options
    color_correction: bool = True
    enable_smoothing: bool = True

    # Model paths (optional)
    detection_model_path: Optional[str] = None
    swap_model_path: Optional[str] = None

    # Explicit CPU realtime mode (overrides quality presets for video)
    fast_cpu: bool = False
    detect_every_n: int = 3
    # Live webcam / stream: enable detect-every-N + ROI track + skip heavy restore
    realtime: bool = False

    # Optional overrides (None = use quality preset defaults)
    enhance_method: Optional[str] = None
    swap_model: Optional[str] = None  # inswapper | hyperswap
    pixel_boost: Optional[int] = None  # 0 off; seamless default 1024

    # Target face selection: all | largest | first | index | pose
    face_select: str = "all"
    face_index: int = 0
    max_faces: int = 0  # 0 = unlimited

    def to_pipeline_config(self) -> PipelineConfig:
        """Convert to internal PipelineConfig."""
        config = PipelineConfig(
            device=self.device,
            color_correction=self.color_correction,
        )

        # Quality presets — InSwapper is always 128 (ArcFace crop size).
        if self.quality == "low" or self.quality == "fast_cpu" or self.fast_cpu:
            config.crop_size = 128
            config.blend_mode = "alpha"
            config.enable_temporal = False
            config.enable_enhance = False
            config.enable_color_match = not self.fast_cpu and self.quality != "fast_cpu"
            config.fast_video = FastVideoConfig(
                enabled=True,
                detect_every_n=(
                    self.detect_every_n
                    if self.device == "cpu"
                    or self.fast_cpu
                    or self.quality == "fast_cpu"
                    else 1
                ),
                det_size=(320, 320),
                detect_max_side=640,
                skip_enhance=True,
                skip_color_match=self.fast_cpu or self.quality == "fast_cpu",
                max_faces=1,
                ort_intra_threads=4,
            )
        elif self.quality == "medium":
            config.crop_size = 128
            config.blend_mode = "alpha"
            config.enable_temporal = self.enable_smoothing
            config.enable_color_match = True
            config.enable_enhance = True
            config.enhance_method = "opencv"
        elif self.quality == "seamless":
            # FaceFusion-style stack: HyperSwap-256 + GFPGAN + XSeg + lighting
            config.crop_size = 128
            config.swap_model = "hyperswap"
            config.blend_mode = "feather"
            config.enable_temporal = self.enable_smoothing
            config.enable_color_match = True
            config.enable_enhance = True
            config.enhance_method = "gfpgan"
            config.enhance_blend = 0.70
            config.enhance_target_px = 1024  # FaceFusion-style tiled pixel boost
            config.use_occlusion_mask = True
            config.color_match_strength = 1.0
            config.preserve_lower_face = "auto"
            config.enable_grain_match = True
            config.use_xseg_occlusion = True
            config.enable_lighting_match = True
            config.lighting_match_strength = 0.75
            # Video temporal: detect every 2nd frame + flow blend + face EMA
            config.video_detect_every_n = 2
            config.video_face_ema_alpha = 0.35
            config.video_flow_blend = 0.28
        else:  # high
            config.crop_size = 128
            config.blend_mode = "feather"
            config.enable_temporal = self.enable_smoothing
            config.enable_color_match = True
            config.enable_enhance = True
            config.enhance_method = "opencv"
            config.color_match_strength = 1.0

        config.use_native_inswapper = True
        config.enable_quality_gate = False

        if self.enhance_method:
            config.enhance_method = self.enhance_method
            config.enable_enhance = True

        if self.swap_model:
            config.swap_model = self.swap_model

        if self.swap_model_path:
            config.swap_model_path = self.swap_model_path

        if self.pixel_boost is not None:
            config.enhance_target_px = max(0, int(self.pixel_boost))

        config.face_select = self.face_select
        config.face_index = int(self.face_index)
        config.max_faces = int(self.max_faces or 0)

        if self.realtime:
            n = max(1, int(self.detect_every_n))
            # Live path: prefer speed; keep color match, drop heavy restore
            config.fast_video = FastVideoConfig(
                enabled=True,
                detect_every_n=n,
                det_size=(320, 320),
                detect_max_side=640,
                skip_enhance=True,
                skip_color_match=False,
                max_faces=max(1, int(self.max_faces or 1)),
                roi_track=True,
                roi_pad_frac=0.65,
                ort_intra_threads=4,
            )
            config.video_detect_every_n = n
            config.enable_enhance = False
            config.enhance_target_px = 0
            config.use_xseg_occlusion = False
            config.enable_temporal = self.enable_smoothing

        return config


def swap_image(
    source_img: Union[str, Frame],
    target_img: Union[str, Frame],
    config: Optional[FaceSwapConfig] = None,
) -> Frame:
    """
    Swap a face from source image onto target image.

    High-level API as per PRD Section 9.1.

    Args:
        source_img: Source image path or array containing the face to swap
        target_img: Target image path or array to swap face onto
        config: Optional configuration

    Returns:
        Output image with swapped face

    Example:
        >>> result = swap_image("source.jpg", "target.jpg")
        >>> cv2.imwrite("output.jpg", result)
    """
    # Load images if paths provided
    source_img = _load_image(source_img, "source")

    target_img = _load_image(target_img, "target")

    # Use default config if not provided
    cfg = config or FaceSwapConfig()

    with _pooled_pipeline(cfg, cfg.to_pipeline_config(), "image") as pipeline:
        source_embedding = pipeline.extract_source_embedding(source_img)
        return pipeline.process_frame(target_img, source_embedding)


def swap_video(
    source_img: Union[str, Frame],
    input_video: str,
    output_video: str,
    config: Optional[FaceSwapConfig] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> None:
    """
    Swap a face from source image onto all frames of a video.

    High-level API as per PRD Section 9.1.

    Args:
        source_img: Source image path or array containing the face to swap
        input_video: Path to input video file
        output_video: Path to output video file
        config: Optional configuration
        progress_callback: Optional callback(frame_index, total_frames)

    Example:
        >>> swap_video("source.jpg", "input.mp4", "output.mp4")
    """
    # Load source image
    source_img = _load_image(source_img, "source")

    # Use default config if not provided
    cfg = config or FaceSwapConfig()
    pipeline_config = cfg.to_pipeline_config()
    # Video: enable temporal unless fast_cpu (reuse detections instead)
    if not (cfg.fast_cpu or cfg.quality == "fast_cpu"):
        pipeline_config.enable_temporal = True
        if pipeline_config.video_detect_every_n < 2 and cfg.quality in (
            "seamless",
            "high",
            "medium",
        ):
            pipeline_config.video_detect_every_n = 2
        if pipeline_config.video_flow_blend <= 0:
            pipeline_config.video_flow_blend = 0.25

    with _pooled_pipeline(cfg, pipeline_config, "video") as pipeline:
        source_embedding = pipeline.extract_source_embedding(source_img)
        _run_video(
            pipeline,
            source_embedding,
            input_video,
            output_video,
            progress_callback,
        )


def build_ffmpeg_mux_cmd(
    video_path: str,
    audio_source: str,
    output_path: str,
    ffmpeg: str = "ffmpeg",
) -> List[str]:
    """ffmpeg command: re-encode the swapped video (H.264) and copy source audio."""
    return [
        ffmpeg,
        "-y",
        "-i",
        video_path,
        "-i",
        audio_source,
        "-map",
        "0:v",
        "-map",
        "1:a?",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-shortest",
        output_path,
    ]


def _mux_audio(silent_video: str, audio_source: str, output_video: str) -> bool:
    """Remux audio from ``audio_source`` into the output; False if unavailable."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        logger.warning(
            "ffmpeg not found on PATH: output video has no audio and uses mp4v."
        )
        return False
    cmd = build_ffmpeg_mux_cmd(silent_video, audio_source, output_video, ffmpeg)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except OSError as e:
        logger.warning("ffmpeg could not be started (%s); output has no audio.", e)
        return False
    if proc.returncode != 0:
        logger.warning(
            "ffmpeg remux failed (code %s): %s",
            proc.returncode,
            (proc.stderr or "")[-300:],
        )
        return False
    return True


def _run_video(
    pipeline,
    source_embedding,
    input_video: str,
    output_video: str,
    progress_callback: Optional[Callable[[int, int], None]],
) -> None:
    import tempfile

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {input_video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps <= 0:
        logger.warning("Video reports fps=%s; assuming 25.", fps)
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Write to an ASCII temp path (cv2.VideoWriter is not unicode-safe on
    # Windows), then remux audio / move to the real destination.
    tmp_dir = tempfile.mkdtemp(prefix="ravana_")
    tmp_video = os.path.join(tmp_dir, "video_noaudio.mp4")
    out = cv2.VideoWriter(
        tmp_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not out.isOpened():
        cap.release()
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise ValueError(f"Could not open video writer for: {output_video}")

    try:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            result = pipeline.process_video_frame(frame, source_embedding, frame_idx)
            out.write(result)
            if progress_callback:
                progress_callback(frame_idx, total_frames)
            frame_idx += 1
    finally:
        cap.release()
        out.release()

    try:
        out_dir = os.path.dirname(os.path.abspath(output_video))
        os.makedirs(out_dir, exist_ok=True)
        if not _mux_audio(tmp_video, input_video, output_video):
            shutil.move(tmp_video, output_video)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def start_realtime_swap(
    source_img: Union[str, Frame],
    camera_id: int = 0,
    callback: Optional[Callable[[Frame], None]] = None,
    config: Optional[FaceSwapConfig] = None,
    display_size: tuple = (1280, 720),
) -> None:
    """
    Start real-time face swapping from webcam.

    High-level API as per PRD Section 9.1.

    Args:
        source_img: Source image path or array containing the face to swap
        camera_id: Camera device ID (default 0 for primary webcam)
        callback: Optional callback for each processed frame
        config: Optional configuration
        display_size: Size to display the output

    Example:
        >>> start_realtime_swap("source.jpg", camera_id=0)
    """
    # Load source image
    source_img = _load_image(source_img, "source")

    # Use realtime path by default for webcam
    cfg = config or FaceSwapConfig(
        quality="medium", device="auto", realtime=True, enable_smoothing=True
    )
    if not cfg.realtime and (cfg.device == "cpu" or cfg.fast_cpu):
        # Legacy CPU-only fast path
        pass
    elif not cfg.realtime:
        cfg = FaceSwapConfig(
            quality=cfg.quality,
            device=cfg.device,
            realtime=True,
            detect_every_n=cfg.detect_every_n,
            enable_smoothing=cfg.enable_smoothing,
            enhance_method=cfg.enhance_method,
            face_select=cfg.face_select,
        )

    pipeline_config = cfg.to_pipeline_config()
    if pipeline_config.fast_video is None or not pipeline_config.fast_video.enabled:
        pipeline_config.fast_video = FastVideoConfig(
            enabled=True,
            detect_every_n=max(1, cfg.detect_every_n),
            skip_enhance=True,
            roi_track=True,
            max_faces=1,
        )
        pipeline_config.video_detect_every_n = max(1, cfg.detect_every_n)
        pipeline_config.enable_enhance = False
    pipeline = FaceSwapPipeline(pipeline_config)

    # Extract source embedding
    source_embedding = pipeline.extract_source_embedding(source_img)

    # Open camera
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise ValueError(f"Could not open camera: {camera_id}")

    # Set camera properties for performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, display_size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, display_size[1])

    print("Starting real-time face swap. Press 'q' to quit.")

    frame_idx = 0
    fps_hist = []
    last_t = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            now = time.time()
            if last_t is not None:
                dt = now - last_t
                if dt > 1e-6:
                    fps_hist.append(1.0 / dt)
                    if len(fps_hist) > 30:
                        fps_hist.pop(0)
            last_t = now

            result = pipeline.process_video_frame(frame, source_embedding, frame_idx)

            if fps_hist:
                avg = sum(fps_hist) / len(fps_hist)
                cv2.putText(
                    result,
                    f"FPS: {avg:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            if callback:
                callback(result)

            cv2.imshow("Face Swap - Press q to quit", result)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frame_idx += 1

    finally:
        cap.release()
        cv2.destroyAllWindows()


def batch_swap(
    source_img: Union[str, Frame],
    target_paths: List[str],
    output_dir: str,
    config: Optional[FaceSwapConfig] = None,
) -> List[str]:
    """
    Batch process multiple images with the same source face.

    Args:
        source_img: Source image path or array
        target_paths: List of target image paths
        output_dir: Directory to save output images
        config: Optional configuration

    Returns:
        List of output file paths
    """
    from pathlib import Path

    # Load source image
    source_img = _load_image(source_img, "source")

    # Use default config
    cfg = config or FaceSwapConfig()
    pipeline = FaceSwapPipeline(cfg.to_pipeline_config())

    # Extract source embedding
    source_embedding = pipeline.extract_source_embedding(source_img)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    output_files = []

    for target_path in target_paths:
        # Load target
        target_img = imread_unicode(target_path)
        if target_img is None:
            print(f"Warning: Could not load {target_path}, skipping")
            continue

        # Process
        result = pipeline.process_frame(target_img, source_embedding)

        # Save
        output_file = output_path / f"swapped_{Path(target_path).name}"
        imwrite_unicode(output_file, result)
        output_files.append(str(output_file))

        print(f"Processed: {target_path} -> {output_file}")

    return output_files


# Low-level API exports for advanced users

__all__ = [
    # High-level API
    "swap_image",
    "swap_video",
    "clear_cache",
    "build_ffmpeg_mux_cmd",
    "start_realtime_swap",
    "batch_swap",
    "FaceSwapConfig",
    # Low-level API
    "FaceSwapPipeline",
    "PipelineConfig",
    # Types
    "FaceBBox",
    "Landmarks",
    "AlignedFace",
    "Embedding",
    "SwapResult",
]
