"""QThread workers for swap, score, webcam, batch and model download.

Signals are named ``done`` / ``failed`` (never ``finished``) so they do not
shadow ``QThread.finished``, which the window uses for cleanup.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from demos.gui_app.config import VIDEO_EXTS, build_config
from face_swap import (
    FaceSwapConfig,
    FaceSwapPipeline,
    evaluate_swap,
    swap_image,
    swap_video,
)
from face_swap.core.fast_video import FastVideoConfig
from face_swap.core.model_manager import MODEL_PRESETS, ModelManager

__all__ = [
    "VIDEO_EXTS",
    "build_config",
    "SwapWorker",
    "ScoreWorker",
    "WebcamWorker",
    "BatchWorker",
    "ModelsDownloadWorker",
    "missing_preset_models",
    "imread_any",
    "imwrite_any",
    "Cancelled",
]


class Cancelled(Exception):
    """Raised from a progress callback to stop a running job."""


# ------------------------------------------------------------ unicode-safe IO
def imread_any(path: str) -> Optional[np.ndarray]:
    """cv2.imread that works with non-ASCII Windows paths."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_any(path: str, image: np.ndarray) -> bool:
    ext = Path(path).suffix or ".png"
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        return False
    buf.tofile(path)
    return True


def missing_preset_models(models_dir: str | Path, preset: str = "seamless") -> List[str]:
    """Return model names in ``preset`` that are not present on disk."""
    key = preset.lower().strip()
    if key not in MODEL_PRESETS:
        raise ValueError(f"Unknown preset {preset!r}")
    mgr = ModelManager(str(models_dir))
    missing: List[str] = []
    for name in MODEL_PRESETS[key]:
        info = mgr.get_model(name)
        if info is None or not info.is_downloaded:
            missing.append(name)
    return missing


class ModelsDownloadWorker(QThread):
    progress = Signal(int)
    status = Signal(str)
    done = Signal(str)
    failed = Signal(str)

    def __init__(self, models_dir: str, preset: str = "seamless"):
        super().__init__()
        self.models_dir = models_dir
        self.preset = preset

    def run(self) -> None:
        try:
            key = self.preset.lower().strip()
            if key not in MODEL_PRESETS:
                raise ValueError(f"Unknown preset {self.preset!r}")
            names = list(MODEL_PRESETS[key])
            mgr = ModelManager(self.models_dir)
            total = max(len(names), 1)
            for i, name in enumerate(names):
                self.status.emit(f"Downloading {name} ({i + 1}/{total})")

                def _cb(received: int, total_bytes: Optional[int], *, _i: int = i,
                        _name: str = name) -> None:
                    if total_bytes and total_bytes > 0:
                        frac = min(1.0, received / total_bytes)
                        self.progress.emit(int(((_i + frac) / total) * 100))
                        self.status.emit(
                            f"{_name}: {received / 2**20:.0f} of {total_bytes / 2**20:.0f} MB "
                            f"({_i + 1}/{total})"
                        )

                mgr.ensure_model(name, show_progress=False, progress=_cb)
                self.progress.emit(int((i + 1) / total * 100))
            self.progress.emit(100)
            self.done.emit(f"Models ready ({key}) in {self.models_dir}")
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))


class SwapWorker(QThread):
    """Single image or video swap. Emits ``done(output_path, seconds)``."""

    progress = Signal(int)
    status = Signal(str)
    done = Signal(str, float)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, source: str, target: str, output: str, config: FaceSwapConfig):
        super().__init__()
        self.source = source
        self.target = target
        self.output = output
        self.config = config
        self._running = True

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        t0 = time.perf_counter()
        try:
            if Path(self.target).suffix.lower() in VIDEO_EXTS:
                self.status.emit("Processing video")

                def _cb(i: int, total: int) -> None:
                    if not self._running:
                        raise Cancelled()
                    self.progress.emit(int(i / max(total, 1) * 100))
                    self.status.emit(f"Frame {i} of {total}")

                swap_video(self.source, self.target, self.output, self.config,
                           progress_callback=_cb)
            else:
                self.status.emit("Swapping face")
                self.progress.emit(15)
                src = imread_any(self.source)
                tgt = imread_any(self.target)
                if src is None or tgt is None:
                    raise RuntimeError("Could not read the source or target image.")
                self.progress.emit(35)
                out = swap_image(src, tgt, self.config)
                if not self._running:
                    raise Cancelled()
                self.progress.emit(92)
                if not imwrite_any(self.output, out):
                    raise RuntimeError(f"Could not write {self.output}")
            self.progress.emit(100)
            self.done.emit(self.output, time.perf_counter() - t0)
        except Cancelled:
            self.cancelled.emit()
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))


class ScoreWorker(QThread):
    """Emits ``done(dict)`` with id_similarity, sharpness_gain, passed_id, color_delta."""

    done = Signal(dict)
    failed = Signal(str)

    def __init__(self, source: str, target: str, result: str, device: str):
        super().__init__()
        self.source, self.target, self.result, self.device = source, target, result, device

    def run(self) -> None:
        try:
            m = evaluate_swap(self.source, self.target, self.result, device=self.device)
            self.done.emit({
                "id_similarity": float(m.id_similarity),
                "id_vs_target": float(m.id_vs_target),
                "sharpness_gain": float(m.sharpness_gain),
                "color_delta": float(m.color_delta_lab),
                "passed_id": bool(m.passed_id),
            })
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))


class WebcamWorker(QThread):
    frame_ready = Signal(object)
    fps = Signal(float)
    status = Signal(str)
    failed = Signal(str)

    def __init__(self, source_path: str, config: FaceSwapConfig, camera_id: int = 0):
        super().__init__()
        self.source_path = source_path
        self.config = config
        self.camera_id = camera_id
        self._running = True

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        cap = None
        try:
            cfg = self.config
            if not getattr(cfg, "realtime", False):
                cfg = FaceSwapConfig(
                    quality=cfg.quality, device=cfg.device, enhance_method=cfg.enhance_method,
                    pixel_boost=cfg.pixel_boost, face_select=cfg.face_select, realtime=True,
                )
            pipe_cfg = cfg.to_pipeline_config()
            pipe_cfg.enable_temporal = True
            if pipe_cfg.fast_video is None or not pipe_cfg.fast_video.enabled:
                n = max(1, int(getattr(cfg, "detect_every_n", 3) or 3))
                pipe_cfg.fast_video = FastVideoConfig(
                    enabled=True, detect_every_n=n, det_size=(320, 320), detect_max_side=640,
                    skip_enhance=True, max_faces=1, roi_track=True,
                )
                pipe_cfg.video_detect_every_n = n
                pipe_cfg.enable_enhance = False
            self.status.emit("Loading models")
            pipeline = FaceSwapPipeline(pipe_cfg)
            src = imread_any(self.source_path)
            if src is None:
                raise RuntimeError("Could not read the source image.")
            emb = pipeline.extract_source_embedding(src)
            cap = cv2.VideoCapture(self.camera_id)
            if not cap.isOpened():
                raise RuntimeError(f"Camera {self.camera_id} did not open.")
            self.status.emit("Live")
            frame_i = 0
            t_last = time.perf_counter()
            while self._running:
                ok, frame = cap.read()
                if not ok:
                    break
                out = pipeline.process_video_frame(frame, emb, frame_i)
                frame_i += 1
                self.frame_ready.emit(out.copy())
                if frame_i % 15 == 0:
                    now = time.perf_counter()
                    self.fps.emit(15.0 / max(now - t_last, 1e-6))
                    t_last = now
            self.status.emit("Camera stopped")
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))
        finally:
            if cap is not None:
                cap.release()


class BatchWorker(QThread):
    """Process many targets with one source face.

    Emits ``item_done(index, output_path)`` / ``item_failed(index, error)`` per row
    and ``done(summary)`` at the end.
    """

    progress = Signal(int)
    status = Signal(str)
    item_started = Signal(int)
    item_done = Signal(int, str)
    item_failed = Signal(int, str)
    done = Signal(str)
    failed = Signal(str)

    def __init__(self, source: str, targets: list[str], output_dir: str, config: FaceSwapConfig):
        super().__init__()
        self.source = source
        self.targets = list(targets)
        self.output_dir = output_dir
        self.config = config
        self._running = True

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        try:
            if not self.targets:
                raise RuntimeError("The batch queue is empty.")
            out_dir = Path(self.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            src = imread_any(self.source)
            if src is None:
                raise RuntimeError("Could not read the source image.")
            self.status.emit("Loading models")
            pipeline = FaceSwapPipeline(self.config.to_pipeline_config())
            emb = pipeline.extract_source_embedding(src)

            total = len(self.targets)
            saved = skipped = 0
            for i, target_path in enumerate(self.targets):
                if not self._running:
                    self.status.emit(f"Batch cancelled after {saved} of {total}")
                    break
                name = Path(target_path).name
                self.item_started.emit(i)
                self.status.emit(f"{i + 1} of {total}: {name}")
                self.progress.emit(int(i / max(total, 1) * 100))
                output_file = out_dir / f"swapped_{name}"
                try:
                    if Path(target_path).suffix.lower() in VIDEO_EXTS:
                        swap_video(self.source, target_path, str(output_file), self.config)
                    else:
                        tgt = imread_any(target_path)
                        if tgt is None:
                            raise RuntimeError("unreadable image")
                        result = pipeline.process_frame(tgt, emb)
                        if not imwrite_any(str(output_file), result):
                            raise RuntimeError("could not write output")
                    saved += 1
                    self.item_done.emit(i, str(output_file))
                except Exception as item_err:  # noqa: BLE001
                    skipped += 1
                    self.item_failed.emit(i, str(item_err))
            self.progress.emit(100)
            if not saved:
                self.failed.emit(f"Batch produced no outputs ({skipped} failed).")
                return
            summary = f"{saved} saved" + (f", {skipped} failed" if skipped else "") + f" in {out_dir}"
            self.done.emit(summary)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))
