"""QThread workers for swap, score, and webcam."""

from __future__ import annotations

from pathlib import Path

import cv2
from PySide6.QtCore import QThread, Signal

from demos.gui_app.config import VIDEO_EXTS, build_config
from face_swap import FaceSwapConfig, FaceSwapPipeline, evaluate_swap, swap_image, swap_video
from face_swap.core.fast_video import FastVideoConfig

__all__ = [
    "VIDEO_EXTS",
    "build_config",
    "SwapWorker",
    "ScoreWorker",
    "WebcamWorker",
]


class SwapWorker(QThread):
    progress = Signal(int)
    status = Signal(str)
    finished = Signal(str)
    failed = Signal(str)

    def __init__(self, source: str, target: str, output: str, config: FaceSwapConfig):
        super().__init__()
        self.source = source
        self.target = target
        self.output = output
        self.config = config

    def run(self) -> None:
        try:
            if Path(self.target).suffix.lower() in VIDEO_EXTS:
                self.status.emit("Processing video...")

                def _cb(i: int, total: int) -> None:
                    self.progress.emit(int(i / max(total, 1) * 100))
                    self.status.emit(f"Frame {i}/{total}")

                swap_video(
                    self.source,
                    self.target,
                    self.output,
                    self.config,
                    progress_callback=_cb,
                )
            else:
                self.status.emit("Running face swap...")
                self.progress.emit(20)
                src = cv2.imread(self.source)
                tgt = cv2.imread(self.target)
                if src is None or tgt is None:
                    raise RuntimeError("Could not load source or target image")
                self.progress.emit(40)
                out = swap_image(src, tgt, self.config)
                self.progress.emit(90)
                cv2.imwrite(self.output, out)
            self.progress.emit(100)
            self.finished.emit(self.output)
        except Exception as e:
            self.failed.emit(str(e))


class ScoreWorker(QThread):
    finished = Signal(str)
    failed = Signal(str)

    def __init__(self, source: str, target: str, result: str, device: str):
        super().__init__()
        self.source = source
        self.target = target
        self.result = result
        self.device = device

    def run(self) -> None:
        try:
            m = evaluate_swap(
                self.source, self.target, self.result, device=self.device
            )
            msg = (
                f"id={m.id_similarity:.3f}  vs_tgt={m.id_vs_target:.3f}\n"
                f"sharp={m.sharpness_result:.1f} (d{m.sharpness_gain:+.1f})\n"
                f"dE={m.color_delta_lab:.1f}  "
                f"{'PASS' if m.passed_id else 'WEAK'} id"
            )
            self.finished.emit(msg)
        except Exception as e:
            self.failed.emit(str(e))


class WebcamWorker(QThread):
    frame_ready = Signal(object)
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
        try:
            cfg = self.config
            if not getattr(cfg, "realtime", False):
                cfg = FaceSwapConfig(
                    quality=cfg.quality,
                    device=cfg.device,
                    enhance_method=cfg.enhance_method,
                    pixel_boost=cfg.pixel_boost,
                    face_select=cfg.face_select,
                    realtime=True,
                )
            pipe_cfg = cfg.to_pipeline_config()
            pipe_cfg.enable_temporal = True
            if pipe_cfg.fast_video is None or not pipe_cfg.fast_video.enabled:
                n = max(1, int(getattr(cfg, "detect_every_n", 3) or 3))
                pipe_cfg.fast_video = FastVideoConfig(
                    enabled=True,
                    detect_every_n=n,
                    det_size=(320, 320),
                    detect_max_side=640,
                    skip_enhance=True,
                    max_faces=1,
                    roi_track=True,
                )
                pipe_cfg.video_detect_every_n = n
                pipe_cfg.enable_enhance = False
            pipeline = FaceSwapPipeline(pipe_cfg)
            src = cv2.imread(self.source_path)
            if src is None:
                raise RuntimeError("Could not load source image")
            emb = pipeline.extract_source_embedding(src)
            cap = cv2.VideoCapture(self.camera_id)
            if not cap.isOpened():
                raise RuntimeError(f"Camera {self.camera_id} failed to open")
            self.status.emit("Webcam live")
            frame_i = 0
            while self._running:
                ok, frame = cap.read()
                if not ok:
                    break
                out = pipeline.process_video_frame(frame, emb, frame_i)
                frame_i += 1
                self.frame_ready.emit(out.copy())
            cap.release()
            self.status.emit("Webcam stopped")
        except Exception as e:
            self.failed.emit(str(e))
