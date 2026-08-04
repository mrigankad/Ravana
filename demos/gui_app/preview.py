"""Preview stage: image triple panes + OpenCV video/webcam surface."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}


def bgr_to_qpixmap(frame: np.ndarray, max_side: int = 640) -> QPixmap:
    """Convert a BGR OpenCV frame to a letterboxed QPixmap."""
    if frame is None or frame.size == 0:
        return QPixmap()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
    pix = QPixmap.fromImage(qimg)
    if max(w, h) > max_side:
        pix = pix.scaled(
            max_side,
            max_side,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
    return pix


def load_path_pixmap(path: str, max_side: int = 320) -> QPixmap:
    """Load an image path, or the first frame of a video, as QPixmap."""
    suffix = Path(path).suffix.lower()
    if suffix in VIDEO_EXTS:
        cap = cv2.VideoCapture(path)
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            return QPixmap()
        return bgr_to_qpixmap(frame, max_side)
    img = cv2.imread(path)
    if img is None:
        return QPixmap()
    return bgr_to_qpixmap(img, max_side)


class _Pane(QWidget):
    def __init__(self, title: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        title_lbl = QLabel(title)
        title_lbl.setObjectName("paneTitle")
        title_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image = QLabel("—")
        self.image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image.setMinimumSize(180, 180)
        self.image.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.image.setStyleSheet("background-color: #1c1c1c; border: 1px solid #2a2a2a;")
        layout.addWidget(title_lbl)
        layout.addWidget(self.image, stretch=1)

    def set_pixmap(self, pix: QPixmap) -> None:
        if pix.isNull():
            self.image.setText("—")
            self.image.setPixmap(QPixmap())
            return
        scaled = pix.scaled(
            self.image.size() if self.image.width() > 40 else pix.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.image.setPixmap(scaled)
        self.image.setText("")


class PreviewStage(QWidget):
    """Stacked preview: image triple OR single live/video surface."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._stack = QStackedWidget(self)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(self._stack)

        # Page 0 — source | target | result
        images = QWidget()
        row = QHBoxLayout(images)
        row.setContentsMargins(0, 0, 0, 0)
        self._source_pane = _Pane("Source")
        self._target_pane = _Pane("Target")
        self._result_pane = _Pane("Result")
        row.addWidget(self._source_pane)
        row.addWidget(self._target_pane)
        row.addWidget(self._result_pane)
        self._stack.addWidget(images)

        # Page 1 — video / webcam
        live = QWidget()
        live_l = QVBoxLayout(live)
        live_l.setContentsMargins(0, 0, 0, 0)
        self._live_label = QLabel("Live preview")
        self._live_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._live_label.setMinimumSize(480, 320)
        self._live_label.setStyleSheet(
            "background-color: #1c1c1c; border: 1px solid #2a2a2a;"
        )
        live_l.addWidget(self._live_label)
        self._stack.addWidget(live)

        self._cap: Optional[cv2.VideoCapture] = None
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_timer_tick)

    def show_image_mode(self) -> None:
        self.stop_playback()
        self._stack.setCurrentIndex(0)

    def set_source_image(self, path: Optional[str]) -> None:
        self.show_image_mode()
        if not path:
            self._source_pane.set_pixmap(QPixmap())
            return
        self._source_pane.set_pixmap(load_path_pixmap(path))

    def set_target_image(self, path: Optional[str]) -> None:
        self.show_image_mode()
        if not path:
            self._target_pane.set_pixmap(QPixmap())
            return
        self._target_pane.set_pixmap(load_path_pixmap(path))

    def set_result_image(self, path: Optional[str]) -> None:
        self.show_image_mode()
        if not path:
            self._result_pane.set_pixmap(QPixmap())
            return
        suffix = Path(path).suffix.lower()
        if suffix in VIDEO_EXTS:
            self.show_video_path(path)
            return
        self._result_pane.set_pixmap(load_path_pixmap(path, max_side=640))

    def show_frame_bgr(self, frame: np.ndarray) -> None:
        self._stack.setCurrentIndex(1)
        pix = bgr_to_qpixmap(frame, max_side=960)
        if pix.isNull():
            return
        scaled = pix.scaled(
            self._live_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._live_label.setPixmap(scaled)
        self._live_label.setText("")

    def show_video_path(self, path: str) -> None:
        self.stop_playback()
        self._stack.setCurrentIndex(1)
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            self._live_label.setText("Could not open video")
            return
        self._cap = cap
        self._timer.start(33)

    def stop_playback(self) -> None:
        self._timer.stop()
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def _on_timer_tick(self) -> None:
        if self._cap is None:
            return
        ok, frame = self._cap.read()
        if not ok or frame is None:
            # Loop
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._cap.read()
            if not ok or frame is None:
                self.stop_playback()
                return
        self.show_frame_bgr(frame)
