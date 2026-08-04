"""Preview stage: image triple panes + OpenCV video/webcam surface."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QImage, QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


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
    """Droppable preview pane."""

    file_dropped = Signal(str)

    def __init__(
        self,
        title: str,
        placeholder: str,
        *,
        accept_images: bool = True,
        accept_videos: bool = False,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._placeholder = placeholder
        self._accept_images = accept_images
        self._accept_videos = accept_videos
        self.setAcceptDrops(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)
        title_lbl = QLabel(title)
        title_lbl.setObjectName("paneTitle")
        title_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image = QLabel(placeholder)
        self.image.setObjectName("hint")
        self.image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image.setWordWrap(True)
        self.image.setMinimumSize(200, 220)
        self.image.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.image.setStyleSheet(
            "background-color: #1a1a1a; border: 1px dashed #3a3a3a; border-radius: 6px;"
        )
        layout.addWidget(title_lbl)
        layout.addWidget(self.image, stretch=1)

    def set_pixmap(self, pix: QPixmap) -> None:
        if pix.isNull():
            self.image.setText(self._placeholder)
            self.image.setPixmap(QPixmap())
            self.image.setStyleSheet(
                "background-color: #1a1a1a; border: 1px dashed #3a3a3a; border-radius: 6px;"
            )
            return
        self.image.setStyleSheet(
            "background-color: #1a1a1a; border: 1px solid #2e2e2e; border-radius: 6px;"
        )
        target = self.image.size()
        if target.width() < 40 or target.height() < 40:
            target = pix.size()
        scaled = pix.scaled(
            target,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.image.setPixmap(scaled)
        self.image.setText("")

    def _accepts_path(self, path: str) -> bool:
        ext = Path(path).suffix.lower()
        if self._accept_images and ext in IMAGE_EXTS:
            return True
        if self._accept_videos and ext in VIDEO_EXTS:
            return True
        return False

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls and self._accepts_path(urls[0].toLocalFile()):
                event.acceptProposedAction()
                self.image.setStyleSheet(
                    "background-color: #242018; border: 2px solid #c9a227; border-radius: 6px;"
                )
                return
        event.ignore()

    def dragLeaveEvent(self, event) -> None:  # noqa: N802
        if self.image.pixmap() is None or self.image.pixmap().isNull():
            self.image.setStyleSheet(
                "background-color: #1a1a1a; border: 1px dashed #3a3a3a; border-radius: 6px;"
            )
        else:
            self.image.setStyleSheet(
                "background-color: #1a1a1a; border: 1px solid #2e2e2e; border-radius: 6px;"
            )
        event.accept()

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        urls = event.mimeData().urls()
        if not urls:
            return
        path = urls[0].toLocalFile()
        if self._accepts_path(path):
            self.file_dropped.emit(path)
            event.acceptProposedAction()


class PreviewStage(QWidget):
    """Stacked preview: image triple OR single live/video surface."""

    source_dropped = Signal(str)
    target_dropped = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._stack = QStackedWidget(self)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(self._stack)

        images = QWidget()
        row = QHBoxLayout(images)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        self._source_pane = _Pane(
            "Source",
            "Drop source face\nor click Select",
            accept_images=True,
            accept_videos=False,
        )
        self._target_pane = _Pane(
            "Target",
            "Drop target image/video\nor click Select",
            accept_images=True,
            accept_videos=True,
        )
        self._result_pane = _Pane(
            "Result",
            "Swap output appears here",
            accept_images=False,
            accept_videos=False,
        )
        self._result_pane.setAcceptDrops(False)
        self._source_pane.file_dropped.connect(self.source_dropped.emit)
        self._target_pane.file_dropped.connect(self.target_dropped.emit)
        row.addWidget(self._source_pane)
        row.addWidget(self._target_pane)
        row.addWidget(self._result_pane)
        self._stack.addWidget(images)

        live = QWidget()
        live_l = QVBoxLayout(live)
        live_l.setContentsMargins(0, 0, 0, 0)
        self._live_title = QLabel("LIVE / VIDEO")
        self._live_title.setObjectName("paneTitle")
        self._live_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._live_label = QLabel("Live preview")
        self._live_label.setObjectName("hint")
        self._live_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._live_label.setMinimumSize(480, 320)
        self._live_label.setStyleSheet(
            "background-color: #1a1a1a; border: 1px solid #2e2e2e; border-radius: 6px;"
        )
        live_l.addWidget(self._live_title)
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
        self._live_title.setText("RESULT VIDEO")
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
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._cap.read()
            if not ok or frame is None:
                self.stop_playback()
                return
        self.show_frame_bgr(frame)
