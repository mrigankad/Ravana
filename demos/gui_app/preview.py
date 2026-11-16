"""Image helpers for the GUI: BGR frames and paths to QPixmaps (unicode-safe)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def bgr_to_qpixmap(frame: np.ndarray, max_side: int = 640) -> QPixmap:
    """Convert a BGR OpenCV frame to a QPixmap, downscaled to ``max_side``."""
    if frame is None or frame.size == 0:
        return QPixmap()
    h, w = frame.shape[:2]
    if max(w, h) > max_side:
        s = max_side / max(w, h)
        frame = cv2.resize(frame, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)


def _imread(path: str) -> np.ndarray | None:
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def load_path_pixmap(path: str, max_side: int = 320) -> QPixmap:
    """Load an image path, or the first frame of a video, as a QPixmap."""
    if Path(path).suffix.lower() in VIDEO_EXTS:
        cap = cv2.VideoCapture(path)
        try:
            ok, frame = cap.read()
        finally:
            cap.release()
        return bgr_to_qpixmap(frame, max_side) if ok and frame is not None else QPixmap()
    img = _imread(path)
    return QPixmap() if img is None else bgr_to_qpixmap(img, max_side)


__all__ = ["VIDEO_EXTS", "IMAGE_EXTS", "bgr_to_qpixmap", "load_path_pixmap", "Qt"]
