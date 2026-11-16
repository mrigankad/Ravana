"""Unicode-safe image IO (cv2.imread/imwrite fail on non-ASCII Windows paths)."""

from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np


def imread_unicode(path, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    """Read an image from any path; returns None if missing or undecodable."""
    try:
        data = np.fromfile(os.fspath(path), dtype=np.uint8)
    except (OSError, ValueError):
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwrite_unicode(path, img: np.ndarray, params: Optional[list] = None) -> bool:
    """Write an image to any path (extension picks the codec)."""
    path = os.fspath(path)
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, img, params or [])
    if not ok:
        return False
    try:
        buf.tofile(path)
    except OSError:
        return False
    return True
