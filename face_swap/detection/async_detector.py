"""
Asynchronous (threaded) face detection for real-time mode.

As per PRD Sections 5.3 and 5.9:
  - In real-time mode, support decoupling detection from swapping
    using a separate detection thread and caching to keep overall
    FPS high.
  - Run detection at a lower frequency and reuse cached detections.

This module wraps any ``FaceDetector`` and runs it in a background
thread, returning the most recent cached detections on every call
so that the main swap loop is never blocked by the detector.
"""

import threading
import time
from typing import List, Optional

import numpy as np

from ..core.types import FaceBBox, Frame
from .base import FaceDetector


class AsyncFaceDetector:
    """
    Wrapper that runs a ``FaceDetector`` in a background thread.

    The caller sees near-zero latency on ``detect()`` because it
    always returns cached results.  The background thread keeps
    updating those results at whatever rate the detector can manage.

    Usage:
        >>> async_det = AsyncFaceDetector(RetinaFaceDetector())
        >>> async_det.start()
        >>> bboxes = async_det.detect(frame)  # instant
        >>> async_det.stop()
    """

    def __init__(
        self,
        detector: FaceDetector,
        detect_interval: float = 0.0,
    ):
        """
        Args:
            detector: Underlying face detector.
            detect_interval: Minimum seconds between detections
                             (0 = as fast as possible).
        """
        self._detector = detector
        self._detect_interval = detect_interval

        # Shared state protected by a lock
        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._cached_bboxes: List[FaceBBox] = []
        self._frame_id: int = 0
        self._processed_frame_id: int = -1

        # Thread management
        self._thread: Optional[threading.Thread] = None
        self._running = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background detection thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._detection_loop, daemon=True, name="AsyncDetector"
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the background thread (blocks until it exits)."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    @property
    def is_running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------
    # Public detection API (main / render thread)
    # ------------------------------------------------------------------

    def detect(self, frame: Frame) -> List[FaceBBox]:
        """
        Submit *frame* for detection and return cached results immediately.

        The background thread will pick up the frame and update the
        cache asynchronously; calling code always gets the most recent
        successfully processed detections without blocking.
        """
        with self._lock:
            self._latest_frame = frame
            self._frame_id += 1
            return list(self._cached_bboxes)

    def detect_single(self, frame: Frame) -> Optional[FaceBBox]:
        """Return the largest cached face (convenience wrapper)."""
        bboxes = self.detect(frame)
        if not bboxes:
            return None
        return max(bboxes, key=lambda b: b.width * b.height)

    @property
    def cached_bboxes(self) -> List[FaceBBox]:
        """Read-only snapshot of the current cached detections."""
        with self._lock:
            return list(self._cached_bboxes)

    # ------------------------------------------------------------------
    # Background loop
    # ------------------------------------------------------------------

    def _detection_loop(self) -> None:
        """Runs in the background thread."""
        while self._running:
            # Grab the latest frame
            with self._lock:
                if self._frame_id == self._processed_frame_id:
                    # No new frame submitted — sleep briefly
                    pass
                else:
                    frame = self._latest_frame
                    fid = self._frame_id

            if frame is None or fid == self._processed_frame_id:
                time.sleep(0.001)
                continue

            # Run detection (potentially slow — this is the whole point)
            try:
                bboxes = self._detector.detect(frame)
            except Exception:
                bboxes = []

            # Publish results
            with self._lock:
                self._cached_bboxes = bboxes
                self._processed_frame_id = fid

            if self._detect_interval > 0:
                time.sleep(self._detect_interval)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "AsyncFaceDetector":
        self.start()
        return self

    def __exit__(self, *_exc) -> None:
        self.stop()
