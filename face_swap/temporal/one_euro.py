"""One-Euro filter (Casiez et al. 2012), vectorised, with per-track state.

Cutoffs are in cycles per ``dt`` unit; with the default ``dt=1`` (one frame)
the defaults suit pixel keypoints in video (min_cutoff 0.05, beta 0.02).

Adaptive low-pass: heavy smoothing when the signal is still (kills jitter),
light smoothing when it moves fast (kills lag).
"""

from __future__ import annotations

from typing import Dict, Hashable, Optional

import numpy as np


def _alpha(cutoff: np.ndarray, dt: float) -> np.ndarray:
    tau = 1.0 / (2.0 * np.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """One-Euro filter over an arbitrary-shaped numpy array (e.g. (5, 2) kps)."""

    def __init__(
        self,
        min_cutoff: float = 0.05,
        beta: float = 0.02,
        d_cutoff: float = 1.0,
    ):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x: Optional[np.ndarray] = None
        self._dx: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._x = None
        self._dx = None

    def __call__(self, x, dt: float = 1.0) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        dt = max(float(dt), 1e-6)
        if self._x is None or self._x.shape != x.shape:
            self._x = x.copy()
            self._dx = np.zeros_like(x)
            return x.astype(np.float32)

        dx = (x - self._x) / dt
        a_d = _alpha(np.asarray(self.d_cutoff), dt)
        self._dx = a_d * dx + (1.0 - a_d) * self._dx
        cutoff = self.min_cutoff + self.beta * np.abs(self._dx)
        a = _alpha(cutoff, dt)
        self._x = a * x + (1.0 - a) * self._x
        return self._x.astype(np.float32)


class OneEuroTracks:
    """Independent One-Euro state per track id (e.g. per face)."""

    def __init__(self, **kwargs):
        self._kwargs = kwargs
        self._filters: Dict[Hashable, OneEuroFilter] = {}

    def filter(self, track_id: Hashable, x, dt: float = 1.0) -> np.ndarray:
        f = self._filters.get(track_id)
        if f is None:
            f = self._filters[track_id] = OneEuroFilter(**self._kwargs)
        return f(x, dt)

    def prune(self, keep) -> None:
        keep = set(keep)
        for k in list(self._filters):
            if k not in keep:
                del self._filters[k]

    def reset(self) -> None:
        self._filters.clear()
