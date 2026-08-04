"""GUI workers and FaceSwapConfig mapping (Qt-free helpers live here too)."""

from __future__ import annotations

from face_swap import FaceSwapConfig


def build_config(
    quality: str,
    device: str,
    enhance: str,
    pixel_boost: str,
    face_select: str,
    *,
    realtime: bool = False,
) -> FaceSwapConfig:
    """Map GUI control strings to FaceSwapConfig (parity with legacy Tkinter GUI)."""
    return FaceSwapConfig(
        quality=quality,
        device=device,
        enhance_method=None if enhance == "default" else enhance,
        pixel_boost=None if pixel_boost == "default" else int(pixel_boost),
        face_select=face_select,
        realtime=realtime,
    )
