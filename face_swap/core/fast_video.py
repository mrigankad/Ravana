"""
CPU-fast realtime video face-swap strategies.

Research basis (MobileFaceSwap AAAI'22, Deep-Live-Cam, FaceSwap Kilat,
DEV.to Rust+ONNX pipeline):

1. Cache source embedding once (never recompute per frame)
2. Detect every N frames; reuse landmarks between detections
3. Skip GFPGAN / heavy restore on the hot path
4. Smaller det_size (320) for close-up webcam faces
5. ONNX Runtime session thread tuning
6. Optional downscale of full frame before detect, swap at native

This module holds the config + helpers; the pipeline applies them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class FastVideoConfig:
    """
    Knobs for CPU-oriented realtime / batch video.

    Defaults target ~interactive CPU rates on modern laptops
    (trade quality for speed; disable enhance).
    """

    enabled: bool = False
    # Re-run FaceAnalysis every N frames (1 = every frame)
    detect_every_n: int = 3
    # Preferred detector window for webcam / portrait video
    det_size: Tuple[int, int] = (320, 320)
    # Max long-edge for detection (0 = no downscale)
    detect_max_side: int = 640
    # Skip OpenCV/GFPGAN enhance on video hot path
    skip_enhance: bool = True
    # Skip Reinhard color match (small CPU save)
    skip_color_match: bool = False
    # Process at most this many faces per frame
    max_faces: int = 1
    # ONNX Runtime intra-op threads (0 = library default)
    ort_intra_threads: int = 4
    ort_inter_threads: int = 1


def scale_faces_from_detect_space(
    faces: List[Any],
    scale: float,
) -> List[Any]:
    """Map face geometry from a downscaled detect image back to full-res."""
    if abs(scale - 1.0) < 1e-3:
        return faces
    out = []
    for face in faces:
        # Shallow-copy geometry fields we care about
        face.bbox = face.bbox.astype(np.float32) / scale
        if getattr(face, "kps", None) is not None:
            face.kps = face.kps.astype(np.float32) / scale
        out.append(face)
    return out


def maybe_downscale_for_detect(
    frame: np.ndarray,
    max_side: int,
) -> Tuple[np.ndarray, float]:
    """
    Downscale frame for detection only.

    Returns (image_for_detect, scale) where full = detect / scale
    i.e. scale = detect_size / full_size (< 1 when downscaled).
    """
    if max_side <= 0:
        return frame, 1.0
    h, w = frame.shape[:2]
    long = max(h, w)
    if long <= max_side:
        return frame, 1.0
    scale = max_side / float(long)
    new_w = int(w * scale)
    new_h = int(h * scale)
    small = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return small, scale


def configure_ort_session_options(
    intra_threads: int = 4,
    inter_threads: int = 1,
):
    """Build onnxruntime.SessionOptions tuned for CPU video loops."""
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if intra_threads > 0:
        opts.intra_op_num_threads = intra_threads
    if inter_threads > 0:
        opts.inter_op_num_threads = inter_threads
    opts.enable_mem_pattern = True
    opts.enable_cpu_mem_arena = True
    return opts
