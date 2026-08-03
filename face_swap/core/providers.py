"""
ONNX Runtime execution-provider selection.

Supports CUDA, DirectML (AMD/Intel/NVIDIA on Windows), and CPU.
DirectML requires the ``onnxruntime-directml`` package (replaces ``onnxruntime``).
"""

from __future__ import annotations

import logging
from typing import List, Sequence

logger = logging.getLogger("face_swap.providers")

# Normalized device aliases → canonical name
_DEVICE_ALIASES = {
    "cuda": "cuda",
    "gpu": "cuda",
    "nvidia": "cuda",
    "dml": "dml",
    "directml": "dml",
    "amd": "dml",
    "cpu": "cpu",
    "auto": "auto",
}


def normalize_device(device: str) -> str:
    key = (device or "cpu").strip().lower()
    return _DEVICE_ALIASES.get(key, key)


def available_providers() -> List[str]:
    try:
        import onnxruntime as ort

        return list(ort.get_available_providers())
    except Exception:
        return ["CPUExecutionProvider"]


def resolve_ort_providers(device: str) -> List[str]:
    """
    Return an ordered ONNX Runtime provider list for ``device``.

    ``auto`` prefers CUDA, then DirectML, then CPU.
    Unknown / unavailable accelerators fall back to CPU with a warning.
    """
    device = normalize_device(device)
    avail = available_providers()
    cpu = ["CPUExecutionProvider"]

    def pick(primary: Sequence[str], label: str) -> List[str]:
        for name in primary:
            if name in avail:
                return [name, "CPUExecutionProvider"]
        logger.warning(
            "%s requested but not available (have %s); using CPU. "
            "For AMD GPUs on Windows: pip uninstall onnxruntime && "
            "pip install onnxruntime-directml",
            label,
            avail,
        )
        return cpu

    if device == "cpu":
        return cpu
    if device == "cuda":
        return pick(["CUDAExecutionProvider"], "CUDA")
    if device == "dml":
        return pick(["DmlExecutionProvider"], "DirectML")
    if device == "auto":
        for name, label in (
            ("CUDAExecutionProvider", "CUDA"),
            ("DmlExecutionProvider", "DirectML"),
        ):
            if name in avail:
                logger.info("auto device selected %s", label)
                return [name, "CPUExecutionProvider"]
        return cpu

    logger.warning("Unknown device %r; using CPU.", device)
    return cpu


def insightface_ctx_id(device: str) -> int:
    """InsightFace ctx_id: GPU=0, CPU=-1. DirectML still uses CPU ctx for FaceAnalysis."""
    device = normalize_device(device)
    if device == "cuda":
        return 0
    # FaceAnalysis CUDA ctx does not apply to DML; ORT providers handle acceleration.
    return -1
