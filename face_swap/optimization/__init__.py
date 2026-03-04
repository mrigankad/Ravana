"""
TensorRT optimization and inference module.

As per PRD Section 5.9:
  - Use optimized inference runtimes such as TensorRT
    for deployment builds.
  - Allow configuration of performance vs. quality.

This package provides:
  - ONNX → TensorRT engine export.
  - TensorRT-accelerated inference runtime.
  - INT8/FP16 calibration helpers.
"""

from .export import ExportConfig, TensorRTExporter
from .runtime import TensorRTRuntime

__all__ = [
    "TensorRTExporter",
    "ExportConfig",
    "TensorRTRuntime",
]
