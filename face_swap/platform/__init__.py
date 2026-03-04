"""
Platform-specific support: macOS Metal, mobile (Android/iOS), device detection.
"""

from .apple import (
    detect_apple_device,
    get_best_device,
    AppleDeviceInfo,
    CoreMLExporter,
    MPSInferenceRuntime,
    setup_onnxruntime_coreml,
)
from .mobile import MobileExporter, MobileExportConfig

__all__ = [
    "detect_apple_device",
    "get_best_device",
    "AppleDeviceInfo",
    "CoreMLExporter",
    "MPSInferenceRuntime",
    "setup_onnxruntime_coreml",
    "MobileExporter",
    "MobileExportConfig",
]
