"""
Watermarking and provenance module.

As per PRD Section 6.3, this provides invisible watermarking hooks
and metadata injection for traceability of manipulated content.
"""

from .watermarker import InvisibleWatermarker, WatermarkConfig

__all__ = [
    "InvisibleWatermarker",
    "WatermarkConfig",
]
