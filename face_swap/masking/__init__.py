"""Face masking helpers (occlusion / XSeg)."""

from .xseg import XSegOccluder, apply_occlusion_blend

__all__ = ["XSegOccluder", "apply_occlusion_blend"]
