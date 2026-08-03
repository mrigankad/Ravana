"""
Face enhancement / super-resolution module.

As per PRD Section 5.6, provides optional GAN-based refiners and
super-resolution enhancers for post-processing swapped faces.
"""

from .enhancer import (
    CodeFormerEnhancer,
    CodeFormerOnnxEnhancer,
    EnhancementConfig,
    FaceEnhancer,
    GFPGANEnhancer,
    GFPGANOnnxEnhancer,
    GPENOnnxEnhancer,
    OpenCVEnhancer,
    RealESRGANEnhancer,
    create_enhancer,
    enhance_face_region,
    enhance_with_tiles,
)

__all__ = [
    "FaceEnhancer",
    "EnhancementConfig",
    "GFPGANEnhancer",
    "GFPGANOnnxEnhancer",
    "GPENOnnxEnhancer",
    "RealESRGANEnhancer",
    "CodeFormerEnhancer",
    "CodeFormerOnnxEnhancer",
    "OpenCVEnhancer",
    "create_enhancer",
    "enhance_face_region",
    "enhance_with_tiles",
]
