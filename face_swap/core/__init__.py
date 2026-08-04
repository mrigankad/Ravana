"""
Core types, utilities, and infrastructure for the face swap pipeline.
"""

from .adaptive import (
    AdaptiveConfig,
    FaceQuality,
    assess_face,
    choose_det_size,
    detect_faces_adaptive,
    identity_preserved,
    reinhard_color_match,
)
from .config_loader import load_config, load_face_swap_config, load_pipeline_config
from .metrics import (
    MetricsAnalyzer,
    SwapMetrics,
    VariantResult,
    VariantSpec,
    evaluate_swap,
    expand_variant_grid,
    laplacian_sharpness,
    summarize_variant_rows,
)
from .model_manager import (
    MODEL_PRESETS,
    ModelInfo,
    ModelManager,
    download_with_progress,
    ensure_downloaded,
)
from .profiler import BenchmarkReport, PipelineProfiler, StageTimings
from .quality import QualityCode, QualityReport, QualityValidator
from .types import (
    AlignedFace,
    Embedding,
    FaceBBox,
    Frame,
    Landmarks,
    PipelineResult,
    Point,
    SwapResult,
)

__all__ = [
    # Types
    "Point",
    "FaceBBox",
    "Landmarks",
    "AlignedFace",
    "Embedding",
    "SwapResult",
    "PipelineResult",
    "Frame",
    # Quality
    "QualityValidator",
    "QualityCode",
    "QualityReport",
    # Metrics
    "SwapMetrics",
    "MetricsAnalyzer",
    "evaluate_swap",
    "laplacian_sharpness",
    "VariantSpec",
    "VariantResult",
    "expand_variant_grid",
    "summarize_variant_rows",
    # Adaptive
    "AdaptiveConfig",
    "FaceQuality",
    "assess_face",
    "choose_det_size",
    "detect_faces_adaptive",
    "identity_preserved",
    "reinhard_color_match",
    # Profiler
    "PipelineProfiler",
    "StageTimings",
    "BenchmarkReport",
    # Model management
    "ModelManager",
    "ModelInfo",
    "MODEL_PRESETS",
    "ensure_downloaded",
    "download_with_progress",
    # Config
    "load_config",
    "load_pipeline_config",
    "load_face_swap_config",
]
