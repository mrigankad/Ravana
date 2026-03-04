"""
Face Swap SDK - Real-Time Face Swapping System

A production-ready SDK for face swapping on images, pre-recorded video, and live webcam streams.
"""

__version__ = "0.2.0"
__author__ = "Face Swap Team"

from .api import (
    swap_image,
    swap_video,
    start_realtime_swap,
    batch_swap,
    FaceSwapConfig,
)

from .pipeline import FaceSwapPipeline, PipelineConfig

from .core.types import (
    FaceBBox,
    Landmarks,
    AlignedFace,
    Embedding,
    SwapResult,
    PipelineResult,
)

from .core.quality import QualityValidator, QualityCode, QualityReport
from .core.profiler import PipelineProfiler, StageTimings, BenchmarkReport
from .core.model_manager import ModelManager, ModelInfo
from .core.config_loader import load_config, load_pipeline_config, load_face_swap_config
from .core.model_router import ModelRouter, ModelProfile, SceneType
from .watermark import InvisibleWatermarker, WatermarkConfig

# Phase 2 modules
from .enhancement import (
    FaceEnhancer, EnhancementConfig,
    GFPGANEnhancer, RealESRGANEnhancer, CodeFormerEnhancer,
    create_enhancer,
)
from .plugins import PluginRegistry, PluginInfo, get_registry, register_plugin
from .filters import ARFilterEngine, FilterPreset, FilterGallery, OverlayMode
from .audio import AudioProcessor
from .temporal import OpticalFlowSmoother, FlowGuidedBlender, OpticalFlowConfig

# Optional: TensorRT optimization (only if tensorrt is installed)
try:
    from .optimization import TensorRTExporter, ExportConfig, TensorRTRuntime
except ImportError:
    TensorRTExporter = None
    ExportConfig = None
    TensorRTRuntime = None

# Optional: Native C API bindings (only if compiled library exists)
try:
    from .native import NativeFaceSwap
except (ImportError, FileNotFoundError):
    NativeFaceSwap = None

# Optional: Training (only if torch is installed)
try:
    from .training import FaceSwapTrainer, TrainingConfig, TrainingState
except ImportError:
    FaceSwapTrainer = None
    TrainingConfig = None
    TrainingState = None

__all__ = [
    # High-level API
    "swap_image",
    "swap_video",
    "start_realtime_swap",
    "batch_swap",
    "FaceSwapConfig",

    # Pipeline
    "FaceSwapPipeline",
    "PipelineConfig",

    # Types
    "FaceBBox",
    "Landmarks",
    "AlignedFace",
    "Embedding",
    "SwapResult",
    "PipelineResult",

    # Quality
    "QualityValidator",
    "QualityCode",
    "QualityReport",

    # Profiling
    "PipelineProfiler",
    "StageTimings",
    "BenchmarkReport",

    # Models
    "ModelManager",
    "ModelInfo",
    "ModelRouter",
    "ModelProfile",
    "SceneType",

    # Config
    "load_config",
    "load_pipeline_config",
    "load_face_swap_config",

    # Watermark
    "InvisibleWatermarker",
    "WatermarkConfig",

    # Enhancement (Phase 2)
    "FaceEnhancer",
    "EnhancementConfig",
    "GFPGANEnhancer",
    "RealESRGANEnhancer",
    "CodeFormerEnhancer",
    "create_enhancer",

    # Plugins (Phase 2)
    "PluginRegistry",
    "PluginInfo",
    "get_registry",
    "register_plugin",

    # AR Filters (Phase 2)
    "ARFilterEngine",
    "FilterPreset",
    "FilterGallery",
    "OverlayMode",

    # Audio (Phase 2)
    "AudioProcessor",

    # Advanced Temporal (Phase 2)
    "OpticalFlowSmoother",
    "FlowGuidedBlender",
    "OpticalFlowConfig",

    # Training (Phase 2, optional)
    "FaceSwapTrainer",
    "TrainingConfig",
    "TrainingState",

    # Optimization (optional)
    "TensorRTExporter",
    "ExportConfig",
    "TensorRTRuntime",

    # Native bindings (optional)
    "NativeFaceSwap",
]


