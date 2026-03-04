"""
Model training pipeline for custom face swap models.

As per PRD Section 8.1 / 8.2 (Phase 2):
  - Custom model training with identity / adversarial / perceptual losses.
  - Mixed-precision training, checkpointing, ONNX export.
"""

from .trainer import FaceSwapTrainer, TrainingConfig, TrainingState

__all__ = [
    "FaceSwapTrainer",
    "TrainingConfig",
    "TrainingState",
]
