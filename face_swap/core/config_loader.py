"""
YAML configuration loader.

As per PRD Section 9.2, this loads pipeline configuration from YAML files
(e.g., configs/default.yaml) and merges with runtime overrides.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import yaml

from ..pipeline import PipelineConfig

if TYPE_CHECKING:
    from ..api import FaceSwapConfig

_DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent.parent / "configs" / "default.yaml"
)


def load_config(path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Load raw configuration dictionary from a YAML file.

    Args:
        path: Path to YAML file. Defaults to ``configs/default.yaml``.

    Returns:
        Parsed configuration dictionary.
    """
    path = Path(path) if path else _DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    return data


def load_pipeline_config(
    path: Optional[Union[str, Path]] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> PipelineConfig:
    """
    Build a ``PipelineConfig`` from a YAML file with optional overrides.

    Args:
        path: YAML config path.
        overrides: Dict of key=value overrides applied after loading.

    Returns:
        Fully populated ``PipelineConfig``.
    """
    data = load_config(path)

    # Apply overrides
    if overrides:
        _deep_merge(data, overrides)

    return _dict_to_pipeline_config(data)


def load_face_swap_config(
    path: Optional[Union[str, Path]] = None,
    quality: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> "FaceSwapConfig":
    """
    Build a user-friendly ``FaceSwapConfig`` from a YAML file.

    Args:
        path: YAML config path.
        quality: Optional quality preset override (``low`` / ``medium`` / ``high``).
        overrides: Additional overrides.

    Returns:
        ``FaceSwapConfig`` instance.
    """
    data = load_config(path)

    if overrides:
        _deep_merge(data, overrides)

    # Select quality preset if requested
    if quality and "quality_presets" in data:
        preset = data["quality_presets"].get(quality, {})
        _deep_merge(data, preset)

    device = data.get("device", "cuda")
    blending = data.get("blending", {})
    temporal = data.get("temporal", {})
    swap = data.get("swap", {})

    from ..api import FaceSwapConfig

    return FaceSwapConfig(
        quality=quality or "high",
        device=device,
        color_correction=blending.get("color_correction", True),
        enable_smoothing=temporal.get("enabled", True),
        swap_model_path=swap.get("model_path"),
    )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _dict_to_pipeline_config(data: Dict[str, Any]) -> PipelineConfig:
    """Map a flat / nested dict into a ``PipelineConfig`` dataclass."""

    detection = data.get("detection", {})
    alignment = data.get("alignment", {})
    swap = data.get("swap", {})
    blending = data.get("blending", {})
    temporal = data.get("temporal", {})
    performance = data.get("performance", {})

    return PipelineConfig(
        device=data.get("device", "cuda"),
        detection_model=detection.get("model", "retinaface"),
        det_confidence_threshold=detection.get("confidence_threshold", 0.5),
        crop_size=alignment.get("crop_size", 256),
        swap_model=swap.get("model", "inswapper"),
        swap_model_path=swap.get("model_path"),
        blend_mode=blending.get("mode", "alpha"),
        color_correction=blending.get("color_correction", True),
        enable_temporal=temporal.get("enabled", True),
        temporal_smooth_factor=temporal.get("smooth_factor", 0.7),
        batch_size=performance.get("batch_size", 1),
    )


def _deep_merge(base: dict, override: dict) -> None:
    """Recursively merge *override* into *base* in place."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
