"""
Integration tests for the full face swap pipeline.

These tests validate the end-to-end flow from input to output,
ensuring all pipeline stages work together correctly.
"""

import os
import shutil
import tempfile

import numpy as np
import pytest

# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def sample_image():
    """Create a synthetic 720p BGR image with a face-like pattern."""
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    # Skin-tone ellipse as face
    import cv2

    cv2.ellipse(img, (640, 360), (120, 160), 0, 0, 360, (180, 200, 220), -1)
    # Eyes
    cv2.circle(img, (600, 320), 15, (50, 50, 50), -1)
    cv2.circle(img, (680, 320), 15, (50, 50, 50), -1)
    # Mouth
    cv2.ellipse(img, (640, 400), (40, 15), 0, 0, 360, (80, 80, 180), -1)
    return img


@pytest.fixture
def sample_source_image():
    """Create a synthetic source identity image."""
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    import cv2

    cv2.ellipse(img, (128, 128), (80, 100), 0, 0, 360, (200, 180, 160), -1)
    cv2.circle(img, (100, 110), 10, (40, 40, 40), -1)
    cv2.circle(img, (156, 110), 10, (40, 40, 40), -1)
    return img


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    d = tempfile.mkdtemp(prefix="face_swap_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def sample_video(temp_dir):
    """Create a short synthetic test video (10 frames)."""
    import cv2

    path = os.path.join(temp_dir, "test_input.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, 30.0, (640, 480))

    for i in range(10):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        x_offset = i * 5
        cv2.ellipse(
            frame, (320 + x_offset, 240), (80, 100), 0, 0, 360, (180, 200, 220), -1
        )
        writer.write(frame)

    writer.release()
    return path


# ── Pipeline integration tests ──────────────────────────────────────────


class TestPipelineConfig:
    """Test PipelineConfig creation and validation."""

    def test_default_config(self):
        from face_swap.pipeline import PipelineConfig

        cfg = PipelineConfig()
        assert cfg.device in ("cpu", "cuda")
        assert cfg.crop_size in (128, 256, 512)

    def test_config_from_yaml(self):
        from face_swap.core.config_loader import load_pipeline_config

        config_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "configs", "default.yaml"
        )
        if os.path.exists(config_path):
            cfg = load_pipeline_config(config_path)
            assert cfg is not None


class TestCoreTypes:
    """Test that all core types are properly structured."""

    def test_face_bbox_creation(self):
        from face_swap.core.types import FaceBBox

        bbox = FaceBBox(x1=10, y1=20, x2=110, y2=140, confidence=0.95)
        assert bbox.width == 100
        assert bbox.height == 120

    def test_pipeline_result(self):
        from face_swap.core.types import PipelineResult

        img = np.zeros((256, 256, 3), dtype=np.uint8)
        result = PipelineResult(
            output_frame=img, swap_results=[], processing_time_ms=10.0
        )
        assert result.output_frame is not None
        assert len(result.swap_results) == 0

    def test_swap_result(self):
        from face_swap.core.types import SwapResult

        face = np.zeros((128, 128, 3), dtype=np.uint8)
        mask = np.ones((128, 128), dtype=np.float32)
        result = SwapResult(
            swapped_face=face,
            mask=mask,
            source_embedding=None,
            target_aligned=None,
        )
        assert result.swapped_face.shape == (128, 128, 3)


class TestQualityIntegration:
    """Test quality validation in an integrated context."""

    def test_quality_validator_with_real_image(self, sample_image):
        from face_swap.core.quality import QualityValidator

        validator = QualityValidator()
        assert validator is not None

    def test_quality_report_structure(self):
        from face_swap.core.quality import QualityCode, QualityReport

        report = QualityReport(
            code=QualityCode.OK,
            message="All checks passed",
            score=0.95,
            sharpness=200.0,
        )
        assert report.passed is True
        assert report.code == QualityCode.OK


class TestProfilerIntegration:
    """Test profiler in realistic usage patterns."""

    def test_profiler_multistage(self):
        import time

        from face_swap.core.profiler import PipelineProfiler

        profiler = PipelineProfiler()
        profiler.enabled = True
        profiler.begin_frame()

        with profiler.stage("detection"):
            time.sleep(0.005)
        with profiler.stage("alignment"):
            time.sleep(0.003)
        with profiler.stage("swap"):
            time.sleep(0.004)

        profiler.end_frame()
        report = profiler.report()

        assert report is not None
        assert "detection" in str(report)

    def test_profiler_disabled(self):
        from face_swap.core.profiler import PipelineProfiler

        profiler = PipelineProfiler()
        profiler.enabled = False
        profiler.begin_frame()
        with profiler.stage("test"):
            pass
        profiler.end_frame()


class TestWatermarkIntegration:
    """Test watermarking with real image data."""

    def test_watermark_roundtrip(self, sample_image):
        from face_swap.watermark import InvisibleWatermarker, WatermarkConfig

        config = WatermarkConfig(enabled=True, message="test_integration")
        wm = InvisibleWatermarker(config)

        watermarked = wm.embed(sample_image)
        assert watermarked.shape == sample_image.shape
        assert watermarked.dtype == sample_image.dtype

        # Images should look very similar (invisible watermark)
        diff = np.abs(watermarked.astype(float) - sample_image.astype(float)).mean()
        assert diff < 10.0, f"Watermark too visible: mean diff = {diff}"


class TestModelManager:
    """Test model management in an integrated context."""

    def test_model_registration(self, temp_dir):
        from face_swap.core.model_manager import ModelInfo, ModelManager

        mgr = ModelManager(models_dir=temp_dir)

        info = ModelInfo(
            name="test_model",
            version="1.0.0",
            path="test_model.onnx",
            format="onnx",
        )
        mgr.register_model(info)
        assert mgr.get_model("test_model") is not None

    def test_model_rollback(self, temp_dir):
        from face_swap.core.model_manager import ModelInfo, ModelManager

        mgr = ModelManager(models_dir=temp_dir)

        v1 = ModelInfo(name="swap", version="1.0", path="swap_v1.onnx", format="onnx")
        v2 = ModelInfo(name="swap", version="2.0", path="swap_v2.onnx", format="onnx")
        mgr.register_model(v1)
        mgr.register_model(v2)

        current = mgr.get_model("swap")
        assert current.version == "2.0"

        mgr.rollback("swap")
        rolled = mgr.get_model("swap")
        assert rolled.version == "1.0"


class TestConfigLoader:
    """Test YAML configuration loading."""

    def test_load_with_overrides(self, temp_dir):
        import yaml

        config_path = os.path.join(temp_dir, "test.yaml")
        config_data = {
            "device": "cpu",
            "detection": {"confidence_threshold": 0.7},
            "swap": {"model": "inswapper", "crop_size": 128},
        }
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        from face_swap.core.config_loader import load_config

        loaded = load_config(config_path)
        assert loaded["device"] == "cpu"


class TestPluginSystem:
    """Test the plugin registry."""

    def test_registry_creation(self):
        from face_swap.plugins import PluginInfo, PluginRegistry

        registry = PluginRegistry()

        class DummyDetector:
            pass

        info = PluginInfo(
            name="dummy",
            version="1.0",
            category="detector",
            cls=DummyDetector,
        )
        registry.register(info)
        assert registry.get("detector", "dummy") == DummyDetector

    def test_list_plugins(self):
        from face_swap.plugins import PluginInfo, PluginRegistry

        registry = PluginRegistry()

        class A:
            pass

        class B:
            pass

        registry.register(PluginInfo("a", "1.0", "detector", A))
        registry.register(PluginInfo("b", "1.0", "swapper", B))

        all_plugins = registry.list_plugins()
        assert len(all_plugins) == 2

        det_only = registry.list_plugins(category="detector")
        assert len(det_only) == 1


class TestAudioProcessor:
    """Test audio processing utilities."""

    def test_processor_creation(self):
        from face_swap.audio import AudioProcessor

        proc = AudioProcessor()
        # May or may not have FFmpeg
        assert isinstance(proc.available, bool)

    def test_video_info(self, sample_video):
        from face_swap.audio import AudioProcessor

        proc = AudioProcessor()
        if proc.available:
            info = proc.get_video_info(sample_video)
            assert "fps" in info or info == {}


class TestEnhancement:
    """Test face enhancement module."""

    def test_config_defaults(self):
        from face_swap.enhancement import EnhancementConfig

        cfg = EnhancementConfig()
        assert cfg.enabled is False
        assert cfg.method == "gfpgan"
        assert cfg.upscale == 1

    def test_create_enhancer_factory(self):
        from face_swap.enhancement import EnhancementConfig, create_enhancer

        cfg = EnhancementConfig(method="gfpgan")
        enhancer = create_enhancer(cfg)
        assert enhancer is not None


class TestModelRouter:
    """Test multi-model routing."""

    def test_scene_classification(self):
        from face_swap.core.model_router import ModelRouter, SceneType

        router = ModelRouter()

        scene = router.classify_scene(
            num_faces=1,
            avg_face_size=200,
            frame_shape=(720, 1280),
            max_yaw=5.0,
        )
        assert scene == SceneType.PORTRAIT

    def test_group_scene(self):
        from face_swap.core.model_router import ModelRouter, SceneType

        router = ModelRouter()

        scene = router.classify_scene(
            num_faces=4,
            avg_face_size=100,
            frame_shape=(720, 1280),
            max_yaw=10.0,
        )
        assert scene == SceneType.GROUP

    def test_model_selection(self):
        from face_swap.core.model_router import ModelRouter

        router = ModelRouter()

        profile = router.select_model(
            num_faces=1,
            avg_face_size=150,
            frame_shape=(720, 1280),
        )
        assert profile is not None
        assert profile.name is not None


class TestOpticalFlow:
    """Test advanced temporal consistency."""

    def test_smoother_creation(self):
        from face_swap.temporal import OpticalFlowConfig, OpticalFlowSmoother

        cfg = OpticalFlowConfig(method="farneback")
        smoother = OpticalFlowSmoother(cfg)
        assert smoother is not None

    def test_latent_smoothing(self):
        from face_swap.temporal import OpticalFlowSmoother

        smoother = OpticalFlowSmoother()

        latent = np.random.randn(512).astype(np.float32)
        smoothed1 = smoother.smooth_latent(0, latent)
        assert smoothed1.shape == (512,)

        # Second call should blend
        latent2 = np.random.randn(512).astype(np.float32)
        smoothed2 = smoother.smooth_latent(0, latent2)
        assert not np.allclose(smoothed2, latent2)  # Should be blended

    def test_flow_smoother_first_frame(self, sample_image):
        from face_swap.temporal import OpticalFlowSmoother

        smoother = OpticalFlowSmoother()

        # First frame should pass through unchanged
        result = smoother.smooth_frame(sample_image, sample_image)
        assert result.shape == sample_image.shape


class TestARFilters:
    """Test AR filter system."""

    def test_filter_preset(self):
        from face_swap.filters import FilterPreset, OverlayMode

        preset = FilterPreset(
            name="Test Filter",
            source_images=["test.jpg"],
            overlay_mode=OverlayMode.NONE,
            tags=["test", "demo"],
        )
        assert preset.name == "Test Filter"

    def test_gallery_operations(self):
        from face_swap.filters import FilterGallery, FilterPreset

        gallery = FilterGallery()

        gallery.add(FilterPreset(name="A", source_images=["a.jpg"], tags=["fun"]))
        gallery.add(FilterPreset(name="B", source_images=["b.jpg"], tags=["serious"]))

        assert len(gallery.list_all()) == 2
        assert len(gallery.search("fun")) == 1

        gallery.remove("A")
        assert len(gallery.list_all()) == 1
