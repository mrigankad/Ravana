"""Tests for ONNX RestoreFormer enhancer (no model download for unit tests)."""

from unittest.mock import MagicMock

import numpy as np

from face_swap.api import FaceSwapConfig
from face_swap.enhancement import (
    EnhancementConfig,
    RestoreFormerOnnxEnhancer,
    create_enhancer,
)


class TestRestoreFormerOnnxFactory:
    def test_restoreformer_method_uses_onnx(self):
        enh = create_enhancer(EnhancementConfig(method="restoreformer", device="cpu"))
        assert isinstance(enh, RestoreFormerOnnxEnhancer)

    def test_alias(self):
        enh = create_enhancer(
            EnhancementConfig(method="restoreformer_plus_plus", device="cpu")
        )
        assert isinstance(enh, RestoreFormerOnnxEnhancer)

    def test_enhance_method_override_on_seamless(self):
        pipe = FaceSwapConfig(
            quality="seamless", enhance_method="restoreformer"
        ).to_pipeline_config()
        assert pipe.enhance_method == "restoreformer"

    def test_enhance_with_mock_session(self):
        cfg = EnhancementConfig(method="restoreformer", device="cpu")
        enh = RestoreFormerOnnxEnhancer(cfg)

        class FakeSession:
            def get_inputs(self):
                m = MagicMock()
                m.name = "input"
                return [m]

            def get_providers(self):
                return ["CPUExecutionProvider"]

            def run(self, _out, feeds):
                x = feeds["input"]
                assert x.shape == (1, 3, 512, 512)
                return [x]

        enh._session = FakeSession()
        enh._input_name = "input"
        face = np.random.randint(0, 255, (90, 70, 3), dtype=np.uint8)
        out = enh.enhance(face)
        assert out.shape == face.shape
        assert out.dtype == np.uint8
