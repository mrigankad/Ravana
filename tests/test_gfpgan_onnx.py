"""Tests for ONNX GFPGAN enhancer (no model download required for unit tests)."""

from unittest.mock import MagicMock, patch

import numpy as np
import cv2

from face_swap.enhancement import EnhancementConfig, GFPGANOnnxEnhancer, create_enhancer


class TestGFPGANOnnxFactory:
    def test_gfpgan_method_uses_onnx(self):
        enh = create_enhancer(EnhancementConfig(method="gfpgan", device="cpu"))
        assert isinstance(enh, GFPGANOnnxEnhancer)

    def test_enhance_with_mock_session(self):
        cfg = EnhancementConfig(method="gfpgan", device="cpu")
        enh = GFPGANOnnxEnhancer(cfg)

        # Fake ORT session: return NCHW [-1,1] blob matching input size
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
                return [x]  # identity restore

        enh._session = FakeSession()
        enh._input_name = "input"

        face = np.random.randint(0, 255, (100, 80, 3), dtype=np.uint8)
        out = enh.enhance(face)
        assert out.shape == face.shape
        assert out.dtype == np.uint8
