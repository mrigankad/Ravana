"""Tests for ONNX CodeFormer enhancer (mocked ORT — no download)."""

from unittest.mock import MagicMock

import numpy as np

from face_swap.api import FaceSwapConfig
from face_swap.enhancement import (
    CodeFormerOnnxEnhancer,
    EnhancementConfig,
    create_enhancer,
)


class TestCodeFormerOnnxFactory:
    def test_codeformer_method_uses_onnx(self):
        enh = create_enhancer(EnhancementConfig(method="codeformer", device="cpu"))
        assert isinstance(enh, CodeFormerOnnxEnhancer)

    def test_enhance_method_override_on_seamless(self):
        pipe = FaceSwapConfig(
            quality="seamless", enhance_method="codeformer"
        ).to_pipeline_config()
        assert pipe.enhance_method == "codeformer"
        assert pipe.swap_model == "hyperswap"

    def test_swapper_override(self):
        pipe = FaceSwapConfig(
            quality="seamless", swap_model="inswapper"
        ).to_pipeline_config()
        assert pipe.swap_model == "inswapper"

    def test_enhance_with_mock_session(self):
        cfg = EnhancementConfig(method="codeformer", device="cpu", quality=0.6)
        enh = CodeFormerOnnxEnhancer(cfg)

        class FakeSession:
            def get_inputs(self):
                inp = MagicMock()
                inp.name = "input"
                w = MagicMock()
                w.name = "weight"
                return [inp, w]

            def get_providers(self):
                return ["CPUExecutionProvider"]

            def run(self, _out, feeds):
                assert feeds["input"].shape == (1, 3, 512, 512)
                assert "weight" in feeds
                assert feeds["weight"].dtype == np.float64
                assert abs(float(feeds["weight"][0]) - 0.6) < 1e-6
                return [feeds["input"]]

        enh._session = FakeSession()
        enh._input_name = "input"
        enh._weight_name = "weight"
        enh._has_weight = True

        face = np.random.randint(0, 255, (90, 70, 3), dtype=np.uint8)
        out = enh.enhance(face)
        assert out.shape == face.shape
        assert out.dtype == np.uint8
