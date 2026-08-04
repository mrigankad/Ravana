"""Tests for HyperSwap helpers (mocked ORT — no 400 MB download)."""

from unittest.mock import MagicMock

import numpy as np

from face_swap.api import FaceSwapConfig
from face_swap.swap.hyperswap import HyperSwapModel, _paste_back_insightface


class TestHyperSwapPrepare:
    def test_prepare_target_shape(self):
        crop = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        blob = HyperSwapModel._prepare_target(crop)
        assert blob.shape == (1, 3, 256, 256)
        assert blob.dtype == np.float32
        assert float(blob.min()) >= -1.05
        assert float(blob.max()) <= 1.05

    def test_normalize_output_roundtrip_range(self):
        pred = np.zeros((1, 3, 256, 256), dtype=np.float32)  # mid gray after *0.5+0.5
        out = HyperSwapModel._normalize_output(pred)
        assert out.shape == (256, 256, 3)
        assert out.dtype == np.uint8
        assert 120 <= int(out.mean()) <= 135

    def test_source_latent_uses_normed_embedding(self):
        class F:
            embedding = np.array([3.0, 4.0] + [0.0] * 510, dtype=np.float32)

            @property
            def normed_embedding(self):
                e = self.embedding
                return e / np.linalg.norm(e)

        src = F()
        tgt = F()
        lat = HyperSwapModel._source_latent(src, tgt, swap_weight=0.5)
        assert lat.shape == (1, 512)
        assert abs(float(np.linalg.norm(lat)) - 1.0) < 1e-5

    def test_paste_back_preserves_shape(self):
        frame = np.zeros((120, 100, 3), dtype=np.uint8)
        aimg = np.full((64, 64, 3), 80, dtype=np.uint8)
        fake = np.full((64, 64, 3), 200, dtype=np.uint8)
        M = np.array([[1.0, 0.0, 20.0], [0.0, 1.0, 30.0]], dtype=np.float64)
        out = _paste_back_insightface(frame, fake, aimg, M)
        assert out.shape == frame.shape


class TestHyperSwapMockSession:
    def test_swap_face_with_mock(self):
        model = HyperSwapModel(device="cpu")

        class FakeSession:
            def get_inputs(self):
                s = MagicMock()
                s.name = "source"
                s.shape = [1, 512]
                t = MagicMock()
                t.name = "target"
                t.shape = [1, 3, 256, 256]
                return [s, t]

            def get_outputs(self):
                o = MagicMock()
                o.name = "output"
                return [o]

            def get_providers(self):
                return ["CPUExecutionProvider"]

            def run(self, _outs, feeds):
                assert "source" in feeds and "target" in feeds
                assert feeds["source"].shape == (1, 512)
                assert feeds["target"].shape == (1, 3, 256, 256)
                # Return mid gray in CHW
                return [np.zeros((1, 3, 256, 256), dtype=np.float32)]

        model._session = FakeSession()
        model._source_name = "source"
        model._target_name = "target"

        class Face:
            def __init__(self):
                self.kps = np.array(
                    [[40, 40], [60, 40], [50, 55], [42, 70], [58, 70]], dtype=np.float32
                )
                self.embedding = np.random.randn(512).astype(np.float32)

            @property
            def normed_embedding(self):
                e = self.embedding
                return e / (np.linalg.norm(e) + 1e-8)

        frame = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)
        out = model.swap_face(frame, Face(), Face(), paste_back=True)
        assert out.shape == frame.shape


class TestSeamlessHyperSwapConfig:
    def test_seamless_selects_hyperswap(self):
        pipe = FaceSwapConfig(quality="seamless").to_pipeline_config()
        assert pipe.swap_model == "hyperswap"
