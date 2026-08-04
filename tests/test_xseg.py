"""Tests for XSeg occlusion helpers (mocked ORT — no model download)."""

from unittest.mock import MagicMock

import numpy as np

from face_swap.api import FaceSwapConfig
from face_swap.masking.xseg import XSegOccluder, apply_occlusion_blend


class TestApplyOcclusionBlend:
    def test_mask_one_keeps_swapped(self):
        original = np.zeros((64, 64, 3), dtype=np.uint8)
        swapped = np.full((64, 64, 3), 200, dtype=np.uint8)
        mask = np.ones((40, 40), dtype=np.float32)
        out = apply_occlusion_blend(
            original, swapped, (12, 12, 52, 52), mask, pad_frac=0.0
        )
        assert np.allclose(out[20:40, 20:40], 200)

    def test_mask_zero_keeps_original(self):
        original = np.full((64, 64, 3), 50, dtype=np.uint8)
        swapped = np.full((64, 64, 3), 200, dtype=np.uint8)
        mask = np.zeros((40, 40), dtype=np.float32)
        out = apply_occlusion_blend(
            original, swapped, (12, 12, 52, 52), mask, pad_frac=0.0
        )
        assert np.allclose(out[20:40, 20:40], 50)

    def test_half_mask_blends(self):
        original = np.zeros((32, 32, 3), dtype=np.uint8)
        swapped = np.full((32, 32, 3), 100, dtype=np.uint8)
        mask = np.full((20, 20), 0.5, dtype=np.float32)
        out = apply_occlusion_blend(
            original, swapped, (6, 6, 26, 26), mask, pad_frac=0.0
        )
        mid = float(out[16, 16, 0])
        assert 40 < mid < 60


class TestXSegOccluderMock:
    def test_face_mask_shape_with_mock_session(self):
        oc = XSegOccluder(device="cpu")

        class FakeSession:
            def get_inputs(self):
                m = MagicMock()
                m.name = "input"
                m.shape = [1, 256, 256, 3]
                return [m]

            def get_providers(self):
                return ["CPUExecutionProvider"]

            def run(self, _outs, feeds):
                x = feeds["input"]
                assert x.shape == (1, 256, 256, 3)
                # Return mid mask
                return [np.full((1, 256, 256, 1), 0.75, dtype=np.float32)]

        oc._session = FakeSession()
        oc._input_name = "input"
        oc._input_size = (256, 256)
        oc._nhwc = True

        crop = np.random.randint(0, 255, (80, 60, 3), dtype=np.uint8)
        mask = oc.face_mask(crop)
        assert mask.shape == (80, 60)
        assert 0.7 < float(mask.mean()) < 0.8


class TestSeamlessXSegConfig:
    def test_seamless_enables_xseg(self):
        pipe = FaceSwapConfig(quality="seamless").to_pipeline_config()
        assert pipe.use_xseg_occlusion is True
        assert pipe.swap_model == "hyperswap"

    def test_default_device_auto(self):
        assert FaceSwapConfig().device == "auto"
