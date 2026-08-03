"""Temporal face EMA + seamless video config tests."""

import numpy as np

from face_swap.api import FaceSwapConfig
from face_swap.temporal.smoother import ema_smooth_insight_faces


class _Face:
    def __init__(self, bbox, kps):
        self.bbox = np.asarray(bbox, dtype=np.float32)
        self.kps = np.asarray(kps, dtype=np.float32)


class TestEmaSmoothFaces:
    def test_first_frame_passthrough(self):
        f = _Face([10, 10, 50, 60], [[20, 25], [40, 25], [30, 35], [22, 50], [38, 50]])
        out = ema_smooth_insight_faces([f], None, alpha=0.35)
        assert len(out) == 1
        assert np.allclose(out[0].bbox, f.bbox)

    def test_ema_pulls_toward_previous(self):
        prev = _Face([0, 0, 100, 100], [[20, 30], [80, 30], [50, 50], [30, 80], [70, 80]])
        cur = _Face([20, 20, 120, 120], [[40, 50], [100, 50], [70, 70], [50, 100], [90, 100]])
        expected = 0.5 * cur.bbox + 0.5 * prev.bbox
        out = ema_smooth_insight_faces([cur], [prev], alpha=0.5)
        assert np.allclose(out[0].bbox, expected, atol=1e-4)


class TestSeamlessVideoTemporal:
    def test_seamless_sets_video_temporal_knobs(self):
        pipe = FaceSwapConfig(quality="seamless").to_pipeline_config()
        assert pipe.enable_temporal is True
        assert pipe.video_detect_every_n == 2
        assert pipe.video_flow_blend > 0
        assert 0 < pipe.video_face_ema_alpha < 1
