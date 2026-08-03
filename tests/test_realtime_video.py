"""Tests for realtime / ROI video helpers."""

from unittest.mock import MagicMock

import numpy as np

from face_swap.api import FaceSwapConfig
from face_swap.core.fast_video import (
    FastVideoConfig,
    offset_faces,
    union_face_roi,
)


class _Face:
    def __init__(self, bbox, kps=None, score=0.99):
        self.bbox = np.asarray(bbox, dtype=np.float32)
        self.kps = None if kps is None else np.asarray(kps, dtype=np.float32)
        self.det_score = score


class TestUnionFaceRoi:
    def test_pads_and_clips(self):
        face = _Face([100, 100, 200, 220])
        roi = union_face_roi([face], (480, 640, 3), pad_frac=0.5)
        assert roi is not None
        x1, y1, x2, y2 = roi
        assert x1 < 100 and y1 < 100
        assert x2 > 200 and y2 > 220
        assert x1 >= 0 and y1 >= 0
        assert x2 <= 640 and y2 <= 480

    def test_empty_faces(self):
        assert union_face_roi([], (100, 100, 3)) is None


class TestOffsetFaces:
    def test_shifts_bbox_and_kps(self):
        face = _Face(
            [10, 20, 50, 60],
            kps=[[15, 25], [40, 25], [30, 35], [20, 50], [40, 50]],
        )
        out = offset_faces([face], 100, 50)
        assert np.allclose(out[0].bbox, [110, 70, 150, 110])
        assert np.allclose(out[0].kps[0], [115, 75])


class TestRealtimeConfig:
    def test_realtime_enables_fast_video(self):
        pipe = FaceSwapConfig(
            quality="medium", realtime=True, detect_every_n=4, device="auto"
        ).to_pipeline_config()
        assert pipe.fast_video is not None
        assert pipe.fast_video.enabled is True
        assert pipe.fast_video.detect_every_n == 4
        assert pipe.fast_video.roi_track is True
        assert pipe.fast_video.skip_enhance is True
        assert pipe.enable_enhance is False
        assert pipe.video_detect_every_n == 4

    def test_non_realtime_medium_unchanged(self):
        pipe = FaceSwapConfig(quality="medium", realtime=False).to_pipeline_config()
        assert pipe.enable_enhance is True
        assert pipe.enhance_method == "opencv"


class TestGetFacesVideoRoi:
    def test_interim_uses_roi_when_available(self):
        from face_swap.pipeline import FaceSwapPipeline, PipelineConfig

        cfg = PipelineConfig(
            device="cpu",
            use_native_inswapper=False,
            enable_enhance=False,
            fast_video=FastVideoConfig(
                enabled=True, detect_every_n=3, roi_track=True, max_faces=1
            ),
        )
        pipe = FaceSwapPipeline(cfg)
        pipe._initialized = True
        pipe._fast = cfg.fast_video

        cached = [_Face([40, 40, 120, 140])]
        pipe._cached_faces = cached
        pipe._cache_frame_idx = 0

        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        roi_hit = [_Face([5, 5, 80, 90])]

        pipe._get_faces = MagicMock(return_value=[_Face([1, 1, 2, 2])])
        pipe._detect_faces_in_roi = MagicMock(return_value=roi_hit)

        # Frame 1 is interim (detect every 3, last full at 0)
        out = pipe._get_faces_video(frame, frame_number=1)
        pipe._detect_faces_in_roi.assert_called_once()
        pipe._get_faces.assert_not_called()
        assert out[0].bbox[0] == 5

        # Frame 3 triggers full detect again
        pipe._get_faces_video(frame, frame_number=3)
        pipe._get_faces.assert_called()
