"""Tests for multi-face selection helpers."""

import numpy as np

from face_swap.api import FaceSwapConfig
from face_swap.core.adaptive import face_area, select_faces


class _Face:
    def __init__(self, bbox, kps=None, det_score=0.9):
        self.bbox = np.asarray(bbox, dtype=np.float32)
        self.kps = (
            np.asarray(kps, dtype=np.float32)
            if kps is not None
            else np.array(
                [[bbox[0] + 10, bbox[1] + 10], [bbox[2] - 10, bbox[1] + 10],
                 [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2],
                 [bbox[0] + 10, bbox[3] - 10], [bbox[2] - 10, bbox[3] - 10]],
                dtype=np.float32,
            )
        )
        self.det_score = det_score


class TestSelectFaces:
    def test_largest(self):
        small = _Face([0, 0, 40, 40])
        big = _Face([0, 0, 120, 140])
        out = select_faces([small, big], mode="largest")
        assert len(out) == 1
        assert face_area(out[0]) == face_area(big)

    def test_index(self):
        a = _Face([0, 0, 50, 50], det_score=0.99)
        b = _Face([10, 10, 80, 90], det_score=0.8)
        out = select_faces([a, b], mode="index", index=1)
        assert out[0] is b

    def test_all_with_max(self):
        faces = [_Face([0, 0, 30, 30]), _Face([0, 0, 40, 40]), _Face([0, 0, 50, 50])]
        out = select_faces(faces, mode="all", max_faces=2)
        assert len(out) == 2

    def test_pose_picks_compatible(self):
        # Frontal source
        src = _Face(
            [0, 0, 100, 120],
            [[30, 40], [70, 40], [50, 60], [35, 80], [65, 80]],
        )
        frontal = _Face(
            [0, 0, 100, 120],
            [[32, 42], [68, 41], [50, 62], [36, 82], [64, 81]],
        )
        profile = _Face(
            [0, 0, 100, 120],
            [[48, 40], [55, 40], [52, 60], [45, 80], [58, 80]],
        )
        out = select_faces([profile, frontal], mode="pose", source_face=src)
        assert out[0] is frontal


class TestFaceSelectConfig:
    def test_api_maps_face_select(self):
        pipe = FaceSwapConfig(
            quality="high", face_select="largest", face_index=2, max_faces=1
        ).to_pipeline_config()
        assert pipe.face_select == "largest"
        assert pipe.face_index == 2
        assert pipe.max_faces == 1
