"""Pose / tiny-face adaptive helpers."""

import numpy as np

from face_swap.core.adaptive import (
    pose_compatibility,
    rank_sources_by_pose,
    upscale_frame_for_tiny_faces,
    yaw_proxy_from_face,
)


class _Face:
    def __init__(self, bbox, kps):
        self.bbox = np.asarray(bbox, dtype=np.float32)
        self.kps = np.asarray(kps, dtype=np.float32)


class TestPoseMatch:
    def test_yaw_proxy_frontal_higher_than_profile(self):
        # Wide eye span → frontal
        frontal = _Face([0, 0, 100, 120], [[30, 40], [70, 40], [50, 60], [35, 80], [65, 80]])
        profile = _Face([0, 0, 100, 120], [[48, 40], [55, 40], [52, 60], [45, 80], [58, 80]])
        assert yaw_proxy_from_face(frontal) > yaw_proxy_from_face(profile)

    def test_pose_compatibility_prefers_match(self):
        a = _Face([0, 0, 100, 120], [[30, 40], [70, 40], [50, 60], [35, 80], [65, 80]])
        b = _Face([0, 0, 100, 120], [[32, 42], [68, 41], [50, 62], [36, 82], [64, 81]])
        c = _Face([0, 0, 100, 120], [[48, 40], [55, 40], [52, 60], [45, 80], [58, 80]])
        assert pose_compatibility(a, b) > pose_compatibility(a, c)

    def test_rank_sources(self):
        tgt = _Face([0, 0, 100, 120], [[30, 40], [70, 40], [50, 60], [35, 80], [65, 80]])
        good = _Face([0, 0, 100, 120], [[31, 40], [69, 40], [50, 60], [35, 80], [65, 80]])
        bad = _Face([0, 0, 100, 120], [[48, 40], [54, 40], [51, 60], [46, 80], [56, 80]])
        ranked = rank_sources_by_pose(tgt, [bad, good])
        assert ranked[0][0] == 1


class TestTinyFaceUpscale:
    def test_no_boost_when_large(self):
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        face = _Face([20, 20, 140, 160], [[50, 60], [110, 60], [80, 90], [55, 120], [105, 120]])
        out, faces, scale = upscale_frame_for_tiny_faces(img, [face], min_side_trigger=64)
        assert scale == 1.0
        assert out.shape == img.shape

    def test_boosts_tiny_face(self):
        img = np.zeros((200, 300, 3), dtype=np.uint8)
        # ~40px face
        face = _Face([100, 80, 140, 130], [[110, 95], [130, 95], [120, 105], [112, 118], [128, 118]])
        out, faces, scale = upscale_frame_for_tiny_faces(
            img, [face], ideal_face_px=128, max_upscale=4.0, min_side_trigger=64
        )
        assert scale > 1.5
        assert out.shape[0] > img.shape[0]
        assert abs(float(faces[0].bbox[2] - faces[0].bbox[0]) - 40 * scale) < 2
