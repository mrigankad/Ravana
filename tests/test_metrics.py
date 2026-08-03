"""Unit tests for swap quality metrics (no FaceAnalysis required)."""

from unittest.mock import MagicMock, patch

import numpy as np
import cv2

from face_swap.core.metrics import (
    SwapMetrics,
    laplacian_sharpness,
    mean_lab_delta,
    evaluate_swap,
)


class TestLaplacianSharpness:
    def test_blur_lower_than_sharp(self):
        sharp = np.zeros((64, 64), dtype=np.uint8)
        sharp[::2, ::2] = 255
        blur = cv2.GaussianBlur(sharp, (9, 9), 0)
        assert laplacian_sharpness(cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)) > (
            laplacian_sharpness(cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR))
        )

    def test_bbox_crop(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[40:60, 40:60] = 200
        s = laplacian_sharpness(img, bbox=(40, 40, 60, 60))
        assert s >= 0.0


class TestLabDelta:
    def test_identical_zero(self):
        a = np.full((40, 40, 3), 120, dtype=np.uint8)
        assert mean_lab_delta(a, a.copy()) < 1e-3

    def test_different_positive(self):
        a = np.full((40, 40, 3), 50, dtype=np.uint8)
        b = np.full((40, 40, 3), 200, dtype=np.uint8)
        assert mean_lab_delta(a, b) > 10


class TestSwapMetrics:
    def test_summary_line(self):
        m = SwapMetrics(id_similarity=0.5, sharpness_result=40.0, passed_id=True)
        line = m.summary_line()
        assert "0.500" in line
        assert "PASS" in line

    def test_to_dict(self):
        m = SwapMetrics(id_similarity=0.4)
        d = m.to_dict()
        assert d["id_similarity"] == 0.4


class TestEvaluateMocked:
    def test_evaluate_swap_with_mock_faces(self):
        src = np.full((120, 120, 3), 80, dtype=np.uint8)
        tgt = np.full((120, 120, 3), 90, dtype=np.uint8)
        res = np.full((120, 120, 3), 100, dtype=np.uint8)

        emb_src = np.zeros(512, dtype=np.float32)
        emb_src[0] = 1.0
        emb_res = emb_src.copy()
        emb_tgt = np.zeros(512, dtype=np.float32)
        emb_tgt[1] = 1.0

        def fake_face(emb, box=(20, 20, 100, 100)):
            f = MagicMock()
            f.normed_embedding = emb
            f.embedding = emb
            f.bbox = np.array(box, dtype=np.float32)
            return f

        analyzer = MagicMock()
        analyzer.evaluate = MagicMock(
            wraps=None,
        )

        from face_swap.core.metrics import MetricsAnalyzer

        with patch.object(MetricsAnalyzer, "_ensure_app", lambda self: None), patch.object(
            MetricsAnalyzer,
            "detect",
            side_effect=[
                [fake_face(emb_src)],
                [fake_face(emb_tgt)],
                [fake_face(emb_res)],
            ],
        ):
            metrics = MetricsAnalyzer(device="cpu").evaluate(src, tgt, res)

        assert metrics.id_similarity > 0.99
        assert metrics.passed_id
        assert metrics.faces_source == 1
        assert metrics.faces_result == 1
