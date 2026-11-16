"""P0 engine fixes: tiles, ROI paste-back, unicode IO, ffmpeg cmd, One-Euro, cache."""

import numpy as np
import pytest

from face_swap import api
from face_swap.api import FaceSwapConfig, build_ffmpeg_mux_cmd, clear_cache
from face_swap.core.io import imread_unicode, imwrite_unicode
from face_swap.enhancement.enhancer import _tile_starts
from face_swap.swap.hyperswap import _paste_back_insightface
from face_swap.temporal.one_euro import OneEuroFilter, OneEuroTracks
from face_swap.temporal.optical_flow import OpticalFlowConfig, OpticalFlowSmoother
from face_swap.temporal.smoother import ema_smooth_insight_faces


class TestTileStarts:
    @pytest.mark.parametrize("length", [1, 100, 512])
    def test_small(self, length):
        assert _tile_starts(length, 512, 64) == [0]

    def test_exact_multiple_no_overlap(self):
        assert _tile_starts(1024, 512, 0) == [0, 512]

    def test_1024_512_64(self):
        s = _tile_starts(1024, 512, 64)
        assert len(s) == 3 and s == sorted(set(s)) and s[0] == 0 and s[-1] == 512

    @pytest.mark.parametrize("length", [513, 700, 1000, 1024, 1500, 2048, 4097])
    def test_coverage_unique_overlap(self, length):
        s = _tile_starts(length, 512, 64)
        assert s == sorted(set(s))
        assert s[-1] + 512 == length
        for a, b in zip(s, s[1:]):
            assert b - a <= 512 - 64  # consecutive tiles overlap >= 64


class TestRoiPasteBack:
    def test_only_roi_changes(self):
        frame = np.full((400, 500, 3), 50, dtype=np.uint8)
        aimg = np.full((128, 128, 3), 50, dtype=np.uint8)
        fake = np.full((128, 128, 3), 220, dtype=np.uint8)
        # identity-scale affine mapping frame (200,150) to crop origin
        M = np.array([[1, 0, -200], [0, 1, -150]], dtype=np.float64)
        out = _paste_back_insightface(frame, fake, aimg, M)
        changed = (out != frame).any(axis=2)
        ys, xs = np.where(changed)
        assert changed.any()
        assert xs.min() >= 198 and xs.max() <= 200 + 130
        assert ys.min() >= 148 and ys.max() <= 150 + 130
        assert out[215, 265, 0] > 200  # interior fully swapped
        assert out[0, 0, 0] == 50
        assert frame[215, 265, 0] == 50  # input not mutated

    def test_matches_fullframe_reference(self):
        import cv2

        rng = np.random.default_rng(0)
        frame = rng.integers(0, 255, (300, 300, 3), dtype=np.uint8)
        aimg = np.zeros((128, 128, 3), np.uint8)
        fake = rng.integers(0, 255, (128, 128, 3), dtype=np.uint8)
        M = np.array([[1, 0, -90], [0, 1, -80]], dtype=np.float64)
        out = _paste_back_insightface(frame, fake, aimg, M)
        IM = cv2.invertAffineTransform(M)
        h, w = frame.shape[:2]
        bw = cv2.warpAffine(fake, IM, (w, h))
        white = cv2.warpAffine(np.full((128, 128), 255, np.float32), IM, (w, h))
        white[white > 20] = 255
        ys, xs = np.where(white == 255)
        ms = int(np.sqrt(max(1, (ys.max() - ys.min()) * (xs.max() - xs.min()))))
        m = cv2.erode(white, np.ones((max(ms // 10, 10),) * 2, np.uint8))
        k = max(ms // 20, 5)
        m = cv2.GaussianBlur(m, (2 * k + 1, 2 * k + 1), 0)[:, :, None] / 255.0
        ref = (m * bw + (1 - m) * frame).astype(np.uint8)
        assert np.abs(out.astype(int) - ref.astype(int)).max() <= 3

    def test_offscreen_returns_frame(self):
        frame = np.zeros((50, 50, 3), np.uint8)
        M = np.array([[1, 0, 1000], [0, 1, 1000]], dtype=np.float64)
        out = _paste_back_insightface(
            frame, np.zeros((64, 64, 3), np.uint8), np.zeros((64, 64, 3), np.uint8), M
        )
        assert out.shape == frame.shape


class TestUnicodeIO:
    def test_roundtrip(self, tmp_path):
        d = tmp_path / "foto_日本語"
        d.mkdir()
        p = d / "ñandú.png"
        img = np.random.default_rng(1).integers(0, 255, (20, 30, 3), dtype=np.uint8)
        assert imwrite_unicode(p, img)
        back = imread_unicode(str(p))
        assert back is not None and np.array_equal(back, img)

    def test_missing_returns_none(self, tmp_path):
        assert imread_unicode(tmp_path / "nope.png") is None

    def test_api_error_message(self, tmp_path):
        with pytest.raises(ValueError, match="source image: .*nope"):
            api._load_image(str(tmp_path / "nope.png"), "source")


class TestFfmpegCmd:
    def test_command(self):
        cmd = build_ffmpeg_mux_cmd("v.mp4", "src.mp4", "out.mp4")
        assert cmd[0] == "ffmpeg" and cmd[-1] == "out.mp4"
        assert cmd.count("-i") == 2
        for tok in ["0:v", "1:a?", "libx264", "18", "aac", "-shortest"]:
            assert tok in cmd


class TestOneEuro:
    def test_constant_converges(self):
        f = OneEuroFilter()
        out = None
        for _ in range(50):
            out = f(np.array([10.0, 20.0]))
        assert np.allclose(out, [10, 20], atol=1e-4)

    def test_step_lags_less_than_ema(self):
        f = OneEuroFilter(min_cutoff=0.05, beta=0.05)
        tau = 1 / (2 * np.pi * 0.05)
        a = 1 / (1 + tau)  # EMA with the same resting smoothing
        f(np.array([0.0]))
        ema = 0.0
        err_f = err_e = 0.0
        for _ in range(10):
            y = float(f(np.array([100.0]))[0])
            ema = a * 100.0 + (1 - a) * ema
            err_f += abs(100 - y)
            err_e += abs(100 - ema)
        assert err_f < err_e

    def test_jitter_reduced(self):
        rng = np.random.default_rng(0)
        f = OneEuroFilter()
        noise = rng.normal(0, 1.0, 200)
        out = np.array([float(f(np.array([n]))[0]) for n in noise])
        assert out[50:].std() < noise[50:].std() * 0.6

    def test_tracks_independent(self):
        t = OneEuroTracks()
        t.filter(1, np.zeros(2))
        t.filter(2, np.full(2, 100.0))
        assert np.allclose(t.filter(1, np.zeros(2)), 0)
        t.prune([1])
        assert 2 not in t._filters


class _Face(dict):
    __getattr__ = dict.get

    def __setattr__(self, k, v):
        self[k] = v


class TestSmootherIntegration:
    def test_kps_one_euro_tracks(self):
        tracks = OneEuroTracks()
        bbox = np.array([10, 10, 60, 60], np.float32)
        kps = np.random.default_rng(0).uniform(10, 60, (5, 2)).astype(np.float32)
        prev = []
        for _ in range(3):
            faces = [_Face(bbox=bbox.copy(), kps=kps.copy())]
            prev = ema_smooth_insight_faces(faces, prev, kps_tracks=tracks)
        assert prev[0].track_id is not None
        assert np.allclose(prev[0].kps, kps, atol=1e-3)


class TestFlowRoi:
    def test_outside_roi_untouched(self):
        sm = OpticalFlowSmoother(OpticalFlowConfig(warp_blend=0.5))
        rng = np.random.default_rng(0)
        f0 = rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)
        f1 = rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)
        o0 = np.full_like(f0, 10)
        o1 = np.full_like(f0, 200)
        sm.smooth_frame_roi(o0, f0, (40, 30, 120, 90))
        out = sm.smooth_frame_roi(o1, f1, (40, 30, 120, 90))
        assert np.array_equal(out[:30], o1[:30])
        assert np.array_equal(out[:, :40], o1[:, :40])
        assert (out[60, 80] < 200).all()  # ROI centre blended with previous


class TestPipelineCache:
    def test_reuse_and_clear(self, monkeypatch):
        built = []

        class FakePipe:
            def __init__(self, cfg):
                built.append(cfg)

            def reset_temporal_state(self):
                pass

            def extract_source_embedding(self, img):
                return "emb"

            def process_frame(self, img, emb):
                return img

            def cleanup(self):
                pass

        monkeypatch.setattr(api, "FaceSwapPipeline", FakePipe)
        clear_cache()
        img = np.zeros((8, 8, 3), np.uint8)
        cfg = FaceSwapConfig(quality="low", device="cpu")
        api.swap_image(img, img, cfg)
        api.swap_image(img, img, cfg)
        assert len(built) == 1
        api.swap_image(img, img, FaceSwapConfig(quality="medium", device="cpu"))
        api.swap_image(img, img, FaceSwapConfig(quality="high", device="cpu"))
        assert len(api._cache) == 2  # LRU bounded
        clear_cache()
        assert len(api._cache) == 0
        api.swap_image(img, img, cfg)
        assert len(built) == 4


class TestLazyModular:
    def test_native_path_skips_modular(self):
        from face_swap.pipeline import FaceSwapPipeline

        p = FaceSwapPipeline(FaceSwapConfig(quality="low").to_pipeline_config())
        assert p._detector is None and p._embedder is None
