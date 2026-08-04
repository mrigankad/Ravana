"""Tests for seamless quality preset and enhance_face_region boost/blend."""

import numpy as np

from face_swap.api import FaceSwapConfig
from face_swap.enhancement import (
    EnhancementConfig,
    OpenCVEnhancer,
    enhance_face_region,
)


class TestSeamlessPreset:
    def test_seamless_pipeline_fields(self):
        cfg = FaceSwapConfig(quality="seamless", device="cpu")
        pipe = cfg.to_pipeline_config()
        assert pipe.enhance_method == "gfpgan"
        assert pipe.enhance_blend == 0.70
        assert pipe.enhance_target_px == 1024
        assert pipe.use_occlusion_mask is True
        assert pipe.enable_enhance is True
        assert pipe.enable_color_match is True
        assert pipe.crop_size == 128
        assert pipe.color_match_strength == 1.0
        assert pipe.preserve_lower_face == "auto"
        assert pipe.enable_grain_match is True
        assert pipe.use_xseg_occlusion is True
        assert pipe.enable_lighting_match is True
        assert pipe.lighting_match_strength > 0
        assert pipe.swap_model == "hyperswap"
        assert pipe.video_detect_every_n == 2
        assert pipe.video_flow_blend > 0

    def test_high_unchanged(self):
        pipe = FaceSwapConfig(quality="high").to_pipeline_config()
        assert pipe.enhance_method == "opencv"
        assert pipe.enhance_blend == 1.0
        assert pipe.enhance_target_px == 0
        assert pipe.use_occlusion_mask is False


class TestEnhanceFaceRegion:
    def _frame_and_bbox(self):
        frame = np.full((120, 120, 3), 40, dtype=np.uint8)
        # Bright face-like patch
        frame[30:90, 30:90] = 180
        bbox = (30, 30, 90, 90)
        return frame, bbox

    def test_blend_weight_zero_keeps_crop(self):
        frame, bbox = self._frame_and_bbox()
        enh = OpenCVEnhancer(EnhancementConfig(method="opencv", enabled=True))
        enh.load_model()
        out = enhance_face_region(
            frame, bbox, enh, feather=5, blend_weight=0.0, target_face_px=0
        )
        x1, y1, x2, y2 = bbox
        # Center of face region should stay close to original (no restore mix)
        assert np.allclose(out[60, 60], frame[60, 60], atol=8)

    def test_blend_weight_one_changes_face(self):
        frame, bbox = self._frame_and_bbox()
        enh = OpenCVEnhancer(EnhancementConfig(method="opencv", enabled=True))
        enh.load_model()
        out = enhance_face_region(
            frame, bbox, enh, feather=5, blend_weight=1.0, target_face_px=0
        )
        # OpenCV enhancer alters interior pixels
        assert (
            not np.array_equal(out[60, 60], frame[60, 60]) or out.mean() != frame.mean()
        )

    def test_target_px_boost_preserves_frame_shape(self):
        frame, bbox = self._frame_and_bbox()
        enh = OpenCVEnhancer(EnhancementConfig(method="opencv", enabled=True))
        enh.load_model()
        out = enhance_face_region(
            frame,
            bbox,
            enh,
            feather=5,
            blend_weight=0.85,
            target_face_px=128,
        )
        assert out.shape == frame.shape
        assert out.dtype == np.uint8

    def test_region_mask_reduces_edge_effect(self):
        frame, bbox = self._frame_and_bbox()
        x1, y1, x2, y2 = 30, 30, 90, 90
        pad = int(0.08 * 60)
        px1, py1 = max(0, x1 - pad), max(0, y1 - pad)
        px2, py2 = min(120, x2 + pad), min(120, y2 + pad)
        rh, rw = py2 - py1, px2 - px1

        # Mask only center; edges of crop should stay closer to original
        region_mask = np.zeros((rh, rw), dtype=np.float32)
        cy, cx = rh // 2, rw // 2
        region_mask[cy - 10 : cy + 10, cx - 10 : cx + 10] = 1.0

        enh = OpenCVEnhancer(EnhancementConfig(method="opencv", enabled=True))
        enh.load_model()

        full = enhance_face_region(
            frame, bbox, enh, feather=3, blend_weight=1.0, region_mask=None
        )
        gated = enhance_face_region(
            frame, bbox, enh, feather=3, blend_weight=1.0, region_mask=region_mask
        )

        # Corner of padded crop: gated should be closer to original than full enhance
        corner = (py1 + 2, px1 + 2)
        d_full = np.abs(
            full[corner].astype(np.float32) - frame[corner].astype(np.float32)
        ).mean()
        d_gated = np.abs(
            gated[corner].astype(np.float32) - frame[corner].astype(np.float32)
        ).mean()
        assert d_gated <= d_full + 1e-3


class TestPixelBoostTiles:
    def test_pixel_boost_override(self):
        pipe = FaceSwapConfig(quality="seamless", pixel_boost=512).to_pipeline_config()
        assert pipe.enhance_target_px == 512
        pipe0 = FaceSwapConfig(quality="seamless", pixel_boost=0).to_pipeline_config()
        assert pipe0.enhance_target_px == 0

    def test_enhance_with_tiles_covers_large_image(self):
        from face_swap.enhancement import enhance_with_tiles

        class _MarkEnhancer:
            FACE_SIZE = 64

            def enhance(self, face, upscale=1):
                out = face.copy()
                out[:] = (out.astype(np.int16) + 10).clip(0, 255).astype(np.uint8)
                return out

        img = np.full((120, 100, 3), 50, dtype=np.uint8)
        out = enhance_with_tiles(img, _MarkEnhancer(), tile_size=64, overlap=16)
        assert out.shape == img.shape
        # Interior should be brightened by every overlapping tile path
        assert int(out[60, 50].mean()) > int(img[60, 50].mean())

    def test_tile_starts_cover_edge(self):
        from face_swap.enhancement.enhancer import _tile_starts

        starts = _tile_starts(100, 64, 16)
        assert starts[0] == 0
        assert starts[-1] == 100 - 64
        assert all(s + 64 <= 100 for s in starts)
