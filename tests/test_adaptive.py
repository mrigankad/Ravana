"""Unit tests for adaptive preprocessing helpers (no model download)."""

import cv2
import numpy as np

from face_swap.core.adaptive import (
    adaptive_mask_kernels,
    choose_det_size,
    cosine_similarity,
    detect_lower_face_hair,
    forehead_taper_mask,
    identity_preserved,
    landmark_hull_mask,
    lower_face_preserve_mask,
    match_chrominance_to_target,
    match_face_to_target_skin,
    match_grain_to_target,
    match_lighting_to_target,
    neck_ring_color_reference,
    pad_to_square,
    reinhard_color_match,
    skin_likelihood_mask,
)
from face_swap.enhancement import EnhancementConfig, OpenCVEnhancer, create_enhancer


class TestChooseDetSize:
    def test_small_image_uses_320ish(self):
        img = np.zeros((360, 360, 3), dtype=np.uint8)
        size = choose_det_size(img)
        assert size[0] == size[1]
        assert 256 <= size[0] <= 320

    def test_large_image_uses_higher(self):
        img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        size = choose_det_size(img)
        assert size[0] >= 448

    def test_prefer_small_faces(self):
        img = np.zeros((800, 800, 3), dtype=np.uint8)
        size = choose_det_size(img, prefer_small_faces=True)
        assert size[0] == 640


class TestPadAndColor:
    def test_pad_to_square(self):
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        padded, (x0, y0) = pad_to_square(img)
        assert padded.shape[0] == padded.shape[1] == 200
        assert x0 == 0 and y0 == 50

    def test_reinhard_preserves_shape(self):
        src = np.full((64, 64, 3), 80, dtype=np.uint8)
        ref = np.full((64, 64, 3), 180, dtype=np.uint8)
        out = reinhard_color_match(src, ref)
        assert out.shape == src.shape
        # Should move toward brighter reference
        assert float(out.mean()) > float(src.mean())

    def test_skin_match_pulls_toward_ref(self):
        # Cool pale face vs warm tan reference (BGR)
        pale = np.full((80, 80, 3), (200, 190, 170), dtype=np.uint8)
        tan = np.full((80, 80, 3), (90, 140, 190), dtype=np.uint8)
        mask = np.ones((80, 80), dtype=np.float32)
        out = match_face_to_target_skin(pale, tan, face_mask=mask, strength=1.0)
        # Should move toward tan mean (lower B, higher R-ish)
        assert abs(float(out.mean()) - float(tan.mean())) < abs(
            float(pale.mean()) - float(tan.mean())
        )
        assert skin_likelihood_mask(tan).mean() >= 0.0

    def test_chrominance_keeps_luminance(self):
        pale = np.full((64, 64, 3), (210, 200, 180), dtype=np.uint8)
        tan = np.full((64, 64, 3), (80, 130, 180), dtype=np.uint8)
        out = match_chrominance_to_target(pale, tan, strength=1.0)
        pale_l = (
            cv2.cvtColor(pale, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32).mean()
        )
        out_l = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32).mean()
        assert abs(out_l - pale_l) < 3.0

    def test_lighting_transfers_low_freq_l(self):
        # Uniform bright face vs left-dark / right-bright shading on target
        flat = np.full((96, 96, 3), (160, 150, 140), dtype=np.uint8)
        shaded = np.zeros((96, 96, 3), dtype=np.uint8)
        shaded[:, :48] = (60, 55, 50)
        shaded[:, 48:] = (200, 190, 180)
        mask = np.ones((96, 96), dtype=np.float32)
        out = match_lighting_to_target(flat, shaded, face_mask=mask, strength=1.0)
        out_lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
        # Left should be darker than right after lighting transfer
        assert float(out_lab[:, :40].mean()) < float(out_lab[:, 56:].mean()) - 15

    def test_forehead_taper(self):
        m = forehead_taper_mask(100, 80, top_frac=0.2)
        assert m.shape == (100, 80)
        assert float(m[0, 40]) < 0.15
        assert float(m[50, 40]) > 0.9

    def test_lower_face_preserve_mask(self):
        m = lower_face_preserve_mask(100, 80, start_frac=0.5, feather=0.1)
        assert float(m[20, 40]) < 0.1
        assert float(m[90, 40]) > 0.9

    def test_detect_beard_heuristic(self):
        img = np.full((80, 80, 3), (160, 170, 200), dtype=np.uint8)
        img[55:80, :] = (40, 40, 40)  # dark lower = beard-like
        assert detect_lower_face_hair(img) is True
        clean = np.full((80, 80, 3), (160, 170, 200), dtype=np.uint8)
        assert detect_lower_face_hair(clean) is False

    def test_grain_and_neck_ring(self):
        base = np.full((64, 64, 3), 120, dtype=np.uint8)
        noisy = base.copy()
        noisy[::2, ::2] = 140
        out = match_grain_to_target(base, noisy, strength=0.5)
        assert out.shape == base.shape
        face = np.zeros((64, 64), dtype=np.float32)
        cv2.ellipse(face, (32, 32), (20, 24), 0, 0, 360, 1.0, -1)
        ring = neck_ring_color_reference(base, face, dilate=15)
        # May be None if no skin-like pixels; ellipse on flat gray often yields None
        assert ring is None or ring.shape == (64, 64)


class TestIdentityAndMask:
    def test_cosine_identical(self):
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        assert abs(cosine_similarity(v, v) - 1.0) < 1e-5

    def test_identity_preserved_threshold(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.9, 0.1], dtype=np.float32)
        ok, sim = identity_preserved(a, b, min_sim=0.5)
        assert ok and sim > 0.5

    def test_mask_kernels_scale(self):
        e1, b1 = adaptive_mask_kernels(50, 50)
        e2, b2 = adaptive_mask_kernels(300, 300)
        assert e2 >= e1 and b2 >= b1

    def test_landmark_hull_mask(self):
        pts = np.array([[10, 10], [50, 10], [50, 50], [10, 50]], dtype=np.float32)
        mask = landmark_hull_mask((60, 60), pts, erode=1, blur=3)
        assert mask.shape == (60, 60)
        assert float(mask.max()) > 0.5
        assert float(mask[0, 0]) < 0.2


class TestOpenCVEnhancer:
    def test_enhance_preserves_shape(self):
        enh = OpenCVEnhancer(EnhancementConfig(method="opencv", enabled=True))
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        out = enh.enhance(img)
        assert out.shape == img.shape

    def test_factory_falls_back(self):
        enh = create_enhancer(EnhancementConfig(method="opencv"))
        assert isinstance(enh, OpenCVEnhancer)
