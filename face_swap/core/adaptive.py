"""
Adaptive preprocessing, auto-tuning, and quality checks for variable inputs.

Implements industry best practices drawn from InsightFace / FaceSwapLab /
FaceDancer / production swap pipelines:

1. **Adaptive det_size** — scale detection resolution to image size and
   face scale (InsightFace #2512, FaceSwapLab auto_det_size).
2. **Detection retry ladder** — 640 → pad → 512 → 320 → 256 when no face
   found (ComfyUI IP-Adapter / FaceSwapLab strategy).
3. **Source face upscale** — boost tiny portraits before embedding so
   ArcFace / InSwapper get enough pixels.
4. **Pre/post quality gates** — det_score, face area, sharpness, ArcFace
   identity cosine (FaceDancer / DiffSwap metrics).
5. **Reinhard LAB color match** — match swapped crop stats to target
   (Reinhard et al. 2001; FaceSwap color_transfer).
6. **Size-adaptive paste mask** — erode/blur kernels scale with face
   bbox (official InSwapper paste-back).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import cv2
import numpy as np

logger = logging.getLogger("face_swap.adaptive")

# Detection size ladder used when auto-retrying (largest → smallest).
DET_SIZE_LADDER: Tuple[Tuple[int, int], ...] = (
    (640, 640),
    (512, 512),
    (448, 448),
    (320, 320),
    (256, 256),
)


@dataclass
class FaceQuality:
    """Pre-swap assessment of a detected face."""

    ok: bool
    det_score: float
    face_w: int
    face_h: int
    area_ratio: float
    sharpness: float
    yaw_proxy: float  # |dx| of eye centers / face width (rough pose)
    reasons: List[str]

    @property
    def min_side(self) -> int:
        return min(self.face_w, self.face_h)


@dataclass
class AdaptiveConfig:
    """Tunables for adaptive preprocessing."""

    min_det_score: float = 0.45
    min_face_px: int = 48
    ideal_face_px: int = 128
    min_area_ratio: float = 0.002  # face / image area
    max_area_ratio: float = 0.85
    min_sharpness: float = 20.0
    max_yaw_proxy: float = 0.55
    id_similarity_min: float = 0.25  # post-swap ArcFace cosine floor
    enable_color_match: bool = True
    enable_source_upscale: bool = True
    max_upscale: float = 3.0
    auto_det_size: bool = True


def choose_det_size(
    image: np.ndarray, prefer_small_faces: bool = False
) -> Tuple[int, int]:
    """
    Pick a detection window from image dimensions.

    Rules of thumb (InsightFace maintainers + FaceSwapLab):
      - Large close-up faces → smaller det_size (~320) works better
      - Distant / multi-face scenes → larger det_size (~640)
      - Very small images → clamp to image side (min 256)
    """
    h, w = image.shape[:2]
    short = min(h, w)
    long = max(h, w)

    if prefer_small_faces or long > 1600:
        target = 640
    elif short < 400:
        # Small images / big faces relative to frame
        target = 320
    elif short < 720:
        target = 448
    else:
        target = 512

    # Never ask detector for more than the image provides (plus modest pad room)
    target = int(max(256, min(target, short)))
    # SCRFD prefers multiples of 32
    target = (target // 32) * 32
    return (target, target)


def pad_to_square(
    image: np.ndarray, pad_value: int = 0
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Pad to square; returns (padded, (pad_x, pad_y) top-left offset)."""
    h, w = image.shape[:2]
    side = max(h, w)
    canvas = np.full((side, side, 3), pad_value, dtype=image.dtype)
    y0 = (side - h) // 2
    x0 = (side - w) // 2
    canvas[y0 : y0 + h, x0 : x0 + w] = image
    return canvas, (x0, y0)


def _shift_face(face: Any, dx: int, dy: int) -> Any:
    """Translate face bbox/kps after undoing padding."""
    face.bbox = face.bbox.copy()
    face.bbox[0] -= dx
    face.bbox[2] -= dx
    face.bbox[1] -= dy
    face.bbox[3] -= dy
    if getattr(face, "kps", None) is not None:
        face.kps = face.kps.copy()
        face.kps[:, 0] -= dx
        face.kps[:, 1] -= dy
    return face


def detect_faces_adaptive(
    face_app: Any,
    image: np.ndarray,
    config: Optional[AdaptiveConfig] = None,
) -> List[Any]:
    """
    Detect faces with auto det_size + retry ladder.

    Tries the recommended size first, then pads-to-square, then walks
    the DET_SIZE_LADDER downward until a face is found.
    """
    cfg = config or AdaptiveConfig()
    primary = choose_det_size(image) if cfg.auto_det_size else (640, 640)

    attempts: List[Tuple[str, Tuple[int, int], np.ndarray, Tuple[int, int]]] = [
        ("primary", primary, image, (0, 0)),
    ]

    padded, offset = pad_to_square(image)
    if padded.shape[:2] != image.shape[:2]:
        attempts.append(("padded", primary, padded, offset))

    for size in DET_SIZE_LADDER:
        if size != primary:
            attempts.append((f"retry-{size[0]}", size, image, (0, 0)))

    seen = set()
    last_size = getattr(face_app, "_ravana_det_size", None)
    # Prefer last successful size first (video / batch speed)
    if last_size and last_size != primary:
        attempts.insert(0, ("cached", last_size, image, (0, 0)))

    for label, det_size, img, (dx, dy) in attempts:
        key = (label, det_size)
        if key in seen:
            continue
        seen.add(key)

        if getattr(face_app, "_ravana_det_size", None) != det_size:
            try:
                ctx = getattr(face_app, "ctx_id", -1)
                face_app.prepare(ctx_id=ctx, det_size=det_size)
                face_app._ravana_det_size = det_size
            except Exception:
                try:
                    face_app.prepare(ctx_id=-1, det_size=det_size)
                    face_app._ravana_det_size = det_size
                except Exception as e:
                    logger.debug("prepare(%s) failed: %s", det_size, e)
                    continue

        faces = face_app.get(img) or []
        if not faces:
            logger.debug("No faces with det_size=%s (%s)", det_size, label)
            continue

        if dx or dy:
            faces = [_shift_face(f, dx, dy) for f in faces]

        face_app._ravana_det_size = det_size
        logger.debug(
            "Detected %d face(s) via %s det_size=%s", len(faces), label, det_size
        )
        return faces

    return []


def face_sharpness(image: np.ndarray, bbox: Sequence[float]) -> float:
    """Laplacian variance inside bbox — higher = sharper."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 + 2 or y2 <= y1 + 2:
        return 0.0
    crop = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def assess_face(
    face: Any,
    image: np.ndarray,
    config: Optional[AdaptiveConfig] = None,
) -> FaceQuality:
    """Pre-swap quality gate for a single InsightFace Face."""
    cfg = config or AdaptiveConfig()
    h, w = image.shape[:2]
    bbox = face.bbox.astype(np.float32)
    fw = int(max(1, bbox[2] - bbox[0]))
    fh = int(max(1, bbox[3] - bbox[1]))
    area_ratio = (fw * fh) / float(h * w)
    score = float(getattr(face, "det_score", 0.0))
    sharp = face_sharpness(image, bbox)

    yaw = 0.0
    kps = getattr(face, "kps", None)
    if kps is not None and len(kps) >= 2:
        # InsightFace kps: 0=left eye, 1=right eye
        eye_dx = abs(float(kps[1][0] - kps[0][0]))
        yaw = eye_dx / max(fw, 1)

    reasons: List[str] = []
    if score < cfg.min_det_score:
        reasons.append(f"low_det_score({score:.2f})")
    if min(fw, fh) < cfg.min_face_px:
        reasons.append(f"face_too_small({fw}x{fh})")
    if area_ratio < cfg.min_area_ratio:
        reasons.append(f"area_ratio_low({area_ratio:.4f})")
    if area_ratio > cfg.max_area_ratio:
        reasons.append(f"area_ratio_high({area_ratio:.4f})")
    if sharp < cfg.min_sharpness:
        reasons.append(f"blurry({sharp:.1f})")
    if yaw > 0 and yaw < 0.15:
        # eyes almost stacked → extreme profile
        reasons.append(f"extreme_pose(yaw_proxy={yaw:.2f})")

    return FaceQuality(
        ok=len(reasons) == 0,
        det_score=score,
        face_w=fw,
        face_h=fh,
        area_ratio=area_ratio,
        sharpness=sharp,
        yaw_proxy=yaw,
        reasons=reasons,
    )


def yaw_proxy_from_face(face: Any) -> float:
    """
    Eye-span / face-width proxy for yaw (InsightFace 5-point kps).

    Near 0.45–0.55 ≈ frontal; lower values ≈ profile. Returns 0 if unknown.
    """
    bbox = getattr(face, "bbox", None)
    kps = getattr(face, "kps", None)
    if bbox is None or kps is None or len(kps) < 2:
        return 0.0
    fw = float(max(1.0, bbox[2] - bbox[0]))
    eye_dx = abs(float(kps[1][0] - kps[0][0]))
    return eye_dx / fw


def pose_compatibility(source_face: Any, target_face: Any) -> float:
    """
    Score how well source pose matches target (1 = great, 0 = poor).

    Uses yaw_proxy difference. Prefer frontal pairs; penalize source-profile
    onto frontal target (and vice versa).
    """
    ys = yaw_proxy_from_face(source_face)
    yt = yaw_proxy_from_face(target_face)
    if ys <= 0 or yt <= 0:
        return 0.5  # unknown — neutral
    diff = abs(ys - yt)
    # Typical frontal ~0.45; allow ±0.12 as excellent
    score = float(np.clip(1.0 - diff / 0.35, 0.0, 1.0))
    return score


def rank_sources_by_pose(
    target_face: Any,
    source_faces: Sequence[Any],
) -> List[Tuple[int, float]]:
    """
    Rank source face indices by pose match to ``target_face``.

    Returns list of (index, compatibility) sorted best-first.
    """
    scored = [
        (i, pose_compatibility(src, target_face)) for i, src in enumerate(source_faces)
    ]
    scored.sort(key=lambda t: t[1], reverse=True)
    return scored


def face_area(face: Any) -> float:
    """Bounding-box area in pixels."""
    bbox = getattr(face, "bbox", None)
    if bbox is None:
        return 0.0
    return float(max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1]))


def select_faces(
    faces: Sequence[Any],
    mode: str = "all",
    index: int = 0,
    max_faces: int = 0,
    source_face: Any = None,
) -> List[Any]:
    """
    Choose which detected faces to swap.

    Modes:
      - ``all``: keep all (optionally truncated by ``max_faces``)
      - ``largest``: single biggest bbox
      - ``first``: highest det_score (assumes pre-sorted) / first item
      - ``index``: ``faces[index]`` (clamped)
      - ``pose``: best pose match vs ``source_face`` (falls back to largest)
    """
    if not faces:
        return []

    mode = (mode or "all").lower().strip()
    items = list(faces)

    if mode == "largest":
        items = sorted(items, key=face_area, reverse=True)[:1]
    elif mode == "first":
        items = items[:1]
    elif mode == "index":
        i = int(index)
        if i < 0:
            i = len(items) + i
        i = int(np.clip(i, 0, len(items) - 1))
        items = [items[i]]
    elif mode == "pose":
        if source_face is None:
            items = sorted(items, key=face_area, reverse=True)[:1]
        else:
            ranked = sorted(
                items,
                key=lambda f: pose_compatibility(source_face, f),
                reverse=True,
            )
            items = ranked[:1]
    # else "all" — keep order

    if max_faces and max_faces > 0:
        items = items[: int(max_faces)]
    return items


def upscale_frame_for_tiny_faces(
    image: np.ndarray,
    faces: Sequence[Any],
    ideal_face_px: int = 128,
    max_upscale: float = 4.0,
    min_side_trigger: int = 64,
) -> Tuple[np.ndarray, List[Any], float]:
    """
    Upscale a target frame when the largest face is below ``min_side_trigger``.

    HyperSwap/InSwapper struggle below ~64px; Messi-style full-body shots
    need a temporary boost. Caller should resize the result back by 1/scale
    and preferably re-detect faces on the upscaled frame.
    """
    if image is None or image.size == 0 or not faces:
        return image, list(faces), 1.0

    sides = []
    for f in faces:
        bw = float(f.bbox[2] - f.bbox[0])
        bh = float(f.bbox[3] - f.bbox[1])
        sides.append(min(bw, bh))
    side = max(sides) if sides else 0.0
    if side >= float(min_side_trigger):
        return image, list(faces), 1.0

    scale = min(float(max_upscale), float(ideal_face_px) / max(side, 1.0))
    if scale <= 1.05:
        return image, list(faces), 1.0

    new_w = int(round(image.shape[1] * scale))
    new_h = int(round(image.shape[0] * scale))
    scaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    scaled_faces: List[Any] = []
    for f in faces:
        try:
            from insightface.app.common import Face

            nf = Face(dict(f))
        except Exception:
            nf = f
        nf.bbox = np.asarray(f.bbox, dtype=np.float32) * scale
        if getattr(f, "kps", None) is not None:
            nf.kps = np.asarray(f.kps, dtype=np.float32) * scale
        scaled_faces.append(nf)

    logger.debug(
        "Upscaled target frame ×%.2f for tiny face (~%dpx → ~%dpx)",
        scale,
        int(side),
        int(side * scale),
    )
    return scaled, scaled_faces, float(scale)


def upscale_if_small_face(
    image: np.ndarray,
    face: Any,
    config: Optional[AdaptiveConfig] = None,
) -> Tuple[np.ndarray, Any, float]:
    """
    Upscale image so the detected face reaches ``ideal_face_px``.

    Returns (image, face_with_scaled_coords, scale_used).
    """
    cfg = config or AdaptiveConfig()
    if not cfg.enable_source_upscale:
        return image, face, 1.0

    fw = float(face.bbox[2] - face.bbox[0])
    fh = float(face.bbox[3] - face.bbox[1])
    side = min(fw, fh)
    if side >= cfg.ideal_face_px:
        return image, face, 1.0

    scale = min(cfg.max_upscale, cfg.ideal_face_px / max(side, 1.0))
    if scale <= 1.05:
        return image, face, 1.0

    new_w = int(image.shape[1] * scale)
    new_h = int(image.shape[0] * scale)
    scaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    face = face  # mutate a shallow copy of geometry
    face.bbox = face.bbox.astype(np.float32) * scale
    if getattr(face, "kps", None) is not None:
        face.kps = face.kps.astype(np.float32) * scale

    logger.debug(
        "Upscaled source face ×%.2f (%dpx → ~%dpx)", scale, int(side), int(side * scale)
    )
    return scaled, face, float(scale)


def reinhard_color_match(
    source_bgr: np.ndarray,
    reference_bgr: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Reinhard et al. color transfer in LAB space.

    Matches mean/std of ``source_bgr`` to ``reference_bgr``.
    Optional mask (H×W float 0-1) limits statistics to the face region.
    """
    src = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    if mask is not None:
        m = mask.astype(np.float32)
        if m.ndim == 3:
            m = m[:, :, 0]
        m = (m > 0.1).astype(bool)
        if m.sum() < 50:
            m = None
    else:
        m = None

    out = src.copy()
    for c in range(3):
        s = src[:, :, c]
        r = ref[:, :, c]
        if m is not None:
            s_mean, s_std = float(s[m].mean()), float(s[m].std() + 1e-6)
            r_mean, r_std = float(r[m].mean()), float(r[m].std() + 1e-6)
        else:
            s_mean, s_std = float(s.mean()), float(s.std() + 1e-6)
            r_mean, r_std = float(r.mean()), float(r.std() + 1e-6)
        out[:, :, c] = (s - s_mean) * (r_std / s_std) + r_mean

    out = np.clip(out, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_LAB2BGR)


def skin_likelihood_mask(bgr: np.ndarray) -> np.ndarray:
    """
    Soft skin-tone likelihood in YCrCb (excludes dark hair / bright bg).

    Returns float mask H×W in [0, 1].
    """
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    # Classic skin range (tolerant)
    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)
    hard = cv2.inRange(ycrcb, lower, upper)
    # Soften + fill small holes
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    hard = cv2.morphologyEx(hard, cv2.MORPH_CLOSE, k)
    soft = cv2.GaussianBlur(hard, (11, 11), 0).astype(np.float32) / 255.0
    return soft


def match_face_to_target_skin(
    swapped_bgr: np.ndarray,
    original_bgr: np.ndarray,
    face_mask: Optional[np.ndarray] = None,
    strength: float = 1.0,
) -> np.ndarray:
    """
    Strong skin color transfer: Reinhard on skin-like pixels, then blend.

    Uses original target skin (inside face_mask ∩ skin) as reference so
    the swapped face matches neck / ear tone instead of a full-bbox average
    that includes hair and background.
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength < 1e-3:
        return swapped_bgr

    skin_ref = skin_likelihood_mask(original_bgr)
    skin_src = skin_likelihood_mask(swapped_bgr)
    stats = np.minimum(skin_ref, skin_src)
    if face_mask is not None:
        fm = face_mask.astype(np.float32)
        if fm.ndim == 3:
            fm = fm[:, :, 0]
        stats = stats * fm

    if float(stats.sum()) < 80:
        # Fallback: whole face mask or full image
        stats = face_mask if face_mask is not None else None
        matched = reinhard_color_match(swapped_bgr, original_bgr, mask=stats)
    else:
        matched = reinhard_color_match(swapped_bgr, original_bgr, mask=stats)
        # Second pass on a*b* only using skin stats (keeps luminance detail)
        matched = reinhard_color_match(matched, original_bgr, mask=stats)

    if strength >= 0.999:
        return matched
    a = strength
    return (
        a * matched.astype(np.float32) + (1.0 - a) * swapped_bgr.astype(np.float32)
    ).astype(np.uint8)


def match_chrominance_to_target(
    swapped_bgr: np.ndarray,
    original_bgr: np.ndarray,
    face_mask: Optional[np.ndarray] = None,
    strength: float = 1.0,
    stats_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Transfer only LAB a*/b* (color) from target; keep L from swapped/enhanced.

    Used after GFPGAN so restore detail/sharpness is preserved while skin
    hue/saturation re-aligns to the target neck/face.

    ``stats_mask`` (optional) selects pixels on the *reference* for mean/std
    (e.g. neck ring). Application still uses ``face_mask`` when blending.
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength < 1e-3:
        return swapped_bgr

    src = cv2.cvtColor(swapped_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    def _bool_mask(raw, fallback_skin: bool = True):
        if raw is None:
            if fallback_skin:
                skin = skin_likelihood_mask(original_bgr)
                use = skin > 0.15
                if int(use.sum()) >= 50:
                    return use
            return None
        m = raw.astype(np.float32)
        if m.ndim == 3:
            m = m[:, :, 0]
        use = m > 0.15
        if int(use.sum()) < 50 and fallback_skin:
            skin = skin_likelihood_mask(original_bgr)
            use = (m > 0.05) & (skin > 0.15)
        return use if int(use.sum()) >= 50 else None

    ref_use = _bool_mask(stats_mask if stats_mask is not None else face_mask)
    src_use = _bool_mask(face_mask)

    out = src.copy()
    for c in (1, 2):  # a*, b* only
        s = src[:, :, c]
        r = ref[:, :, c]
        if ref_use is not None:
            r_mean, r_std = float(r[ref_use].mean()), float(r[ref_use].std() + 1e-6)
        else:
            r_mean, r_std = float(r.mean()), float(r.std() + 1e-6)
        if src_use is not None:
            s_mean, s_std = float(s[src_use].mean()), float(s[src_use].std() + 1e-6)
        else:
            s_mean, s_std = float(s.mean()), float(s.std() + 1e-6)
        transferred = (s - s_mean) * (r_std / s_std) + r_mean
        out[:, :, c] = (1.0 - strength) * s + strength * transferred

    out = np.clip(out, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_LAB2BGR)


def forehead_taper_mask(h: int, w: int, top_frac: float = 0.22) -> np.ndarray:
    """
    Soft vertical fade that zeros the top of a face crop (hairline / hat).

    Returns float HxW in [0, 1].
    """
    mask = np.ones((h, w), dtype=np.float32)
    band = max(1, int(h * top_frac))
    fade = np.linspace(0.0, 1.0, band, dtype=np.float32)
    mask[:band, :] *= fade[:, None]
    return mask


def lower_face_preserve_mask(
    h: int, w: int, start_frac: float = 0.52, feather: float = 0.12
) -> np.ndarray:
    """
    Soft mask that is 1 on the lower face (mouth/chin/beard zone).

    ``start_frac`` is where the fade begins (0=top). Used to blend the
    original target back so beards / jaw stay coherent.
    """
    mask = np.zeros((h, w), dtype=np.float32)
    y0 = int(np.clip(start_frac, 0.0, 0.9) * h)
    band = max(1, int(feather * h))
    # Ramp 0→1 over band, then solid 1
    for i in range(band):
        y = y0 + i
        if y >= h:
            break
        mask[y, :] = (i + 1) / float(band)
    if y0 + band < h:
        mask[y0 + band :, :] = 1.0
    # Horizontal soft edges
    side = max(1, w // 10)
    left = np.linspace(0.0, 1.0, side, dtype=np.float32)
    mask[:, :side] *= left[None, :]
    mask[:, -side:] *= left[::-1][None, :]
    return mask


def detect_lower_face_hair(
    original_bgr: np.ndarray, face_mask: Optional[np.ndarray] = None
) -> bool:
    """
    Heuristic: lower third much darker than mid-face → likely beard/stubble.
    """
    lab = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2LAB)
    lum = lab[:, :, 0].astype(np.float32)
    h, w = lum.shape
    if face_mask is not None:
        m = face_mask.astype(np.float32)
        if m.ndim == 3:
            m = m[:, :, 0]
        m = m > 0.2
    else:
        m = np.ones((h, w), dtype=bool)

    mid = np.zeros_like(m)
    low = np.zeros_like(m)
    mid[int(0.30 * h) : int(0.55 * h), :] = True
    low[int(0.60 * h) : int(0.95 * h), :] = True
    mid &= m
    low &= m
    if mid.sum() < 40 or low.sum() < 40:
        return False
    mid_l = float(lum[mid].mean())
    low_l = float(lum[low].mean())
    # Beard: lower zone ≥12 L darker
    return (mid_l - low_l) >= 12.0


def match_lighting_to_target(
    swapped_bgr: np.ndarray,
    original_bgr: np.ndarray,
    face_mask: Optional[np.ndarray] = None,
    strength: float = 0.75,
    blur_frac: float = 0.22,
) -> np.ndarray:
    """
    Transfer low-frequency shading / lighting from the target onto the swap.

    Keeps high-frequency identity detail from ``swapped_bgr`` (pores, edges)
    while replacing the blurred luminance field with the target's. This is the
    main remaining gap after Reinhard chrominance match — shadows on cheeks,
    forehead hotspots, and neck-adjacent brightness.
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength < 1e-3:
        return swapped_bgr
    if swapped_bgr.shape[:2] != original_bgr.shape[:2]:
        return swapped_bgr

    src = cv2.cvtColor(swapped_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    h, w = src.shape[:2]
    # Kernel ~ face fraction; odd, clamped
    k = int(max(h, w) * float(np.clip(blur_frac, 0.08, 0.45)))
    k = max(15, k | 1)  # odd
    if k % 2 == 0:
        k += 1
    k = min(k, (min(h, w) | 1))
    if k < 3:
        return swapped_bgr

    Ls = src[:, :, 0]
    Lr = ref[:, :, 0]
    Ls_low = cv2.GaussianBlur(Ls, (k, k), 0)
    Lr_low = cv2.GaussianBlur(Lr, (k, k), 0)
    detail = Ls - Ls_low
    # Mild gain so target shading isn't crushed when swap is much brighter/darker
    gain = (Lr_low + 1e-3) / (Ls_low + 1e-3)
    gain = np.clip(gain, 0.55, 1.75)
    L_lit = np.clip(Lr_low + detail * np.sqrt(gain), 0, 255)

    out = src.copy()
    out[:, :, 0] = (1.0 - strength) * Ls + strength * L_lit

    if face_mask is not None:
        m = face_mask.astype(np.float32)
        if m.ndim == 3:
            m = m[:, :, 0]
        # Soften mask slightly so lighting doesn't hard-cut at edges
        m = cv2.GaussianBlur(m, (0, 0), max(1.0, min(h, w) * 0.02))
        m = np.clip(m, 0.0, 1.0)
        m3 = m[:, :, None]
        out = m3 * out + (1.0 - m3) * src

    return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def match_grain_to_target(
    swapped_bgr: np.ndarray,
    original_bgr: np.ndarray,
    face_mask: Optional[np.ndarray] = None,
    strength: float = 0.35,
) -> np.ndarray:
    """
    Mild high-frequency grain transfer so restored faces match target film grain.

    Extracts residual of original L channel and adds a scaled copy onto swapped L.
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength < 1e-3:
        return swapped_bgr

    src = cv2.cvtColor(swapped_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref_blur = cv2.GaussianBlur(ref[:, :, 0], (0, 0), 1.2)
    grain = ref[:, :, 0] - ref_blur

    out = src.copy()
    out[:, :, 0] = np.clip(src[:, :, 0] + strength * grain, 0, 255)

    if face_mask is not None:
        m = face_mask.astype(np.float32)
        if m.ndim == 3:
            m = m[:, :, 0]
        m3 = m[:, :, None]
        blended = m3 * out + (1.0 - m3) * src
        out = blended

    return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def neck_ring_color_reference(
    original_bgr: np.ndarray,
    face_mask: np.ndarray,
    dilate: int = 25,
) -> Optional[np.ndarray]:
    """
    Soft mask for skin just outside the face hull (neck / ears / jaw sides).

    Better Reinhard reference than interior face when the swap already replaced
    cheek pixels. Returns float mask or None if too few pixels.
    """
    m = face_mask.astype(np.float32)
    if m.ndim == 3:
        m = m[:, :, 0]
    hard = (m > 0.4).astype(np.uint8) * 255
    k = dilate if dilate % 2 == 1 else dilate + 1
    dilated = cv2.dilate(hard, np.ones((k, k), np.uint8), iterations=1)
    ring = (dilated > 0).astype(np.float32) - (hard > 0).astype(np.float32)
    ring = np.clip(ring, 0.0, 1.0)
    skin = skin_likelihood_mask(original_bgr)
    ring = ring * skin
    if float(ring.sum()) < 60:
        return None
    return ring


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32).ravel()
    b = b.astype(np.float32).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def identity_preserved(
    source_embedding: np.ndarray,
    result_embedding: np.ndarray,
    min_sim: float = 0.25,
) -> Tuple[bool, float]:
    """Post-swap ArcFace cosine check (FaceDancer / DiffSwap style)."""
    sim = cosine_similarity(source_embedding, result_embedding)
    return sim >= min_sim, sim


def landmark_hull_mask(
    image_shape: Tuple[int, int],
    landmarks_xy: np.ndarray,
    erode: int = 3,
    blur: int = 15,
) -> np.ndarray:
    """
    Soft mask from a convex hull of facial landmarks (106/68/5-pt).

    Used by LivePortrait-style paste gating to keep hair/hands outside
    the swap region.
    """
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.asarray(landmarks_xy, dtype=np.int32)
    if pts.ndim != 2 or pts.shape[0] < 3:
        return mask.astype(np.float32)
    hull = cv2.convexHull(pts)
    cv2.fillConvexPoly(mask, hull, 255)
    if erode > 0:
        k = erode if erode % 2 == 1 else erode + 1
        mask = cv2.erode(mask, np.ones((k, k), np.uint8), iterations=1)
    if blur > 0:
        b = blur if blur % 2 == 1 else blur + 1
        mask = cv2.GaussianBlur(mask, (b, b), 0)
    return mask.astype(np.float32) / 255.0


def adaptive_mask_kernels(face_w: int, face_h: int) -> Tuple[int, int]:
    """
    Size-adaptive erode / blur kernels (InsightFace InSwapper paste-back).

    Returns (erode_k, blur_k) both odd-ish positive ints.
    """
    mask_size = int(np.sqrt(max(face_w, 1) * max(face_h, 1)))
    erode_k = max(mask_size // 10, 10)
    blur_k = max(mask_size // 20, 5)
    return erode_k, blur_k
