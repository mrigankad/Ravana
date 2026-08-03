"""
Post-swap quality metrics: identity match, sharpness, color drift.

Complements the frame-level QualityValidator gate with measurable scores
for A/B comparisons (GFPGAN vs GPEN, pixel-boost levels, etc.).
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from .adaptive import cosine_similarity, select_faces
from .providers import resolve_ort_providers

logger = logging.getLogger("face_swap.metrics")

PathOrArray = Union[str, np.ndarray]


@dataclass
class SwapMetrics:
    """Scores for one source→target swap evaluation."""

    id_similarity: float = 0.0  # ArcFace cosine: source vs result face
    id_vs_target: float = 0.0  # ArcFace cosine: result vs original target face
    sharpness_result: float = 0.0  # Laplacian variance on result face crop
    sharpness_target: float = 0.0  # Same on original target face
    sharpness_gain: float = 0.0  # result - target
    color_delta_lab: float = 0.0  # mean |LAB| on face crop vs target
    faces_source: int = 0
    faces_target: int = 0
    faces_result: int = 0
    passed_id: bool = False
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def summary_line(self) -> str:
        return (
            f"id={self.id_similarity:.3f} (vs_tgt={self.id_vs_target:.3f})  "
            f"sharp={self.sharpness_result:.1f} (d{self.sharpness_gain:+.1f})  "
            f"dE={self.color_delta_lab:.1f}  "
            f"{'PASS' if self.passed_id else 'WEAK'} id"
        )


def laplacian_sharpness(
    image_bgr: np.ndarray, bbox: Optional[Sequence[float]] = None
) -> float:
    """Laplacian variance; higher = sharper. Optional bbox crop with pad."""
    if image_bgr is None or image_bgr.size == 0:
        return 0.0
    crop = image_bgr
    if bbox is not None:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = image_bgr.shape[:2]
        pad = int(0.05 * max(x2 - x1, y2 - y1))
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
        if x2 <= x1 + 4 or y2 <= y1 + 4:
            return 0.0
        crop = image_bgr[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def mean_lab_delta(
    a_bgr: np.ndarray,
    b_bgr: np.ndarray,
    bbox: Optional[Sequence[float]] = None,
) -> float:
    """Mean absolute LAB difference on aligned crops (rough color drift)."""
    if a_bgr is None or b_bgr is None or a_bgr.size == 0 or b_bgr.size == 0:
        return 0.0
    if bbox is not None:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = a_bgr.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 + 4 or y2 <= y1 + 4:
            return 0.0
        a_bgr = a_bgr[y1:y2, x1:x2]
        b_bgr = b_bgr[y1:y2, x1:x2]
    if a_bgr.shape[:2] != b_bgr.shape[:2]:
        b_bgr = cv2.resize(b_bgr, (a_bgr.shape[1], a_bgr.shape[0]))
    a = cv2.cvtColor(a_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    b = cv2.cvtColor(b_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    return float(np.mean(np.abs(a - b)))


def _load_bgr(img: PathOrArray) -> np.ndarray:
    if isinstance(img, str):
        out = cv2.imread(img)
        if out is None:
            raise ValueError(f"Could not load image: {img}")
        return out
    return img


def _face_embedding(face: Any) -> Optional[np.ndarray]:
    emb = getattr(face, "normed_embedding", None)
    if emb is None:
        emb = getattr(face, "embedding", None)
    if emb is None:
        return None
    return np.asarray(emb, dtype=np.float32).ravel()


def _bbox_of(face: Any) -> Optional[Tuple[float, float, float, float]]:
    box = getattr(face, "bbox", None)
    if box is None:
        return None
    return tuple(float(v) for v in box[:4])  # type: ignore[return-value]


class MetricsAnalyzer:
    """
    FaceAnalysis-backed metric extractor (buffalo_l).

    Reuse one instance across many evaluate() calls to avoid reload cost.
    """

    def __init__(self, device: str = "auto", det_size: Tuple[int, int] = (640, 640)):
        self.device = device
        self.det_size = det_size
        self._app = None

    def _ensure_app(self) -> None:
        if self._app is not None:
            return
        from insightface.app import FaceAnalysis

        providers = resolve_ort_providers(self.device)
        self._app = FaceAnalysis(name="buffalo_l", root="./models", providers=providers)
        # DirectML / CUDA ctx_id 0; CPU -1
        ctx = -1 if "CPUExecutionProvider" in providers and len(providers) == 1 else 0
        self._app.prepare(ctx_id=ctx, det_size=self.det_size)

    def detect(self, image_bgr: np.ndarray) -> List[Any]:
        self._ensure_app()
        assert self._app is not None
        faces = self._app.get(image_bgr)
        return list(faces or [])

    def evaluate(
        self,
        source: PathOrArray,
        target: PathOrArray,
        result: PathOrArray,
        *,
        min_id_sim: float = 0.25,
        face_select: str = "largest",
    ) -> SwapMetrics:
        """
        Score a completed swap.

        ``id_similarity`` — source identity vs result face (want high).
        ``id_vs_target`` — result vs original target (often lower after strong swap).
        """
        src = _load_bgr(source)
        tgt = _load_bgr(target)
        res = _load_bgr(result)

        src_faces = select_faces(self.detect(src), mode=face_select)
        tgt_faces = select_faces(self.detect(tgt), mode=face_select)
        res_faces = select_faces(self.detect(res), mode=face_select)

        metrics = SwapMetrics(
            faces_source=len(src_faces),
            faces_target=len(tgt_faces),
            faces_result=len(res_faces),
        )

        if not src_faces or not res_faces:
            metrics.message = "Missing face on source or result"
            return metrics

        src_emb = _face_embedding(src_faces[0])
        res_emb = _face_embedding(res_faces[0])
        if src_emb is None or res_emb is None:
            metrics.message = "Missing embeddings"
            return metrics

        metrics.id_similarity = cosine_similarity(src_emb, res_emb)
        metrics.passed_id = metrics.id_similarity >= min_id_sim

        if tgt_faces:
            tgt_emb = _face_embedding(tgt_faces[0])
            if tgt_emb is not None:
                metrics.id_vs_target = cosine_similarity(res_emb, tgt_emb)

        res_bbox = _bbox_of(res_faces[0])
        tgt_bbox = _bbox_of(tgt_faces[0]) if tgt_faces else res_bbox

        metrics.sharpness_result = laplacian_sharpness(res, res_bbox)
        metrics.sharpness_target = laplacian_sharpness(tgt, tgt_bbox)
        metrics.sharpness_gain = metrics.sharpness_result - metrics.sharpness_target

        # Color drift on result bbox vs original target (same box)
        if res_bbox is not None:
            metrics.color_delta_lab = mean_lab_delta(res, tgt, res_bbox)

        metrics.message = metrics.summary_line()
        return metrics


def evaluate_swap(
    source: PathOrArray,
    target: PathOrArray,
    result: PathOrArray,
    *,
    device: str = "auto",
    min_id_sim: float = 0.25,
) -> SwapMetrics:
    """One-shot evaluate (loads FaceAnalysis once)."""
    return MetricsAnalyzer(device=device).evaluate(
        source, target, result, min_id_sim=min_id_sim
    )


@dataclass
class VariantSpec:
    """One cell in an A/B metrics matrix."""

    name: str
    enhance_method: str = "gfpgan"
    pixel_boost: int = 512
    quality: str = "seamless"


@dataclass
class VariantResult:
    variant: str
    pair: str
    metrics: SwapMetrics
    elapsed_s: float = 0.0
    output_path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = self.metrics.to_dict()
        d.update(
            {
                "variant": self.variant,
                "pair": self.pair,
                "elapsed_s": round(self.elapsed_s, 3),
                "output_path": self.output_path,
            }
        )
        return d


DEFAULT_VARIANTS: List[VariantSpec] = [
    VariantSpec("gfpgan_512", "gfpgan", 512),
    VariantSpec("gfpgan_1024", "gfpgan", 1024),
    VariantSpec("gpen_512", "gpen", 512),
    VariantSpec("codeformer_512", "codeformer", 512),
    VariantSpec("opencv_0", "opencv", 0),
]


def expand_variant_grid(
    enhances: Sequence[str],
    boosts: Sequence[int],
    quality: str = "seamless",
) -> List[VariantSpec]:
    """Build VariantSpec list from enhance × boost axes."""
    out: List[VariantSpec] = []
    for method in enhances:
        for boost in boosts:
            name = f"{method}_{boost}"
            out.append(
                VariantSpec(
                    name=name,
                    enhance_method=method,
                    pixel_boost=int(boost),
                    quality=quality,
                )
            )
    return out


def summarize_variant_rows(rows: Sequence[VariantResult]) -> List[Dict[str, Any]]:
    """Average metrics per variant name across pairs."""
    by: Dict[str, List[VariantResult]] = {}
    for row in rows:
        by.setdefault(row.variant, []).append(row)
    summary = []
    for name, items in sorted(by.items()):
        n = len(items)
        summary.append(
            {
                "variant": name,
                "pairs": n,
                "id_similarity_mean": round(
                    sum(i.metrics.id_similarity for i in items) / n, 4
                ),
                "sharpness_gain_mean": round(
                    sum(i.metrics.sharpness_gain for i in items) / n, 2
                ),
                "color_delta_lab_mean": round(
                    sum(i.metrics.color_delta_lab for i in items) / n, 2
                ),
                "elapsed_s_mean": round(sum(i.elapsed_s for i in items) / n, 3),
                "passed_id_rate": round(
                    sum(1 for i in items if i.metrics.passed_id) / n, 3
                ),
            }
        )
    summary.sort(key=lambda r: (-r["id_similarity_mean"], -r["sharpness_gain_mean"]))
    return summary
