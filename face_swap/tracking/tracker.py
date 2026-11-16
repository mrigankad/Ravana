"""Multi-face tracker: stable IDs across video frames.

Without it, face_selection "first" or "index" picks whichever face the detector
ranks first in this frame, so the swapped person can flip mid-clip.

Algorithm (SORT-style, no motion model needed at video frame rates):
  cost[i, j] = w_iou * (1 - IoU(track_i, det_j)) + w_app * cosine_distance(emb_i, emb_j)
  Hungarian assignment on the cost matrix (scipy linear_sum_assignment, O(n^3)
  but n is the face count, so microseconds), gated by IoU or appearance.
  Unmatched detections open tracks; tracks unseen for max_age frames close.

Track IDs follow first appearance, so index=0 means the first person who
appeared, which is what users expect.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment as _lsa
except Exception:  # pragma: no cover - scipy is a declared dependency
    _lsa = None

_BLOCK = 1e6


def iou_matrix(a, b) -> np.ndarray:
    """Pairwise IoU between (N,4) and (M,4) xyxy boxes, vectorised."""
    a = np.asarray(a, np.float32).reshape(-1, 4)
    b = np.asarray(b, np.float32).reshape(-1, 4)
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)), np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-6)


def _normed(e) -> Optional[np.ndarray]:
    if e is None:
        return None
    e = np.asarray(e, np.float32).ravel()
    return e / (np.linalg.norm(e) + 1e-8)


def _assign(cost: np.ndarray):
    if _lsa is not None:
        return _lsa(cost)
    rows, cols, used_r, used_c = [], [], set(), set()
    for flat in np.argsort(cost, axis=None):  # greedy fallback: cheapest pairs first
        r, c = divmod(int(flat), cost.shape[1])
        if r not in used_r and c not in used_c:
            rows.append(r)
            cols.append(c)
            used_r.add(r)
            used_c.add(c)
    return np.array(rows, int), np.array(cols, int)


@dataclass
class Track:
    id: int
    box: np.ndarray
    emb: Optional[np.ndarray] = None
    hits: int = 1
    age: int = 0  # frames since last match
    first_frame: int = 0
    det_index: int = -1  # index into the latest detections, -1 if unmatched
    extra: dict = field(default_factory=dict)


class FaceTrackSet:
    def __init__(
        self,
        max_age: int = 15,
        iou_gate: float = 0.2,
        app_gate: float = 0.55,
        w_iou: float = 0.5,
        w_app: float = 0.5,
        emb_momentum: float = 0.9,
    ):
        self.max_age = max_age
        self.iou_gate = iou_gate
        self.app_gate = app_gate  # max cosine distance for an appearance-only match
        self.w_iou, self.w_app = w_iou, w_app
        self.emb_momentum = emb_momentum
        self.tracks: List[Track] = []
        self._next_id = 0
        self.frame = 0

    def reset(self) -> None:
        self.tracks.clear()
        self._next_id = 0
        self.frame = 0

    def update(self, boxes: Sequence, embeddings: Optional[Sequence[Any]] = None) -> List[int]:
        """Match detections to tracks. Returns a track id per detection, in input order."""
        self.frame += 1
        boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
        n = len(boxes)
        embs = [_normed(e) for e in embeddings] if embeddings is not None else [None] * n
        for t in self.tracks:
            t.age += 1
            t.det_index = -1
        ids = [-1] * n

        if self.tracks and n:
            iou = iou_matrix(np.stack([t.box for t in self.tracks]), boxes)
            app = np.ones_like(iou)
            have = np.zeros(iou.shape, bool)
            for i, t in enumerate(self.tracks):
                if t.emb is None:
                    continue
                for j, e in enumerate(embs):
                    if e is not None:
                        app[i, j] = 1.0 - float(t.emb @ e)
                        have[i, j] = True
            cost = np.where(have, self.w_iou * (1.0 - iou) + self.w_app * app, 1.0 - iou)
            allowed = (iou >= self.iou_gate) | (have & (app <= self.app_gate))
            allowed &= ~(have & (app > 0.8))  # overlapping, but clearly another person
            cost = np.where(allowed, cost, _BLOCK)
            for r, c in zip(*_assign(cost)):
                if cost[r, c] >= _BLOCK:
                    continue
                t = self.tracks[r]
                t.box, t.hits, t.age, t.det_index = boxes[c], t.hits + 1, 0, int(c)
                if embs[c] is not None:
                    t.emb = embs[c] if t.emb is None else _normed(
                        self.emb_momentum * t.emb + (1 - self.emb_momentum) * embs[c])
                ids[c] = t.id

        for j in range(n):
            if ids[j] == -1:
                t = Track(self._next_id, boxes[j], embs[j], first_frame=self.frame, det_index=j)
                self._next_id += 1
                self.tracks.append(t)
                ids[j] = t.id
        self.tracks = [t for t in self.tracks if t.age <= self.max_age]
        return ids

    def tag_faces(self, faces: List[Any]) -> List[Any]:
        """Set face.track_id on InsightFace-style faces; return them sorted by track id."""
        if not faces:
            self.update([])
            return []
        boxes = [np.asarray(f.bbox, np.float32)[:4] for f in faces]
        embs = [getattr(f, "normed_embedding", None) for f in faces]
        ids = self.update(boxes, None if all(e is None for e in embs) else embs)
        for f, i in zip(faces, ids):
            try:
                f.track_id = i
            except Exception:
                pass
        return [faces[k] for k in np.argsort(ids, kind="stable")]
