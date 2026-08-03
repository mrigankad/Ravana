"""
Live regression harness for Ravana seamless quality.

Usage:
  .\\.venv\\Scripts\\python.exe scripts/live_regression.py
  .\\.venv\\Scripts\\python.exe scripts/live_regression.py --pairs 8

Writes reports + side-by-side strips under data/samples/seamless_compare_live/.
Reuses one pipeline (avoids DirectML multi-session crashes).
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np

from face_swap.api import FaceSwapConfig
from face_swap.core.adaptive import pose_compatibility, yaw_proxy_from_face
from face_swap.pipeline import FaceSwapPipeline


def _fit(img, h=280):
    r = h / img.shape[0]
    return cv2.resize(img, (max(1, int(img.shape[1] * r)), h))


def _label(img, text):
    o = img.copy()
    cv2.rectangle(o, (0, 0), (o.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(
        o, text, (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA
    )
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quality", default="seamless")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--min-bytes", type=int, default=15000)
    ap.add_argument("--out", default="data/samples/seamless_compare_live")
    args = ap.parse_args()

    base = Path("data/samples")
    folders = [base / "live_test_v2", base / "live_test"]
    out_root = Path(args.out)
    for sub in ("seamless", "side_by_side", "reports"):
        (out_root / sub).mkdir(parents=True, exist_ok=True)

    candidates = []
    for folder in folders:
        if not folder.exists():
            continue
        for p in sorted(folder.glob("*")):
            if p.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
                continue
            if folder.name == "live_test_v2" and p.stat().st_size < args.min_bytes:
                continue
            # Skip previous swap outputs
            if "_on_" in p.stem or p.stem.startswith(("fixed_", "seamless_", "v2_")):
                continue
            candidates.append(p)

    pipe = FaceSwapPipeline(
        FaceSwapConfig(quality=args.quality, device=args.device).to_pipeline_config()
    )
    pipe.initialize()
    fa = pipe._ensure_face_app()

    usable = {}
    detect_rows = []
    for p in candidates:
        img = cv2.imread(str(p))
        if img is None:
            continue
        faces = fa.get(img) or []
        if not faces:
            detect_rows.append({"file": p.name, "ok": False})
            continue
        f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
        detect_rows.append(
            {
                "file": p.name,
                "ok": True,
                "face_w": float(f.bbox[2] - f.bbox[0]),
                "face_h": float(f.bbox[3] - f.bbox[1]),
                "yaw_proxy": yaw_proxy_from_face(f),
                "score": float(f.det_score),
            }
        )
        usable[p.name] = (img, f)

    # Default pairs: classics + pose-ranked pravatar
    pairs = []
    for a, b in [
        ("woman1.jpg", "man1.jpg"),
        ("woman3.jpg", "lena.jpg"),
        ("man2.jpg", "woman2.jpg"),
        ("lena.jpg", "messi.jpg"),
        ("woman1.jpg", "messi.jpg"),
        ("man3.jpg", "woman1.jpg"),
    ]:
        if a in usable and b in usable:
            pairs.append((a, b))

    pv = sorted(n for n in usable if n.startswith("pv_"))
    for i in range(0, min(len(pv) - 1, 8), 2):
        pairs.append((pv[i], pv[i + 1]))

    # Best pose match among new faces onto man1 / lena
    for tgt_name in ("man1.jpg", "lena.jpg"):
        if tgt_name not in usable or not pv:
            continue
        tgt_face = usable[tgt_name][1]
        ranked = sorted(
            pv,
            key=lambda n: pose_compatibility(usable[n][1], tgt_face),
            reverse=True,
        )
        pairs.append((ranked[0], tgt_name))

    # Dedupe
    seen = set()
    uniq = []
    for a, b in pairs:
        if (a, b) not in seen and a != b:
            uniq.append((a, b))
            seen.add((a, b))
    pairs = uniq

    rows = []
    print(f"usable={len(usable)} pairs={len(pairs)}", flush=True)
    for src_name, tgt_name in pairs:
        tag = f"{Path(src_name).stem}_on_{Path(tgt_name).stem}"
        src, src_f = usable[src_name]
        tgt, tgt_f = usable[tgt_name]
        pose = pose_compatibility(src_f, tgt_f)
        try:
            t0 = time.time()
            emb = pipe.extract_source_embedding(src)
            out = pipe.process_frame(tgt, emb)
            dt = time.time() - t0
            diff = float(np.abs(out.astype(np.float32) - tgt.astype(np.float32)).mean())
            faces_out = len(fa.get(out) or [])
            row = {
                "tag": tag,
                "src": src_name,
                "tgt": tgt_name,
                "sec": round(dt, 2),
                "mean_diff": round(diff, 2),
                "faces_out": faces_out,
                "pose": round(pose, 3),
                "ok": True,
                "error": "",
            }
            cv2.imwrite(str(out_root / "seamless" / f"{tag}.jpg"), out)
            strip = np.hstack(
                [
                    _label(_fit(src), "source"),
                    _label(_fit(tgt), "original"),
                    _label(_fit(out), "seamless"),
                ]
            )
            cv2.imwrite(str(out_root / "side_by_side" / f"{tag}.jpg"), strip)
            print(
                f"OK {tag} {dt:.1f}s diff={diff:.1f} pose={pose:.2f} faces={faces_out}",
                flush=True,
            )
        except Exception as e:
            row = {
                "tag": tag,
                "src": src_name,
                "tgt": tgt_name,
                "sec": 0,
                "mean_diff": 0,
                "faces_out": 0,
                "pose": round(pose, 3),
                "ok": False,
                "error": str(e),
            }
            print(f"FAIL {tag}: {e}", flush=True)
        rows.append(row)

    json.dump(detect_rows, open(out_root / "reports" / "detect.json", "w"), indent=2)
    json.dump(rows, open(out_root / "reports" / "swaps.json", "w"), indent=2)
    with open(out_root / "reports" / "swaps.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    ok = [r for r in rows if r["ok"]]
    print(
        f"SUMMARY {len(ok)}/{len(rows)} avg_sec={np.mean([r['sec'] for r in ok] or [0]):.2f} "
        f"avg_diff={np.mean([r['mean_diff'] for r in ok] or [0]):.2f}",
        flush=True,
    )
    pipe.cleanup()


if __name__ == "__main__":
    main()
