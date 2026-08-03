"""
A/B metrics batch: compare restore methods × pixel-boost on live pairs.

Reuses one FaceSwapPipeline (swapper stays loaded) and hot-swaps the enhancer
to avoid DirectML multi-session crashes.

Usage:
  .\\.venv\\Scripts\\python.exe scripts/metrics_batch.py
  .\\.venv\\Scripts\\python.exe scripts/metrics_batch.py --pairs 2 --boosts 512 --enhance gfpgan,gpen
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2

from face_swap.api import FaceSwapConfig
from face_swap.core.metrics import (
    DEFAULT_VARIANTS,
    MetricsAnalyzer,
    VariantResult,
    VariantSpec,
    expand_variant_grid,
    summarize_variant_rows,
)
from face_swap.enhancement import EnhancementConfig, create_enhancer
from face_swap.pipeline import FaceSwapPipeline


def _discover_portraits(folders: List[Path]) -> List[Path]:
    skip_prefixes = ("fixed_", "seamless_", "v2_", "bench_", "video_", "swapped")
    out: List[Path] = []
    for folder in folders:
        if not folder.is_dir():
            continue
        for p in sorted(folder.glob("*")):
            if p.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
                continue
            if "_on_" in p.stem:
                continue
            if p.stem.startswith(skip_prefixes) or p.stem in ("two_faces",):
                continue
            out.append(p)
    return out


def _default_pairs(portraits: List[Path], limit: int) -> List[Tuple[Path, Path, str]]:
    """Curated pairs when available; otherwise first N sequential pairs."""
    by_stem = {p.stem: p for p in portraits}
    curated = [
        ("woman1", "man1"),
        ("woman3", "lena"),
        ("man2", "woman2"),
        ("man1", "lena"),
    ]
    pairs: List[Tuple[Path, Path, str]] = []
    for src, tgt in curated:
        if src in by_stem and tgt in by_stem:
            pairs.append((by_stem[src], by_stem[tgt], f"{src}_on_{tgt}"))
        if len(pairs) >= limit:
            return pairs[:limit]

    # Fallback: consecutive portraits
    for i in range(0, max(0, len(portraits) - 1), 2):
        a, b = portraits[i], portraits[i + 1]
        pairs.append((a, b, f"{a.stem}_on_{b.stem}"))
        if len(pairs) >= limit:
            break
    return pairs[:limit]


def _enhancer_device(pipeline_device: str) -> str:
    """
    Hot-swapping multiple ORT restore sessions on DirectML is unstable.

    Keep the swapper on DML/CUDA; run GFPGAN/GPEN/CodeFormer on CPU EP.
    """
    d = (pipeline_device or "auto").lower()
    if d in ("dml", "directml", "amd", "auto"):
        return "cpu"
    return pipeline_device


def _set_enhancer(pipe: FaceSwapPipeline, method: str, boost: int, device: str) -> None:
    cfg = pipe.config
    cfg.enhance_method = method
    cfg.enhance_target_px = max(0, int(boost))
    cfg.enable_enhance = True
    enh_device = _enhancer_device(device)
    enh_cfg = EnhancementConfig(
        enabled=True,
        method=method,
        device=enh_device,
        upscale=1,
        blend_weight=float(getattr(cfg, "enhance_blend", 0.7) or 0.7),
        target_face_px=cfg.enhance_target_px,
        quality=float(getattr(cfg, "enhance_fidelity", 0.5) or 0.5),
    )
    # Drop previous session before loading another ORT graph
    pipe._enhancer = None
    enhancer = create_enhancer(enh_cfg)
    try:
        enhancer.load_model()
    except Exception as e:
        print(f"  warn: {method} load failed ({e}); using opencv", flush=True)
        enh_cfg.method = "opencv"
        enhancer = create_enhancer(enh_cfg)
        enhancer.load_model()
    pipe._enhancer = enhancer


def main() -> None:
    ap = argparse.ArgumentParser(description="A/B face-swap metrics batch")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--pairs", type=int, default=2, help="Number of image pairs")
    ap.add_argument(
        "--enhance",
        default="gfpgan,gpen,opencv",
        help="Comma list of enhance methods",
    )
    ap.add_argument(
        "--boosts",
        default="512,1024",
        help="Comma list of pixel-boost sizes (0 = off)",
    )
    ap.add_argument(
        "--defaults",
        action="store_true",
        help="Use DEFAULT_VARIANTS instead of --enhance/--boosts grid",
    )
    ap.add_argument(
        "--out",
        default="data/samples/metrics_batch",
        help="Output directory for images + CSV/JSON",
    )
    ap.add_argument("--min-id", type=float, default=0.25)
    args = ap.parse_args()

    out_root = Path(args.out)
    img_dir = out_root / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    folders = [
        Path("data/samples/live_test"),
        Path("data/samples/live_test_v2"),
    ]
    portraits = _discover_portraits(folders)
    pairs = _default_pairs(portraits, args.pairs)
    if not pairs:
        raise SystemExit("No portrait pairs found under data/samples/live_test*")

    if args.defaults:
        variants = list(DEFAULT_VARIANTS)
    else:
        enhances = [x.strip() for x in args.enhance.split(",") if x.strip()]
        boosts = [int(x.strip()) for x in args.boosts.split(",") if x.strip()]
        # Avoid huge GPEN×1024 grid by default when user passes gpen — still ok
        variants = expand_variant_grid(enhances, boosts)

    print(f"Pairs ({len(pairs)}): {[p[2] for p in pairs]}", flush=True)
    print(f"Variants ({len(variants)}): {[v.name for v in variants]}", flush=True)

    # Base seamless pipeline once (HyperSwap); enhancer swapped per variant
    base_cfg = FaceSwapConfig(
        quality="seamless",
        device=args.device,
        enhance_method="opencv",
        pixel_boost=0,
    ).to_pipeline_config()
    # Start light; _set_enhancer replaces restore each cell
    base_cfg.enable_enhance = True
    base_cfg.enhance_method = "opencv"
    base_cfg.enhance_target_px = 0
    pipe = FaceSwapPipeline(base_cfg)
    analyzer = MetricsAnalyzer(device=args.device)

    rows: List[VariantResult] = []
    for src_path, tgt_path, pair_name in pairs:
        src = cv2.imread(str(src_path))
        tgt = cv2.imread(str(tgt_path))
        if src is None or tgt is None:
            print(f"skip unreadable pair {pair_name}", flush=True)
            continue
        print(f"\n=== pair {pair_name} ===", flush=True)
        emb = pipe.extract_source_embedding(src)

        for spec in variants:
            print(f"  {spec.name} ...", flush=True)
            _set_enhancer(pipe, spec.enhance_method, spec.pixel_boost, args.device)
            t0 = time.perf_counter()
            result = pipe.process_frame(tgt, emb)
            elapsed = time.perf_counter() - t0
            out_path = img_dir / f"{pair_name}__{spec.name}.jpg"
            cv2.imwrite(str(out_path), result)
            metrics = analyzer.evaluate(src, tgt, result, min_id_sim=args.min_id)
            row = VariantResult(
                variant=spec.name,
                pair=pair_name,
                metrics=metrics,
                elapsed_s=elapsed,
                output_path=str(out_path),
            )
            rows.append(row)
            print(
                f"    {metrics.summary_line()}  time={elapsed:.2f}s",
                flush=True,
            )

    summary = summarize_variant_rows(rows)
    report = {
        "pairs": [p[2] for p in pairs],
        "variants": [v.name for v in variants],
        "rows": [r.to_dict() for r in rows],
        "summary": summary,
    }
    json_path = out_root / "report.json"
    csv_path = out_root / "report.csv"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    fieldnames = [
        "variant",
        "pair",
        "id_similarity",
        "id_vs_target",
        "sharpness_result",
        "sharpness_gain",
        "color_delta_lab",
        "passed_id",
        "elapsed_s",
        "output_path",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r.to_dict())

    print("\n=== summary (sorted by id then sharpness) ===", flush=True)
    for s in summary:
        print(
            f"  {s['variant']:20s}  id={s['id_similarity_mean']:.3f}  "
            f"sharp_d={s['sharpness_gain_mean']:+.1f}  "
            f"dE={s['color_delta_lab_mean']:.1f}  "
            f"t={s['elapsed_s_mean']:.2f}s  pass={s['passed_id_rate']:.0%}",
            flush=True,
        )
    print(f"\nWrote {json_path}", flush=True)
    print(f"Wrote {csv_path}", flush=True)


if __name__ == "__main__":
    main()
