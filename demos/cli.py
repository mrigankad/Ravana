"""
Command-line interface for face swapping.

As per PRD Section 4.1, this provides CLI and minimal GUI for batch processing.
"""

import argparse
import glob
import os
import sys
from pathlib import Path
from typing import List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from face_swap import swap_image, swap_video, batch_swap, FaceSwapConfig
import cv2


def main():
    """Main entry point for CLI."""
    if len(sys.argv) > 1 and sys.argv[1] == "models":
        return models_main(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == "evaluate":
        return evaluate_main(sys.argv[2:])

    parser = argparse.ArgumentParser(
        description="Ravana - Real-time face swapping",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Swap face on a single image
  python -m demos.cli -s source.jpg -t target.jpg -o output.jpg

  # Best quality (HyperSwap + GFPGAN + XSeg); AMD/NVIDIA auto
  python -m demos.cli -s source.jpg -t target.jpg -o out.jpg -q seamless --device auto

  # Score identity + sharpness after a swap
  python -m demos.cli evaluate -s source.jpg -t target.jpg -o out.jpg -q seamless

  # Score an existing result without re-swapping
  python -m demos.cli evaluate -s source.jpg -t target.jpg -r result.jpg

  # Seamless with GPEN restore
  python -m demos.cli -s source.jpg -t target.jpg -o out.jpg -q seamless --enhance gpen

  # Pre-download seamless weights (progress bar)
  python -m demos.cli models download --preset seamless

  # List local model status
  python -m demos.cli models list

  # Swap face on a video
  python -m demos.cli -s source.jpg -t input.mp4 -o output.mp4

  # Batch process multiple images
  python -m demos.cli -s source.jpg --batch "images/*.jpg" -o ./output/

  # Real-time webcam demo
  python -m demos.cli -s source.jpg --webcam --camera 0
        """
    )
    
    # Input/output arguments
    parser.add_argument(
        "-s", "--source",
        required=True,
        help="Source image containing the face to swap"
    )
    parser.add_argument(
        "-t", "--target",
        help="Target image or video to swap face onto"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output file or directory"
    )
    
    # Mode arguments
    parser.add_argument(
        "--batch",
        help="Batch process files matching pattern (e.g., 'images/*.jpg')"
    )
    parser.add_argument(
        "--webcam",
        action="store_true",
        help="Start real-time webcam demo"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device ID (default: 0)"
    )
    
    # Quality arguments
    parser.add_argument(
        "-q", "--quality",
        choices=["low", "fast_cpu", "medium", "high", "seamless"],
        default="medium",
        help="Quality preset (seamless = HyperSwap+GFPGAN+XSeg; default: medium)"
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu", "dml", "auto"],
        default="auto",
        help="Device: cuda, cpu, dml (AMD DirectML), or auto (default)"
    )
    parser.add_argument(
        "--enhance",
        choices=["gfpgan", "gpen", "codeformer", "opencv"],
        default=None,
        help="Override face restore for seamless/high (gfpgan | gpen | codeformer | opencv)"
    )
    parser.add_argument(
        "--swapper",
        choices=["inswapper", "hyperswap"],
        default=None,
        help="Override swap model (seamless defaults to hyperswap)"
    )
    parser.add_argument(
        "--face",
        choices=["all", "largest", "first", "index", "pose"],
        default="all",
        help="Which target face(s) to swap (default: all)"
    )
    parser.add_argument(
        "--face-index",
        type=int,
        default=0,
        help="Face index when --face index (0 = highest conf)"
    )
    parser.add_argument(
        "--max-faces",
        type=int,
        default=0,
        help="Cap number of faces swapped (0 = unlimited)"
    )
    parser.add_argument(
        "--pixel-boost",
        type=int,
        default=None,
        metavar="PX",
        help="Face restore pixel boost side length (0=off; seamless default 1024)",
    )
    parser.add_argument(
        "--no-color-correction",
        action="store_true",
        help="Disable color correction"
    )
    parser.add_argument(
        "--no-smoothing",
        action="store_true",
        help="Disable temporal smoothing (for video)"
    )
    
    # Model arguments
    parser.add_argument(
        "--swap-model",
        help="Path to face swap ONNX model (optional)"
    )
    
    args = parser.parse_args()
    
    # Create config
    config = FaceSwapConfig(
        quality=args.quality,
        device=args.device,
        color_correction=not args.no_color_correction,
        enable_smoothing=not args.no_smoothing,
        swap_model_path=args.swap_model,
        enhance_method=args.enhance,
        swap_model=args.swapper,
        face_select=args.face,
        face_index=args.face_index,
        max_faces=args.max_faces,
        pixel_boost=args.pixel_boost,
    )
    
    try:
        if args.webcam:
            # Real-time webcam mode
            run_webcam(args.source, args.camera, config)
        
        elif args.batch:
            # Batch processing mode
            if not args.output:
                print("Error: --output required for batch processing")
                sys.exit(1)
            
            # Expand glob pattern
            target_files = glob.glob(args.batch)
            if not target_files:
                print(f"No files found matching pattern: {args.batch}")
                sys.exit(1)
            
            print(f"Batch processing {len(target_files)} files...")
            output_files = batch_swap(args.source, target_files, args.output, config)
            print(f"Completed. Output files saved to: {args.output}")
        
        elif args.target:
            # Single file mode
            if not args.output:
                print("Error: --output required")
                sys.exit(1)
            
            target_path = Path(args.target)
            
            if target_path.suffix.lower() in ['.mp4', '.mov', '.avi', '.mkv']:
                # Video mode
                print(f"Processing video: {args.target}")
                swap_video(
                    args.source,
                    args.target,
                    args.output,
                    config,
                    progress_callback=lambda idx, total: print(f"\rProgress: {idx}/{total} frames", end="")
                )
                print(f"\nOutput saved to: {args.output}")
            else:
                # Image mode
                print(f"Processing image: {args.target}")
                result = swap_image(args.source, args.target, config)
                cv2.imwrite(args.output, result)
                print(f"Output saved to: {args.output}")
        
        else:
            parser.print_help()
            sys.exit(1)
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def run_webcam(source_path: str, camera_id: int, config: FaceSwapConfig):
    """Run real-time webcam demo."""
    from face_swap import start_realtime_swap
    
    print(f"Starting webcam demo (Camera {camera_id})")
    print("Press 'q' in the video window to quit")
    
    start_realtime_swap(
        source_img=source_path,
        camera_id=camera_id,
        config=config
    )


def models_main(argv: List[str]) -> None:
    """``python -m demos.cli models …`` — list / download / ensure weights."""
    from face_swap.core.model_manager import MODEL_PRESETS, ModelManager

    parser = argparse.ArgumentParser(
        prog="python -m demos.cli models",
        description="Manage face-swap model weights",
    )
    parser.add_argument(
        "--models-dir",
        default="./models",
        help="Models directory (default: ./models)",
    )
    sub = parser.add_subparsers(dest="action", required=True)

    list_p = sub.add_parser("list", help="Show registered models and local status")
    list_p.add_argument(
        "--presets",
        action="store_true",
        help="Also print preset → model name mapping",
    )

    dl_p = sub.add_parser(
        "download",
        help="Download models by name and/or preset",
    )
    dl_p.add_argument(
        "names",
        nargs="*",
        help="Model names (inswapper, hyperswap, gfpgan, gpen, codeformer, xseg)",
    )
    dl_p.add_argument(
        "--preset",
        choices=sorted(MODEL_PRESETS.keys()),
        help="Download a preset bundle (core|seamless|enhance|all)",
    )
    dl_p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    ens_p = sub.add_parser(
        "ensure",
        help="Alias for download (idempotent)",
    )
    ens_p.add_argument("names", nargs="*")
    ens_p.add_argument(
        "--preset",
        choices=sorted(MODEL_PRESETS.keys()),
        default=None,
    )
    ens_p.add_argument("--quiet", action="store_true")

    args = parser.parse_args(argv)
    mgr = ModelManager(models_dir=args.models_dir)

    if args.action == "list":
        rows = mgr.status()
        name_w = max((len(str(r["name"])) for r in rows), default=8)
        print(f"{'NAME'.ljust(name_w)}  {'VER':8}  STATUS   SIZE")
        for r in rows:
            status = "OK" if r["present"] else ("MISS*" if r["downloadable"] else "MISS")
            size = (
                f"{int(r['bytes']) / (1024 * 1024):.1f} MB"
                if r["present"]
                else "-"
            )
            print(
                f"{str(r['name']).ljust(name_w)}  {str(r['version']):8}  "
                f"{status:7}  {size}"
            )
        if args.presets:
            print("\nPresets:")
            for key, names in sorted(mgr.list_presets().items()):
                print(f"  {key}: {', '.join(names)}")
        print("\n* MISS* = missing but downloadable via `models download`")
        return

    # download / ensure
    show = not args.quiet
    if not args.names and not args.preset:
        args.preset = "seamless"
        print("No names given — using preset 'seamless'", flush=True)

    done = []
    if args.preset:
        done.extend(mgr.ensure_preset(args.preset, show_progress=show))
    if args.names:
        done.extend(mgr.ensure_models(args.names, show_progress=show))

    # Deduplicate by name
    seen = set()
    unique = []
    for info in done:
        if info.name in seen:
            continue
        seen.add(info.name)
        unique.append(info)

    print(f"Ready ({len(unique)} model(s)):")
    for info in unique:
        mb = os.path.getsize(info.path) / (1024 * 1024) if info.is_downloaded else 0
        print(f"  {info.name} {info.version} -> {info.path} ({mb:.1f} MB)")


def evaluate_main(argv: List[str]) -> None:
    """``python -m demos.cli evaluate …`` — swap (optional) + ID/sharpness report."""
    from face_swap import FaceSwapConfig, evaluate_swap, swap_image

    parser = argparse.ArgumentParser(
        prog="python -m demos.cli evaluate",
        description="Measure ArcFace identity match and face sharpness for a swap",
    )
    parser.add_argument("-s", "--source", required=True, help="Source face image")
    parser.add_argument("-t", "--target", required=True, help="Target image")
    parser.add_argument(
        "-r",
        "--result",
        default=None,
        help="Existing swapped image (skip running a new swap)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Write swapped image here when running a new swap",
    )
    parser.add_argument(
        "-q",
        "--quality",
        choices=["low", "fast_cpu", "medium", "high", "seamless"],
        default="seamless",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu", "dml", "auto"],
        default="auto",
    )
    parser.add_argument(
        "--enhance",
        choices=["gfpgan", "gpen", "codeformer", "opencv"],
        default=None,
    )
    parser.add_argument(
        "--min-id",
        type=float,
        default=0.25,
        help="Pass threshold for ArcFace cosine (default 0.25)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print metrics as JSON",
    )
    args = parser.parse_args(argv)

    result_path = args.result
    if result_path is None:
        cfg = FaceSwapConfig(
            quality=args.quality,
            device=args.device,
            enhance_method=args.enhance,
        )
        print(f"Swapping ({args.quality}) ...", flush=True)
        out = swap_image(args.source, args.target, cfg)
        if args.output:
            cv2.imwrite(args.output, out)
            result_path = args.output
            print(f"Wrote {args.output}", flush=True)
        else:
            # Keep array path via temp write for evaluate_swap path API
            import tempfile

            fd, result_path = tempfile.mkstemp(suffix=".jpg")
            os.close(fd)
            cv2.imwrite(result_path, out)

    metrics = evaluate_swap(
        args.source,
        args.target,
        result_path,
        device=args.device,
        min_id_sim=args.min_id,
    )

    if args.json:
        import json

        print(json.dumps(metrics.to_dict(), indent=2))
    else:
        print("Swap metrics")
        print(f"  faces: src={metrics.faces_source} tgt={metrics.faces_target} "
              f"out={metrics.faces_result}")
        print(f"  id_similarity (src-out): {metrics.id_similarity:.4f}")
        print(f"  id_vs_target  (out-tgt): {metrics.id_vs_target:.4f}")
        print(f"  sharpness out/tgt/d:     "
              f"{metrics.sharpness_result:.1f} / {metrics.sharpness_target:.1f} / "
              f"{metrics.sharpness_gain:+.1f}")
        print(f"  color dE (LAB mean):     {metrics.color_delta_lab:.1f}")
        print(f"  id gate (>={args.min_id}): "
              f"{'PASS' if metrics.passed_id else 'WEAK'}")


if __name__ == "__main__":
    main()
