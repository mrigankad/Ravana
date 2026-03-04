"""
CLI tool for exporting ONNX models to TensorRT engines.

Usage:
    python -m face_swap.optimization.export_cli \
        --onnx models/inswapper_128.onnx \
        --engine models/inswapper_128_fp16.engine \
        --precision fp16

    python -m face_swap.optimization.export_cli \
        --onnx models/simswap_256.onnx \
        --engine models/simswap_256_int8.engine \
        --precision int8 \
        --calibration-data ./calibration_images/
"""

import argparse
import logging
import sys

from .export import ExportConfig, TensorRTExporter


def main():
    parser = argparse.ArgumentParser(
        description="Export ONNX models to TensorRT engines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--onnx",
        required=True,
        help="Path to source ONNX model",
    )
    parser.add_argument(
        "--engine",
        required=True,
        help="Path for output TensorRT engine",
    )
    parser.add_argument(
        "--precision",
        choices=["fp32", "fp16", "int8"],
        default="fp16",
        help="Inference precision (default: fp16)",
    )
    parser.add_argument(
        "--workspace",
        type=float,
        default=4.0,
        help="Max workspace in GB (default: 4.0)",
    )
    parser.add_argument(
        "--max-batch",
        type=int,
        default=1,
        help="Maximum batch size (default: 1)",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Enable dynamic batch axes",
    )
    parser.add_argument(
        "--calibration-data",
        help="Path to calibration images directory (required for INT8)",
    )
    parser.add_argument(
        "--calibration-count",
        type=int,
        default=500,
        help="Number of calibration images (default: 500)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose TensorRT logging",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run a quick latency benchmark after export",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    exporter = TensorRTExporter()

    if not exporter.available:
        print("ERROR: TensorRT is not installed.")
        print("Install with: pip install tensorrt")
        sys.exit(1)

    config = ExportConfig(
        precision=args.precision,
        max_workspace_gb=args.workspace,
        max_batch_size=args.max_batch,
        dynamic_axes=args.dynamic,
        calibration_data=args.calibration_data,
        calibration_count=args.calibration_count,
        verbose=args.verbose,
    )

    try:
        engine_path = exporter.export(args.onnx, args.engine, config)
        print(f"\n✅ Engine exported: {engine_path}")
    except Exception as e:
        print(f"\n❌ Export failed: {e}")
        sys.exit(1)

    # Optional benchmark
    if args.benchmark:
        from .runtime import TensorRTRuntime

        print("\nRunning benchmark...")
        rt = TensorRTRuntime(engine_path)
        results = rt.benchmark(warmup=20, iterations=200)

        print(f"  Average latency: {results['avg_ms']:.2f} ms")
        print(f"  Min latency:     {results['min_ms']:.2f} ms")
        print(f"  Max latency:     {results['max_ms']:.2f} ms")
        print(f"  Throughput:      {results['fps']:.1f} FPS")

        if results["avg_ms"] <= 10.0:
            print("  ✅ Meets PRD swap-step target (≤ 10 ms)")
        elif results["avg_ms"] <= 40.0:
            print("  ✅ Meets PRD per-frame target (≤ 40 ms)")
        else:
            print("  ⚠️  Exceeds PRD latency target (> 40 ms)")

        rt.unload()


if __name__ == "__main__":
    main()
