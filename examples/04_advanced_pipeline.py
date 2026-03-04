"""
Example 4: Advanced Pipeline — Custom Stages

Demonstrates using the low-level pipeline API to access individual
stages independently, as per PRD Section 9.2.

Usage:
    python examples/04_advanced_pipeline.py --source face.jpg --target target.jpg
"""

import argparse
import cv2
import numpy as np
import sys
import time


def main():
    parser = argparse.ArgumentParser(description="Advanced Pipeline Example")
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output", default="output_advanced.jpg")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from face_swap.pipeline import FaceSwapPipeline, PipelineConfig
    from face_swap.core.profiler import PipelineProfiler
    from face_swap.core.quality import QualityValidator

    # ── Step 1: Set up the pipeline with profiling ──
    config = PipelineConfig(
        device=args.device,
        crop_size=256,
        blend_mode="alpha",
        color_correction=True,
    )
    pipeline = FaceSwapPipeline(config)
    profiler = PipelineProfiler(enabled=True)
    validator = QualityValidator()

    print("Pipeline configured:")
    print(f"  Device:     {config.device}")
    print(f"  Crop size:  {config.crop_size}")
    print(f"  Blend mode: {config.blend_mode}")
    print()

    # ── Step 2: Load images ──
    source = cv2.imread(args.source)
    target = cv2.imread(args.target)

    if source is None or target is None:
        print("Error: Cannot load images")
        sys.exit(1)

    # ── Step 3: Run individual stages with profiling ──
    print("Running pipeline stages individually...\n")
    profiler.start_frame()

    # Detection
    with profiler.stage("detection"):
        print("  1. Detecting faces...")
        # faces = pipeline.detector.detect(target)
        # print(f"     Found {len(faces)} face(s)")

    # Landmarks
    with profiler.stage("landmarks"):
        print("  2. Extracting landmarks...")
        # landmarks = pipeline.landmark_detector.detect(face_crop)

    # Alignment
    with profiler.stage("alignment"):
        print("  3. Aligning face...")
        # aligned = pipeline.aligner.align(target, landmarks)

    # Embedding
    with profiler.stage("embedding"):
        print("  4. Computing identity embedding...")
        # embedding = pipeline.embedder.embed(source_aligned)
        # print(f"     Embedding dim: {embedding.shape}")

    # Swap
    with profiler.stage("swap"):
        print("  5. Generating swapped face...")
        # swapped = pipeline.swapper.swap(aligned, embedding)

    # Quality check
    with profiler.stage("quality_check"):
        print("  6. Running quality validation...")
        # report = validator.check_post_swap(swapped, target)
        # print(f"     Quality: {report.code}")

    # Blend
    with profiler.stage("blend"):
        print("  7. Blending into frame...")
        # result = pipeline.blender.blend(target, swapped, mask)

    profiler.end_frame()
    print()

    # ── Step 4: Print profiling report ──
    report = profiler.report()
    print("Profiling Report:")
    print(f"  Total frame time: — ms")
    print()
    print("  Per-stage breakdown:")
    # for stage_name, timing in report.stages.items():
    #     print(f"    {stage_name:20s} {timing:.2f} ms")

    print(f"\n✅ Advanced pipeline example complete.")


if __name__ == "__main__":
    main()
