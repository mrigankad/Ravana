"""
Example 1: Basic Image Swap

As per PRD Section 9.3, this demonstrates the simplest face swap workflow.

Usage:
    python examples/01_basic_image_swap.py \
        --source path/to/source_face.jpg \
        --target path/to/target_image.jpg \
        --output path/to/output.jpg
"""

import argparse
import cv2
import sys


def main():
    parser = argparse.ArgumentParser(description="Basic Image Face Swap Example")
    parser.add_argument("--source", required=True, help="Source face image")
    parser.add_argument("--target", required=True, help="Target image")
    parser.add_argument("--output", default="output_swap.jpg", help="Output path")
    parser.add_argument("--quality", default="high", choices=["low", "medium", "high"])
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    # ── Step 1: Import the SDK ──
    from face_swap import swap_image, FaceSwapConfig

    # ── Step 2: Configure ──
    config = FaceSwapConfig(
        quality=args.quality,
        device=args.device,
    )

    # ── Step 3: Load images ──
    source = cv2.imread(args.source)
    target = cv2.imread(args.target)

    if source is None:
        print(f"Error: Cannot load source image: {args.source}")
        sys.exit(1)
    if target is None:
        print(f"Error: Cannot load target image: {args.target}")
        sys.exit(1)

    print(f"Source: {args.source} ({source.shape[1]}×{source.shape[0]})")
    print(f"Target: {args.target} ({target.shape[1]}×{target.shape[0]})")

    # ── Step 4: Swap! ──
    print("Running face swap...")
    result = swap_image(source, target, config)

    # ── Step 5: Save ──
    cv2.imwrite(args.output, result)
    print(f"✅ Output saved: {args.output}")


if __name__ == "__main__":
    main()
