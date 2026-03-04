"""
Example 3: Real-Time Webcam Face Swap

Demonstrates the live webcam face swap experience with
FPS overlay and keyboard controls.

Usage:
    python examples/03_realtime_webcam.py --source face.jpg

Controls:
    q — Quit
    s — Save screenshot
    + — Increase quality
    - — Decrease quality
"""

import argparse
import cv2
import sys
import time


def main():
    parser = argparse.ArgumentParser(description="Real-Time Webcam Face Swap")
    parser.add_argument("--source", required=True, help="Source face image")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from face_swap import start_realtime_swap, FaceSwapConfig

    # Load source image
    source = cv2.imread(args.source)
    if source is None:
        print(f"Error: Cannot load source: {args.source}")
        sys.exit(1)

    print(f"Source loaded: {args.source}")
    print(f"Camera: {args.camera}")
    print(f"Device: {args.device}")
    print()
    print("Controls:")
    print("  q — Quit")
    print("  s — Save screenshot")
    print()

    config = FaceSwapConfig(
        device=args.device,
        quality="medium",
    )

    # Frame counter for screenshots
    screenshot_counter = [0]

    def on_frame(frame, frame_idx):
        """Optional per-frame callback."""
        pass

    print("Starting webcam... Press 'q' to quit.")
    start_realtime_swap(
        source_img=args.source,
        camera_id=args.camera,
        config=config,
    )


if __name__ == "__main__":
    main()
