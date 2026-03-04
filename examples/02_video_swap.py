"""
Example 2: Video Face Swap with Audio Preservation

Demonstrates offline video processing with progress tracking
and automatic audio re-muxing.

Usage:
    python examples/02_video_swap.py \
        --source face.jpg \
        --target input_video.mp4 \
        --output swapped_video.mp4
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Video Face Swap Example")
    parser.add_argument("--source", required=True, help="Source face image")
    parser.add_argument("--target", required=True, help="Input video file")
    parser.add_argument("--output", default="output_video.mp4")
    parser.add_argument("--quality", default="medium", choices=["low", "medium", "high"])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from face_swap import swap_video, FaceSwapConfig
    from face_swap.audio import AudioProcessor

    # Configure
    config = FaceSwapConfig(
        quality=args.quality,
        device=args.device,
    )

    # Progress callback
    def on_progress(frame_idx, total_frames):
        pct = (frame_idx / max(total_frames, 1)) * 100
        bar = "█" * int(pct // 2) + "░" * (50 - int(pct // 2))
        print(f"\r  [{bar}] {pct:.0f}% ({frame_idx}/{total_frames})", end="", flush=True)

    print(f"Input:  {args.target}")
    print(f"Source: {args.source}")
    print(f"Processing video...")

    # Step 1: Swap faces in video (produces video without audio)
    swap_video(args.source, args.target, args.output + ".tmp.mp4", config)
    print()  # Newline after progress bar

    # Step 2: Re-mux with original audio
    audio = AudioProcessor()
    if audio.available:
        print("Re-muxing audio...")
        audio.swap_video_with_audio(args.target, args.output + ".tmp.mp4", args.output)
        import os
        os.remove(args.output + ".tmp.mp4")
        print(f"✅ Output with audio: {args.output}")
    else:
        import shutil
        shutil.move(args.output + ".tmp.mp4", args.output)
        print(f"⚠️  FFmpeg not found — output has no audio: {args.output}")


if __name__ == "__main__":
    main()
