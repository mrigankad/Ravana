"""
Command-line interface for face swapping.

As per PRD Section 4.1, this provides CLI and minimal GUI for batch processing.
"""

import argparse
import sys
from pathlib import Path
from typing import List
import glob

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from face_swap import swap_image, swap_video, batch_swap, FaceSwapConfig
import cv2


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Ravana - Real-time face swapping",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Swap face on a single image
  python -m demos.cli -s source.jpg -t target.jpg -o output.jpg
  
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
        choices=["low", "medium", "high"],
        default="medium",
        help="Quality level (default: medium)"
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda",
        help="Device to use for inference (default: cuda)"
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
        help="Path to face swap model"
    )
    
    args = parser.parse_args()
    
    # Create config
    config = FaceSwapConfig(
        quality=args.quality,
        device=args.device,
        color_correction=not args.no_color_correction,
        enable_smoothing=not args.no_smoothing,
        swap_model_path=args.swap_model
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


if __name__ == "__main__":
    main()
