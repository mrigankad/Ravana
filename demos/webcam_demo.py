"""
Real-time webcam face swap demo.

As per PRD Section 4.1, this provides a simple webcam demo app.
"""

import sys
from pathlib import Path
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from face_swap import FaceSwapPipeline, PipelineConfig, FaceSwapConfig


class WebcamDemo:
    """
    Interactive webcam face swap demo.
    
    Features:
    - Real-time face swapping
    - FPS counter
    - Screenshot capture
    - Multiple source face support
    """
    
    def __init__(self, config: FaceSwapConfig = None):
        self.config = config or FaceSwapConfig(quality="medium")
        self.pipeline = None
        self.source_embedding = None
        self.frame_count = 0
        self.fps_history = []
        self.last_time = None
        
    def initialize(self, source_image: np.ndarray):
        """Initialize the pipeline with a source image."""
        print("Initializing face swap pipeline...")
        
        pipeline_config = self.config.to_pipeline_config()
        pipeline_config.enable_temporal = True
        
        self.pipeline = FaceSwapPipeline(pipeline_config)
        self.source_embedding = self.pipeline.extract_source_embedding(source_image)
        
        print("Pipeline ready!")
    
    def run(self, camera_id: int = 0, display_size: tuple = (1280, 720)):
        """
        Run the webcam demo.
        
        Args:
            camera_id: Camera device ID
            display_size: Display resolution
        """
        if self.pipeline is None:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")
        
        # Open camera
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera {camera_id}")
        
        # Set resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, display_size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, display_size[1])
        
        print("\nControls:")
        print("  'q' - Quit")
        print("  's' - Save screenshot")
        print("  'f' - Toggle fullscreen")
        print("  'r' - Reset temporal smoothing")
        
        window_name = "Face Swap - Real-time Demo"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        screenshot_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to capture frame")
                    break
                
                # Calculate FPS
                import time
                current_time = time.time()
                if self.last_time is not None:
                    fps = 1.0 / (current_time - self.last_time)
                    self.fps_history.append(fps)
                    if len(self.fps_history) > 30:
                        self.fps_history.pop(0)
                self.last_time = current_time
                
                # Process frame
                result = self.pipeline.process_video_frame(
                    frame,
                    self.source_embedding,
                    self.frame_count
                )
                
                # Add FPS overlay
                if self.fps_history:
                    avg_fps = sum(self.fps_history) / len(self.fps_history)
                    fps_text = f"FPS: {avg_fps:.1f}"
                    cv2.putText(
                        result,
                        fps_text,
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )
                
                # Show result
                cv2.imshow(window_name, result)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save screenshot
                    filename = f"screenshot_{screenshot_count:04d}.jpg"
                    cv2.imwrite(filename, result)
                    print(f"Screenshot saved: {filename}")
                    screenshot_count += 1
                elif key == ord('f'):
                    # Toggle fullscreen
                    is_full = cv2.getWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN)
                    cv2.setWindowProperty(
                        window_name,
                        cv2.WND_PROP_FULLSCREEN,
                        cv2.WINDOW_FULLSCREEN if is_full != 1 else cv2.WINDOW_NORMAL
                    )
                elif key == ord('r'):
                    # Reset temporal smoothing
                    if self.pipeline.temporal_smoother:
                        self.pipeline.temporal_smoother.clear_cache()
                    print("Temporal smoothing reset")
                
                self.frame_count += 1
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("Demo ended")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Real-time Face Swap Webcam Demo")
    parser.add_argument(
        "-s", "--source",
        required=True,
        help="Source image containing the face to swap"
    )
    parser.add_argument(
        "-c", "--camera",
        type=int,
        default=0,
        help="Camera device ID (default: 0)"
    )
    parser.add_argument(
        "-q", "--quality",
        choices=["low", "medium", "high"],
        default="medium",
        help="Quality level (default: medium)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Camera width (default: 1280)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Camera height (default: 720)"
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Use CPU instead of CUDA"
    )
    
    args = parser.parse_args()
    
    # Load source image
    source_img = cv2.imread(args.source)
    if source_img is None:
        print(f"Error: Could not load source image: {args.source}")
        sys.exit(1)
    
    # Create config
    config = FaceSwapConfig(
        quality=args.quality,
        device="cpu" if args.cpu else "cuda"
    )
    
    # Run demo
    demo = WebcamDemo(config)
    demo.initialize(source_img)
    demo.run(args.camera, (args.width, args.height))


if __name__ == "__main__":
    main()
