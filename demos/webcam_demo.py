"""
Real-time webcam face swap demo.

As per PRD Section 4.1, this provides a simple webcam demo app.
"""

import argparse
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np

from face_swap import FaceSwapConfig, FaceSwapPipeline
from face_swap.core.fast_video import FastVideoConfig


class WebcamDemo:
    """
    Interactive webcam face swap demo.

    Features:
    - Real-time face swapping with detect-every-N + ROI tracking
    - FPS / HUD overlay
    - Screenshot capture
    - Hotkeys for detect interval and enhance toggle
    """

    def __init__(self, config: FaceSwapConfig = None):
        self.config = config or FaceSwapConfig(
            quality="medium", realtime=True, device="auto"
        )
        self.pipeline = None
        self.source_embedding = None
        self.frame_count = 0
        self.fps_history = []
        self.last_time = None
        self._show_hud = True

    def initialize(self, source_image: np.ndarray):
        """Initialize the pipeline with a source image."""
        print("Initializing face swap pipeline...")

        pipeline_config = self.config.to_pipeline_config()
        pipeline_config.enable_temporal = True
        # Ensure live knobs even if caller forgot realtime=
        if pipeline_config.fast_video is None or not pipeline_config.fast_video.enabled:
            n = max(1, int(self.config.detect_every_n))
            pipeline_config.fast_video = FastVideoConfig(
                enabled=True,
                detect_every_n=n,
                det_size=(320, 320),
                detect_max_side=640,
                skip_enhance=True,
                max_faces=1,
                roi_track=True,
            )
            pipeline_config.video_detect_every_n = n
            pipeline_config.enable_enhance = False

        self.pipeline = FaceSwapPipeline(pipeline_config)
        self.source_embedding = self.pipeline.extract_source_embedding(source_image)

        print("Pipeline ready!")

    def _avg_fps(self) -> float:
        if not self.fps_history:
            return 0.0
        return sum(self.fps_history) / len(self.fps_history)

    def _draw_hud(self, frame: np.ndarray) -> None:
        if not self._show_hud:
            return
        cfg = self.pipeline.config if self.pipeline else None
        fast = getattr(cfg, "fast_video", None) if cfg else None
        n = 1
        if fast and fast.enabled:
            n = int(fast.detect_every_n)
        elif cfg:
            n = int(getattr(cfg, "video_detect_every_n", 1) or 1)
        roi = bool(fast and getattr(fast, "roi_track", False))
        enh = (
            "on" if (cfg and cfg.enable_enhance and self.pipeline._enhancer) else "off"
        )
        lines = [
            f"FPS: {self._avg_fps():.1f}",
            f"q={self.config.quality}  dev={self.config.device}",
            f"detect_every={n}  ROI={'on' if roi else 'off'}  enh={enh}",
            "keys: q quit | s shot | d detectN | e enhance | h hud | r reset",
        ]
        y = 28
        for line in lines:
            cv2.putText(
                frame,
                line,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65 if y == 28 else 0.5,
                (0, 255, 0) if y == 28 else (220, 220, 220),
                2 if y == 28 else 1,
                cv2.LINE_AA,
            )
            y += 24

    def _cycle_detect_every_n(self) -> None:
        if self.pipeline is None:
            return
        cfg = self.pipeline.config
        options = (1, 2, 3, 4, 5)
        cur = 1
        if cfg.fast_video and cfg.fast_video.enabled:
            cur = int(cfg.fast_video.detect_every_n)
        else:
            cur = int(cfg.video_detect_every_n or 1)
        nxt = options[(options.index(cur) + 1) % len(options)] if cur in options else 2
        if cfg.fast_video is None:
            cfg.fast_video = FastVideoConfig(
                enabled=True, detect_every_n=nxt, roi_track=True
            )
            self.pipeline._fast = cfg.fast_video
        else:
            cfg.fast_video.enabled = True
            cfg.fast_video.detect_every_n = nxt
            cfg.fast_video.roi_track = True
            self.pipeline._fast = cfg.fast_video
        cfg.video_detect_every_n = nxt
        self.pipeline._cached_faces = []
        self.pipeline._cache_frame_idx = -999
        print(f"detect_every_n -> {nxt}")

    def _toggle_enhance(self) -> None:
        """Toggle lightweight OpenCV enhance on the live path (no GFPGAN reload)."""
        if self.pipeline is None:
            return
        from face_swap.enhancement import EnhancementConfig, OpenCVEnhancer

        cfg = self.pipeline.config
        if self.pipeline._enhancer is not None:
            self.pipeline._enhancer = None
            cfg.enable_enhance = False
            if cfg.fast_video:
                cfg.fast_video.skip_enhance = True
            print("enhance -> off")
            return

        enh = OpenCVEnhancer(
            EnhancementConfig(method="opencv", enabled=True, blend_weight=0.55)
        )
        enh.load_model()
        self.pipeline._enhancer = enh
        cfg.enable_enhance = True
        if cfg.fast_video:
            cfg.fast_video.skip_enhance = False
        print("enhance -> opencv")

    def run(self, camera_id: int = 0, display_size: tuple = (1280, 720)):
        """
        Run the webcam demo.

        Args:
            camera_id: Camera device ID
            display_size: Display resolution
        """
        if self.pipeline is None:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")

        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera {camera_id}")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, display_size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, display_size[1])

        print("\nControls:")
        print("  q - Quit")
        print("  s - Save screenshot")
        print("  f - Toggle fullscreen")
        print("  r - Reset temporal / face cache")
        print("  d - Cycle detect_every_n (1-5)")
        print("  e - Toggle OpenCV enhance")
        print("  h - Toggle HUD")

        window_name = "Face Swap - Real-time Demo"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        screenshot_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to capture frame")
                    break

                current_time = time.time()
                if self.last_time is not None:
                    dt = current_time - self.last_time
                    if dt > 1e-6:
                        self.fps_history.append(1.0 / dt)
                        if len(self.fps_history) > 30:
                            self.fps_history.pop(0)
                self.last_time = current_time

                result = self.pipeline.process_video_frame(
                    frame,
                    self.source_embedding,
                    self.frame_count,
                )
                self._draw_hud(result)
                cv2.imshow(window_name, result)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("s"):
                    filename = f"screenshot_{screenshot_count:04d}.jpg"
                    cv2.imwrite(filename, result)
                    print(f"Screenshot saved: {filename}")
                    screenshot_count += 1
                elif key == ord("f"):
                    is_full = cv2.getWindowProperty(
                        window_name, cv2.WND_PROP_FULLSCREEN
                    )
                    cv2.setWindowProperty(
                        window_name,
                        cv2.WND_PROP_FULLSCREEN,
                        cv2.WINDOW_FULLSCREEN if is_full != 1 else cv2.WINDOW_NORMAL,
                    )
                elif key == ord("r"):
                    if self.pipeline.temporal_smoother:
                        self.pipeline.temporal_smoother.clear_cache()
                    self.pipeline._cached_faces = []
                    self.pipeline._cache_frame_idx = -999
                    self.pipeline._smooth_faces_prev = []
                    print("Temporal / face cache reset")
                elif key == ord("d"):
                    self._cycle_detect_every_n()
                elif key == ord("e"):
                    self._toggle_enhance()
                elif key == ord("h"):
                    self._show_hud = not self._show_hud

                self.frame_count += 1

        finally:
            cap.release()
            cv2.destroyAllWindows()
            print(
                f"Demo ended ({self.frame_count} frames, avg FPS {self._avg_fps():.1f})"
            )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Real-time Face Swap Webcam Demo")
    parser.add_argument(
        "-s", "--source", required=True, help="Source image containing the face to swap"
    )
    parser.add_argument("-c", "--camera", type=int, default=0, help="Camera device ID")
    parser.add_argument(
        "-q",
        "--quality",
        choices=["low", "fast_cpu", "medium", "high", "seamless"],
        default="medium",
        help="Quality preset (realtime skips heavy restore; default: medium)",
    )
    parser.add_argument("--width", type=int, default=1280, help="Camera width")
    parser.add_argument("--height", type=int, default=720, help="Camera height")
    parser.add_argument(
        "--detect-every",
        type=int,
        default=3,
        metavar="N",
        help="Full face detect every N frames (default: 3)",
    )
    parser.add_argument(
        "--no-realtime",
        action="store_true",
        help="Disable realtime speed path (full quality per frame)",
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu", "dml", "auto"],
        default="auto",
        help="Device (default: auto)",
    )

    args = parser.parse_args()

    source_img = cv2.imread(args.source)
    if source_img is None:
        print(f"Error: Could not load source image: {args.source}")
        sys.exit(1)

    config = FaceSwapConfig(
        quality=args.quality,
        device="cpu" if args.cpu else args.device,
        realtime=not args.no_realtime,
        detect_every_n=max(1, args.detect_every),
        enable_smoothing=True,
    )

    demo = WebcamDemo(config)
    demo.initialize(source_img)
    demo.run(args.camera, (args.width, args.height))


if __name__ == "__main__":
    main()
