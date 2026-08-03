# Quick Start Guide

This guide will walk you through the most common Ravana use cases.

## 1. Single Image Swap

The easiest way to swap a face in an image:

```python
import cv2
from face_swap import swap_image, FaceSwapConfig

# Configure the swap
config = FaceSwapConfig(
    quality="seamless",  # low | fast_cpu | medium | high | seamless
    device="auto",       # auto | cuda | dml | cpu
    # enhance_method="codeformer",  # optional; seamless defaults to gfpgan
)

# Load your images
source = cv2.imread("source_face.jpg")
target = cv2.imread("target_image.jpg")

# Run the swap
output = swap_image(source, target, config)

# Save the result
cv2.imwrite("swapped.jpg", output)
```

`quality="seamless"` enables HyperSwap-256, GFPGAN restore, XSeg occlusion, and lighting match. Models download to `models/` on first run (or run `python -m demos.cli models download --preset seamless`).

## 2. Video Processing

Processing a video file offline. The SDK will automatically handle progress tracking and audio re-muxing when ffmpeg is available.

```python
from face_swap.api import swap_video
from face_swap import FaceSwapConfig

config = FaceSwapConfig(quality="medium", device="auto")

swap_video(
    "source_face.jpg",
    "input_video.mp4",
    "swapped_video.mp4",
    config=config,
)
```

## 3. Live Webcam

For a real-time face swap experience using your webcam:

```python
from face_swap.api import start_realtime_swap
from face_swap import FaceSwapConfig

config = FaceSwapConfig(
    quality="fast_cpu",  # or low — prioritize FPS
    device="auto",
)

start_realtime_swap(
    source_img="source_face.jpg",
    camera_id=0,
    config=config,
)
```

## 4. Advanced: Using the Pipeline Directly

For custom logic, use the lower-level pipeline and optional quality validation:

```python
import cv2
from face_swap.pipeline import FaceSwapPipeline, PipelineConfig
from face_swap.core.quality import QualityValidator

cfg = PipelineConfig(device="auto", swap_model="hyperswap", enable_enhance=True)
pipeline = FaceSwapPipeline(cfg)
pipeline.initialize()

source = cv2.imread("source_face.jpg")
target = cv2.imread("target_image.jpg")
pipeline.extract_source_embedding(source)
output = pipeline.process_frame(target)

validator = QualityValidator()
report = validator.validate(source, target, output)
print(report)
```

## Next Steps

- See [Installation](installation.md) for GPU / DirectML setup.
- Dive into the [Configuration Docs](configuration.md) to tweak performance and quality.
