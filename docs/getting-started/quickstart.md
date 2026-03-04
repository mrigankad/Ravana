# Quick Start Guide

This guide will walk you through the most common Ravana use cases.

## 1. Single Image Swap

The easiest way to swap a face in an image:

```python
import cv2
from face_swap import swap_image, FaceSwapConfig

# Configure the swap
config = FaceSwapConfig(
    quality="high",    # "low" (fastest), "medium" (balanced), "high" (best)
    device="cuda",     # "cuda", "cpu", or "mps" (macOS)
)

# Load your images
source = cv2.imread("source_face.jpg")
target = cv2.imread("target_image.jpg")

# Run the swap
output = swap_image(source, target, config)

# Save the result
cv2.imwrite("swapped.jpg", output)
```

## 2. Video Processing

Processing a video file offline. The SDK will automatically handle progress tracking and audio re-muxing.

```python
from face_swap.api import swap_video
from face_swap import FaceSwapConfig

config = FaceSwapConfig(quality="medium", device="cuda")

# Swap faces and preserve audio
swap_video(
    source_img_path="source_face.jpg",
    target_video_path="input_video.mp4",
    output_video_path="swapped_video.mp4",
    config=config
)
```

## 3. Live Webcam

For a real-time face swap experience using your webcam:

```python
from face_swap.api import start_realtime_swap
from face_swap import FaceSwapConfig

config = FaceSwapConfig(
    quality="low",  # Lower quality ensures higher FPS
    device="cuda",  # CUDA is recommended for real-time
)

start_realtime_swap(
    source_img="source_face.jpg",
    camera_id=0,
    config=config
)
```

## 4. Advanced: Using the Pipeline Directly

For custom logic, you can use the lower-level pipeline directly. Validating the quality of a swap before displaying it is a good practice.

```python
import cv2
from face_swap.pipeline import FaceSwapPipeline, PipelineConfig
from face_swap.core.quality import QualityValidator

# Initialize pipeline
cfg = PipelineConfig(device="cuda", crop_size=256)
pipeline = FaceSwapPipeline(cfg)
pipeline.initialize()

# Extract Identity
source_img = cv2.imread("source.jpg")
embedding = pipeline.extract_source_embedding(source_img)

# Process a frame
target_img = cv2.imread("target.jpg")
result = pipeline.process_video_frame(target_img, embedding)

# Validate quality
validator = QualityValidator()
report = validator.check_post_swap(result, target_img)

if report.passed:
    cv2.imshow("Result", result)
else:
    print(f"Swap rejected: {report.message}")

pipeline.cleanup()
```

## Next Steps
- Dive into the [Configuration Docs](configuration.md) to tweak performance and quality.
- Check out the [AR Filters](../advanced/ar-filters.md) to add visual effects.
- See how to use [Plugins](../advanced/plugins.md) to replace internal components.
