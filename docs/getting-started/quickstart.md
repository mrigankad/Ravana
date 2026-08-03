# Quick Start Guide

This guide walks through the most common Ravana use cases.

## 0. First run (models + health check)

```bash
# Prefetch seamless weights (~HyperSwap + GFPGAN + XSeg)
python -m demos.cli models download --preset seamless

# Or run the health check (optional --download)
python scripts/first_run_check.py --download
```

Models also auto-download on first swap if missing.

## 1. Single Image Swap

```python
import cv2
from face_swap import swap_image, FaceSwapConfig

config = FaceSwapConfig(
    quality="seamless",  # low | fast_cpu | medium | high | seamless
    device="auto",       # auto | cuda | dml | cpu
    # enhance_method="gpen",       # or codeformer / gfpgan / opencv
    # pixel_boost=1024,            # seamless default
)

source = cv2.imread("source_face.jpg")
target = cv2.imread("target_image.jpg")
output = swap_image(source, target, config)
cv2.imwrite("swapped.jpg", output)
```

`quality="seamless"` enables HyperSwap-256, GFPGAN restore, XSeg occlusion, lighting match, and tiled pixel-boost.

```bash
python -m demos.cli -s source.jpg -t target.jpg -o out.jpg -q seamless --device auto
```

## 2. Score a swap (identity + sharpness)

```bash
python -m demos.cli evaluate -s source.jpg -t target.jpg -o out.jpg -q seamless
python -m demos.cli evaluate-batch --pairs 2 --enhance gfpgan,gpen --boosts 512
```

```python
from face_swap import evaluate_swap

m = evaluate_swap("source.jpg", "target.jpg", "out.jpg", device="auto")
print(m.summary_line())
```

## 3. Video Processing

```python
from face_swap import FaceSwapConfig, swap_video

config = FaceSwapConfig(quality="medium", device="auto")
swap_video("source_face.jpg", "input_video.mp4", "swapped_video.mp4", config=config)
```

## 4. Live Webcam

Realtime mode skips heavy restore and uses detect-every-N + ROI tracking:

```python
from face_swap import FaceSwapConfig, start_realtime_swap

config = FaceSwapConfig(
    quality="medium",
    device="auto",
    realtime=True,
    detect_every_n=3,
)

start_realtime_swap("source_face.jpg", camera_id=0, config=config)
```

```bash
python -m demos.webcam_demo -s source.jpg --device auto --detect-every 3
# keys: q quit | d cycle detect N | e OpenCV enhance | h HUD
```

## 5. Pipeline + metrics

```python
import cv2
from face_swap import FaceSwapConfig, evaluate_swap
from face_swap.pipeline import FaceSwapPipeline

cfg = FaceSwapConfig(quality="seamless", device="auto").to_pipeline_config()
pipeline = FaceSwapPipeline(cfg)

source = cv2.imread("source_face.jpg")
target = cv2.imread("target_image.jpg")
emb = pipeline.extract_source_embedding(source)
output = pipeline.process_frame(target, emb)
cv2.imwrite("swapped.jpg", output)

print(evaluate_swap(source, target, output).summary_line())
```

## Next Steps

- [Installation](installation.md) for GPU / DirectML setup
- `python -m demos.cli models list --presets` for weight status
