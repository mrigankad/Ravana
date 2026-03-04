<div align="center">
  <img src="assets/mascot.png" width="300" alt="Ravana Mascot">
</div>

# Ravana

**Production-ready SDK for real-time face swapping on images, video, and live webcam streams.**

## Overview

Ravana provides a modular, high-performance pipeline for face swapping with:

- **🖼️ Image swap** — Single-image face replacement
- **🎬 Video swap** — Offline video processing with audio preservation
- **📹 Real-time** — Live webcam face swap at 25+ FPS
- **🎭 AR Filters** — Fun filter-style experience with overlays
- **🔌 Plugins** — Extensible architecture for custom components

## Quick Start

```python
from face_swap import swap_image, FaceSwapConfig

config = FaceSwapConfig(quality="high", device="cuda")
result = swap_image("source.jpg", "target.jpg", config)
```

## Architecture

```mermaid
graph LR
    A[Input] --> B[Detection]
    B --> C[Landmarks]
    C --> D[Alignment]
    D --> E[Embedding]
    E --> F[Swap Model]
    F --> G[Enhancement]
    G --> H[Blending]
    H --> I[Output]
```

## Features

| Feature | Status |
|---------|--------|
| Image face swap | ✅ |
| Video face swap | ✅ |
| Real-time webcam | ✅ |
| Multi-face support | ✅ |
| Temporal smoothing | ✅ |
| Quality validation | ✅ |
| Invisible watermarking | ✅ |
| GAN enhancement (GFPGAN) | ✅ |
| Plugin system | ✅ |
| AR filter engine | ✅ |
| TensorRT optimization | ✅ |
| C/C++ native API | ✅ |
| Model training pipeline | ✅ |
| macOS Metal support | ✅ |
| Mobile export (Android/iOS) | ✅ |

## Installation

```bash
pip install -e .
```

## Navigation

- [Getting Started](getting-started/installation.md) — Setup & first swap
- [API Reference](api/high-level.md) — Detailed API documentation
- [Advanced](advanced/plugins.md) — Plugins, training, & deployment
- [Examples](examples/image-swap.md) — Code samples & tutorials
