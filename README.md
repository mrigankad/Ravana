<div align="center">
  <img src="docs/assets/mascot.png" width="280" alt="Ravana real-time face-swap SDK mascot">
  <h1>Ravana</h1>
  <p><b>Open-source Python face-swap SDK</b> for images, video, and webcam.<br/>
  HyperSwap / InSwapper · GFPGAN / GPEN / CodeFormer · ONNX Runtime on <b>CUDA</b>, <b>AMD DirectML</b>, or CPU.</p>

  <p>
    <a href="https://github.com/mrigankad/Ravana/stargazers"><img src="https://img.shields.io/github/stars/mrigankad/Ravana?style=social" alt="GitHub stars"></a>
    <a href="https://github.com/mrigankad/Ravana/releases"><img src="https://img.shields.io/github/v/release/mrigankad/Ravana?display_name=tag" alt="Latest release"></a>
    <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python 3.9+">
    <img src="https://img.shields.io/badge/License-MIT-purple.svg" alt="MIT License">
    <img src="https://img.shields.io/badge/ONNX-CUDA%20%7C%20DirectML%20%7C%20CPU-orange.svg" alt="ONNX backends">
  </p>
</div>

```bash
# From GitHub (latest main)
pip install "git+https://github.com/mrigankad/Ravana.git"

# Or clone for extras / demos
git clone https://github.com/mrigankad/Ravana.git && cd Ravana
pip install -e ".[directml]"   # AMD Windows
# pip install -e ".[gpu]"      # NVIDIA
# pip install -e .             # CPU
```

> **Package name:** install/import as `ravana-sdk` on PyPI (the name `ravana` is taken). Python imports stay `from face_swap import ...`.

---

## Features (0.3.3)

| Area | What’s included |
|------|-----------------|
| **Quality** | `seamless` preset: HyperSwap-256, GFPGAN (default), XSeg occlusion, lighting/grain match, tiled pixel-boost 1024 |
| **Restores** | GFPGAN · GPEN · CodeFormer · RestoreFormer++ · OpenCV (`--enhance`) |
| **Realtime** | `realtime=True`: detect-every-N, ROI track, skip heavy restore, webcam FPS HUD |
| **Faces** | Select `all` / `largest` / `first` / `index` / `pose` |
| **Metrics** | ArcFace ID + sharpness (`evaluate` / `evaluate-batch` / GUI Score) |
| **Models** | Progress download CLI + presets (`core`, `seamless`, `enhance`, `all`) |
| **Device** | `device="auto"` → CUDA → DirectML → CPU |
| **Desktop GUI** | PySide6 app — drag-drop, batch, webcam, score (`pip install -e ".[gui]"`) |

---

## Install

```bash
git clone https://github.com/mrigankad/Ravana.git
cd Ravana
python -m venv .venv
# Windows: .\.venv\Scripts\activate
pip install -e .
# GPU extras:
pip install -e ".[gpu]"       # NVIDIA
pip install -e ".[directml]"  # AMD Windows
```

CLI entry points after install: `ravana` or `face-swap`.

**First run** prefetch weights (or they download on first swap):

```bash
python scripts/first_run_check.py --download
# same as: python -m demos.cli models download --preset seamless
```

ONNX restores work on Python 3.13 without `basicsr`. Optional `.[enhancement]` PyTorch wheels are not required.

---

## Quick start

### Python

```python
import cv2
from face_swap import FaceSwapConfig, swap_image

config = FaceSwapConfig(
    quality="seamless",
    device="auto",
    # enhance_method="gpen",       # or restoreformer / codeformer / opencv
    # pixel_boost=1024,            # seamless default
    # face_select="largest",
)

out = swap_image(
    cv2.imread("source.jpg"),
    cv2.imread("target.jpg"),
    config,
)
cv2.imwrite("output.jpg", out)
```

### CLI

```bash
# Image
python -m demos.cli -s source.jpg -t target.jpg -o out.jpg -q seamless --device auto

# Restore override
python -m demos.cli -s source.jpg -t target.jpg -o out.jpg -q seamless --enhance gpen

# Video (audio remuxed when ffmpeg is available)
python -m demos.cli -s source.jpg -t input.mp4 -o output.mp4 -q medium

# Webcam (realtime path)
python -m demos.webcam_demo -s source.jpg --device auto --detect-every 3
# keys: q quit | d detect-N | e enhance | h HUD | r reset

# Models
python -m demos.cli models list --presets
python -m demos.cli models download --preset seamless

# Quality metrics
python -m demos.cli evaluate -s source.jpg -t target.jpg -o out.jpg -q seamless
python -m demos.cli evaluate-batch --pairs 2 --enhance gfpgan,gpen --boosts 512
```

### GUI

PySide6 desktop app with branded layout, drag-and-drop, source/target/result previews, batch processing, in-app video playback, and embedded webcam.

```bash
pip install -e ".[gui]"   # PySide6
python -m demos.gui
```

<p align="center">
  <img src="docs/assets/gui/gui_swap_result.png" width="900" alt="Ravana desktop GUI showing source, target, seamless result, and score metrics">
</p>

<p align="center">
  <img src="docs/assets/gui/gui_overview.png" width="900" alt="Ravana desktop GUI empty state with drop zones and controls">
</p>

Quality / device / restore / pixel-boost / face select, plus **Score**, **Run batch**, **Webcam**, and **Save As**.

---

## Gallery

Higher-resolution Unsplash demo portraits (replacing the old tiny thumbnails), swapped locally with `quality=seamless`:

<p align="center">
  <img src="docs/assets/gui/runs_strip.jpg" width="900" alt="Source, target, seamless, GPEN, and pixel-boost 1024 outputs">
</p>

<p align="center">
  <img src="docs/assets/gui/runs_strip_b.jpg" width="700" alt="Second demo pair: source, target, seamless result">
</p>

| Face select `all` | Face select `largest` |
|:-----------------:|:---------------------:|
| ![all](docs/assets/gui/two_all_preview.jpg) | ![largest](docs/assets/gui/two_largest_preview.jpg) |

---

## Comparisons

Illustrative Unsplash portraits for demo quality only — not for identity claims. Regenerate locally with the CLI.

### Source → target → seamless

| Source | Target | Seamless (`quality=seamless`) |
|:------:|:------:|:-----------------------------:|
| ![source](docs/assets/comparisons/woman1.jpg) | ![target](docs/assets/comparisons/man1.jpg) | ![seamless](docs/assets/comparisons/seamless_woman1_on_man1.jpg) |

Also available: [GPEN restore](docs/assets/comparisons/gpen_woman1_on_man1.jpg) on the same pair.

### Alternate pair

| Source | Target | Seamless |
|:------:|:------:|:--------:|
| ![source](docs/assets/comparisons/source_b.jpg) | ![target](docs/assets/comparisons/target_b.jpg) | ![seamless](docs/assets/comparisons/seamless_b.jpg) |

### Pixel boost 512 vs 1024

| Boost 512 | Boost 1024 |
|:---------:|:----------:|
| ![512](docs/assets/comparisons/boost_512.jpg) | ![1024](docs/assets/comparisons/boost_1024.jpg) |

### Face select: all vs largest

| `face_select=all` | `face_select=largest` |
|:-----------------:|:---------------------:|
| ![all](docs/assets/comparisons/two_all.jpg) | ![largest](docs/assets/comparisons/two_largest.jpg) |

### Metrics (local A/B, DirectML)

`evaluate-batch` on `woman1 → man1`, pixel-boost 512:

| Variant | ID similarity | Sharpness Δ | Time |
|---------|--------------:|------------:|-----:|
| GFPGAN | 0.309 | +16.9 | ~2.3 s |
| OpenCV | 0.324 | +6.5 | ~0.37 s |

Higher sharpness gain with GFPGAN; OpenCV is faster with slightly higher ArcFace ID on this pair. Re-run with your hardware:

```bash
python -m demos.cli evaluate-batch --pairs 1 --enhance gfpgan,opencv --boosts 512
```

---

## Pipeline

```mermaid
flowchart LR
  IN[Image / Video / Webcam] --> DET[Detect buffalo_l]
  DET --> SEL[Face select]
  SEL --> SWP[HyperSwap / InSwapper]
  SWP --> MASK[XSeg + lighting]
  MASK --> ENH[Restore + pixel boost]
  ENH --> TMP[Temporal EMA / flow]
  TMP --> OUT[Output]
```

| Preset | Behavior |
|--------|----------|
| `seamless` | HyperSwap + GFPGAN + XSeg + lighting + boost 1024 + video temporal |
| `high` / `medium` | InSwapper + lighter restore |
| `fast_cpu` / `low` | Speed path, enhance off |
| `realtime=True` | Any quality + detect-every-N + ROI + skip heavy restore (webcam) |

---

## Docs & tests

```bash
# Unit tests
python -m pytest tests/ -q

# MkDocs (optional)
pip install mkdocs-material mkdocstrings[python]
mkdocs serve
```

- Getting started: [docs/getting-started/quickstart.md](docs/getting-started/quickstart.md)
- Changelog: [CHANGELOG.md](CHANGELOG.md)

---

## Ethical use

For entertainment, VFX, privacy, and creative work **with consent**. Do not use for non-consensual deepfakes, impersonation, or fraud. You are responsible for complying with local law. Optional invisible DCT watermarking is available for provenance.

Third-party model licenses (InsightFace, FaceFusion HyperSwap ResearchRAIL, GFPGAN/GPEN/etc.) apply to downloaded weights review those before commercial use.

---

## License

MIT see [LICENSE](LICENSE).
