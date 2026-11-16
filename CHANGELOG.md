# Changelog

## 0.4.0 (unreleased) — 2026-11-16

### Desktop app (redesign)
- Light theme with Inter bundled; new "ten faces" mark and app icon
- Title-bar tabs (Swap, Batch, Live, Models), context rail, compare canvas with draggable divider, Result and Side-by-side views
- Automatic ID and sharpness scoring after every image swap; swap time readout
- Inline banners replace modal dialogs; one-time consent gate before the first swap
- Cancel any running job with Esc; batch queue shows per-row status
- Responsive layout from 1440 px down to ~780 px
- 78-icon kit and Dasha character states (`design/kit/`)

### Engine
- Pipeline is cached between `swap_image` / `swap_video` calls (`face_swap.clear_cache()` to reset)
- Pixel-boost tiles no longer repeat; paste-back limited to the face ROI
- Unicode-safe image IO across the SDK and CLI
- RetinaFace receives BGR as InsightFace expects; modular components load lazily
- Video output is remuxed with source audio (H.264 + AAC) when ffmpeg is on PATH
- One-Euro keypoint smoothing; optical-flow blending limited to the face ROI
- torch is optional unless you load SimSwap PyTorch weights

### Fixes
- GUI worker signals no longer shadow `QThread.finished`; camera always released

## 0.3.4 — 2026-08-04

### Packaging
- Windows one-folder GUI executable via PyInstaller (`scripts/build_exe.ps1` → `dist/Ravana/`)
- Release zip helper (`scripts/package_release.ps1`); models stay outside the bundle

### GUI
- First-run prompt + Help → Download models (seamless preset into `./models` next to the exe)
- Frozen path helpers so mascot icon and `./models` work in the packaged app

## 0.3.3 — 2026-08-04

### GUI
- Drag-and-drop source/target panes, sidebar thumbnails, File/Run/Help menus + shortcuts
- Batch processing panel (add files/folder, output dir, progress, cancel)
- Webcam camera ID + detect-every-N; Open folder; inline score strip; mascot window icon (`.ico`)
- README gallery: desktop screenshots + run strip

## 0.3.2 — 2026-08-04

### GUI
- PySide6 desktop app replaces Tkinter: branded layout, image previews, in-app video playback, embedded webcam
- Optional install: `pip install -e ".[gui]"`

## 0.3.1 — 2026-08-04

Realtime polish, metrics A/B batch, RestoreFormer++, GUI Score, and first-run hardening.

### Packaging
- PyPI distribution name **`ravana-sdk`** (PyPI `ravana` is taken); console scripts `ravana` and `face-swap`
- README mascot header, comparison showcase, GitHub release artifacts

### Realtime
- `FaceSwapConfig(realtime=True)` — detect-every-N, skip heavy restore, ROI track between full detects
- Webcam HUD (FPS / detect N / ROI / enhance) + hotkeys `d` `e` `h` `r`
- `start_realtime_swap` defaults to realtime path with FPS overlay

### Metrics
- `evaluate-batch` / `scripts/metrics_batch.py` — enhance × pixel-boost matrix → CSV/JSON
- Restore ORT models forced to CPU when swapper is on DirectML (stable multi-variant runs)
- GUI **Score** button — ArcFace ID + sharpness on last output

### Quality
- **RestoreFormer++** ONNX restore (`--enhance restoreformer`)

### DX
- `scripts/first_run_check.py` — version, providers, model status, optional `--download`
- Quickstart docs: first-run, evaluate, realtime webcam

## 0.3.0 — 2026-08-04

Seamless quality stack and developer UX for the Ravana face-swap SDK.

### Quality
- `quality="seamless"` preset: HyperSwap-256, GFPGAN restore, XSeg occlusion, lighting match, grain, temporal video knobs
- Pixel boost default **1024** with overlapping restore tiles
- Optional restores: **CodeFormer**, **GPEN-BFR-512** (`--enhance`)
- Multi-face selection: `all | largest | first | index | pose`
- Tiny-face upscale + pose ranking helpers
- Video temporal: face EMA, optical-flow blend, detect-every-N

### Device & models
- `device="auto"` (CUDA → DirectML → CPU)
- Unified model download with progress (`ensure_downloaded` / `ModelManager`)
- CLI: `python -m demos.cli models st|download|ensure`
- Presets: `core`, `seamless`, `enhance`, `all`

### Metrics & DX
- `evaluate_swap` / CLI `evaluate` — ArcFace ID, sharpness, LAB drift
- LICENSE (MIT), packaging via `pyproject.toml`, docs/README updates
- GUI wiring for quality, enhance, pixel boost, face select

### Tests
- Expanded suite covering seamless, HyperSwap, XSeg, GPEN, metrics, face select, temporal, realtime

## 0.2.0

Initial packaged SDK baseline (pipeline, demos, CI scaffolding).
