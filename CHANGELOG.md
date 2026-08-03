# Changelog

## 0.3.1 — 2026-08-04

Realtime polish, metrics A/B batch, RestoreFormer++, GUI Score, and first-run hardening.

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
- CLI: `python -m demos.cli models list|download|ensure`
- Presets: `core`, `seamless`, `enhance`, `all`

### Metrics & DX
- `evaluate_swap` / CLI `evaluate` — ArcFace ID, sharpness, LAB drift
- LICENSE (MIT), packaging via `pyproject.toml`, docs/README updates
- GUI wiring for quality, enhance, pixel boost, face select

### Tests
- Expanded suite covering seamless, HyperSwap, XSeg, GPEN, metrics, face select, temporal, realtime

## 0.2.0

Initial packaged SDK baseline (pipeline, demos, CI scaffolding).
