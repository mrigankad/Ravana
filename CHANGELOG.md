# Changelog

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
- Expanded suite (~161 tests) covering seamless, HyperSwap, XSeg, GPEN, metrics, face select, temporal

## 0.2.0

Initial packaged SDK baseline (pipeline, demos, CI scaffolding).
