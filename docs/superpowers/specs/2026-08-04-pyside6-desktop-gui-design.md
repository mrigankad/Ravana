# Ravana Desktop GUI — PySide6 Redesign

**Date:** 2026-08-04  
**Status:** Draft for review  
**Approach:** A — single PySide6 app replaces Tkinter  

## Goal

Ship a branded, usable desktop GUI with **in-app** image preview, video playback, and live webcam — same swap/score/settings surface as today’s Tkinter app, without spawning a separate OpenCV window for webcam.

## Context

- Current UI: `demos/gui.py` (Tkinter). Webcam calls `WebcamDemo` and blocks on an external OpenCV window. Preview is a placeholder label (no real thumbnails).
- Core APIs stay unchanged: `FaceSwapConfig`, `swap_image`, `swap_video`, `evaluate_swap`, realtime webcam path via existing demo/pipeline helpers.
- Packaging: optional extra `.[gui]` → `PySide6` (LGPL; fits MIT SDK). Entry remains `python -m demos.gui`.

## Non-goals

- Web/Gradio UI, batch queue, training UI, cloud drag-drop
- Rewriting the face-swap pipeline
- Shipping PySide6 as a hard core dependency (must stay optional)

## UX layout

One main window (~1200×800, min ~960×640), dark theme with gold/red accents (mascot-aligned; avoid purple-default AI chrome).

| Zone | Responsibility |
|------|----------------|
| **Header** | Small mascot (`docs/assets/mascot.png` if present), title “Ravana”, version badge |
| **Left rail** | Source + target file pickers with thumbnails; settings: Quality, Device, Restore, Pixel boost, Face select |
| **Center stage** | Mode-dependent preview (see below) |
| **Action bar** | Swap, Score, Webcam (start/stop), Save As / Reveal in folder |
| **Footer** | Determinate progress + status text |

### Center stage modes

1. **Idle / image** — three panes: Source | Target | Result (QLabel + QPixmap, letterboxed).
2. **Video result** — embedded player for last output video (prefer `QMediaPlayer` + `QVideoWidget`; fallback OpenCV decode → `QImage` on a `QTimer` if media backend unavailable).
3. **Webcam live** — OpenCV capture + swap on a worker; paint frames into the same stage via `QImage` (no `cv2.imshow`). Start/Stop toggles; optional FPS overlay.

## Behavior

| Action | Behavior |
|--------|----------|
| Select source | Image only; show thumbnail; clear stale result |
| Select target | Image or video; thumbnail or first-frame still |
| Swap (image) | `QThread` worker → `swap_image`; update Result pane; enable Score/Save |
| Swap (video) | Worker → `swap_video` with progress callback → UI progress bar; then play result in stage |
| Score | `evaluate_swap` on last result vs source/target; dialog or status with ID / sharpness |
| Webcam | Require source; run realtime path into stage; Stop ends capture cleanly |
| Save As | Copy/export last result path via file dialog |

Settings map 1:1 to existing `FaceSwapConfig` fields used by the current GUI (`quality`, `device`, `enhance_method`, `pixel_boost`, `face_select`, `realtime` for webcam).

## Architecture

```
demos/gui.py              # entry: check PySide6, launch MainWindow
demos/gui_app/
  __init__.py
  main_window.py          # layout, wiring, theme
  workers.py              # SwapWorker, ScoreWorker, WebcamWorker (QThread)
  preview.py              # image panes + video/webcam surface
  theme.py                # colors, fonts, stylesheet
```

- **Workers** emit Qt signals (`progress`, `finished`, `error`) — never touch widgets from the worker thread.
- **WebcamWorker** owns `VideoCapture`; main thread only receives RGB frames as `QImage`/`numpy` copies.
- Keep Tkinter code out of the tree once Qt path works (or leave a one-line stub that errors with install hint — prefer full replace of `demos/gui.py` body).

## Packaging & docs

- `pyproject.toml`: `[project.optional-dependencies] gui = ["PySide6>=6.5.0"]`; document `pip install -e ".[gui]"`.
- README: GUI section → PySide6 install + `python -m demos.gui`.
- Changelog: note GUI rewrite under next patch (e.g. 0.3.2) when released.

## Testing

- Manual: image swap, video swap + play, score, webcam start/stop, missing-PySide6 message.
- Automated (light): import-guard unit test that mocks absence of PySide6 / or smoke-import workers’ config mapping without launching QApplication (optional; not blocking).

## Success criteria

1. No separate OpenCV window for webcam or result viewing.
2. Source/target/result visible as real images for still swaps.
3. Video progress + in-app playback of output.
4. Parity with current settings + Score.
5. `.[gui]` optional; core install without PySide6 still works for CLI.

## Risks

| Risk | Mitigation |
|------|------------|
| `QMediaPlayer` codecs missing on Windows | OpenCV timer fallback for result playback |
| Webcam blocking UI | Dedicated `QThread` + signalled frames |
| Large dependency | Optional extra only |
