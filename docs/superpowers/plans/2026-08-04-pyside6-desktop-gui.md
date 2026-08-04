# PySide6 Desktop GUI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Tkinter `demos/gui.py` with a PySide6 desktop app that shows real image previews, in-app video playback, and live webcam in the same window.

**Architecture:** Thin `demos/gui.py` entry checks for PySide6 and launches `MainWindow`. UI modules live under `demos/gui_app/` (theme, preview surface, QThread workers, main window). Workers call existing `face_swap` APIs (`FaceSwapConfig`, `swap_image`, `swap_video`, `evaluate_swap`) and never touch widgets directly.

**Tech Stack:** Python 3.9+, PySide6 ≥ 6.5 (optional extra), OpenCV, NumPy, existing `face_swap` package.

## Global Constraints

- PySide6 is **optional** via `pip install -e ".[gui]"` — not a core dependency.
- Entry point remains `python -m demos.gui`.
- No `cv2.imshow` for webcam or result viewing.
- Settings parity with current GUI: quality, device, enhance_method, pixel_boost, face_select.
- Dark theme with gold/red accents (mascot-aligned); avoid purple AI-default chrome.
- Workers use Qt signals; UI updates only on the main thread.
- Spec: `docs/superpowers/specs/2026-08-04-pyside6-desktop-gui-design.md`.

## File map

| Path | Role |
|------|------|
| `demos/gui.py` | Entry: import guard + `QApplication` + `MainWindow` |
| `demos/gui_app/__init__.py` | Package marker |
| `demos/gui_app/theme.py` | Colors + QSS stylesheet |
| `demos/gui_app/preview.py` | Image triple panes + OpenCV video/webcam `QLabel` surface |
| `demos/gui_app/workers.py` | `build_config`, `SwapWorker`, `ScoreWorker`, `WebcamWorker` |
| `demos/gui_app/main_window.py` | Layout, file dialogs, wiring |
| `tests/test_gui_config.py` | Pure-Python config mapping tests (no QApplication) |
| `pyproject.toml` | `gui` optional dependency |
| `README.md` | GUI install/run blurb |
| `CHANGELOG.md` | Note under Unreleased / 0.3.2 |

---

### Task 1: Config helper + unit test (no Qt)

**Files:**
- Create: `demos/gui_app/__init__.py`
- Create: `demos/gui_app/workers.py` (config helper only first)
- Create: `tests/test_gui_config.py`
- Modify: `pyproject.toml` (add `gui` extra)

**Interfaces:**
- Produces: `build_config(quality: str, device: str, enhance: str, pixel_boost: str, face_select: str, *, realtime: bool = False) -> FaceSwapConfig`
- Mapping: `enhance == "default"` → `enhance_method=None`; `pixel_boost == "default"` → `pixel_boost=None`; else `int(pixel_boost)`.

- [ ] **Step 1: Write failing test**

```python
# tests/test_gui_config.py
from demos.gui_app.workers import build_config


def test_build_config_defaults():
    cfg = build_config("seamless", "auto", "default", "default", "all")
    assert cfg.quality == "seamless"
    assert cfg.device == "auto"
    assert cfg.enhance_method is None
    assert cfg.pixel_boost is None
    assert cfg.face_select == "all"
    assert cfg.realtime is False


def test_build_config_overrides_and_realtime():
    cfg = build_config("high", "dml", "gpen", "1024", "largest", realtime=True)
    assert cfg.enhance_method == "gpen"
    assert cfg.pixel_boost == 1024
    assert cfg.face_select == "largest"
    assert cfg.realtime is True
```

- [ ] **Step 2: Run test — expect fail**

```bash
python -m pytest tests/test_gui_config.py -v
```

Expected: `ModuleNotFoundError` or import error for `demos.gui_app.workers`.

- [ ] **Step 3: Implement helper**

```python
# demos/gui_app/__init__.py
"""PySide6 desktop GUI package for Ravana."""

# demos/gui_app/workers.py
from __future__ import annotations

from face_swap import FaceSwapConfig


def build_config(
    quality: str,
    device: str,
    enhance: str,
    pixel_boost: str,
    face_select: str,
    *,
    realtime: bool = False,
) -> FaceSwapConfig:
    return FaceSwapConfig(
        quality=quality,
        device=device,
        enhance_method=None if enhance == "default" else enhance,
        pixel_boost=None if pixel_boost == "default" else int(pixel_boost),
        face_select=face_select,
        realtime=realtime,
    )
```

Add to `pyproject.toml` under `[project.optional-dependencies]`:

```toml
gui = ["PySide6>=6.5.0"]
```

Update `all` extra to include `gui` only if it already lists other extras by name — append `,gui` inside the `ravana-sdk[...]` string.

- [ ] **Step 4: Run test — expect pass**

```bash
python -m pytest tests/test_gui_config.py -v
```

- [ ] **Step 5: Commit**

```bash
git add demos/gui_app/__init__.py demos/gui_app/workers.py tests/test_gui_config.py pyproject.toml
git commit -m "feat(gui): add FaceSwapConfig builder and gui optional extra"
```

---

### Task 2: Theme + preview widgets

**Files:**
- Create: `demos/gui_app/theme.py`
- Create: `demos/gui_app/preview.py`

**Interfaces:**
- Produces: `APP_STYLESHEET: str`, `COLORS: dict`
- Produces: `class PreviewStage(QWidget)` with methods:
  - `set_source_image(path: str | None) -> None`
  - `set_target_image(path: str | None) -> None`  # image path or video first-frame
  - `set_result_image(path: str | None) -> None`
  - `show_image_mode() -> None`
  - `show_video_path(path: str) -> None`  # OpenCV+QTimer playback
  - `show_frame_bgr(frame) -> None`  # webcam live
  - `stop_playback() -> None`

- [ ] **Step 1: Implement `theme.py`**

```python
# demos/gui_app/theme.py
COLORS = {
    "bg": "#121212",
    "card": "#1c1c1c",
    "text": "#f2f2f2",
    "muted": "#9a9a9a",
    "accent": "#c9a227",  # gold
    "danger": "#c0392b",  # red
    "border": "#2a2a2a",
}

APP_STYLESHEET = f"""
QMainWindow, QWidget {{ background-color: {COLORS['bg']}; color: {COLORS['text']}; }}
QPushButton {{
  background-color: {COLORS['card']}; color: {COLORS['text']};
  border: 1px solid {COLORS['border']}; padding: 8px 14px; border-radius: 4px;
}}
QPushButton#primaryButton {{
  background-color: {COLORS['danger']}; border: none; font-weight: 600;
}}
QPushButton#accentButton {{
  background-color: {COLORS['accent']}; color: #111; border: none; font-weight: 600;
}}
QComboBox, QProgressBar {{ background-color: {COLORS['card']}; border: 1px solid {COLORS['border']}; }}
QProgressBar::chunk {{ background-color: {COLORS['accent']}; }}
QLabel#muted {{ color: {COLORS['muted']}; }}
QGroupBox {{ border: 1px solid {COLORS['border']}; margin-top: 8px; padding-top: 8px; }}
"""
```

- [ ] **Step 2: Implement `preview.py`**

Use `QStackedWidget`: page 0 = horizontal `QHBoxLayout` of three `QLabel`s (Source/Target/Result); page 1 = single `QLabel` for video/webcam frames.

Helpers (same file):

```python
def bgr_to_qpixmap(frame, max_side: int = 640) -> QPixmap:
    import cv2
    from PySide6.QtGui import QImage, QPixmap
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
    pix = QPixmap.fromImage(qimg)
    if max(w, h) > max_side:
        pix = pix.scaled(max_side, max_side, Qt.AspectRatioMode.KeepAspectRatio,
                         Qt.TransformationMode.SmoothTransformation)
    return pix


def load_path_pixmap(path: str, max_side: int = 320) -> QPixmap:
    import cv2
    from pathlib import Path
    suffix = Path(path).suffix.lower()
    if suffix in {".mp4", ".mov", ".avi", ".mkv"}:
        cap = cv2.VideoCapture(path)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            return QPixmap()
        return bgr_to_qpixmap(frame, max_side)
    img = cv2.imread(path)
    if img is None:
        return QPixmap()
    return bgr_to_qpixmap(img, max_side)
```

`show_video_path`: open `cv2.VideoCapture`, `QTimer` ~33ms, read frame → `show_frame_bgr`; on EOF restart or stop. `stop_playback` releases capture and stops timer.

- [ ] **Step 3: Smoke import (manual)**

```bash
python -c "from demos.gui_app.theme import APP_STYLESHEET; from demos.gui_app import preview; print('ok', len(APP_STYLESHEET))"
```

Requires PySide6 installed: `pip install -e ".[gui]"`.

- [ ] **Step 4: Commit**

```bash
git add demos/gui_app/theme.py demos/gui_app/preview.py
git commit -m "feat(gui): add theme and preview stage widgets"
```

---

### Task 3: QThread workers (swap, score, webcam)

**Files:**
- Modify: `demos/gui_app/workers.py`

**Interfaces:**
- Consumes: `build_config(...)`
- Produces:
  - `SwapWorker(source, target, output, config)` signals: `progress(int)`, `status(str)`, `finished(str)`, `failed(str)`
  - `ScoreWorker(source, target, result, device)` signals: `finished(str)`, `failed(str)`
  - `WebcamWorker(source_path, config, camera_id=0)` signals: `frame_ready(object)`, `status(str)`, `failed(str)`; method `stop()`

- [ ] **Step 1: Add worker classes to `workers.py`**

```python
from PySide6.QtCore import QThread, Signal
import cv2
from pathlib import Path
from face_swap import swap_image, swap_video, evaluate_swap, FaceSwapPipeline
from face_swap.core.fast_video import FastVideoConfig

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}


class SwapWorker(QThread):
    progress = Signal(int)
    status = Signal(str)
    finished = Signal(str)
    failed = Signal(str)

    def __init__(self, source: str, target: str, output: str, config):
        super().__init__()
        self.source, self.target, self.output, self.config = source, target, output, config

    def run(self):
        try:
            if Path(self.target).suffix.lower() in VIDEO_EXTS:
                self.status.emit("Processing video...")
                def _cb(i, total):
                    self.progress.emit(int(i / max(total, 1) * 100))
                    self.status.emit(f"Frame {i}/{total}")
                swap_video(self.source, self.target, self.output, self.config, progress_callback=_cb)
            else:
                self.status.emit("Running face swap...")
                self.progress.emit(20)
                src = cv2.imread(self.source)
                tgt = cv2.imread(self.target)
                if src is None or tgt is None:
                    raise RuntimeError("Could not load source or target image")
                self.progress.emit(40)
                out = swap_image(src, tgt, self.config)
                self.progress.emit(90)
                cv2.imwrite(self.output, out)
            self.progress.emit(100)
            self.finished.emit(self.output)
        except Exception as e:
            self.failed.emit(str(e))


class ScoreWorker(QThread):
    finished = Signal(str)
    failed = Signal(str)

    def __init__(self, source: str, target: str, result: str, device: str):
        super().__init__()
        self.source, self.target, self.result, self.device = source, target, result, device

    def run(self):
        try:
            m = evaluate_swap(self.source, self.target, self.result, device=self.device)
            msg = (
                f"id={m.id_similarity:.3f}  vs_tgt={m.id_vs_target:.3f}\n"
                f"sharp={m.sharpness_result:.1f} (d{m.sharpness_gain:+.1f})\n"
                f"dE={m.color_delta_lab:.1f}  "
                f"{'PASS' if m.passed_id else 'WEAK'} id"
            )
            self.finished.emit(msg)
        except Exception as e:
            self.failed.emit(str(e))


class WebcamWorker(QThread):
    frame_ready = Signal(object)  # BGR ndarray
    status = Signal(str)
    failed = Signal(str)

    def __init__(self, source_path: str, config, camera_id: int = 0):
        super().__init__()
        self.source_path = source_path
        self.config = config
        self.camera_id = camera_id
        self._running = True

    def stop(self):
        self._running = False

    def run(self):
        try:
            cfg = self.config
            if not getattr(cfg, "realtime", False):
                cfg = FaceSwapConfig(
                    quality=cfg.quality,
                    device=cfg.device,
                    enhance_method=cfg.enhance_method,
                    pixel_boost=cfg.pixel_boost,
                    face_select=cfg.face_select,
                    realtime=True,
                )
            pipe_cfg = cfg.to_pipeline_config()
            pipe_cfg.enable_temporal = True
            if pipe_cfg.fast_video is None or not pipe_cfg.fast_video.enabled:
                n = max(1, int(getattr(cfg, "detect_every_n", 3) or 3))
                pipe_cfg.fast_video = FastVideoConfig(
                    enabled=True,
                    detect_every_n=n,
                    det_size=(320, 320),
                    detect_max_side=640,
                    skip_enhance=True,
                    max_faces=1,
                    roi_track=True,
                )
                pipe_cfg.video_detect_every_n = n
                pipe_cfg.enable_enhance = False
            pipeline = FaceSwapPipeline(pipe_cfg)
            src = cv2.imread(self.source_path)
            if src is None:
                raise RuntimeError("Could not load source image")
            emb = pipeline.extract_source_embedding(src)
            cap = cv2.VideoCapture(self.camera_id)
            if not cap.isOpened():
                raise RuntimeError(f"Camera {self.camera_id} failed to open")
            self.status.emit("Webcam live")
            frame_i = 0
            while self._running:
                ok, frame = cap.read()
                if not ok:
                    break
                out = pipeline.process_video_frame(frame, emb, frame_i)
                frame_i += 1
                self.frame_ready.emit(out.copy())
            cap.release()
            self.status.emit("Webcam stopped")
        except Exception as e:
            self.failed.emit(str(e))
```

Import `FaceSwapConfig` at top of file (already used by `build_config`). Use `FaceSwapPipeline.process_video_frame(frame, embedding, frame_number)` — same as `demos/webcam_demo.py` (returns BGR `ndarray`).

- [ ] **Step 2: Confirm signature**

```bash
python -c "import inspect; from face_swap.pipeline import FaceSwapPipeline; print(inspect.signature(FaceSwapPipeline.process_video_frame))"
```

Expected: `(self, frame, source_embedding, frame_number=0)`.

- [ ] **Step 3: Commit**

```bash
git add demos/gui_app/workers.py
git commit -m "feat(gui): add swap, score, and webcam QThread workers"
```

---

### Task 4: MainWindow + entry point

**Files:**
- Create: `demos/gui_app/main_window.py`
- Rewrite: `demos/gui.py`

**Interfaces:**
- Consumes: `APP_STYLESHEET`, `PreviewStage`, `build_config`, workers
- Produces: `class MainWindow(QMainWindow)`, `main()` launches app

- [ ] **Step 1: Implement `MainWindow`**

Layout:
- Header: mascot `QLabel` (load `docs/assets/mascot.png` if exists, height 48) + title + `v0.3.1`
- Left `QGroupBox` “Files”: Source button + thumb label; Target button + thumb label
- Left `QGroupBox` “Settings”: `QComboBox` for quality (`low/medium/high/seamless`), device (`auto/dml/cuda/cpu`), enhance (`default/gfpgan/gpen/restoreformer/codeformer/opencv`), pixel boost (`default/512/1024/0`), face select (`all/largest/first/pose`)
- Center: `PreviewStage`
- Bottom actions: Swap (`objectName=primaryButton`), Score, Webcam toggle (`objectName=accentButton`), Save As
- Footer: `QProgressBar` + status `QLabel`

State: `_source_path`, `_target_path`, `_output_path`, `_swap_worker`, `_score_worker`, `_webcam_worker`.

Wiring:
- Swap → validate paths → default output `{stem}_swapped{suffix}` → start `SwapWorker` → on finished set result image or `show_video_path`
- Score → `ScoreWorker` → `QMessageBox.information`
- Webcam → if running, `stop()`; else require source, `build_config(..., realtime=True)`, start `WebcamWorker`, connect `frame_ready` → `preview.show_frame_bgr`
- Save As → `QFileDialog.getSaveFileName` + `shutil.copy2`

Disable Swap while processing; re-enable on finished/failed.

- [ ] **Step 2: Rewrite `demos/gui.py`**

```python
"""Ravana desktop GUI (PySide6)."""

def main():
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        raise SystemExit(
            "PySide6 is required for the GUI.\n"
            'Install with:  pip install -e ".[gui]"'
        ) from None

    import sys
    from demos.gui_app.main_window import MainWindow
    from demos.gui_app.theme import APP_STYLESHEET

    app = QApplication(sys.argv)
    app.setApplicationName("Ravana")
    app.setStyleSheet(APP_STYLESHEET)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Manual launch**

```bash
pip install -e ".[gui]"
python -m demos.gui
```

Expected: window opens, no traceback; missing PySide6 path shows install hint if tested in a clean env.

- [ ] **Step 4: Commit**

```bash
git add demos/gui.py demos/gui_app/main_window.py
git commit -m "feat(gui): PySide6 MainWindow with in-app preview and webcam"
```

---

### Task 5: Docs + changelog

**Files:**
- Modify: `README.md` (GUI section)
- Modify: `CHANGELOG.md` (Unreleased or 0.3.2 notes)

- [ ] **Step 1: Update README GUI section**

Replace:

```bash
python -m demos.gui
```

with:

```bash
pip install -e ".[gui]"   # PySide6
python -m demos.gui
```

One sentence: desktop app with in-app image/video preview and live webcam (no separate OpenCV window).

- [ ] **Step 2: CHANGELOG**

Under `## Unreleased` (create if missing):

```markdown
### GUI
- PySide6 desktop app replaces Tkinter: branded layout, image previews, in-app video playback, embedded webcam
- Optional install: `pip install -e ".[gui]"`
```

- [ ] **Step 3: Run unit tests**

```bash
python -m pytest tests/test_gui_config.py tests/ -q --ignore=tests/test_integration.py
```

(Or full `pytest tests/ -q` if integration is fast enough.)

- [ ] **Step 4: Commit**

```bash
git add README.md CHANGELOG.md
git commit -m "docs: document PySide6 GUI install and changelog"
```

---

## Spec coverage checklist

| Spec requirement | Task |
|------------------|------|
| Optional `[gui]` / PySide6 | 1, 5 |
| Header mascot + title | 4 |
| Left rail files + settings | 4 |
| Image triple preview | 2, 4 |
| Video in-app playback | 2, 4 |
| Webcam in-app (no imshow) | 3, 4 |
| Swap / Score / Save As | 3, 4 |
| Progress + status | 4 |
| Config parity | 1, 4 |
| README + changelog | 5 |
| Light automated test | 1 |

## Placeholder / consistency review

- No TBD steps; `WebcamWorker` must mirror `webcam_demo.py` if `swap_face` name differs (Task 3 Step 2 verifies).
- `build_config` signature shared by MainWindow and tests.
- Video extensions set `VIDEO_EXTS` used in worker and preview first-frame loader.
