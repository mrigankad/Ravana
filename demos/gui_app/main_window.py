"""Ravana PySide6 main window."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from demos.gui_app.config import VIDEO_EXTS, build_config
from demos.gui_app.preview import IMAGE_EXTS, PreviewStage, load_path_pixmap
from demos.gui_app.workers import ScoreWorker, SwapWorker, WebcamWorker

VERSION = "0.3.2"


def _mascot_path() -> Path:
    return Path(__file__).resolve().parents[2] / "docs" / "assets" / "mascot.png"


def _mascot_icon_path() -> Path:
    root = Path(__file__).resolve().parents[2] / "docs" / "assets"
    ico = root / "mascot.ico"
    return ico if ico.is_file() else root / "mascot.png"


def load_mascot_pixmap(max_side: int = 48) -> QPixmap:
    path = _mascot_path()
    if not path.is_file():
        return QPixmap()
    pix = QPixmap(str(path))
    if pix.isNull():
        return QPixmap()
    return pix.scaled(
        max_side,
        max_side,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )


def load_app_icon():
    """Windows taskbar/title-bar icon (prefers .ico)."""
    from PySide6.QtGui import QIcon

    path = _mascot_icon_path()
    if not path.is_file():
        return QIcon()
    return QIcon(str(path))


def _reveal_in_explorer(path: str) -> None:
    p = Path(path)
    if sys.platform == "win32":
        subprocess.run(["explorer", "/select,", str(p)], check=False)
    elif sys.platform == "darwin":
        subprocess.run(["open", "-R", str(p)], check=False)
    else:
        subprocess.run(["xdg-open", str(p.parent)], check=False)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"Ravana {VERSION}")
        self.resize(1280, 840)
        self.setMinimumSize(1000, 680)
        icon = load_app_icon()
        if not icon.isNull():
            self.setWindowIcon(icon)

        self._source_path: Optional[str] = None
        self._target_path: Optional[str] = None
        self._output_path: Optional[str] = None
        self._swap_worker: Optional[SwapWorker] = None
        self._score_worker: Optional[ScoreWorker] = None
        self._webcam_worker: Optional[WebcamWorker] = None
        self._last_dir = str(Path.home())

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(14, 10, 14, 12)
        root.setSpacing(10)

        self._build_menu()
        root.addLayout(self._build_header())

        body = QHBoxLayout()
        body.setSpacing(12)
        body.addWidget(self._build_left_rail(), stretch=0)
        self.preview = PreviewStage()
        self.preview.source_dropped.connect(self._apply_source)
        self.preview.target_dropped.connect(self._apply_target)
        body.addWidget(self.preview, stretch=1)
        root.addLayout(body, stretch=1)

        root.addWidget(self._build_score_strip())
        root.addLayout(self._build_actions())
        root.addLayout(self._build_footer())

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        open_src = QAction("Open &source…", self)
        open_src.setShortcut(QKeySequence("Ctrl+O"))
        open_src.triggered.connect(self._select_source)
        open_tgt = QAction("Open &target…", self)
        open_tgt.setShortcut(QKeySequence("Ctrl+T"))
        open_tgt.triggered.connect(self._select_target)
        save = QAction("&Save result as…", self)
        save.setShortcut(QKeySequence("Ctrl+S"))
        save.triggered.connect(self._save_as)
        reveal = QAction("Reveal result in folder", self)
        reveal.triggered.connect(self._reveal_result)
        quit_act = QAction("&Quit", self)
        quit_act.setShortcut(QKeySequence("Ctrl+Q"))
        quit_act.triggered.connect(self.close)
        file_menu.addAction(open_src)
        file_menu.addAction(open_tgt)
        file_menu.addSeparator()
        file_menu.addAction(save)
        file_menu.addAction(reveal)
        file_menu.addSeparator()
        file_menu.addAction(quit_act)

        run_menu = self.menuBar().addMenu("&Run")
        swap = QAction("Start face &swap", self)
        swap.setShortcut(QKeySequence("Ctrl+Return"))
        swap.triggered.connect(self._start_swap)
        score = QAction("S&core result", self)
        score.setShortcut(QKeySequence("Ctrl+E"))
        score.triggered.connect(self._score_result)
        cam = QAction("Toggle &webcam", self)
        cam.setShortcut(QKeySequence("Ctrl+W"))
        cam.triggered.connect(self._toggle_webcam)
        run_menu.addAction(swap)
        run_menu.addAction(score)
        run_menu.addAction(cam)

        help_menu = self.menuBar().addMenu("&Help")
        about = QAction("&About Ravana", self)
        about.triggered.connect(self._about)
        help_menu.addAction(about)

    def _build_header(self) -> QHBoxLayout:
        row = QHBoxLayout()
        mascot = QLabel()
        pix = load_mascot_pixmap(52)
        if not pix.isNull():
            mascot.setPixmap(pix)
        title = QLabel("Ravana")
        title.setStyleSheet("font-size: 22px; font-weight: 700;")
        tag = QLabel("Face-swap SDK")
        tag.setObjectName("muted")
        ver = QLabel(f"v{VERSION}")
        ver.setObjectName("muted")
        row.addWidget(mascot)
        row.addWidget(title)
        row.addWidget(tag)
        row.addStretch(1)
        row.addWidget(ver)
        return row

    def _build_left_rail(self) -> QWidget:
        rail = QWidget()
        rail.setFixedWidth(300)
        layout = QVBoxLayout(rail)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        brand = QLabel()
        brand.setAlignment(Qt.AlignmentFlag.AlignCenter)
        brand_pix = load_mascot_pixmap(110)
        if not brand_pix.isNull():
            brand.setPixmap(brand_pix)
        layout.addWidget(brand)

        files = QGroupBox("Files")
        fl = QVBoxLayout(files)
        fl.setSpacing(6)

        self.source_thumb = QLabel("Source preview")
        self.source_thumb.setObjectName("thumb")
        self.source_thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.source_btn = QPushButton("Select source image")
        self.source_btn.clicked.connect(self._select_source)
        self.source_name = QLabel("No source — drag onto Source pane")
        self.source_name.setObjectName("muted")
        self.source_name.setWordWrap(True)

        self.target_thumb = QLabel("Target preview")
        self.target_thumb.setObjectName("thumb")
        self.target_thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.target_btn = QPushButton("Select target image/video")
        self.target_btn.clicked.connect(self._select_target)
        self.target_name = QLabel("No target — drag onto Target pane")
        self.target_name.setObjectName("muted")
        self.target_name.setWordWrap(True)

        fl.addWidget(self.source_thumb)
        fl.addWidget(self.source_btn)
        fl.addWidget(self.source_name)
        fl.addSpacing(4)
        fl.addWidget(self.target_thumb)
        fl.addWidget(self.target_btn)
        fl.addWidget(self.target_name)
        layout.addWidget(files)

        settings = QGroupBox("Settings")
        form = QFormLayout(settings)
        form.setSpacing(8)

        self.quality = QComboBox()
        self.quality.addItem("Fast", "low")
        self.quality.addItem("Balanced", "medium")
        self.quality.addItem("Quality", "high")
        self.quality.addItem("Seamless", "seamless")
        self.quality.setCurrentIndex(3)  # seamless default

        self.device = QComboBox()
        self.device.addItem("Auto", "auto")
        self.device.addItem("AMD (DirectML)", "dml")
        self.device.addItem("GPU (CUDA)", "cuda")
        self.device.addItem("CPU", "cpu")

        self.enhance = QComboBox()
        for label, val in [
            ("Preset", "default"),
            ("GFPGAN", "gfpgan"),
            ("GPEN", "gpen"),
            ("RF++", "restoreformer"),
            ("CodeFormer", "codeformer"),
            ("OpenCV", "opencv"),
        ]:
            self.enhance.addItem(label, val)

        self.pixel_boost = QComboBox()
        for label, val in [
            ("Preset", "default"),
            ("512", "512"),
            ("1024", "1024"),
            ("Off", "0"),
        ]:
            self.pixel_boost.addItem(label, val)

        self.face_select = QComboBox()
        for label, val in [
            ("All", "all"),
            ("Largest", "largest"),
            ("First", "first"),
            ("Pose match", "pose"),
        ]:
            self.face_select.addItem(label, val)

        form.addRow("Quality", self.quality)
        form.addRow("Device", self.device)
        form.addRow("Restore", self.enhance)
        form.addRow("Pixel boost", self.pixel_boost)
        form.addRow("Target face", self.face_select)
        layout.addWidget(settings)

        live = QGroupBox("Webcam")
        live_form = QFormLayout(live)
        self.camera_id = QSpinBox()
        self.camera_id.setRange(0, 8)
        self.camera_id.setValue(0)
        self.detect_every = QSpinBox()
        self.detect_every.setRange(1, 10)
        self.detect_every.setValue(3)
        self.detect_every.setToolTip("Detect faces every N frames (higher = faster)")
        live_form.addRow("Camera ID", self.camera_id)
        live_form.addRow("Detect every N", self.detect_every)
        layout.addWidget(live)

        layout.addStretch(1)
        return rail

    def _build_score_strip(self) -> QLabel:
        self.score_chip = QLabel("Score: run a swap, then Score (Ctrl+E)")
        self.score_chip.setObjectName("scoreChip")
        self.score_chip.setWordWrap(True)
        return self.score_chip

    def _build_actions(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(8)
        self.swap_btn = QPushButton("Start face swap")
        self.swap_btn.setObjectName("primaryButton")
        self.swap_btn.setToolTip("Ctrl+Enter")
        self.swap_btn.clicked.connect(self._start_swap)

        self.score_btn = QPushButton("Score")
        self.score_btn.setToolTip("Ctrl+E")
        self.score_btn.clicked.connect(self._score_result)

        self.webcam_btn = QPushButton("Webcam")
        self.webcam_btn.setObjectName("accentButton")
        self.webcam_btn.setToolTip("Ctrl+W")
        self.webcam_btn.clicked.connect(self._toggle_webcam)

        self.save_btn = QPushButton("Save As…")
        self.save_btn.clicked.connect(self._save_as)

        self.reveal_btn = QPushButton("Open folder")
        self.reveal_btn.setObjectName("ghostButton")
        self.reveal_btn.clicked.connect(self._reveal_result)

        row.addWidget(self.swap_btn, stretch=2)
        row.addWidget(self.score_btn)
        row.addWidget(self.webcam_btn)
        row.addWidget(self.save_btn)
        row.addWidget(self.reveal_btn)
        return row

    def _build_footer(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.status = QLabel("Ready — drop files or use File menu")
        self.status.setObjectName("muted")
        row.addWidget(self.progress, stretch=1)
        row.addWidget(self.status)
        return row

    def _combo_data(self, box: QComboBox) -> str:
        return str(box.currentData())

    def _current_config(self, *, realtime: bool = False):
        cfg = build_config(
            self._combo_data(self.quality),
            self._combo_data(self.device),
            self._combo_data(self.enhance),
            self._combo_data(self.pixel_boost),
            self._combo_data(self.face_select),
            realtime=realtime,
        )
        cfg.detect_every_n = int(self.detect_every.value())
        return cfg

    def _set_status(self, text: str) -> None:
        self.status.setText(text)

    def _set_thumb(self, label: QLabel, path: Optional[str]) -> None:
        if not path:
            label.setPixmap(QPixmap())
            label.setText("No preview")
            return
        pix = load_path_pixmap(path, max_side=140)
        if pix.isNull():
            label.setText(Path(path).name)
            return
        label.setPixmap(
            pix.scaled(
                140,
                68,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        label.setText("")

    def _apply_source(self, path: str) -> None:
        if Path(path).suffix.lower() not in IMAGE_EXTS:
            QMessageBox.warning(self, "Invalid source", "Source must be an image.")
            return
        self._source_path = path
        self._output_path = None
        self._last_dir = str(Path(path).parent)
        self.source_name.setText(Path(path).name)
        self._set_thumb(self.source_thumb, path)
        self.preview.set_source_image(path)
        self.preview.set_result_image(None)
        self.score_chip.setText("Score: run a swap, then Score (Ctrl+E)")
        self._set_status(f"Source: {Path(path).name}")

    def _apply_target(self, path: str) -> None:
        ext = Path(path).suffix.lower()
        if ext not in IMAGE_EXTS and ext not in VIDEO_EXTS:
            QMessageBox.warning(self, "Invalid target", "Target must be image or video.")
            return
        self._target_path = path
        self._output_path = None
        self._last_dir = str(Path(path).parent)
        kind = "video" if ext in VIDEO_EXTS else "image"
        self.target_name.setText(f"{Path(path).name}  ({kind})")
        self._set_thumb(self.target_thumb, path)
        self.preview.set_target_image(path)
        self.preview.set_result_image(None)
        self.score_chip.setText("Score: run a swap, then Score (Ctrl+E)")
        self._set_status(f"Target: {Path(path).name}")

    def _select_source(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select source face image",
            self._last_dir,
            "Images (*.png *.jpg *.jpeg *.bmp *.webp);;All files (*.*)",
        )
        if path:
            self._apply_source(path)

    def _select_target(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select target image or video",
            self._last_dir,
            "Images & Videos (*.png *.jpg *.jpeg *.webp *.mp4 *.mov *.avi *.mkv);;"
            "Images (*.png *.jpg *.jpeg *.bmp *.webp);;"
            "Videos (*.mp4 *.mov *.avi *.mkv);;All files (*.*)",
        )
        if path:
            self._apply_target(path)

    def _start_swap(self) -> None:
        if self._swap_worker and self._swap_worker.isRunning():
            QMessageBox.information(self, "Busy", "Processing already in progress.")
            return
        if self._webcam_worker and self._webcam_worker.isRunning():
            QMessageBox.information(self, "Busy", "Stop webcam before swapping.")
            return
        if not self._source_path:
            QMessageBox.warning(self, "Missing input", "Select a source face image.")
            return
        if not self._target_path:
            QMessageBox.warning(self, "Missing input", "Select a target image or video.")
            return

        target = Path(self._target_path)
        self._output_path = str(target.parent / f"{target.stem}_swapped{target.suffix}")
        self.swap_btn.setEnabled(False)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self._set_status("Processing...")

        worker = SwapWorker(
            self._source_path,
            self._target_path,
            self._output_path,
            self._current_config(),
        )
        worker.progress.connect(self.progress.setValue)
        worker.status.connect(self._set_status)
        worker.finished.connect(self._on_swap_finished)
        worker.failed.connect(self._on_swap_failed)
        self._swap_worker = worker
        worker.start()

    def _on_swap_finished(self, output: str) -> None:
        self.swap_btn.setEnabled(True)
        self._output_path = output
        self.preview.set_result_image(output)
        self._set_status(f"Done: {Path(output).name}")

    def _on_swap_failed(self, error: str) -> None:
        self.swap_btn.setEnabled(True)
        self.progress.setValue(0)
        self._set_status("Error")
        QMessageBox.critical(self, "Swap failed", error)

    def _score_result(self) -> None:
        if not self._source_path or not self._target_path:
            QMessageBox.warning(self, "Missing input", "Select source and target first.")
            return
        if not self._output_path or not Path(self._output_path).is_file():
            QMessageBox.warning(self, "No result", "Run a face swap first.")
            return
        if Path(self._output_path).suffix.lower() in VIDEO_EXTS:
            QMessageBox.information(
                self, "Score", "Score currently supports image results only."
            )
            return

        self._set_status("Scoring...")
        self.score_chip.setText("Score: computing…")
        worker = ScoreWorker(
            self._source_path,
            self._target_path,
            self._output_path,
            self._combo_data(self.device),
        )
        worker.finished.connect(self._on_score_done)
        worker.failed.connect(self._on_score_failed)
        self._score_worker = worker
        worker.start()

    def _on_score_done(self, msg: str) -> None:
        self._set_status("Score complete")
        self.score_chip.setText("Score: " + msg.replace("\n", "  |  "))
        QMessageBox.information(self, "Swap score", msg)

    def _on_score_failed(self, error: str) -> None:
        self._set_status("Score failed")
        self.score_chip.setText(f"Score failed: {error}")
        QMessageBox.critical(self, "Score error", error)

    def _toggle_webcam(self) -> None:
        if self._webcam_worker and self._webcam_worker.isRunning():
            self._webcam_worker.stop()
            self.webcam_btn.setText("Webcam")
            self.progress.setRange(0, 100)
            self.progress.setValue(0)
            self._set_status("Stopping webcam...")
            return
        if not self._source_path:
            QMessageBox.warning(self, "Missing input", "Select a source face image first.")
            return
        if self._swap_worker and self._swap_worker.isRunning():
            QMessageBox.information(self, "Busy", "Wait for swap to finish.")
            return

        self.preview.stop_playback()
        self.webcam_btn.setText("Stop webcam")
        self.progress.setRange(0, 0)  # indeterminate
        self._set_status("Starting webcam...")
        worker = WebcamWorker(
            self._source_path,
            self._current_config(realtime=True),
            camera_id=int(self.camera_id.value()),
        )
        worker.frame_ready.connect(self.preview.show_frame_bgr)
        worker.status.connect(self._set_status)
        worker.failed.connect(self._on_webcam_failed)
        worker.finished.connect(self._on_webcam_thread_finished)
        self._webcam_worker = worker
        worker.start()

    def _on_webcam_failed(self, error: str) -> None:
        self.webcam_btn.setText("Webcam")
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self._set_status("Webcam error")
        QMessageBox.critical(self, "Webcam error", error)

    def _on_webcam_thread_finished(self) -> None:
        self.webcam_btn.setText("Webcam")
        self.progress.setRange(0, 100)
        self.progress.setValue(0)

    def _save_as(self) -> None:
        if not self._output_path or not Path(self._output_path).is_file():
            QMessageBox.warning(self, "No result", "Nothing to save yet.")
            return
        src = Path(self._output_path)
        dest, _ = QFileDialog.getSaveFileName(
            self,
            "Save result",
            str(Path(self._last_dir) / src.name),
            f"*{src.suffix};;All files (*.*)",
        )
        if not dest:
            return
        shutil.copy2(src, dest)
        self._set_status(f"Copied to {Path(dest).name}")

    def _reveal_result(self) -> None:
        if not self._output_path or not Path(self._output_path).is_file():
            QMessageBox.warning(self, "No result", "Nothing to reveal yet.")
            return
        _reveal_in_explorer(self._output_path)

    def _about(self) -> None:
        QMessageBox.about(
            self,
            "About Ravana",
            f"<b>Ravana</b> v{VERSION}<br/>"
            "Open-source face-swap SDK<br/>"
            "HyperSwap / InSwapper · GFPGAN / GPEN · ONNX<br/><br/>"
            "Shortcuts: Ctrl+O source · Ctrl+T target · "
            "Ctrl+Enter swap · Ctrl+E score · Ctrl+W webcam",
        )

    def closeEvent(self, event) -> None:  # noqa: N802
        if self._webcam_worker and self._webcam_worker.isRunning():
            self._webcam_worker.stop()
            self._webcam_worker.wait(3000)
        self.preview.stop_playback()
        super().closeEvent(event)
