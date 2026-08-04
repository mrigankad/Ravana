"""Ravana PySide6 main window."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
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
    QVBoxLayout,
    QWidget,
)

from demos.gui_app.config import VIDEO_EXTS, build_config
from demos.gui_app.preview import PreviewStage
from demos.gui_app.workers import ScoreWorker, SwapWorker, WebcamWorker

VERSION = "0.3.2"


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"Ravana {VERSION}")
        self.resize(1200, 800)
        self.setMinimumSize(960, 640)

        self._source_path: Optional[str] = None
        self._target_path: Optional[str] = None
        self._output_path: Optional[str] = None
        self._swap_worker: Optional[SwapWorker] = None
        self._score_worker: Optional[ScoreWorker] = None
        self._webcam_worker: Optional[WebcamWorker] = None

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        root.addLayout(self._build_header())

        body = QHBoxLayout()
        body.addWidget(self._build_left_rail(), stretch=0)
        self.preview = PreviewStage()
        body.addWidget(self.preview, stretch=1)
        root.addLayout(body, stretch=1)

        root.addLayout(self._build_actions())
        root.addLayout(self._build_footer())

    def _build_header(self) -> QHBoxLayout:
        row = QHBoxLayout()
        mascot = QLabel()
        mascot_path = Path(__file__).resolve().parents[2] / "docs" / "assets" / "mascot.png"
        if mascot_path.is_file():
            pix = QPixmap(str(mascot_path))
            if not pix.isNull():
                mascot.setPixmap(
                    pix.scaled(
                        48,
                        48,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                )
        title = QLabel("Ravana")
        title.setStyleSheet("font-size: 22px; font-weight: 700;")
        ver = QLabel(f"v{VERSION}")
        ver.setObjectName("muted")
        row.addWidget(mascot)
        row.addWidget(title)
        row.addWidget(ver)
        row.addStretch(1)
        return row

    def _build_left_rail(self) -> QWidget:
        rail = QWidget()
        rail.setFixedWidth(280)
        layout = QVBoxLayout(rail)
        layout.setContentsMargins(0, 0, 0, 0)

        files = QGroupBox("Files")
        fl = QVBoxLayout(files)
        self.source_btn = QPushButton("Select source image")
        self.source_btn.clicked.connect(self._select_source)
        self.source_name = QLabel("No source")
        self.source_name.setObjectName("muted")
        self.source_name.setWordWrap(True)
        self.target_btn = QPushButton("Select target image/video")
        self.target_btn.clicked.connect(self._select_target)
        self.target_name = QLabel("No target")
        self.target_name.setObjectName("muted")
        self.target_name.setWordWrap(True)
        fl.addWidget(self.source_btn)
        fl.addWidget(self.source_name)
        fl.addWidget(self.target_btn)
        fl.addWidget(self.target_name)
        layout.addWidget(files)

        settings = QGroupBox("Settings")
        form = QFormLayout(settings)
        self.quality = QComboBox()
        self.quality.addItem("Fast", "low")
        self.quality.addItem("Balanced", "medium")
        self.quality.addItem("Quality", "high")
        self.quality.addItem("Seamless", "seamless")
        self.quality.setCurrentIndex(1)

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
        layout.addStretch(1)
        return rail

    def _build_actions(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self.swap_btn = QPushButton("Start face swap")
        self.swap_btn.setObjectName("primaryButton")
        self.swap_btn.clicked.connect(self._start_swap)

        self.score_btn = QPushButton("Score")
        self.score_btn.clicked.connect(self._score_result)

        self.webcam_btn = QPushButton("Webcam")
        self.webcam_btn.setObjectName("accentButton")
        self.webcam_btn.clicked.connect(self._toggle_webcam)

        self.save_btn = QPushButton("Save As…")
        self.save_btn.clicked.connect(self._save_as)

        row.addWidget(self.swap_btn, stretch=2)
        row.addWidget(self.score_btn)
        row.addWidget(self.webcam_btn)
        row.addWidget(self.save_btn)
        return row

    def _build_footer(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.status = QLabel("Ready")
        self.status.setObjectName("muted")
        row.addWidget(self.progress, stretch=1)
        row.addWidget(self.status)
        return row

    def _combo_data(self, box: QComboBox) -> str:
        return str(box.currentData())

    def _current_config(self, *, realtime: bool = False):
        return build_config(
            self._combo_data(self.quality),
            self._combo_data(self.device),
            self._combo_data(self.enhance),
            self._combo_data(self.pixel_boost),
            self._combo_data(self.face_select),
            realtime=realtime,
        )

    def _set_status(self, text: str) -> None:
        self.status.setText(text)

    def _select_source(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select source face image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp);;All files (*.*)",
        )
        if not path:
            return
        self._source_path = path
        self._output_path = None
        self.source_name.setText(Path(path).name)
        self.preview.set_source_image(path)
        self.preview.set_result_image(None)
        self._set_status(f"Source: {Path(path).name}")

    def _select_target(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select target image or video",
            "",
            "Images & Videos (*.png *.jpg *.jpeg *.mp4 *.mov *.avi *.mkv);;"
            "Images (*.png *.jpg *.jpeg *.bmp);;"
            "Videos (*.mp4 *.mov *.avi *.mkv);;All files (*.*)",
        )
        if not path:
            return
        self._target_path = path
        self._output_path = None
        self.target_name.setText(Path(path).name)
        self.preview.set_target_image(path)
        self.preview.set_result_image(None)
        self._set_status(f"Target: {Path(path).name}")

    def _start_swap(self) -> None:
        if self._swap_worker and self._swap_worker.isRunning():
            QMessageBox.information(self, "Busy", "Processing already in progress.")
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
        QMessageBox.information(self, "Complete", f"Saved to:\n{output}")

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
        QMessageBox.information(self, "Swap score", msg)

    def _on_score_failed(self, error: str) -> None:
        self._set_status("Score failed")
        QMessageBox.critical(self, "Score error", error)

    def _toggle_webcam(self) -> None:
        if self._webcam_worker and self._webcam_worker.isRunning():
            self._webcam_worker.stop()
            self.webcam_btn.setText("Webcam")
            self._set_status("Stopping webcam...")
            return
        if not self._source_path:
            QMessageBox.warning(self, "Missing input", "Select a source face image first.")
            return

        self.preview.stop_playback()
        self.webcam_btn.setText("Stop webcam")
        self._set_status("Starting webcam...")
        worker = WebcamWorker(
            self._source_path, self._current_config(realtime=True), camera_id=0
        )
        worker.frame_ready.connect(self.preview.show_frame_bgr)
        worker.status.connect(self._set_status)
        worker.failed.connect(self._on_webcam_failed)
        worker.finished.connect(self._on_webcam_thread_finished)
        self._webcam_worker = worker
        worker.start()

    def _on_webcam_failed(self, error: str) -> None:
        self.webcam_btn.setText("Webcam")
        self._set_status("Webcam error")
        QMessageBox.critical(self, "Webcam error", error)

    def _on_webcam_thread_finished(self) -> None:
        self.webcam_btn.setText("Webcam")

    def _save_as(self) -> None:
        if not self._output_path or not Path(self._output_path).is_file():
            QMessageBox.warning(self, "No result", "Nothing to save yet.")
            return
        src = Path(self._output_path)
        dest, _ = QFileDialog.getSaveFileName(
            self,
            "Save result",
            str(src.name),
            f"*{src.suffix};;All files (*.*)",
        )
        if not dest:
            return
        shutil.copy2(src, dest)
        self._set_status(f"Copied to {Path(dest).name}")

    def closeEvent(self, event) -> None:  # noqa: N802
        if self._webcam_worker and self._webcam_worker.isRunning():
            self._webcam_worker.stop()
            self._webcam_worker.wait(3000)
        self.preview.stop_playback()
        super().closeEvent(event)
