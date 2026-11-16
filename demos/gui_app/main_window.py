"""Ravana 2.0 main window (light, Inter, responsive).

Layout mirrors the Paper design "Ravana — Redesign" / 01 Desktop app:
title bar with tabs, a context rail (Swap / Batch / Live / Models), and a stage
with a compare canvas, live scores and one primary action.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QSettings, QSize, Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from demos.gui_app.config import VIDEO_EXTS, build_config
from demos.gui_app.paths import models_dir
from demos.gui_app.preview import IMAGE_EXTS, bgr_to_qpixmap, load_path_pixmap
from demos.gui_app.theme import COLORS
from demos.gui_app.widgets import (
    DoctorPanel,
    Banner,
    CompareCanvas,
    DropCard,
    Metric,
    Segmented,
    SelectRow,
    app_icon,
    eyebrow,
    svg_icon,
    svg_pixmap,
)
from demos.gui_app.workers import (
    BatchWorker,
    ModelsDownloadWorker,
    ScoreWorker,
    SwapWorker,
    WebcamWorker,
    missing_preset_models,
)

VERSION = "0.4.0"

QUALITY_ITEMS = [("Fast", "low"), ("Medium", "medium"), ("High", "high"), ("Seamless", "seamless")]
QUALITY_HELP = {
    "low": "Fastest path, restore off. Good for previews and CPU-only machines.",
    "medium": "InSwapper with light restore. A good default for video.",
    "high": "InSwapper with GFPGAN restore and occlusion masks.",
    "seamless": "HyperSwap 256 with GFPGAN, XSeg occlusion, lighting match and 1024 pixel boost.",
}
RESTORE_ITEMS = [("Preset", "default"), ("GFPGAN", "gfpgan"), ("GPEN", "gpen"),
                 ("RestoreFormer++", "restoreformer"), ("CodeFormer", "codeformer"),
                 ("OpenCV (fast)", "opencv")]
BOOST_ITEMS = [("Preset", "default"), ("512", "512"), ("1024", "1024"), ("Off", "0")]
FACES_ITEMS = [("Largest", "largest"), ("All", "all"), ("First", "first"), ("Pose match", "pose")]
DEVICE_ITEMS = [("Auto", "auto"), ("NVIDIA (CUDA)", "cuda"), ("DirectML", "dml"), ("CPU", "cpu")]


def load_app_icon():
    """Kept for demos/gui.py compatibility."""
    return app_icon()


def detect_device_label() -> str:
    """Human label for the best available ONNX Runtime provider."""
    try:
        import onnxruntime as ort

        providers = ort.get_available_providers()
    except Exception:  # noqa: BLE001
        return "CPU"
    if "TensorrtExecutionProvider" in providers:
        return "TensorRT"
    if "CUDAExecutionProvider" in providers:
        return "CUDA"
    if "DmlExecutionProvider" in providers:
        return "DirectML"
    if "CoreMLExecutionProvider" in providers:
        return "CoreML"
    return "CPU only"


def _open_folder(path: str) -> None:
    if sys.platform == "win32":
        os.startfile(path)  # noqa: S606
    elif sys.platform == "darwin":
        subprocess.run(["open", path], check=False)
    else:
        subprocess.run(["xdg-open", path], check=False)


def _reveal(path: str) -> None:
    if sys.platform == "win32":
        subprocess.run(["explorer", "/select,", str(Path(path))], check=False)
    else:
        _open_folder(str(Path(path).parent))


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Ravana")
        self.setWindowIcon(app_icon())
        self.resize(1440, 900)
        self.setMinimumSize(780, 560)
        self._settings = QSettings("Ravana", "Ravana")

        self._source_path: Optional[str] = None
        self._target_path: Optional[str] = None
        self._output_path: Optional[str] = None
        self._swap_worker: Optional[SwapWorker] = None
        self._score_worker: Optional[ScoreWorker] = None
        self._webcam_worker: Optional[WebcamWorker] = None
        self._batch_worker: Optional[BatchWorker] = None
        self._models_worker: Optional[ModelsDownloadWorker] = None
        self._batch_output_dir: Optional[str] = None
        self._last_dir = str(Path.home())
        self._pending_after_consent = None
        self._compact = None

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._build_title_bar())

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        body.addWidget(self._build_rail())
        body.addWidget(self._build_stage(), 1)
        root.addLayout(body, 1)

        self._build_shortcuts()
        self._apply_responsive(self.width())
        QTimer.singleShot(500, self._check_models)

    # ================================================================ title bar
    def _build_title_bar(self) -> QWidget:
        bar = QWidget()
        bar.setObjectName("titleBar")
        bar.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        bar.setFixedHeight(56)
        row = QHBoxLayout(bar)
        row.setContentsMargins(20, 0, 20, 0)
        row.setSpacing(12)

        mark = QLabel()
        mark.setPixmap(svg_pixmap("mark.svg", QSize(64, 18)))
        word = QLabel("Ravana")
        word.setObjectName("wordmark")
        row.addWidget(mark)
        row.addWidget(word)
        row.addSpacing(16)

        self._nav = QButtonGroup(self)
        self._nav.setExclusive(True)
        for i, (label, icon) in enumerate(
            [("Swap", "swap.svg"), ("Batch", "batch.svg"), ("Live", "live.svg"), ("Models", "models.svg")]
        ):
            b = QPushButton(label)
            b.setObjectName("navTab")
            b.setCheckable(True)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.setIcon(svg_icon(icon, 16, COLORS["muted"]))
            b.setIconSize(QSize(16, 16))
            b.clicked.connect(lambda _=False, idx=i: self._go(idx))
            self._nav.addButton(b, i)
            row.addWidget(b)
        self._nav.button(0).setChecked(True)
        row.addStretch(1)

        chip = QFrame()
        chip.setObjectName("chip")
        cl = QHBoxLayout(chip)
        cl.setContentsMargins(10, 5, 10, 5)
        cl.setSpacing(8)
        dot = QLabel()
        dot.setFixedSize(7, 7)
        dev = detect_device_label()
        dot.setStyleSheet(
            f"background: {COLORS['brass'] if dev != 'CPU only' else COLORS['faint']}; border-radius: 3px;"
        )
        self._device_label = QLabel(dev)
        self._device_label.setObjectName("label")
        self._device_label.setToolTip("Click for the hardware doctor")
        chip.setCursor(Qt.CursorShape.PointingHandCursor)
        chip.mousePressEvent = lambda _e: self._open_doctor()
        self._chip, self._chip_dot, self._doctor_report = chip, dot, None
        cl.addWidget(dot)
        cl.addWidget(self._device_label)
        chip.setFixedHeight(30)
        row.addWidget(chip, 0, Qt.AlignmentFlag.AlignVCenter)
        ver = QLabel(f"v{VERSION}")
        ver.setObjectName("version")
        row.addWidget(ver)
        return bar

    def _go(self, idx: int) -> None:
        self._nav.button(idx).setChecked(True)
        self._rail_stack.setCurrentIndex(idx)
        self._refresh_empty_state()

    def _refresh_empty_state(self) -> None:
        """Pick Dasha's pose and copy for the canvas when it has nothing to show."""
        idx = self._rail_stack.currentIndex() if hasattr(self, "_rail_stack") else 0
        if idx == 1:
            n = self.batch_list.count()
            self.canvas.set_empty_state(
                "batch", "Queue targets to swap in one go" if not n else f"{n} in the queue",
                "Add files or a whole folder. Results land in batch_out.")
        elif idx == 2:
            self.canvas.set_empty_state("live", "Your camera, live", "Start the camera to swap in real time.")
        elif idx == 3 or self._models_missing:
            self.canvas.set_empty_state("offline", "Models are not downloaded yet",
                                        "Download the seamless preset once. It stays on this machine.")
        else:
            self.canvas.set_empty_state("idle", "Drop a source face and a target to begin",
                                        "Images or video. Nothing leaves this machine.")

    # ================================================================ rail
    def _build_rail(self) -> QWidget:
        self._rail = QWidget()
        self._rail.setObjectName("rail")
        self._rail.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._rail.setFixedWidth(320)
        outer = QVBoxLayout(self._rail)
        outer.setContentsMargins(0, 0, 0, 0)
        self._rail_stack = QStackedWidget()
        for page in (self._swap_page(), self._batch_page(), self._live_page(), self._models_page()):
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            scroll.setWidget(page)
            self._rail_stack.addWidget(scroll)
        outer.addWidget(self._rail_stack)
        return self._rail

    def _page(self) -> tuple[QWidget, QVBoxLayout]:
        w = QWidget()
        col = QVBoxLayout(w)
        col.setContentsMargins(20, 24, 20, 20)
        col.setSpacing(28)
        return w, col

    def _section(self, title: str, *widgets: QWidget, spacing: int = 12) -> QWidget:
        box = QWidget()
        col = QVBoxLayout(box)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(spacing)
        col.addWidget(eyebrow(title))
        for w in widgets:
            col.addWidget(w)
        return box

    def _primary(self, text: str, hint: str = "") -> QPushButton:
        b = QPushButton(text + (f"    {hint}" if hint else ""))
        b.setObjectName("primary")
        b.setCursor(Qt.CursorShape.PointingHandCursor)
        b.setMinimumHeight(48)
        return b

    # ---------------------------------------------------------- swap page
    def _swap_page(self) -> QWidget:
        page, col = self._page()
        cards = QWidget()
        cr = QHBoxLayout(cards)
        cr.setContentsMargins(0, 0, 0, 0)
        cr.setSpacing(10)
        self.source_card = DropCard("Source face", "One clear face", images=True)
        self.target_card = DropCard("Target", "Image or video", images=True, videos=True)
        self.source_card.clicked.connect(self._select_source)
        self.target_card.clicked.connect(self._select_target)
        self.source_card.dropped.connect(self._apply_source)
        self.target_card.dropped.connect(self._apply_target)
        cr.addWidget(self.source_card)
        cr.addWidget(self.target_card)
        col.addWidget(self._section("Inputs", cards))

        self.quality = Segmented(QUALITY_ITEMS, "seamless", fill=True)
        self.quality_help = QLabel(QUALITY_HELP["seamless"])
        self.quality_help.setObjectName("muted")
        self.quality_help.setWordWrap(True)
        self.quality.changed.connect(lambda v: self.quality_help.setText(QUALITY_HELP[v]))
        col.addWidget(self._section("Quality", self.quality, self.quality_help))

        tune = QWidget()
        tl = QVBoxLayout(tune)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.setSpacing(0)
        self.restore = SelectRow("Restore", RESTORE_ITEMS, "default")
        self.boost = SelectRow("Pixel boost", BOOST_ITEMS, "default")
        self.faces = SelectRow("Faces", FACES_ITEMS, "largest")
        self.device = SelectRow("Device", DEVICE_ITEMS, "auto", last=True)
        for r in (self.restore, self.boost, self.faces, self.device):
            tl.addWidget(r)
        col.addWidget(self._section("Fine tune", tune, spacing=4))
        col.addStretch(1)

        self.swap_btn = self._primary("Swap face", "Ctrl ↵")
        self.swap_btn.clicked.connect(self._start_swap)
        note = QLabel("Use only with consent. Results can be watermarked.")
        note.setObjectName("muted")
        note.setAlignment(Qt.AlignmentFlag.AlignCenter)
        note.setWordWrap(True)
        self._consent_note = note
        col.addWidget(self.swap_btn)
        col.addWidget(note)
        col.setSpacing(24)
        return page

    # ---------------------------------------------------------- batch page
    def _batch_page(self) -> QWidget:
        page, col = self._page()
        self.batch_list = QListWidget()
        self.batch_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.batch_list.setMinimumHeight(220)
        self.batch_list.itemDoubleClicked.connect(self._batch_open_item)
        btns = QWidget()
        br = QHBoxLayout(btns)
        br.setContentsMargins(0, 0, 0, 0)
        br.setSpacing(6)
        for text, icon, slot in (("Files", "add-file.svg", self._batch_add_files),
                                 ("Folder", "add-folder.svg", self._batch_add_folder),
                                 ("Remove", "trash.svg", self._batch_remove_selected),
                                 ("Clear", "close.svg", self._batch_clear)):
            icon_only = text in ("Remove", "Clear")
            b = QPushButton("" if icon_only else text)
            b.setToolTip(text)
            b.setIcon(svg_icon(icon, 16))
            b.setIconSize(QSize(16, 16))
            if icon_only:
                b.setFixedWidth(40)
            else:
                b.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
            b.clicked.connect(slot)
            br.addWidget(b)
        hint = QLabel("Uses the source face from the Swap tab and its quality settings. "
                      "Double-click a finished row to open it.")
        hint.setObjectName("muted")
        hint.setWordWrap(True)
        col.addWidget(self._section("Queue", self.batch_list, btns, hint))

        out = QWidget()
        orow = QHBoxLayout(out)
        orow.setContentsMargins(0, 0, 0, 0)
        self.batch_out_label = QLabel("Next to the first target, in batch_out")
        self.batch_out_label.setObjectName("muted")
        self.batch_out_label.setWordWrap(True)
        pick = QPushButton("Change")
        pick.setIcon(svg_icon("folder.svg", 16))
        pick.clicked.connect(self._batch_pick_output)
        orow.addWidget(self.batch_out_label, 1)
        orow.addWidget(pick)
        col.addWidget(self._section("Output folder", out))
        col.addStretch(1)
        self.batch_btn = self._primary("Run batch", "Ctrl B")
        self.batch_btn.clicked.connect(self._start_batch)
        col.addWidget(self.batch_btn)
        return page

    # ---------------------------------------------------------- live page
    def _live_page(self) -> QWidget:
        page, col = self._page()
        info = QLabel("Live swaps your camera feed with the source face from the Swap tab. "
                      "Heavy restore is skipped to keep the frame rate up.")
        info.setObjectName("muted")
        info.setWordWrap(True)
        col.addWidget(info)
        self.camera = SelectRow("Camera", [(f"Camera {i}", str(i)) for i in range(4)], "0")
        self.detect_every = SelectRow(
            "Detect faces", [("Every frame", "1"), ("Every 2nd frame", "2"),
                             ("Every 3rd frame", "3"), ("Every 5th frame", "5")], "3", last=True)
        box = QWidget()
        bl = QVBoxLayout(box)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(0)
        bl.addWidget(self.camera)
        bl.addWidget(self.detect_every)
        col.addWidget(self._section("Camera", box, spacing=4))
        col.addStretch(1)
        self.webcam_btn = self._primary("Start camera", "Ctrl W")
        self.webcam_btn.clicked.connect(self._toggle_webcam)
        col.addWidget(self.webcam_btn)
        return page

    # ---------------------------------------------------------- models page
    def _models_page(self) -> QWidget:
        page, col = self._page()
        self.models_status = QLabel("Checking…")
        self.models_status.setWordWrap(True)
        folder = QPushButton("Open models folder")
        folder.setIcon(svg_icon("folder-open.svg", 16))
        folder.clicked.connect(lambda: (models_dir().mkdir(parents=True, exist_ok=True),
                                        _open_folder(str(models_dir()))))
        col.addWidget(self._section("Seamless preset", self.models_status, folder))
        lic = QLabel(
            "Ravana's code is MIT. Default weights are not: InsightFace models are "
            "non-commercial research, CodeFormer is NTU S-Lab and HyperSwap is ResearchRAIL. "
            "Check each licence before commercial use."
        )
        lic.setObjectName("muted")
        lic.setWordWrap(True)
        col.addWidget(self._section("Licences", lic))
        col.addStretch(1)
        self.models_btn = self._primary("Download models")
        self.models_btn.clicked.connect(self._start_model_download)
        col.addWidget(self.models_btn)
        return page

    # ================================================================ stage
    def _build_stage(self) -> QWidget:
        stage = QWidget()
        col = QVBoxLayout(stage)
        col.setContentsMargins(28, 24, 28, 16)
        col.setSpacing(16)
        self._stage_layout = col

        head = QHBoxLayout()
        head.setSpacing(12)
        titles = QVBoxLayout()
        titles.setSpacing(4)
        self.stage_title = QLabel("New swap")
        self.stage_title.setObjectName("stageTitle")
        self.stage_meta = QLabel("Add a source face and a target")
        self.stage_meta.setObjectName("muted")
        titles.addWidget(self.stage_title)
        titles.addWidget(self.stage_meta)
        head.addLayout(titles, 1)
        self.view_mode = Segmented([("Compare", "compare"), ("Result", "result"), ("Side by side", "side")])
        self.view_mode.changed.connect(lambda m: self.canvas.set_mode(m))
        self.save_btn = QPushButton("Save as…")
        self.save_btn.setIcon(svg_icon("save.svg", 16))
        self.save_btn.clicked.connect(self._save_as)
        self.reveal_btn = QPushButton("Show in folder")
        self.reveal_btn.setIcon(svg_icon("reveal.svg", 16, COLORS["muted"]))
        self.reveal_btn.setObjectName("ghost")
        self.reveal_btn.clicked.connect(self._reveal_result)
        for value, icon in (("compare", "compare.svg"), ("result", "result.svg"), ("side", "side-by-side.svg")):
            btn = self.view_mode._buttons[value]
            btn.setIcon(svg_icon(icon, 15, COLORS["muted"]))
            btn.setIconSize(QSize(15, 15))
        head.addWidget(self.view_mode, 0, Qt.AlignmentFlag.AlignBottom)
        head.addWidget(self.reveal_btn, 0, Qt.AlignmentFlag.AlignBottom)
        head.addWidget(self.save_btn, 0, Qt.AlignmentFlag.AlignBottom)
        col.addLayout(head)

        self.banner = Banner()
        self.banner.action.connect(self._banner_action)
        col.addWidget(self.banner)

        self.canvas = CompareCanvas()
        self.canvas.setAcceptDrops(False)
        col.addWidget(self.canvas, 1)

        read = QHBoxLayout()
        read.setSpacing(36)
        self.m_id = Metric("ID similarity · ArcFace")
        self.m_sharp = Metric("Sharpness gain", "metricValueBrass")
        self.m_time = Metric("Swap time")
        for m in (self.m_id, self.m_sharp, self.m_time):
            read.addWidget(m)
        read.addStretch(1)
        status_col = QVBoxLayout()
        status_col.setSpacing(6)
        self.status = QLabel("Ready")
        self.status.setObjectName("muted")
        self.status.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop)
        self.status.setWordWrap(True)
        self.status.setMinimumWidth(80)
        self.status.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setObjectName("ghost")
        self.cancel_btn.clicked.connect(self._cancel_current)
        self.cancel_btn.hide()
        status_col.addWidget(self.status)
        status_col.addWidget(self.cancel_btn, 0, Qt.AlignmentFlag.AlignRight)
        read.addLayout(status_col, 1)
        col.addLayout(read)

        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        col.addWidget(self.progress)
        self._set_result_actions(False)
        return stage

    # ================================================================ responsive
    def resizeEvent(self, e) -> None:  # noqa: N802
        super().resizeEvent(e)
        self._apply_responsive(self.width())

    def _apply_responsive(self, width: int) -> None:
        mode = "wide" if width >= 1240 else "medium" if width >= 960 else "narrow"
        if mode == self._compact:
            return
        self._compact = mode
        rail_w = {"wide": 320, "medium": 276, "narrow": 264}[mode]
        thumb_h = {"wide": 156, "medium": 112, "narrow": 92}[mode]
        self._rail.setFixedWidth(rail_w)
        self.source_card.set_thumb_height(thumb_h)
        self.target_card.set_thumb_height(thumb_h)
        self.quality_help.setVisible(mode == "wide")
        self.view_mode.set_visible_item("side", mode != "narrow")
        self.reveal_btn.setVisible(mode == "wide")
        self.stage_meta.setVisible(mode != "narrow")
        self.m_time.setVisible(mode != "narrow")
        m = {"wide": (28, 24, 28, 16), "medium": (20, 20, 20, 14), "narrow": (14, 14, 14, 10)}[mode]
        self._stage_layout.setContentsMargins(*m)

    # ================================================================ shortcuts
    def _build_shortcuts(self) -> None:
        for seq, slot in (
            ("Ctrl+O", self._select_source), ("Ctrl+T", self._select_target),
            ("Ctrl+Return", self._start_swap), ("Ctrl+S", self._save_as),
            ("Ctrl+B", self._start_batch), ("Ctrl+W", self._toggle_webcam),
            ("Ctrl+1", lambda: self._go(0)), ("Ctrl+2", lambda: self._go(1)),
            ("Ctrl+3", lambda: self._go(2)), ("Ctrl+4", lambda: self._go(3)),
            ("Escape", self._cancel_current),
        ):
            act = QAction(self)
            act.setShortcut(QKeySequence(seq))
            act.triggered.connect(slot)
            self.addAction(act)

    # ================================================================ helpers
    def _set_status(self, text: str) -> None:
        self.status.setText(text)

    def _error(self, text: str) -> None:
        self.banner.show_message(text, kind="error")
        self._set_status("Something went wrong")

    def _busy(self) -> Optional[str]:
        for w, what in ((self._swap_worker, "a swap"), (self._batch_worker, "a batch"),
                        (self._webcam_worker, "the camera"), (self._models_worker, "a model download")):
            if w is not None and w.isRunning():
                return what
        return None

    def _current_config(self, *, realtime: bool = False):
        cfg = build_config(self.quality.value(), self.device.value(), self.restore.value(),
                           self.boost.value(), self.faces.value(), realtime=realtime)
        cfg.detect_every_n = int(self.detect_every.value())
        return cfg

    def _set_result_actions(self, enabled: bool) -> None:
        self.save_btn.setEnabled(enabled)
        self.reveal_btn.setEnabled(enabled)

    def _start_job_ui(self, label: str) -> None:
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.cancel_btn.show()
        self._set_status(label)

    def _end_job_ui(self) -> None:
        self.cancel_btn.hide()

    def _update_title(self) -> None:
        if self._source_path and self._target_path:
            self.stage_title.setText(f"{Path(self._source_path).stem} → {Path(self._target_path).stem}")
        elif self._source_path:
            self.stage_title.setText(Path(self._source_path).stem)
        cfg_bits = [dict(QUALITY_ITEMS_INV)[self.quality.value()]]
        if self.restore.value() != "default":
            cfg_bits.append(self.restore._label_for(self.restore.value()))
        if self.boost.value() not in ("default",):
            cfg_bits.append(f"boost {self.boost.value()}")
        cfg_bits.append(self._device_label.text())
        self.stage_meta.setText(" · ".join(cfg_bits))

    # ================================================================ inputs
    def _apply_source(self, path: str) -> None:
        if Path(path).suffix.lower() not in IMAGE_EXTS:
            self._error("The source must be an image (PNG, JPG, WEBP or BMP).")
            return
        self._source_path = path
        self._last_dir = str(Path(path).parent)
        self.source_card.set_content(load_path_pixmap(path, max_side=480), Path(path).name)
        self._update_title()
        self._set_status("Source ready")

    def _apply_target(self, path: str) -> None:
        ext = Path(path).suffix.lower()
        if ext not in IMAGE_EXTS and ext not in VIDEO_EXTS:
            self._error("The target must be an image or a video (MP4, MOV, AVI, MKV).")
            return
        self._target_path = path
        self._output_path = None
        self._last_dir = str(Path(path).parent)
        kind = "video" if ext in VIDEO_EXTS else "image"
        pix = load_path_pixmap(path, max_side=1600)
        self.target_card.set_content(load_path_pixmap(path, max_side=480), f"{Path(path).name} · {kind}")
        self.canvas.set_after(QPixmap())
        self.canvas.set_before(pix)
        self._set_result_actions(False)
        for m in (self.m_id, self.m_sharp, self.m_time):
            m.set(None)
        self._update_title()
        self._set_status("Target ready")

    def _select_source(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choose the source face", self._last_dir,
                                              "Images (*.png *.jpg *.jpeg *.bmp *.webp)")
        if path:
            self._apply_source(path)

    def _select_target(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose the target", self._last_dir,
            "Images and videos (*.png *.jpg *.jpeg *.webp *.bmp *.mp4 *.mov *.avi *.mkv)")
        if path:
            self._apply_target(path)

    # ================================================================ consent
    def _has_consent(self) -> bool:
        return bool(self._settings.value("consent_ack", False, type=bool))

    def _ask_consent(self, then) -> None:
        self._pending_after_consent = then
        self.banner.show_message(
            "Before your first swap: confirm you have permission from the person whose face you "
            "are using, and that you will label the result as synthetic where required.",
            kind="info", action="I have consent")

    def _banner_action(self) -> None:
        if self._pending_after_consent is not None:
            self._settings.setValue("consent_ack", True)
            self.banner.hide()
            fn, self._pending_after_consent = self._pending_after_consent, None
            fn()
        elif self._models_missing:
            self.banner.hide()
            self._go(3)
            self._start_model_download()

    # ================================================================ swap
    def _start_swap(self) -> None:
        busy = self._busy()
        if busy:
            self._error(f"Wait for {busy} to finish, or press Esc to cancel it.")
            return
        if not self._source_path or not self._target_path:
            self._error("Add a source face and a target first. You can drop files onto the cards.")
            return
        if not self._has_consent():
            self._ask_consent(self._start_swap)
            return
        self.banner.hide()
        target = Path(self._target_path)
        self._output_path = str(target.parent / f"{target.stem}_swapped{target.suffix}")
        self.swap_btn.setEnabled(False)
        self._start_job_ui("Loading models")
        self._update_title()
        w = SwapWorker(self._source_path, self._target_path, self._output_path, self._current_config())
        w.progress.connect(self.progress.setValue)
        w.status.connect(self._set_status)
        w.done.connect(self._on_swap_done)
        w.failed.connect(self._on_swap_failed)
        w.cancelled.connect(self._on_swap_cancelled)
        w.finished.connect(lambda: (self.swap_btn.setEnabled(True), self._end_job_ui()))
        self._swap_worker = w
        if not self.canvas.has_result() and Path(self._target_path).suffix.lower() in VIDEO_EXTS:
            self.canvas.set_empty_state("working", "Swapping your video", "This can take a while. Esc cancels.")
        w.start()

    def _on_swap_done(self, output: str, seconds: float) -> None:
        self._output_path = output
        self.m_time.set(f"{seconds:.1f} s")
        self._set_result_actions(True)
        if Path(output).suffix.lower() in VIDEO_EXTS:
            self._set_status(f"Saved {Path(output).name}")
            self.canvas.set_after(load_path_pixmap(output, max_side=1600))
            return
        self.canvas.set_after(load_path_pixmap(output, max_side=1600))
        self.view_mode.set_value("compare")
        self.canvas.set_mode("compare")
        self._set_status("Scoring")
        self._score()

    def _on_swap_failed(self, err: str) -> None:
        self.progress.setValue(0)
        if "face" in err.lower() and ("no " in err.lower() or "not found" in err.lower()):
            self.canvas.set_empty_state("no-face", "No face found",
                                        "Try a clearer, front-facing photo or set Faces to All.")
            self._error("No face was found. Use a sharper, front-facing photo, or set Faces to All.")
        else:
            self._error(f"Swap failed: {err}")

    def _on_swap_cancelled(self) -> None:
        self.progress.setValue(0)
        self._set_status("Cancelled")

    # ================================================================ score
    def _score(self) -> None:
        if not (self._output_path and self._source_path and self._target_path):
            return
        w = ScoreWorker(self._source_path, self._target_path, self._output_path, self.device.value())
        w.done.connect(self._on_score_done)
        w.failed.connect(lambda e: self._set_status(f"Saved. Scoring unavailable: {e}"))
        self._score_worker = w
        w.start()

    def _on_score_done(self, m: dict) -> None:
        self.m_id.set(f"{m['id_similarity']:.3f}")
        self.m_sharp.set(f"{m['sharpness_gain']:+.1f}")
        verdict = "Identity looks strong" if m["passed_id"] else "Weak identity. Try High or GPEN."
        self._set_status(verdict)
        self.status.setToolTip(f"Saved to {self._output_path}")

    # ================================================================ webcam
    def _toggle_webcam(self) -> None:
        if self._webcam_worker and self._webcam_worker.isRunning():
            self._webcam_worker.stop()
            self._set_status("Stopping camera")
            return
        busy = self._busy()
        if busy:
            self._error(f"Wait for {busy} to finish first.")
            return
        if not self._source_path:
            self._go(0)
            self._error("Pick a source face on the Swap tab first.")
            return
        if not self._has_consent():
            self._ask_consent(self._toggle_webcam)
            return
        self.webcam_btn.setText("Stop camera    Ctrl W")
        self.progress.setRange(0, 0)
        w = WebcamWorker(self._source_path, self._current_config(realtime=True),
                         camera_id=int(self.camera.value()))
        w.frame_ready.connect(lambda f: self.canvas.set_live(bgr_to_qpixmap(f, max_side=1600)))
        w.fps.connect(lambda v: self.m_time.set(f"{v:.0f} fps"))
        w.status.connect(self._set_status)
        w.failed.connect(lambda e: self._error(f"Camera error: {e}"))
        w.finished.connect(self._on_webcam_ended)
        self._webcam_worker = w
        w.start()

    def _on_webcam_ended(self) -> None:
        self.webcam_btn.setText("Start camera    Ctrl W")
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.canvas.clear_live()

    # ================================================================ batch
    def _batch_paths(self) -> list[str]:
        return [str(self.batch_list.item(i).data(Qt.ItemDataRole.UserRole))
                for i in range(self.batch_list.count())]

    def _batch_add_paths(self, paths: list[str]) -> None:
        existing = set(self._batch_paths())
        for path in paths:
            p = Path(path)
            if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS | VIDEO_EXTS:
                continue
            key = str(p.resolve())
            if key in existing:
                continue
            item = QListWidgetItem(f"○  {p.name}")
            item.setData(Qt.ItemDataRole.UserRole, key)
            item.setToolTip(key)
            self.batch_list.addItem(item)
            existing.add(key)
        if self.batch_list.count() and not self._batch_output_dir:
            first = Path(self._batch_paths()[0])
            self._batch_output_dir = str(first.parent / "batch_out")
            self.batch_out_label.setText(self._batch_output_dir)
        self._set_status(f"{self.batch_list.count()} in queue")
        self._refresh_empty_state()

    def _batch_add_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self, "Add targets", self._last_dir,
            "Images and videos (*.png *.jpg *.jpeg *.webp *.bmp *.mp4 *.mov *.avi *.mkv)")
        if files:
            self._last_dir = str(Path(files[0]).parent)
            self._batch_add_paths(files)

    def _batch_add_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Add a folder of targets", self._last_dir)
        if folder:
            self._last_dir = folder
            self._batch_add_paths([str(p) for p in sorted(Path(folder).iterdir())])

    def _batch_remove_selected(self) -> None:
        for item in self.batch_list.selectedItems():
            self.batch_list.takeItem(self.batch_list.row(item))

    def _batch_clear(self) -> None:
        self.batch_list.clear()

    def _batch_pick_output(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Batch output folder",
                                                  self._batch_output_dir or self._last_dir)
        if folder:
            self._batch_output_dir = folder
            self.batch_out_label.setText(folder)

    def _batch_open_item(self, item: QListWidgetItem) -> None:
        out = item.data(Qt.ItemDataRole.UserRole + 1)
        if out and Path(out).is_file():
            _reveal(out)

    def _batch_mark(self, i: int, glyph: str, tip: Optional[str] = None) -> None:
        item = self.batch_list.item(i)
        if item is None:
            return
        name = Path(str(item.data(Qt.ItemDataRole.UserRole))).name
        item.setText(f"{glyph}  {name}")
        if tip:
            item.setToolTip(tip)

    def _start_batch(self) -> None:
        if self._batch_worker and self._batch_worker.isRunning():
            self._batch_worker.stop()
            self._set_status("Cancelling after the current item")
            return
        busy = self._busy()
        if busy:
            self._error(f"Wait for {busy} to finish first.")
            return
        if not self._source_path:
            self._go(0)
            self._error("Pick a source face on the Swap tab first.")
            return
        targets = self._batch_paths()
        if not targets:
            self._error("The queue is empty. Add files or a folder.")
            return
        if not self._has_consent():
            self._ask_consent(self._start_batch)
            return
        out_dir = self._batch_output_dir or str(Path(targets[0]).parent / "batch_out")
        self._batch_output_dir = out_dir
        for i in range(len(targets)):
            self._batch_mark(i, "○")
        self.batch_btn.setText("Stop batch    Ctrl B")
        self._start_job_ui(f"0 of {len(targets)}")
        w = BatchWorker(self._source_path, targets, out_dir, self._current_config())
        w.progress.connect(self.progress.setValue)
        w.status.connect(self._set_status)
        w.item_started.connect(lambda i: self._batch_mark(i, "◐"))
        w.item_done.connect(self._on_batch_item_done)
        w.item_failed.connect(lambda i, e: self._batch_mark(i, "✕", e))
        w.done.connect(self._on_batch_done)
        w.failed.connect(lambda e: self._error(f"Batch failed: {e}"))
        w.finished.connect(lambda: (self.batch_btn.setText("Run batch    Ctrl B"), self._end_job_ui()))
        self._batch_worker = w
        w.start()

    def _on_batch_item_done(self, i: int, out: str) -> None:
        self._batch_mark(i, "●", out)
        item = self.batch_list.item(i)
        if item is not None:
            item.setData(Qt.ItemDataRole.UserRole + 1, out)

    def _on_batch_done(self, summary: str) -> None:
        self._set_status(summary)
        self.banner.show_message(f"Batch finished: {summary}", kind="ok")

    # ================================================================ models
    _models_missing: list = []

    def _run_doctor(self):
        from face_swap.core import doctor

        try:
            self._doctor_report = doctor.run(str(models_dir()))
        except Exception:
            self._doctor_report = None
        return self._doctor_report

    def _open_doctor(self) -> None:
        rep = self._doctor_report or self._run_doctor()
        if rep is None:
            self._set_status("Hardware check failed. Run `ravana doctor` in a terminal.")
            return
        if rep.status != "ok":
            tone = COLORS["accent"] if rep.status == "fail" else COLORS["brass"]
            self._chip_dot.setStyleSheet(f"background: {tone}; border-radius: 3px;")
        DoctorPanel(rep, self).show_below(self._chip)

    def _check_models(self) -> None:
        try:
            self._models_missing = missing_preset_models(models_dir(), "seamless")
        except Exception as exc:  # noqa: BLE001
            self.models_status.setText(f"Could not check models: {exc}")
            return
        self._refresh_empty_state()
        if self._models_missing:
            self.models_status.setText(
                "Missing: " + ", ".join(self._models_missing) + f"\n\nThey go into {models_dir()}")
            self.banner.show_message(
                "Seamless quality needs model weights that are not downloaded yet.",
                kind="info", action="Download now")
        else:
            self.models_status.setText(f"All seamless models are ready in {models_dir()}")
            self.models_btn.setEnabled(False)

    def _start_model_download(self) -> None:
        busy = self._busy()
        if busy:
            self._error(f"Wait for {busy} to finish first.")
            return
        self.models_btn.setEnabled(False)
        self.swap_btn.setEnabled(False)
        self._start_job_ui("Downloading models")
        self.cancel_btn.hide()
        w = ModelsDownloadWorker(str(models_dir()), "seamless")
        w.progress.connect(self.progress.setValue)
        w.status.connect(self._set_status)
        w.done.connect(lambda s: (self.banner.show_message(s, kind="ok"), self._check_models()))
        w.failed.connect(lambda e: (self._error(f"Download failed: {e}"), self.models_btn.setEnabled(True)))
        w.finished.connect(lambda: (self.swap_btn.setEnabled(True), self._end_job_ui()))
        self._models_worker = w
        w.start()

    # ================================================================ result actions
    def _save_as(self) -> None:
        if not self._output_path or not Path(self._output_path).is_file():
            self._error("There is no result to save yet.")
            return
        src = Path(self._output_path)
        dest, _ = QFileDialog.getSaveFileName(self, "Save result", str(Path(self._last_dir) / src.name),
                                              f"*{src.suffix}")
        if dest:
            shutil.copy2(src, dest)
            self._set_status(f"Saved a copy as {Path(dest).name}")

    def _reveal_result(self) -> None:
        if self._output_path and Path(self._output_path).is_file():
            _reveal(self._output_path)

    def _cancel_current(self) -> None:
        if self._swap_worker and self._swap_worker.isRunning():
            self._swap_worker.stop()
            self._set_status("Cancelling")
        elif self._batch_worker and self._batch_worker.isRunning():
            self._batch_worker.stop()
            self._set_status("Cancelling after the current item")
        elif self._webcam_worker and self._webcam_worker.isRunning():
            self._webcam_worker.stop()
        elif self.banner.isVisible():
            self.banner.hide()

    def closeEvent(self, event) -> None:  # noqa: N802
        for w in (self._batch_worker, self._swap_worker, self._webcam_worker):
            if w is not None and w.isRunning():
                w.stop()
        for w in (self._batch_worker, self._swap_worker, self._webcam_worker,
                  self._score_worker, self._models_worker):
            if w is not None and w.isRunning():
                w.wait(4000)
        super().closeEvent(event)


QUALITY_ITEMS_INV = [(v, k) for k, v in QUALITY_ITEMS]
