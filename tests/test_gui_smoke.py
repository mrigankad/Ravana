"""Offscreen smoke test for the Ravana 2.0 window (no model weights needed)."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")


@pytest.fixture(scope="module")
def app():
    from demos.gui_app.theme import build_stylesheet, load_fonts

    inst = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    inst.setStyleSheet(build_stylesheet(load_fonts()))
    return inst


@pytest.fixture
def win(app, monkeypatch):
    from demos.gui_app import main_window as mw

    monkeypatch.setattr(mw.MainWindow, "_check_models", lambda self: None)
    w = mw.MainWindow()
    yield w
    w.close()


def test_tabs_switch_rail(win):
    for i in range(4):
        win._go(i)
        assert win._rail_stack.currentIndex() == i


@pytest.mark.parametrize("width,rail", [(1440, 320), (1100, 276), (820, 264)])
def test_responsive_rail(win, width, rail):
    win._compact = None
    win._apply_responsive(width)
    assert win._rail.width() == rail or win._rail.minimumWidth() == rail


def test_config_from_controls(win):
    win.quality.set_value("high")
    win.restore.set_value("gpen")
    cfg = win._current_config()
    assert cfg.quality == "high"
    assert cfg.enhance_method == "gpen"


def test_swap_requires_inputs(win):
    win._start_swap()
    assert win.banner.isVisibleTo(win)
    assert win._swap_worker is None


def test_unicode_image_roundtrip(tmp_path):
    import numpy as np

    from demos.gui_app.workers import imread_any, imwrite_any

    p = tmp_path / "चेहरा_测试.png"
    img = np.full((8, 8, 3), 127, np.uint8)
    assert imwrite_any(str(p), img)
    back = imread_any(str(p))
    assert back is not None and back.shape == img.shape


def test_doctor_panel_renders(qapp=None):
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    from demos.gui_app.widgets import DoctorPanel
    from face_swap.core import doctor

    rep = doctor.Report(device="CPU", checks=[
        doctor.Check("providers", doctor.WARN, "CPU only", "pip install x"),
        doctor.Check("disk", doctor.OK, "50 GB free"),
    ])
    p = DoctorPanel(rep)
    p.adjustSize()
    app.processEvents()
    assert p.width() == 440 and p.height() > 100
    p.close()
