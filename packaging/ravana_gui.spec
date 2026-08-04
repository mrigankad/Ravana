# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the Ravana desktop GUI (one-folder build)."""

from __future__ import annotations

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_dynamic_libs

ROOT = Path(SPECPATH).resolve().parent
ICON = ROOT / "docs" / "assets" / "mascot.ico"

block_cipher = None

# Heavy ML / GUI packages need explicit collection on Windows.
_collect_pkgs = (
    "PySide6",
    "onnxruntime",
    "insightface",
    "cv2",
    "skimage",
    "sklearn",
    "scipy",
    "numpy",
    "PIL",
    "imageio",
    "imageio_ffmpeg",
    "mediapipe",
    "torch",
    "torchvision",
)

datas = [
    (str(ROOT / "docs" / "assets" / "mascot.ico"), "docs/assets"),
]
binaries = []
hiddenimports = [
    "demos.gui",
    "demos.gui_app",
    "demos.gui_app.main_window",
    "demos.gui_app.workers",
    "demos.gui_app.preview",
    "demos.gui_app.theme",
    "demos.gui_app.config",
    "demos.gui_app.paths",
    "face_swap",
    "face_swap.api",
    "face_swap.pipeline",
    "onnxruntime",
    "insightface",
    "cv2",
]

# Optional DirectML build (AMD on Windows) — include if installed.
try:
    import onnxruntime_directml  # noqa: F401

    _collect_pkgs = _collect_pkgs + ("onnxruntime_directml",)
    hiddenimports.append("onnxruntime_directml")
except ImportError:
    pass

for pkg in _collect_pkgs:
    try:
        pkg_datas, pkg_bins, pkg_hidden = collect_all(pkg)
        datas += pkg_datas
        binaries += pkg_bins
        hiddenimports += pkg_hidden
    except Exception as exc:  # pragma: no cover - packaging best-effort
        print(f"[ravana_gui.spec] skip collect_all({pkg!r}): {exc}", file=sys.stderr)

# Extra data some packages need beyond collect_all.
for pkg in ("insightface", "mediapipe", "imageio_ffmpeg"):
    try:
        datas += collect_data_files(pkg)
    except Exception:
        pass
    try:
        binaries += collect_dynamic_libs(pkg)
    except Exception:
        pass

# Drop obvious non-runtime bulk if present (training / docs tooling).
excludes = [
    "tkinter",
    "matplotlib",
    "IPython",
    "jupyter",
    "notebook",
    "pytest",
    "tensorboard",
    "mkdocs",
    "black",
    "mypy",
    "flake8",
]

a = Analysis(
    [str(ROOT / "demos" / "gui.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Ravana",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ICON) if ICON.is_file() else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="Ravana",
)
