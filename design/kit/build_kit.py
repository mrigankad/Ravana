"""Build the Ravana character kit + icon kit from source art.

    .venv/Scripts/python design/kit/build_kit.py

Outputs
  design/kit/character/<state>.png       transparent poses, one per app state
  design/kit/icons/<name>.svg            24px icon set (stroke 1.75, round caps)
  design/kit/icons-sheet.png             contact sheet of every icon
  design/kit/character-sheet.png         contact sheet of every pose
  demos/gui_app/assets/{icons,character} copies used by the desktop app
  docs/assets/brand/                     copies used by the README
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
KIT = ROOT / "design" / "kit"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(KIT))

from icons import ICONS, svg  # noqa: E402

# ------------------------------------------------------------------ character
# (row, col) in the 3x5 expression sheet -> app state
POSES = {
    "idle": (0, 0),        # empty canvas, ready
    "working": (0, 1),     # swap / batch in progress
    "success": (0, 2),     # strong identity score
    "no-face": (0, 3),     # no face detected in source or target
    "error": (0, 4),       # any failure banner
    "consent": (2, 0),     # first-run consent gate
    "offline": (2, 1),     # models missing / not downloaded
    "batch": (2, 2),       # batch queue empty state
    "live": (2, 3),        # camera tab
    "sleeping": (2, 4),    # long idle / camera stopped
}


def _bands(mask: np.ndarray, axis: int, min_gap: int, min_size: int) -> list[tuple[int, int]]:
    proj = mask.any(axis=axis)
    out, start, gap = [], None, 0
    for i, v in enumerate(proj):
        if v:
            if start is None:
                start = i
            gap = 0
        elif start is not None:
            gap += 1
            if gap >= min_gap:
                if i - gap - start >= min_size:
                    out.append((start, i - gap + 1))
                start, gap = None, 0
    if start is not None and len(proj) - start >= min_size:
        out.append((start, len(proj)))
    return out


def _to_rgba(tile: np.ndarray) -> np.ndarray:
    """White background -> transparent, by flood-filling from the border."""
    h, w = tile.shape[:2]
    near_white = (tile.min(axis=2) > 235).astype(np.uint8)
    ff = near_white.copy() * 255
    mask = np.zeros((h + 2, w + 2), np.uint8)
    for x, y in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
        if ff[y, x] == 255:
            cv2.floodFill(ff, mask, (x, y), 128)
    bg = (ff == 128).astype(np.float32)
    bg = cv2.GaussianBlur(bg, (3, 3), 0)
    alpha = ((1.0 - bg) * 255).astype(np.uint8)
    rgba = cv2.cvtColor(tile, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = alpha
    return rgba


def build_character(sheet: Path, out_dir: Path) -> dict[str, Path]:
    img = cv2.imread(str(sheet))
    if img is None:
        raise SystemExit(f"missing {sheet}")
    # The sheet is a 3x5 grid whose poses touch, so cut on fixed bands and trim.
    h, w = img.shape[:2]
    row_bands = {0: (0.03, 0.37), 2: (0.62, 0.985)}
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for state, (r, c) in POSES.items():
        y0, y1 = (int(v * h) for v in row_bands[r])
        cw = w / 5
        x0, x1 = max(0, int((c - 0.35) * cw)), min(w, int((c + 1.35) * cw))
        tile = img[y0:y1, x0:x1].copy()
        ink = (tile.min(axis=2) < 225).astype(np.uint8)
        joined = cv2.dilate(ink, np.ones((5, 5), np.uint8))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(joined)
        lo, hi = (c * cw - x0) + cw * 0.2, (c * cw - x0) + cw * 0.8
        keep = np.zeros_like(ink, bool)
        for k in range(1, n):
            bx, by, bw, bh, area = stats[k]
            if area < 30:
                continue
            centre = bx + bw / 2
            if (c * cw - x0) + cw * 0.08 <= centre <= (c * cw - x0) + cw * 0.92:
                keep |= labels == k
        tile[~keep] = 255
        ys, xs = np.where(keep & (ink > 0))
        if len(xs) == 0:
            continue
        pad = 6
        tile = tile[max(0, ys.min() - pad):ys.max() + pad, max(0, xs.min() - pad):xs.max() + pad]
        rgba = _to_rgba(tile)
        # square canvas, centred, 2x upscale for HiDPI use
        side = max(rgba.shape[:2])
        canvas = np.zeros((side, side, 4), np.uint8)
        oy, ox = (side - rgba.shape[0]) // 2, (side - rgba.shape[1]) // 2
        canvas[oy:oy + rgba.shape[0], ox:ox + rgba.shape[1]] = rgba
        canvas = cv2.resize(canvas, (side * 2, side * 2), interpolation=cv2.INTER_CUBIC)
        path = out_dir / f"{state}.png"
        cv2.imwrite(str(path), canvas)
        written[state] = path
    return written


# ------------------------------------------------------------------ sheets (Qt)
def _qt_app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def icon_sheet(icon_dir: Path, out: Path) -> None:
    _qt_app()
    from PySide6.QtCore import QRectF, Qt
    from PySide6.QtGui import QColor, QFont, QFontDatabase, QImage, QPainter
    from PySide6.QtSvg import QSvgRenderer

    fid = QFontDatabase.addApplicationFont(str(ROOT / "demos/gui_app/assets/fonts/Inter-Variable.ttf"))
    fam = (QFontDatabase.applicationFontFamilies(fid) or ["Segoe UI"])[0]
    groups: dict[str, list[str]] = {}
    for name, meta in ICONS.items():
        groups.setdefault(meta["group"], []).append(name)
    cols, cell, label_h, pad, head = 10, 112, 26, 48, 44
    rows_total = sum((len(v) + cols - 1) // cols for v in groups.values())
    W = pad * 2 + cols * cell
    H = pad + 70 + len(groups) * head + rows_total * (cell + label_h) + pad
    img = QImage(W, H, QImage.Format.Format_ARGB32)
    img.fill(QColor("#FFFFFF"))
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    f = QFont(fam)
    f.setPixelSize(30)
    f.setWeight(QFont.Weight.DemiBold)
    p.setFont(f)
    p.setPen(QColor("#15120F"))
    p.drawText(pad, pad + 30, f"Ravana icon kit · {len(ICONS)} icons · 24px grid · 1.75 stroke")
    y = pad + 70
    for group, names in groups.items():
        f.setPixelSize(13)
        f.setWeight(QFont.Weight.DemiBold)
        p.setFont(f)
        p.setPen(QColor("#6B645B"))
        p.drawText(pad, y + 20, group.upper())
        y += head
        for i, name in enumerate(names):
            cx, cy = pad + (i % cols) * cell, y + (i // cols) * (cell + label_h)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor("#F5F3F0"))
            p.drawRoundedRect(QRectF(cx + 6, cy, cell - 12, cell - 12), 12, 12)
            QSvgRenderer(str(icon_dir / f"{name}.svg")).render(
                p, QRectF(cx + cell / 2 - 18, cy + (cell - 12) / 2 - 18, 36, 36))
            f.setPixelSize(12)
            f.setWeight(QFont.Weight.Normal)
            p.setFont(f)
            p.setPen(QColor("#6B645B"))
            p.drawText(QRectF(cx, cy + cell - 8, cell, 20), Qt.AlignmentFlag.AlignCenter, name)
        y += ((len(names) + cols - 1) // cols) * (cell + label_h)
    p.end()
    img.save(str(out))


def character_sheet(poses: dict[str, Path], out: Path) -> None:
    _qt_app()
    from PySide6.QtCore import QRectF, Qt
    from PySide6.QtGui import QColor, QFont, QFontDatabase, QImage, QPainter, QPixmap

    fid = QFontDatabase.addApplicationFont(str(ROOT / "demos/gui_app/assets/fonts/Inter-Variable.ttf"))
    fam = (QFontDatabase.applicationFontFamilies(fid) or ["Segoe UI"])[0]
    cols, cell, pad = 5, 240, 48
    rows = (len(poses) + cols - 1) // cols
    W, H = pad * 2 + cols * cell, pad + 70 + rows * (cell + 40) + pad
    img = QImage(W, H, QImage.Format.Format_ARGB32)
    img.fill(QColor("#FFFFFF"))
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
    f = QFont(fam)
    f.setPixelSize(30)
    f.setWeight(QFont.Weight.DemiBold)
    p.setFont(f)
    p.setPen(QColor("#15120F"))
    p.drawText(pad, pad + 30, "Dasha · ten heads, ten app states")
    for i, (state, path) in enumerate(poses.items()):
        cx, cy = pad + (i % cols) * cell, pad + 70 + (i // cols) * (cell + 40)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor("#F5F3F0"))
        p.drawRoundedRect(QRectF(cx + 8, cy, cell - 16, cell - 16), 16, 16)
        pix = QPixmap(str(path))
        p.drawPixmap(QRectF(cx + 24, cy + 16, cell - 48, cell - 48), pix, QRectF(pix.rect()))
        f.setPixelSize(14)
        f.setWeight(QFont.Weight.Medium)
        p.setFont(f)
        p.setPen(QColor("#15120F"))
        p.drawText(QRectF(cx, cy + cell - 10, cell, 24), Qt.AlignmentFlag.AlignCenter, state)
    p.end()
    img.save(str(out))


def main() -> None:
    icon_dir = KIT / "icons"
    icon_dir.mkdir(parents=True, exist_ok=True)
    for name, meta in ICONS.items():
        (icon_dir / f"{name}.svg").write_text(svg(meta["body"]), encoding="utf-8")
    print(f"icons: {len(ICONS)} -> {icon_dir}")

    poses = build_character(ROOT / "design" / "character" / "dasha-expressions.png", KIT / "character")
    print(f"character: {len(poses)} poses -> {KIT / 'character'}")

    icon_sheet(icon_dir, KIT / "icons-sheet.png")
    character_sheet(poses, KIT / "character-sheet.png")
    print("sheets written")

    app_assets = ROOT / "demos" / "gui_app" / "assets"
    for sub, src in (("icons", icon_dir), ("character", KIT / "character")):
        dst = app_assets / sub
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    brand = ROOT / "docs" / "assets" / "brand"
    brand.mkdir(parents=True, exist_ok=True)
    for f in ("icons-sheet.png", "character-sheet.png"):
        shutil.copy2(KIT / f, brand / f)
    print("copied into app assets and docs/assets/brand")


if __name__ == "__main__":
    main()
