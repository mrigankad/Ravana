"""Ravana 2.0 custom widgets (no stock combo boxes or message boxes)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QDragEnterEvent,
    QDropEvent,
    QFont,
    QIcon,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from demos.gui_app.theme import COLORS, assets_dir

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


# ---------------------------------------------------------------- helpers
def svg_pixmap(name: str, size: QSize, color: Optional[str] = None) -> QPixmap:
    """Render an SVG from assets/ to a HiDPI pixmap, optionally recoloured."""
    from PySide6.QtSvg import QSvgRenderer

    name = {"chevron.svg": "chevron-down.svg"}.get(name, name)
    path = assets_dir() / "icons" / name
    if not path.is_file():
        path = assets_dir() / name
    if not path.is_file():
        return QPixmap()
    data = path.read_text(encoding="utf-8")
    if color:
        data = data.replace("#15120F", color)
    renderer = QSvgRenderer(data.encode("utf-8"))
    dpr = 2.0
    pix = QPixmap(int(size.width() * dpr), int(size.height() * dpr))
    pix.fill(Qt.GlobalColor.transparent)
    p = QPainter(pix)
    renderer.render(p)
    p.end()
    pix.setDevicePixelRatio(dpr)
    return pix


def pose_pixmap(state: str, px: int) -> QPixmap:
    """Dasha character pose for an app state (see design/kit/KIT.md)."""
    path = assets_dir() / "character" / f"{state}.png"
    pix = QPixmap(str(path)) if path.is_file() else QPixmap()
    if pix.isNull():
        return pix
    out = pix.scaled(px * 2, px * 2, Qt.AspectRatioMode.KeepAspectRatio,
                     Qt.TransformationMode.SmoothTransformation)
    out.setDevicePixelRatio(2.0)
    return out


def svg_icon(name: str, px: int = 18, color: Optional[str] = None) -> QIcon:
    return QIcon(svg_pixmap(name, QSize(px, px), color))


def app_icon() -> QIcon:
    """Sindoor rounded square with three heads, matching the Paper app icon."""
    icon = QIcon()
    for s in (16, 24, 32, 48, 64, 128, 256):
        pix = QPixmap(s, s)
        pix.fill(Qt.GlobalColor.transparent)
        p = QPainter(pix)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setBrush(QColor(COLORS["accent"]))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(QRectF(0, 0, s, s), s * 0.23, s * 0.23)
        k = s / 60.0
        cream = QColor("#FFF6EE")
        if s >= 32:
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(cream, max(1.0, 2.5 * k)))
            p.drawEllipse(QPointF(14 * k, 34 * k), 7 * k, 7 * k)
            p.drawEllipse(QPointF(46 * k, 34 * k), 7 * k, 7 * k)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(cream)
            p.drawEllipse(QPointF(30 * k, 28 * k), 11 * k, 11 * k)
        else:
            p.setBrush(cream)
            p.drawEllipse(QPointF(s / 2, s / 2), s * 0.24, s * 0.24)
        p.end()
        icon.addPixmap(pix)
    return icon


def accepts(path: str, *, images: bool, videos: bool) -> bool:
    ext = Path(path).suffix.lower()
    return (images and ext in IMAGE_EXTS) or (videos and ext in VIDEO_EXTS)


def eyebrow(text: str) -> QLabel:
    lbl = QLabel(text.upper())
    lbl.setObjectName("eyebrow")
    return lbl


# ---------------------------------------------------------------- segmented
class Segmented(QFrame):
    """Pill-style single-choice control. Emits the data value of the choice."""

    changed = Signal(str)

    def __init__(self, items: Sequence[tuple[str, str]], current: Optional[str] = None,
                 parent: Optional[QWidget] = None, *, fill: bool = False):
        super().__init__(parent)
        self.setObjectName("segTrack")
        row = QHBoxLayout(self)
        row.setContentsMargins(3, 3, 3, 3)
        row.setSpacing(2)
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._buttons: dict[str, QPushButton] = {}
        for label, value in items:
            b = QPushButton(label)
            b.setObjectName("segItem")
            b.setCheckable(True)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            if fill:  # share the row equally and allow shrinking (rail)
                b.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
                b.setMinimumWidth(0)
                b.setStyleSheet("padding: 6px 2px;")
            b.clicked.connect(lambda _=False, v=value: self.changed.emit(v))
            self._group.addButton(b)
            self._buttons[value] = b
            row.addWidget(b, len(label) + 2 if fill else 0)
        self.set_value(current if current is not None else items[0][1])

    def value(self) -> str:
        for v, b in self._buttons.items():
            if b.isChecked():
                return v
        return next(iter(self._buttons))

    def set_value(self, value: str) -> None:
        if value in self._buttons:
            self._buttons[value].setChecked(True)

    def set_visible_item(self, value: str, visible: bool) -> None:
        if value in self._buttons:
            self._buttons[value].setVisible(visible)


# ---------------------------------------------------------------- select
class SelectRow(QFrame):
    """Label on the left, value + chevron on the right, opens a styled menu."""

    changed = Signal(str)

    def __init__(self, label: str, items: Sequence[tuple[str, str]], current: Optional[str] = None,
                 *, last: bool = False, parent: Optional[QWidget] = None):
        super().__init__(parent)
        if not last:
            self.setObjectName("settingRow")
        self._items = list(items)
        self._value = current if current is not None else self._items[0][1]
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        self.setFixedHeight(40)
        lbl = QLabel(label)
        lbl.setObjectName("label")
        self._btn = QPushButton()
        self._btn.setObjectName("selectButton")
        self._btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
        self._btn.setIcon(svg_icon("chevron.svg", 12, COLORS["muted"]))
        self._btn.setIconSize(QSize(12, 12))
        self._btn.clicked.connect(self._open)
        row.addWidget(lbl)
        row.addStretch(1)
        row.addWidget(self._btn)
        self._sync()

    def _label_for(self, value: str) -> str:
        for lab, v in self._items:
            if v == value:
                return lab
        return value

    def _sync(self) -> None:
        self._btn.setText(self._label_for(self._value) + "  ")

    def _open(self) -> None:
        menu = QMenu(self)
        for lab, v in self._items:
            act = menu.addAction(("✓  " if v == self._value else "    ") + lab)
            act.setData(v)
        chosen = menu.exec(self._btn.mapToGlobal(self._btn.rect().bottomLeft()))
        if chosen is not None:
            self.set_value(str(chosen.data()))
            self.changed.emit(self._value)

    def value(self) -> str:
        return self._value

    def set_value(self, value: str) -> None:
        self._value = value
        self._sync()


# ---------------------------------------------------------------- drop card
class DropCard(QFrame):
    """Clickable, droppable thumbnail card for the source or target input."""

    dropped = Signal(str)
    clicked = Signal()

    def __init__(self, title: str, empty_hint: str, *, images: bool = True, videos: bool = False,
                 thumb_height: int = 156, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._images, self._videos = images, videos
        self._title = title
        self._hint = empty_hint
        self._pix = QPixmap()
        self._thumb_h = thumb_height
        self.setAcceptDrops(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setMinimumWidth(60)
        col = QVBoxLayout(self)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(8)
        self._thumb = _Thumb(self)
        self._thumb.setFixedHeight(thumb_height)
        self._name = QLabel(title)
        self._name.setStyleSheet("font-weight: 500;")
        self._meta = QLabel(empty_hint)
        self._meta.setObjectName("muted")
        self._meta.setWordWrap(False)
        self._meta.setMinimumWidth(10)
        self._meta.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        col.addWidget(self._thumb)
        col.addWidget(self._name)
        col.addWidget(self._meta)
        col.addStretch(1)

    def set_thumb_height(self, h: int) -> None:
        self._thumb.setFixedHeight(h)

    def set_content(self, pix: QPixmap, meta: str) -> None:
        self._thumb.set_pixmap(pix)
        self._meta.setText(meta)

    def clear(self) -> None:
        self._thumb.set_pixmap(QPixmap())
        self._meta.setText(self._hint)

    def mouseReleaseEvent(self, e: QMouseEvent) -> None:  # noqa: N802
        if e.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()

    def dragEnterEvent(self, e: QDragEnterEvent) -> None:  # noqa: N802
        urls = e.mimeData().urls() if e.mimeData().hasUrls() else []
        if urls and accepts(urls[0].toLocalFile(), images=self._images, videos=self._videos):
            self._thumb.hover = True
            self._thumb.update()
            e.acceptProposedAction()
        else:
            e.ignore()

    def dragLeaveEvent(self, e) -> None:  # noqa: N802
        self._thumb.hover = False
        self._thumb.update()

    def dropEvent(self, e: QDropEvent) -> None:  # noqa: N802
        self._thumb.hover = False
        self._thumb.update()
        urls = e.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            if accepts(path, images=self._images, videos=self._videos):
                self.dropped.emit(path)
                e.acceptProposedAction()


class _Thumb(QWidget):
    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self._pix = QPixmap()
        self.hover = False

    def set_pixmap(self, pix: QPixmap) -> None:
        self._pix = pix
        self.update()

    def paintEvent(self, _e) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        path = QPainterPath()
        path.addRoundedRect(r, 8, 8)
        if self._pix.isNull():
            p.fillPath(path, QColor(COLORS["accent_soft"] if self.hover else COLORS["surface"]))
            pen = QPen(QColor(COLORS["accent"] if self.hover else "#CFC8BE"), 1.2)
            pen.setStyle(Qt.PenStyle.DashLine)
            p.setPen(pen)
            p.drawPath(path)
            p.setPen(QColor(COLORS["muted"]))
            f = QFont(self.font())
            f.setPointSizeF(9.5)
            p.setFont(f)
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Drop or click")
        else:
            p.setClipPath(path)
            _draw_cover(p, self._pix, r)
            if self.hover:
                p.setClipping(False)
                p.setPen(QPen(QColor(COLORS["accent"]), 2))
                p.drawPath(path)
        p.end()


def _draw_cover(p: QPainter, pix: QPixmap, rect: QRectF, anchor_x: float = 0.5) -> None:
    """Draw pix to fill rect (object-fit: cover)."""
    pw, ph = pix.width() / pix.devicePixelRatio(), pix.height() / pix.devicePixelRatio()
    if pw <= 0 or ph <= 0:
        return
    scale = max(rect.width() / pw, rect.height() / ph)
    w, h = pw * scale, ph * scale
    x = rect.x() + (rect.width() - w) * anchor_x
    y = rect.y() + (rect.height() - h) / 2
    p.drawPixmap(QRectF(x, y, w, h), pix, QRectF(0, 0, pix.width(), pix.height()))


def _draw_contain(p: QPainter, pix: QPixmap, rect: QRectF) -> QRectF:
    pw, ph = pix.width() / pix.devicePixelRatio(), pix.height() / pix.devicePixelRatio()
    if pw <= 0 or ph <= 0:
        return rect
    scale = min(rect.width() / pw, rect.height() / ph)
    w, h = pw * scale, ph * scale
    target = QRectF(rect.x() + (rect.width() - w) / 2, rect.y() + (rect.height() - h) / 2, w, h)
    p.drawPixmap(target, pix, QRectF(0, 0, pix.width(), pix.height()))
    return target


# ---------------------------------------------------------------- compare canvas
class CompareCanvas(QWidget):
    """Before/after canvas with a draggable divider, plus result-only and side-by-side modes.

    Also hosts live frames (webcam / video playback) in 'live' mode.
    """

    MODES = ("compare", "result", "side")

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._before = QPixmap()
        self._after = QPixmap()
        self._live = QPixmap()
        self._mode = "compare"
        self._split = 0.5
        self._dragging = False
        self._empty_title = "Drop a source face and a target to begin"
        self._empty_sub = "Images or video. Nothing leaves this machine."
        self._pose = "idle"
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)

    # state
    def set_mode(self, mode: str) -> None:
        self._mode = mode
        self.update()

    def set_before(self, pix: QPixmap) -> None:
        self._before = pix
        self._live = QPixmap()
        self.update()

    def set_after(self, pix: QPixmap) -> None:
        self._after = pix
        self._live = QPixmap()
        self.update()

    def set_live(self, pix: QPixmap) -> None:
        self._live = pix
        self.update()

    def clear_live(self) -> None:
        self._live = QPixmap()
        self.update()

    def set_empty_state(self, pose: str, title: str, sub: str) -> None:
        """Character + message shown when there is nothing to display."""
        self._pose, self._empty_title, self._empty_sub = pose, title, sub
        self.update()

    def has_result(self) -> bool:
        return not self._after.isNull()

    # input
    def _divider_x(self) -> float:
        return self.width() * self._split

    def mousePressEvent(self, e: QMouseEvent) -> None:  # noqa: N802
        if self._mode == "compare" and self.has_result() and self._live.isNull():
            self._dragging = True
            self._set_split(e.position().x())

    def mouseMoveEvent(self, e: QMouseEvent) -> None:  # noqa: N802
        near = abs(e.position().x() - self._divider_x()) < 24
        if self._mode == "compare" and self.has_result() and self._live.isNull():
            self.setCursor(Qt.CursorShape.SplitHCursor if near or self._dragging else Qt.CursorShape.ArrowCursor)
        if self._dragging:
            self._set_split(e.position().x())

    def mouseReleaseEvent(self, _e) -> None:  # noqa: N802
        self._dragging = False

    def _set_split(self, x: float) -> None:
        self._split = min(0.98, max(0.02, x / max(1, self.width())))
        self.update()

    # paint
    def paintEvent(self, _e) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        r = QRectF(self.rect())
        clip = QPainterPath()
        clip.addRoundedRect(r, 10, 10)
        p.fillPath(clip, QColor(COLORS["surface"]))
        p.setClipPath(clip)

        if not self._live.isNull():
            _draw_contain(p, self._live, r)
        elif self._after.isNull() and self._before.isNull():
            self._paint_empty(p, r)
        elif self._after.isNull():
            _draw_contain(p, self._before, r)
            self._tag(p, "Target", left=True)
        elif self._mode == "result":
            _draw_contain(p, self._after, r)
            self._tag(p, "Result", left=False)
        elif self._mode == "side":
            gap = 8
            half = (r.width() - gap) / 2
            _draw_contain(p, self._before, QRectF(r.x(), r.y(), half, r.height()))
            _draw_contain(p, self._after, QRectF(r.x() + half + gap, r.y(), half, r.height()))
            self._tag(p, "Target", left=True)
            self._tag(p, "Result", left=False)
        else:
            self._paint_compare(p, r)
        p.end()

    def _paint_compare(self, p: QPainter, r: QRectF) -> None:
        # both images share one contain-rect so the divider lines up pixel-for-pixel
        ref = self._after
        pw, ph = ref.width() / ref.devicePixelRatio(), ref.height() / ref.devicePixelRatio()
        scale = min(r.width() / pw, r.height() / ph) if pw and ph else 1
        w, h = pw * scale, ph * scale
        target = QRectF(r.x() + (r.width() - w) / 2, r.y() + (r.height() - h) / 2, w, h)
        x = self._divider_x()
        p.save()
        p.setClipRect(QRectF(r.x(), r.y(), x - r.x(), r.height()), Qt.ClipOperation.IntersectClip)
        p.drawPixmap(target, self._before, QRectF(0, 0, self._before.width(), self._before.height()))
        p.restore()
        p.save()
        p.setClipRect(QRectF(x, r.y(), r.right() - x, r.height()), Qt.ClipOperation.IntersectClip)
        p.drawPixmap(target, self._after, QRectF(0, 0, self._after.width(), self._after.height()))
        p.restore()
        # divider + handle
        p.setPen(QPen(QColor("#FFFFFF"), 2))
        p.drawLine(QPointF(x, r.top()), QPointF(x, r.bottom()))
        cy = r.center().y()
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(0, 0, 0, 40))
        p.drawEllipse(QPointF(x, cy + 2), 19, 19)
        p.setBrush(QColor("#FFFFFF"))
        p.drawEllipse(QPointF(x, cy), 18, 18)
        p.setPen(QPen(QColor(COLORS["text"]), 1.6))
        p.drawPolyline([QPointF(x - 3, cy - 4), QPointF(x - 7, cy), QPointF(x - 3, cy + 4)])
        p.drawPolyline([QPointF(x + 3, cy - 4), QPointF(x + 7, cy), QPointF(x + 3, cy + 4)])
        self._tag(p, "Target", left=True)
        self._tag(p, "Result", left=False)

    def _tag(self, p: QPainter, text: str, *, left: bool) -> None:
        f = QFont(self.font())
        f.setPointSizeF(9)
        f.setWeight(QFont.Weight.Medium)
        p.setFont(f)
        fm = p.fontMetrics()
        w = fm.horizontalAdvance(text) + 18
        h = 26
        x = 16 if left else self.width() - 16 - w
        rect = QRectF(x, 16, w, h)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(255, 255, 255, 235))
        p.drawRoundedRect(rect, 6, 6)
        p.setPen(QColor(COLORS["text"]))
        p.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)

    def _paint_empty(self, p: QPainter, r: QRectF) -> None:
        cx, cy = r.center().x(), r.center().y()
        side = max(96.0, min(168.0, r.height() * 0.32))
        pose = pose_pixmap(self._pose, int(side))
        if not pose.isNull():
            p.drawPixmap(QRectF(cx - side / 2, cy - side - 8, side, side), pose,
                         QRectF(0, 0, pose.width(), pose.height()))
        else:
            mark = svg_pixmap("mark.svg", QSize(164, 46))
            p.drawPixmap(QRectF(cx - 82, cy - 70, 164, 46), mark, QRectF(0, 0, mark.width(), mark.height()))
        f = QFont(self.font())
        f.setPointSizeF(13)
        f.setWeight(QFont.Weight.DemiBold)
        p.setFont(f)
        p.setPen(QColor(COLORS["text"]))
        p.drawText(QRectF(r.x(), cy + 4, r.width(), 28), Qt.AlignmentFlag.AlignCenter, self._empty_title)
        f.setPointSizeF(10)
        f.setWeight(QFont.Weight.Normal)
        p.setFont(f)
        p.setPen(QColor(COLORS["muted"]))
        p.drawText(QRectF(r.x(), cy + 34, r.width(), 22), Qt.AlignmentFlag.AlignCenter, self._empty_sub)


# ---------------------------------------------------------------- metric
class Metric(QWidget):
    def __init__(self, caption: str, value_object: str = "metricValue", parent: Optional[QWidget] = None):
        super().__init__(parent)
        col = QVBoxLayout(self)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(0)
        self._value_object = value_object
        self.value = QLabel("—")
        self.value.setObjectName("metricValueMuted")
        self.caption = QLabel(caption)
        self.caption.setObjectName("muted")
        col.addWidget(self.value)
        col.addWidget(self.caption)

    def set(self, text: Optional[str]) -> None:
        if text is None:
            self.value.setText("—")
            self.value.setObjectName("metricValueMuted")
        else:
            self.value.setText(text)
            self.value.setObjectName(self._value_object)
        self.value.style().unpolish(self.value)
        self.value.style().polish(self.value)


# ---------------------------------------------------------------- banner
class Banner(QFrame):
    """Inline, dismissible notice used instead of modal message boxes."""

    action = Signal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(14, 10, 10, 10)
        row.setSpacing(12)
        self._dot = QLabel()
        self._dot.setFixedSize(40, 40)
        self._text = QLabel()
        self._text.setWordWrap(True)
        self._btn = QPushButton()
        self._btn.clicked.connect(self.action.emit)
        close = QPushButton("Dismiss")
        close.setObjectName("ghost")
        close.clicked.connect(self.hide)
        row.addWidget(self._dot, 0, Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(self._text, 1)
        row.addWidget(self._btn)
        row.addWidget(close)
        self.hide()

    def show_message(self, text: str, *, kind: str = "info", action: Optional[str] = None) -> None:
        colors = {"info": COLORS["brass"], "error": COLORS["accent"], "ok": COLORS["ok"]}
        bg = {"info": "#FBF6EA", "error": COLORS["accent_soft"], "ok": "#EEF5EC"}[kind]
        self.setStyleSheet(f"Banner {{ background: {bg}; border-radius: 8px; }}")
        pose = {"info": "consent" if action == "I have consent" else "offline" if action else "idle",
                "error": "error", "ok": "success"}[kind]
        self._dot.setPixmap(pose_pixmap(pose, 40))
        self._dot.setStyleSheet("background: transparent;")
        del colors
        self._text.setText(text)
        self._btn.setVisible(bool(action))
        if action:
            self._btn.setText(action)
        self.show()
