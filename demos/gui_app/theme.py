"""Ravana 2.0 light theme: tokens, Inter font loading and the app stylesheet.

Tokens mirror the Paper design file ("Ravana — Redesign", page 00 Brand + kit).
"""

from __future__ import annotations

from pathlib import Path

COLORS = {
    "bg": "#FFFFFF",        # paper: ground
    "surface": "#F5F3F0",   # plaster: wells, segmented tracks, canvas
    "line": "#E7E3DD",      # seam: borders
    "line_soft": "#EFEBE5",  # row dividers
    "text": "#15120F",      # ink
    "text_2": "#3B352E",    # labels
    "muted": "#6B645B",     # ash: secondary text
    "faint": "#B5AEA4",
    "accent": "#E0401F",    # sindoor: one primary action per screen
    "accent_hover": "#C93517",
    "accent_soft": "#FDF1EE",
    "brass": "#A97A12",     # status, metrics
    "ok": "#3C7A3A",
}

FONT_FAMILY = "Inter"
FALLBACK_FAMILY = "Segoe UI"
_ASSETS = Path(__file__).resolve().parent / "assets"


def assets_dir() -> Path:
    return _ASSETS


def load_fonts() -> str:
    """Register the bundled Inter font. Returns the family name to use."""
    from PySide6.QtGui import QFontDatabase

    font_file = _ASSETS / "fonts" / "Inter-Variable.ttf"
    if font_file.is_file():
        fid = QFontDatabase.addApplicationFont(str(font_file))
        if fid >= 0:
            fams = QFontDatabase.applicationFontFamilies(fid)
            if fams:
                return fams[0]
    return FALLBACK_FAMILY


def build_stylesheet(family: str = FONT_FAMILY) -> str:
    c = COLORS
    return f"""
QMainWindow, QWidget {{
  background-color: {c['bg']};
  color: {c['text']};
  font-family: "{family}", "{FALLBACK_FAMILY}", sans-serif;
  font-size: 13px;
}}
QToolTip {{
  background: {c['text']}; color: {c['bg']}; border: none; padding: 6px 8px; border-radius: 6px;
}}
QLabel {{ background: transparent; }}
QLabel#muted {{ color: {c['muted']}; }}
QLabel#label {{ color: {c['text_2']}; }}
QLabel#eyebrow {{ color: {c['muted']}; font-size: 11px; font-weight: 600; letter-spacing: 0.6px; }}
QLabel#wordmark {{ font-size: 20px; font-weight: 600; letter-spacing: -0.4px; }}
QLabel#stageTitle {{ font-size: 26px; font-weight: 600; letter-spacing: -0.6px; }}
QLabel#metricValue {{ font-size: 28px; font-weight: 600; letter-spacing: -0.6px; }}
QLabel#metricValueBrass {{ font-size: 28px; font-weight: 600; letter-spacing: -0.6px; color: {c['brass']}; }}
QLabel#metricValueMuted {{ font-size: 28px; font-weight: 600; letter-spacing: -0.6px; color: {c['faint']}; }}
QLabel#version {{ color: {c['muted']}; font-size: 12px; }}

QWidget#titleBar {{ border-bottom: 1px solid {c['line']}; }}
QWidget#rail {{ border-right: 1px solid {c['line']}; }}
QScrollArea, QScrollArea > QWidget > QWidget {{ border: none; background: {c['bg']}; }}

QPushButton {{
  background: {c['bg']}; color: {c['text']};
  border: 1px solid {c['line']}; border-radius: 8px;
  padding: 8px 14px; font-weight: 500;
}}
QPushButton:hover {{ border-color: {c['faint']}; }}
QPushButton:disabled {{ color: {c['faint']}; border-color: {c['line_soft']}; }}
QPushButton#primary {{
  background: {c['accent']}; color: #FFFFFF; border: none;
  font-size: 15px; font-weight: 600; padding: 14px 18px;
}}
QPushButton#primary:hover {{ background: {c['accent_hover']}; }}
QPushButton#primary:disabled {{ background: #F0B4A6; color: #FFFFFF; }}
QPushButton#ghost {{ border: none; background: transparent; color: {c['muted']}; padding: 6px 8px; }}
QPushButton#ghost:hover {{ color: {c['text']}; }}

QPushButton#navTab {{
  border: none; background: transparent; color: {c['muted']};
  padding: 6px 12px; border-radius: 6px; font-weight: 500;
}}
QPushButton#navTab:hover {{ color: {c['text']}; }}
QPushButton#navTab:checked {{ background: {c['surface']}; color: {c['text']}; }}

QFrame#segTrack {{ background: {c['surface']}; border: 1px solid #ECE8E2; border-radius: 8px; }}
QPushButton#segItem {{
  border: none; background: transparent; color: {c['muted']};
  padding: 6px 10px; border-radius: 6px; font-weight: 500;
}}
QPushButton#segItem:hover {{ color: {c['text']}; }}
QPushButton#segItem:checked {{ background: {c['bg']}; color: {c['text']}; border: 1px solid {c['line']}; }}

QPushButton#selectButton {{
  border: none; background: transparent; padding: 4px 0; text-align: right; font-weight: 500;
}}
QFrame#settingRow {{ border-bottom: 1px solid {c['line_soft']}; }}

QFrame#chip {{ border: 1px solid {c['line']}; border-radius: 6px; }}
QFrame#dropCard {{ background: {c['surface']}; border: 1px dashed #CFC8BE; border-radius: 10px; }}
QFrame#dropCard[filled="true"] {{ border: 1px solid transparent; }}
QFrame#dropCard[hover="true"] {{ border: 2px solid {c['accent']}; background: {c['accent_soft']}; }}

QMenu {{
  background: {c['bg']}; border: 1px solid {c['line']}; border-radius: 8px; padding: 4px;
}}
QMenu::item {{ padding: 7px 26px 7px 12px; border-radius: 6px; }}
QMenu::item:selected {{ background: {c['surface']}; color: {c['text']}; }}
QMenu::indicator {{ width: 0; }}

QListWidget {{
  background: {c['bg']}; border: 1px solid {c['line']}; border-radius: 8px; padding: 4px; outline: none;
}}
QListWidget::item {{ padding: 8px; border-radius: 6px; }}
QListWidget::item:selected {{ background: {c['surface']}; color: {c['text']}; }}

QSpinBox {{
  background: {c['bg']}; border: 1px solid {c['line']}; border-radius: 6px; padding: 4px 8px; min-height: 22px;
}}

QProgressBar {{
  background: {c['surface']}; border: none; border-radius: 2px; max-height: 4px; min-height: 4px;
}}
QProgressBar::chunk {{ background: {c['accent']}; border-radius: 2px; }}

QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {c['line']}; border-radius: 4px; min-height: 30px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: none; }}
"""


# Backwards-compatible name used by demos/gui.py
APP_STYLESHEET = build_stylesheet()
