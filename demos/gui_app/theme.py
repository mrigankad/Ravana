"""Dark gold/red theme for the Ravana desktop GUI."""

COLORS = {
    "bg": "#0e0e0e",
    "card": "#1a1a1a",
    "text": "#f2f2f2",
    "muted": "#8e8e8e",
    "accent": "#c9a227",
    "danger": "#c0392b",
    "border": "#2e2e2e",
    "drop": "#242018",
}

APP_STYLESHEET = f"""
QMainWindow, QWidget {{
  background-color: {COLORS['bg']};
  color: {COLORS['text']};
  font-family: "Segoe UI", "Helvetica Neue", sans-serif;
  font-size: 13px;
}}
QMenuBar {{
  background-color: {COLORS['bg']};
  color: {COLORS['text']};
  border-bottom: 1px solid {COLORS['border']};
  padding: 2px;
}}
QMenuBar::item:selected {{ background-color: {COLORS['card']}; }}
QMenu {{
  background-color: {COLORS['card']};
  border: 1px solid {COLORS['border']};
}}
QMenu::item:selected {{ background-color: {COLORS['accent']}; color: #111; }}
QPushButton {{
  background-color: {COLORS['card']};
  color: {COLORS['text']};
  border: 1px solid {COLORS['border']};
  padding: 9px 14px;
  border-radius: 5px;
}}
QPushButton:hover {{ border-color: {COLORS['accent']}; background-color: #222; }}
QPushButton:disabled {{ color: {COLORS['muted']}; }}
QPushButton#primaryButton {{
  background-color: {COLORS['danger']};
  border: none;
  font-weight: 600;
  font-size: 14px;
  padding: 12px 18px;
}}
QPushButton#primaryButton:hover {{ background-color: #d45045; }}
QPushButton#accentButton {{
  background-color: {COLORS['accent']};
  color: #111111;
  border: none;
  font-weight: 600;
}}
QPushButton#accentButton:hover {{ background-color: #dbb53a; }}
QPushButton#ghostButton {{
  background: transparent;
  border: 1px dashed {COLORS['border']};
  color: {COLORS['muted']};
}}
QComboBox, QSpinBox, QLineEdit {{
  background-color: {COLORS['card']};
  border: 1px solid {COLORS['border']};
  padding: 5px 8px;
  border-radius: 4px;
  min-height: 22px;
}}
QComboBox::drop-down {{ border: none; width: 20px; }}
QComboBox QAbstractItemView {{
  background-color: {COLORS['card']};
  selection-background-color: {COLORS['accent']};
  selection-color: #111;
}}
QProgressBar {{
  background-color: {COLORS['card']};
  border: 1px solid {COLORS['border']};
  border-radius: 4px;
  text-align: center;
  min-height: 16px;
}}
QProgressBar::chunk {{ background-color: {COLORS['accent']}; border-radius: 3px; }}
QLabel#muted {{ color: {COLORS['muted']}; }}
QLabel#hint {{ color: {COLORS['muted']}; font-size: 12px; }}
QLabel#paneTitle {{
  color: {COLORS['accent']};
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}}
QLabel#scoreChip {{
  background-color: {COLORS['card']};
  border: 1px solid {COLORS['border']};
  border-radius: 4px;
  padding: 6px 10px;
  color: {COLORS['text']};
  font-family: Consolas, "Courier New", monospace;
  font-size: 12px;
}}
QLabel#thumb {{
  background-color: {COLORS['card']};
  border: 1px solid {COLORS['border']};
  border-radius: 4px;
  min-height: 72px;
  max-height: 72px;
}}
QGroupBox {{
  border: 1px solid {COLORS['border']};
  border-radius: 6px;
  margin-top: 12px;
  padding-top: 14px;
  font-weight: 600;
}}
QGroupBox::title {{
  subcontrol-origin: margin;
  left: 10px;
  padding: 0 6px;
  color: {COLORS['accent']};
}}
QScrollArea {{ border: none; background: transparent; }}
QStatusBar {{
  background: {COLORS['bg']};
  color: {COLORS['muted']};
}}
"""
