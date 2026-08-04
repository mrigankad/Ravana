"""Dark gold/red theme for the Ravana desktop GUI."""

COLORS = {
    "bg": "#121212",
    "card": "#1c1c1c",
    "text": "#f2f2f2",
    "muted": "#9a9a9a",
    "accent": "#c9a227",
    "danger": "#c0392b",
    "border": "#2a2a2a",
}

APP_STYLESHEET = f"""
QMainWindow, QWidget {{
  background-color: {COLORS['bg']};
  color: {COLORS['text']};
  font-family: "Segoe UI", "Helvetica Neue", sans-serif;
  font-size: 13px;
}}
QPushButton {{
  background-color: {COLORS['card']};
  color: {COLORS['text']};
  border: 1px solid {COLORS['border']};
  padding: 8px 14px;
  border-radius: 4px;
}}
QPushButton:hover {{ border-color: {COLORS['accent']}; }}
QPushButton:disabled {{ color: {COLORS['muted']}; }}
QPushButton#primaryButton {{
  background-color: {COLORS['danger']};
  border: none;
  font-weight: 600;
}}
QPushButton#accentButton {{
  background-color: {COLORS['accent']};
  color: #111111;
  border: none;
  font-weight: 600;
}}
QComboBox, QLineEdit {{
  background-color: {COLORS['card']};
  border: 1px solid {COLORS['border']};
  padding: 4px 8px;
  border-radius: 3px;
}}
QComboBox::drop-down {{ border: none; }}
QProgressBar {{
  background-color: {COLORS['card']};
  border: 1px solid {COLORS['border']};
  border-radius: 3px;
  text-align: center;
  min-height: 14px;
}}
QProgressBar::chunk {{ background-color: {COLORS['accent']}; }}
QLabel#muted {{ color: {COLORS['muted']}; }}
QLabel#paneTitle {{
  color: {COLORS['muted']};
  font-size: 11px;
  font-weight: 600;
}}
QGroupBox {{
  border: 1px solid {COLORS['border']};
  border-radius: 4px;
  margin-top: 10px;
  padding-top: 12px;
  font-weight: 600;
}}
QGroupBox::title {{
  subcontrol-origin: margin;
  left: 10px;
  padding: 0 4px;
  color: {COLORS['accent']};
}}
QScrollArea {{ border: none; }}
"""
