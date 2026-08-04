"""Ravana desktop GUI (PySide6)."""


def main():
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        raise SystemExit(
            "PySide6 is required for the GUI.\n"
            'Install with:  pip install -e ".[gui]"'
        ) from None

    import sys

    from demos.gui_app.main_window import MainWindow, load_app_icon
    from demos.gui_app.theme import APP_STYLESHEET

    app = QApplication(sys.argv)
    app.setApplicationName("Ravana")
    app.setDesktopFileName("Ravana")
    icon = load_app_icon()
    if not icon.isNull():
        app.setWindowIcon(icon)
    app.setStyleSheet(APP_STYLESHEET)
    win = MainWindow()
    if not icon.isNull():
        win.setWindowIcon(icon)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
