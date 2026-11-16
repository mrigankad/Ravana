"""Render the GUI offscreen at several window widths (no models needed).

    python scripts/gui_snapshots.py --out design/exports/qt

Loads the demo comparison images as source/target/result so layout, theme and
responsiveness can be checked in CI or on machines without model weights.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "design" / "exports" / "qt"))
    ap.add_argument("--sizes", default="1440x900,1100x760,820x620")
    args = ap.parse_args()
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from PySide6.QtWidgets import QApplication

    from demos.gui_app.theme import build_stylesheet, load_fonts

    app = QApplication(sys.argv)
    family = load_fonts()
    app.setStyleSheet(build_stylesheet(family))

    from demos.gui_app import main_window as mw
    from demos.gui_app.preview import load_path_pixmap

    mw.MainWindow._check_models = lambda self: None  # keep snapshots deterministic
    comp = ROOT / "docs" / "assets" / "comparisons"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for size in args.sizes.split(","):
        w, h = (int(v) for v in size.split("x"))
        for state in ("empty", "result"):
            win = mw.MainWindow()
            win.resize(w, h)
            win.show()
            if state == "result":
                win._apply_source(str(comp / "woman1.jpg"))
                win._apply_target(str(comp / "man1.jpg"))
                win.canvas.set_after(load_path_pixmap(str(comp / "seamless_woman1_on_man1.jpg"), 1600))
                win.m_id.set("0.309")
                win.m_sharp.set("+16.9")
                win.m_time.set("2.3 s")
                win._set_result_actions(True)
                win._set_status("Weak identity. Try High or GPEN.")
            app.processEvents()
            path = out / f"{state}-{w}.png"
            win.grab().save(str(path))
            print("wrote", path, f"(font: {family})")
            if state == "result" and w >= 1400:
                for idx, tab in ((1, "batch"), (2, "live"), (3, "models")):
                    win._go(idx)
                    app.processEvents()
                    win.grab().save(str(out / f"tab-{tab}-{w}.png"))
                    print("wrote", out / f"tab-{tab}-{w}.png")
            win.close()
            win.deleteLater()
            app.processEvents()


if __name__ == "__main__":
    main()
