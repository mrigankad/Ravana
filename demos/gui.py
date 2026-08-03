"""
Desktop GUI application for Ravana.

As per PRD Section 4.1:
  - Minimal GUI for batch image/video processing.
  - Simple webcam demo app.

This provides a cross-platform Tkinter-based GUI with:
  - Source image selection and preview.
  - Target image/video selection.
  - Quality preset control.
  - Real-time webcam mode.
  - Progress tracking and output preview.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from typing import Optional
import threading
import os
import logging

logger = logging.getLogger("face_swap.gui")


class FaceSwapGUI:
    """
    Desktop GUI for Face Swap operations.

    Features:
      - Drag-and-drop-style file selection
      - Side-by-side before/after preview
      - Quality preset selector
      - Progress bar for video processing
      - Real-time webcam mode toggle
    """

    WINDOW_TITLE = "Ravana"
    WINDOW_SIZE = "1000x700"
    BG_COLOR = "#1a1a2e"
    FG_COLOR = "#e0e0e0"
    ACCENT = "#0f3460"
    HIGHLIGHT = "#e94560"
    CARD_BG = "#16213e"

    def __init__(self):
        self.root = tk.Tk()
        self.root.title(self.WINDOW_TITLE)
        self.root.geometry(self.WINDOW_SIZE)
        self.root.configure(bg=self.BG_COLOR)
        self.root.minsize(800, 600)

        # State
        self._source_path: Optional[str] = None
        self._target_path: Optional[str] = None
        self._output_path: Optional[str] = None
        self._quality = tk.StringVar(value="medium")
        self._device = tk.StringVar(value="auto")
        self._enhance = tk.StringVar(value="default")
        self._pixel_boost = tk.StringVar(value="default")
        self._face_select = tk.StringVar(value="all")
        self._processing = False

        self._build_ui()
        self._apply_styles()

    def run(self):
        """Start the GUI event loop."""
        self.root.mainloop()

    # ── UI Construction ──────────────────────────────────────────────────

    def _build_ui(self):
        """Build all UI components."""
        # Header
        header = tk.Frame(self.root, bg=self.ACCENT, height=60)
        header.pack(fill=tk.X)
        header.pack_propagate(False)

        title_label = tk.Label(
            header, text="🔄 Ravana",
            font=("Segoe UI", 18, "bold"),
            bg=self.ACCENT, fg="white",
        )
        title_label.pack(side=tk.LEFT, padx=20, pady=10)

        version_label = tk.Label(
            header, text="v0.3.1",
            font=("Segoe UI", 10),
            bg=self.ACCENT, fg="#aaaacc",
        )
        version_label.pack(side=tk.LEFT, padx=5, pady=15)

        # Main content
        main_frame = tk.Frame(self.root, bg=self.BG_COLOR)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Left panel — File Selection
        left = tk.Frame(main_frame, bg=self.CARD_BG, width=300, relief=tk.FLAT)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left.pack_propagate(False)

        self._build_file_panel(left)

        # Right panel — Preview / Controls
        right = tk.Frame(main_frame, bg=self.BG_COLOR)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self._build_preview_panel(right)
        self._build_controls_panel(right)

        # Footer
        footer = tk.Frame(self.root, bg=self.BG_COLOR, height=40)
        footer.pack(fill=tk.X, padx=20, pady=(0, 10))

        self.progress = ttk.Progressbar(footer, mode="determinate", length=400)
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

        self.status_label = tk.Label(
            footer, text="Ready",
            font=("Segoe UI", 9),
            bg=self.BG_COLOR, fg="#aaaaaa",
        )
        self.status_label.pack(side=tk.RIGHT)

    def _build_file_panel(self, parent):
        """Build the file selection panel."""
        pad = {"padx": 15, "pady": 5}

        tk.Label(
            parent, text="Input Files",
            font=("Segoe UI", 12, "bold"),
            bg=self.CARD_BG, fg=self.FG_COLOR,
        ).pack(**pad, anchor=tk.W, pady=(15, 5))

        # Source image
        tk.Label(
            parent, text="Source Face:",
            font=("Segoe UI", 9),
            bg=self.CARD_BG, fg="#aaaacc",
        ).pack(**pad, anchor=tk.W)

        self.source_btn = tk.Button(
            parent, text="📷  Select Source Image",
            command=self._select_source,
            bg=self.ACCENT, fg="white",
            font=("Segoe UI", 9),
            relief=tk.FLAT, cursor="hand2",
            activebackground=self.HIGHLIGHT,
        )
        self.source_btn.pack(**pad, fill=tk.X)

        self.source_label = tk.Label(
            parent, text="No file selected",
            font=("Segoe UI", 8),
            bg=self.CARD_BG, fg="#888888",
            wraplength=250,
        )
        self.source_label.pack(**pad, anchor=tk.W)

        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=15, pady=10)

        # Target
        tk.Label(
            parent, text="Target (Image or Video):",
            font=("Segoe UI", 9),
            bg=self.CARD_BG, fg="#aaaacc",
        ).pack(**pad, anchor=tk.W)

        self.target_btn = tk.Button(
            parent, text="🎬  Select Target",
            command=self._select_target,
            bg=self.ACCENT, fg="white",
            font=("Segoe UI", 9),
            relief=tk.FLAT, cursor="hand2",
            activebackground=self.HIGHLIGHT,
        )
        self.target_btn.pack(**pad, fill=tk.X)

        self.target_label = tk.Label(
            parent, text="No file selected",
            font=("Segoe UI", 8),
            bg=self.CARD_BG, fg="#888888",
            wraplength=250,
        )
        self.target_label.pack(**pad, anchor=tk.W)

        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=15, pady=10)

        # Settings
        tk.Label(
            parent, text="Settings",
            font=("Segoe UI", 12, "bold"),
            bg=self.CARD_BG, fg=self.FG_COLOR,
        ).pack(**pad, anchor=tk.W, pady=(5, 5))

        # Quality preset
        tk.Label(
            parent, text="Quality:",
            font=("Segoe UI", 9),
            bg=self.CARD_BG, fg="#aaaacc",
        ).pack(**pad, anchor=tk.W)

        quality_frame = tk.Frame(parent, bg=self.CARD_BG)
        quality_frame.pack(**pad, fill=tk.X)

        for val, label in [
            ("low", "Fast"),
            ("medium", "Balanced"),
            ("high", "Quality"),
            ("seamless", "Seamless"),
        ]:
            tk.Radiobutton(
                quality_frame, text=label, variable=self._quality,
                value=val, bg=self.CARD_BG, fg=self.FG_COLOR,
                selectcolor=self.ACCENT,
                font=("Segoe UI", 9),
                activebackground=self.CARD_BG,
                activeforeground=self.HIGHLIGHT,
            ).pack(side=tk.LEFT, padx=5)

        # Device
        tk.Label(
            parent, text="Device:",
            font=("Segoe UI", 9),
            bg=self.CARD_BG, fg="#aaaacc",
        ).pack(**pad, anchor=tk.W)

        device_frame = tk.Frame(parent, bg=self.CARD_BG)
        device_frame.pack(**pad, fill=tk.X)

        for val, label in [
            ("auto", "Auto"),
            ("dml", "AMD (DirectML)"),
            ("cuda", "GPU (CUDA)"),
            ("cpu", "CPU"),
        ]:
            tk.Radiobutton(
                device_frame, text=label, variable=self._device,
                value=val, bg=self.CARD_BG, fg=self.FG_COLOR,
                selectcolor=self.ACCENT,
                font=("Segoe UI", 9),
                activebackground=self.CARD_BG,
            ).pack(side=tk.LEFT, padx=5)

        # Enhancer override (mainly for seamless)
        tk.Label(
            parent, text="Restore (seamless):",
            font=("Segoe UI", 9),
            bg=self.CARD_BG, fg="#aaaacc",
        ).pack(**pad, anchor=tk.W)

        enhance_frame = tk.Frame(parent, bg=self.CARD_BG)
        enhance_frame.pack(**pad, fill=tk.X)

        for val, label in [
            ("default", "Preset"),
            ("gfpgan", "GFPGAN"),
            ("gpen", "GPEN"),
            ("restoreformer", "RF++"),
            ("codeformer", "CodeFormer"),
            ("opencv", "OpenCV"),
        ]:
            tk.Radiobutton(
                enhance_frame, text=label, variable=self._enhance,
                value=val, bg=self.CARD_BG, fg=self.FG_COLOR,
                selectcolor=self.ACCENT,
                font=("Segoe UI", 9),
                activebackground=self.CARD_BG,
            ).pack(side=tk.LEFT, padx=5)

        # Pixel boost (seamless restore resolution)
        tk.Label(
            parent, text="Pixel boost:",
            font=("Segoe UI", 9),
            bg=self.CARD_BG, fg="#aaaacc",
        ).pack(**pad, anchor=tk.W)

        boost_frame = tk.Frame(parent, bg=self.CARD_BG)
        boost_frame.pack(**pad, fill=tk.X)

        for val, label in [
            ("default", "Preset"),
            ("512", "512"),
            ("1024", "1024"),
            ("0", "Off"),
        ]:
            tk.Radiobutton(
                boost_frame, text=label, variable=self._pixel_boost,
                value=val, bg=self.CARD_BG, fg=self.FG_COLOR,
                selectcolor=self.ACCENT,
                font=("Segoe UI", 9),
                activebackground=self.CARD_BG,
            ).pack(side=tk.LEFT, padx=5)

        # Face selection
        tk.Label(
            parent, text="Target face:",
            font=("Segoe UI", 9),
            bg=self.CARD_BG, fg="#aaaacc",
        ).pack(**pad, anchor=tk.W)

        face_frame = tk.Frame(parent, bg=self.CARD_BG)
        face_frame.pack(**pad, fill=tk.X)

        for val, label in [
            ("all", "All"),
            ("largest", "Largest"),
            ("first", "First"),
            ("pose", "Pose match"),
        ]:
            tk.Radiobutton(
                face_frame, text=label, variable=self._face_select,
                value=val, bg=self.CARD_BG, fg=self.FG_COLOR,
                selectcolor=self.ACCENT,
                font=("Segoe UI", 9),
                activebackground=self.CARD_BG,
            ).pack(side=tk.LEFT, padx=5)

    def _build_preview_panel(self, parent):
        """Build the image preview area."""
        preview_frame = tk.Frame(parent, bg=self.CARD_BG, height=400)
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        self.preview_label = tk.Label(
            preview_frame,
            text="Preview will appear here after processing",
            font=("Segoe UI", 11),
            bg=self.CARD_BG, fg="#666666",
        )
        self.preview_label.pack(expand=True)

    def _build_controls_panel(self, parent):
        """Build the action buttons."""
        btn_frame = tk.Frame(parent, bg=self.BG_COLOR)
        btn_frame.pack(fill=tk.X)

        self.swap_btn = tk.Button(
            btn_frame, text="🚀  Start Face Swap",
            command=self._start_swap,
            bg=self.HIGHLIGHT, fg="white",
            font=("Segoe UI", 12, "bold"),
            relief=tk.FLAT, cursor="hand2",
            activebackground="#ff6b6b",
            height=2,
        )
        self.swap_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        self.score_btn = tk.Button(
            btn_frame, text="Score",
            command=self._score_result,
            bg="#3d3d5c", fg="white",
            font=("Segoe UI", 11),
            relief=tk.FLAT, cursor="hand2",
            height=2,
        )
        self.score_btn.pack(side=tk.LEFT, padx=(0, 5))

        self.webcam_btn = tk.Button(
            btn_frame, text="Webcam",
            command=self._start_webcam,
            bg=self.ACCENT, fg="white",
            font=("Segoe UI", 11),
            relief=tk.FLAT, cursor="hand2",
            height=2,
        )
        self.webcam_btn.pack(side=tk.RIGHT, padx=(5, 0))

    # ── Event Handlers ───────────────────────────────────────────────────

    def _select_source(self):
        path = filedialog.askopenfilename(
            title="Select Source Face Image",
            filetypes=[
                ("Images", "*.png *.jpg *.jpeg *.bmp"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self._source_path = path
            self.source_label.configure(text=Path(path).name)
            self._set_status(f"Source: {Path(path).name}")

    def _select_target(self):
        path = filedialog.askopenfilename(
            title="Select Target Image or Video",
            filetypes=[
                ("Images & Videos", "*.png *.jpg *.jpeg *.mp4 *.mov *.avi"),
                ("Images", "*.png *.jpg *.jpeg *.bmp"),
                ("Videos", "*.mp4 *.mov *.avi *.mkv"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self._target_path = path
            self.target_label.configure(text=Path(path).name)
            self._set_status(f"Target: {Path(path).name}")

    def _start_swap(self):
        if self._processing:
            messagebox.showinfo("Info", "Processing already in progress.")
            return

        if not self._source_path:
            messagebox.showwarning("Missing Input", "Please select a source face image.")
            return
        if not self._target_path:
            messagebox.showwarning("Missing Input", "Please select a target image or video.")
            return

        # Output path
        target = Path(self._target_path)
        self._output_path = str(target.parent / f"{target.stem}_swapped{target.suffix}")

        # Run in background thread
        self._processing = True
        self.swap_btn.configure(state=tk.DISABLED, text="⏳  Processing...")
        self._set_status("Processing...")

        thread = threading.Thread(target=self._run_swap, daemon=True)
        thread.start()

    def _run_swap(self):
        """Run the swap pipeline in a background thread."""
        try:
            target = Path(self._target_path)
            is_video = target.suffix.lower() in (".mp4", ".mov", ".avi", ".mkv")

            if is_video:
                self._run_video_swap()
            else:
                self._run_image_swap()

            self.root.after(0, self._on_swap_complete)

        except Exception as e:
            logger.exception("Swap failed")
            self.root.after(0, lambda: self._on_swap_error(str(e)))

    def _make_config(self):
        """Build FaceSwapConfig from UI controls."""
        from face_swap import FaceSwapConfig

        enhance = self._enhance.get()
        boost = self._pixel_boost.get()
        return FaceSwapConfig(
            quality=self._quality.get(),
            device=self._device.get(),
            enhance_method=None if enhance == "default" else enhance,
            pixel_boost=None if boost == "default" else int(boost),
            face_select=self._face_select.get(),
        )

    def _run_image_swap(self):
        """Process a single image swap."""
        import cv2
        from face_swap import swap_image

        self.root.after(0, lambda: self.progress.configure(value=20))
        self.root.after(0, lambda: self._set_status("Loading images..."))

        source = cv2.imread(self._source_path)
        target = cv2.imread(self._target_path)
        if source is None or target is None:
            raise RuntimeError("Could not load source or target image")

        self.root.after(0, lambda: self.progress.configure(value=40))
        self.root.after(0, lambda: self._set_status("Running face swap..."))

        result = swap_image(source, target, self._make_config())

        self.root.after(0, lambda: self.progress.configure(value=90))
        cv2.imwrite(self._output_path, result)
        self.root.after(0, lambda: self.progress.configure(value=100))

    def _run_video_swap(self):
        """Process a video swap with progress updates."""
        from face_swap import swap_video

        self.root.after(0, lambda: self._set_status("Processing video..."))
        self.root.after(0, lambda: self.progress.configure(value=5))

        def _progress(frame_i: int, total: int):
            pct = int((frame_i / max(total, 1)) * 100)
            self.root.after(0, lambda p=pct: self.progress.configure(value=p))
            self.root.after(
                0,
                lambda i=frame_i, t=total: self._set_status(f"Frame {i}/{t}"),
            )

        swap_video(
            self._source_path,
            self._target_path,
            self._output_path,
            self._make_config(),
            progress_callback=_progress,
        )
        self.root.after(0, lambda: self.progress.configure(value=100))

    def _on_swap_complete(self):
        self._processing = False
        self.swap_btn.configure(state=tk.NORMAL, text="🚀  Start Face Swap")
        self._set_status(f"Done! Output: {Path(self._output_path).name}")
        self.preview_label.configure(text=f"✅ Saved to:\n{self._output_path}")
        messagebox.showinfo("Complete", f"Face swap complete!\n\nSaved to:\n{self._output_path}")

    def _on_swap_error(self, error: str):
        self._processing = False
        self.swap_btn.configure(state=tk.NORMAL, text="🚀  Start Face Swap")
        self._set_status("Error")
        self.progress.configure(value=0)
        messagebox.showerror("Error", f"Face swap failed:\n{error}")

    def _score_result(self):
        """Run ArcFace ID + sharpness metrics on the last output (or target path)."""
        if not self._source_path or not self._target_path:
            messagebox.showwarning(
                "Missing Input", "Select source and target images first."
            )
            return
        result = self._output_path if self._output_path and Path(self._output_path).is_file() else None
        if result is None:
            messagebox.showwarning(
                "No Result", "Run a face swap first so there is an output to score."
            )
            return

        self._set_status("Scoring...")

        def _job():
            try:
                from face_swap import evaluate_swap

                m = evaluate_swap(
                    self._source_path,
                    self._target_path,
                    result,
                    device=self._device.get(),
                )
                msg = (
                    f"id={m.id_similarity:.3f}  vs_tgt={m.id_vs_target:.3f}\n"
                    f"sharp={m.sharpness_result:.1f} (d{m.sharpness_gain:+.1f})\n"
                    f"dE={m.color_delta_lab:.1f}  "
                    f"{'PASS' if m.passed_id else 'WEAK'} id"
                )
                self.root.after(0, lambda: self._on_score_done(msg))
            except Exception as e:
                self.root.after(0, lambda: self._on_score_error(str(e)))

        threading.Thread(target=_job, daemon=True).start()

    def _on_score_done(self, msg: str):
        self._set_status("Score complete")
        self.preview_label.configure(text=msg)
        messagebox.showinfo("Swap Score", msg)

    def _on_score_error(self, error: str):
        self._set_status("Score failed")
        messagebox.showerror("Score Error", error)

    def _start_webcam(self):
        if not self._source_path:
            messagebox.showwarning("Missing Input", "Please select a source face image first.")
            return
        self._set_status("Starting webcam...")
        try:
            from demos.webcam_demo import WebcamDemo

            demo = WebcamDemo(self._make_config())
            import cv2

            src = cv2.imread(self._source_path)
            if src is None:
                raise RuntimeError("Could not load source image")
            demo.initialize(src)
            # Blocks until user quits webcam window
            demo.run(camera_id=0)
            self._set_status("Webcam closed")
        except Exception as e:
            logger.exception("Webcam failed")
            messagebox.showerror("Webcam Error", str(e))
            self._set_status("Error")

    def _set_status(self, text: str):
        self.status_label.configure(text=text)

    def _apply_styles(self):
        """Apply custom ttk styles."""
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "TProgressbar",
            troughcolor=self.CARD_BG,
            background=self.HIGHLIGHT,
            thickness=8,
        )


def main():
    """Launch the Face Swap GUI."""
    app = FaceSwapGUI()
    app.run()


if __name__ == "__main__":
    main()
