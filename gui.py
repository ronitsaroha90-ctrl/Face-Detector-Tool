"""
gui.py
------
Tkinter-based graphical interface for the Face Detection Tool.
Handles all user interactions and delegates detection work to detector.py.
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import threading
import os

from detector import FaceDetector
from utils import bgr_to_tk_image, save_output_image, resize_frame, draw_face_count


# ── Colour palette ──────────────────────────────────────────────────────────
BG_COLOR      = "#1e1e2e"   # dark background
PANEL_COLOR   = "#2a2a3e"   # slightly lighter panel
BTN_COLOR     = "#3a86ff"   # accent blue for primary buttons
BTN_STOP      = "#e63946"   # red for stop/exit
BTN_SAVE      = "#2ec4b6"   # teal for save
BTN_TEXT      = "#ffffff"
LABEL_COLOR   = "#cdd6f4"
TITLE_COLOR   = "#89b4fa"
STATUS_OK     = "#a6e3a1"   # green
STATUS_WARN   = "#fab387"   # orange
STATUS_ERR    = "#f38ba8"   # red
FACE_CTR      = "#f9e2af"   # yellow


class FaceDetectionApp:
    """
    Main application class. Manages the Tkinter window, button callbacks,
    webcam/video loops, and communication with FaceDetector.
    """

    def __init__(self, root):
        self.root = root
        self.root.title("Face Detection Tool")
        self.root.geometry("900x750")
        self.root.configure(bg=BG_COLOR)
        self.root.resizable(True, True)

        # ── State ─────────────────────────────────────────────────────────
        self.detector       = None
        self.cap            = None          # OpenCV VideoCapture object
        self.is_running     = False         # True while camera/video loop is active
        self.current_frame  = None          # Latest annotated frame (for saving)
        self.face_count     = 0

        # ── Load detector ─────────────────────────────────────────────────
        try:
            self.detector = FaceDetector()
        except (FileNotFoundError, ValueError) as e:
            messagebox.showerror("Startup Error", str(e))
            self.root.destroy()
            return

        self._build_ui()

    # ─────────────────────────────────────────────────────────────────────────
    #  UI Construction
    # ─────────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        """Build and lay out all Tkinter widgets."""

        # ── Title bar ─────────────────────────────────────────────────────
        title_frame = tk.Frame(self.root, bg=BG_COLOR)
        title_frame.pack(fill="x", padx=20, pady=(16, 4))

        tk.Label(
            title_frame,
            text="🎯  Face Detection Tool",
            font=("Segoe UI", 18, "bold"),
            bg=BG_COLOR, fg=TITLE_COLOR
        ).pack(side="left")

        tk.Label(
            title_frame,
            text="OpenCV  |  Haar Cascade",
            font=("Segoe UI", 10),
            bg=BG_COLOR, fg="#585b70"
        ).pack(side="right", pady=6)

        # ── Canvas (video display) ─────────────────────────────────────────
        self.canvas = tk.Label(
            self.root,
            bg="#11111b",
            width=640, height=480,
            text="No feed active",
            font=("Segoe UI", 13),
            fg="#585b70"
        )
        self.canvas.pack(padx=20, pady=8, expand=False)

        # ── Info row (status + face counter) ──────────────────────────────
        info_frame = tk.Frame(self.root, bg=BG_COLOR)
        info_frame.pack(fill="x", padx=20, pady=(0, 6))

        self.status_label = tk.Label(
            info_frame,
            text="● Ready",
            font=("Segoe UI", 10),
            bg=BG_COLOR, fg=STATUS_OK,
            anchor="w"
        )
        self.status_label.pack(side="left")

        self.face_label = tk.Label(
            info_frame,
            text="Faces detected: 0",
            font=("Segoe UI", 10, "bold"),
            bg=BG_COLOR, fg=FACE_CTR,
            anchor="e"
        )
        self.face_label.pack(side="right")

        # ── Button panel ──────────────────────────────────────────────────
        btn_frame = tk.Frame(self.root, bg=PANEL_COLOR, pady=12)
        btn_frame.pack(fill="x", padx=0, pady=(4, 0), side="bottom")

        btn_cfg = dict(
            font=("Segoe UI", 10, "bold"),
            relief="flat", cursor="hand2",
            width=16, height=2, bd=0
        )

        self.btn_camera = tk.Button(
            btn_frame, text="📷  Open Camera",
            bg=BTN_COLOR, fg=BTN_TEXT,
            command=self.start_camera, **btn_cfg
        )
        self.btn_camera.grid(row=0, column=0, padx=12, pady=4)

        self.btn_image = tk.Button(
            btn_frame, text="🖼  Detect from Image",
            bg=BTN_COLOR, fg=BTN_TEXT,
            command=self.detect_image, **btn_cfg
        )
        self.btn_image.grid(row=0, column=1, padx=12, pady=4)

        self.btn_video = tk.Button(
            btn_frame, text="🎬  Detect from Video",
            bg=BTN_COLOR, fg=BTN_TEXT,
            command=self.detect_video, **btn_cfg
        )
        self.btn_video.grid(row=0, column=2, padx=12, pady=4)

        self.btn_stop = tk.Button(
            btn_frame, text="⏹  Stop",
            bg="#45475a", fg=BTN_TEXT,
            command=self.stop_feed, state="disabled", **btn_cfg
        )
        self.btn_stop.grid(row=0, column=3, padx=12, pady=4)

        self.btn_save = tk.Button(
            btn_frame, text="💾  Save Output",
            bg=BTN_SAVE, fg=BTN_TEXT,
            command=self.save_output, state="disabled", **btn_cfg
        )
        self.btn_save.grid(row=1, column=0, padx=12, pady=4)

        self.btn_exit = tk.Button(
            btn_frame, text="✖  Exit",
            bg=BTN_STOP, fg=BTN_TEXT,
            command=self.on_exit, **btn_cfg
        )
        self.btn_exit.grid(row=1, column=3, padx=12, pady=4)

        # Centre all columns
        for col in range(4):
            btn_frame.columnconfigure(col, weight=1)

        # ── Footer ────────────────────────────────────────────────────────
        tk.Label(
            self.root,
            text="Ronit Saroha  ·  B.Tech CSE  ·  SMVDU",
            font=("Segoe UI", 8),
            bg=BG_COLOR, fg="#313244"
        ).pack(pady=(2, 8))

        # Intercept window close button
        self.btn_camera.invoke()
        self.root.protocol("WM_DELETE_WINDOW", self.on_exit)

    # ─────────────────────────────────────────────────────────────────────────
    #  Status helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _set_status(self, message, colour=STATUS_OK):
        self.status_label.config(text=f"● {message}", fg=colour)

    def _set_face_count(self, count):
        self.face_count = count
        self.face_label.config(text=f"Faces detected: {count}")

    def _set_buttons_active(self, running):
        """Toggle buttons between idle and running states."""
        state_primary = "disabled" if running else "normal"
        state_stop    = "normal"   if running else "disabled"

        self.btn_camera.config(state=state_primary)
        self.btn_image.config(state=state_primary)
        self.btn_video.config(state=state_primary)
        self.btn_stop.config(state=state_stop)

    # ─────────────────────────────────────────────────────────────────────────
    #  Camera
    # ─────────────────────────────────────────────────────────────────────────

    def start_camera(self):
        """Open the default webcam and begin the detection loop."""
        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            messagebox.showerror(
                "Camera Error",
                "Could not access the webcam.\n"
                "Make sure it is connected and not used by another application."
            )
            return

        self.is_running = True
        self._set_buttons_active(True)
        self._set_status("Webcam active – detecting faces…")
        self.btn_save.config(state="normal")

        # Run the frame loop in a background thread so the GUI stays responsive
        thread = threading.Thread(target=self._webcam_loop, daemon=True)
        thread.start()

    def _webcam_loop(self):
        """Read frames from webcam, detect faces, update canvas. Runs in a thread."""
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                self._set_status("Webcam feed lost.", STATUS_ERR)
                break

            frame = resize_frame(frame)
            annotated, count = self.detector.detect(frame)
            draw_face_count(annotated, count)

            self.current_frame = annotated
            self._update_canvas(annotated)
            self._set_face_count(count)

        self._release_capture()

    # ─────────────────────────────────────────────────────────────────────────
    #  Image detection
    # ─────────────────────────────────────────────────────────────────────────

    def detect_image(self):
        """Let the user pick an image file and run face detection on it."""
        path = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All Files", "*.*")]
        )
        if not path:
            return

        try:
            annotated, count = self.detector.detect_from_image_path(path)
        except (FileNotFoundError, ValueError) as e:
            messagebox.showerror("Image Error", str(e))
            return

        draw_face_count(annotated, count)
        self.current_frame = annotated
        self._update_canvas(annotated)
        self._set_face_count(count)
        self.btn_save.config(state="normal")

        if count == 0:
            self._set_status("Image loaded – no faces detected.", STATUS_WARN)
        else:
            self._set_status(f"Image loaded – {count} face(s) detected.")

    # ─────────────────────────────────────────────────────────────────────────
    #  Video detection
    # ─────────────────────────────────────────────────────────────────────────

    def detect_video(self):
        """Let the user pick a video file and process it frame by frame."""
        path = filedialog.askopenfilename(
            title="Select a Video File",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv *.wmv"), ("All Files", "*.*")]
        )
        if not path:
            return

        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            messagebox.showerror("Video Error", f"Could not open video:\n{path}")
            return

        self.is_running = True
        self._set_buttons_active(True)
        self._set_status("Playing video – detecting faces…")
        self.btn_save.config(state="normal")

        thread = threading.Thread(target=self._video_loop, daemon=True)
        thread.start()

    def _video_loop(self):
        """Process video frames one by one until end or stop. Runs in a thread."""
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                # Reached end of video
                self._set_status("Video finished.")
                break

            frame = resize_frame(frame)
            annotated, count = self.detector.detect(frame)
            draw_face_count(annotated, count)

            self.current_frame = annotated
            self._update_canvas(annotated)
            self._set_face_count(count)

            # ~25 fps playback: wait 40 ms between frames
            if cv2.waitKey(40) & 0xFF == ord("q"):
                break

        self.is_running = False
        self._set_buttons_active(False)
        self._release_capture()

    # ─────────────────────────────────────────────────────────────────────────
    #  Save / Stop / Exit
    # ─────────────────────────────────────────────────────────────────────────

    def save_output(self):
        """Save the most recent annotated frame to the assets/ folder."""
        if self.current_frame is None:
            messagebox.showwarning("Nothing to Save", "No frame available to save yet.")
            return

        try:
            filepath = save_output_image(self.current_frame)
            self._set_status(f"Saved → {os.path.basename(filepath)}")
            messagebox.showinfo("Saved", f"Image saved:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Save Error", str(e))

    def stop_feed(self):
        """Stop the active webcam or video loop."""
        self.is_running = False
        self._set_status("Feed stopped.")
        self._set_buttons_active(False)
        self._set_face_count(0)

    def on_exit(self):
        """Cleanly shut down the application."""
        self.is_running = False
        self._release_capture()
        self.root.destroy()

    # ─────────────────────────────────────────────────────────────────────────
    #  Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _update_canvas(self, frame):
        """Convert a BGR frame and display it on the Tkinter canvas label."""
        tk_img = bgr_to_tk_image(frame, max_width=640, max_height=480)
        # Keep a reference — Tkinter's GC will delete it otherwise
        self.canvas.config(image=tk_img, text="")
        self.canvas.image = tk_img

    def _release_capture(self):
        """Release the OpenCV VideoCapture if open."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
