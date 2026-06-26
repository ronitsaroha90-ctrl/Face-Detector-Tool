"""
utils.py
--------
Helper functions shared across the project.
Handles image conversion, saving output, and frame resizing.
"""

import cv2
import numpy as np
from PIL import Image, ImageTk
import os
from datetime import datetime


def bgr_to_tk_image(frame, max_width=640, max_height=480):
    """
    Convert an OpenCV BGR frame to a Tkinter-compatible PhotoImage.

    Tkinter cannot display OpenCV frames directly, so we:
      1. Convert BGR → RGB
      2. Resize to fit the display canvas
      3. Wrap in a PIL ImageTk object

    Args:
        frame (numpy.ndarray): BGR image from OpenCV.
        max_width (int): Maximum display width in pixels.
        max_height (int): Maximum display height in pixels.

    Returns:
        ImageTk.PhotoImage: Tkinter-compatible image object.
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = frame_rgb.shape[:2]

    # Scale down while maintaining aspect ratio
    scale = min(max_width / w, max_height / h, 1.0)
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        frame_rgb = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

    pil_image = Image.fromarray(frame_rgb)
    return ImageTk.PhotoImage(pil_image)


def save_output_image(frame, output_dir="assets"):
    """
    Save an annotated frame as a timestamped PNG file.

    Args:
        frame (numpy.ndarray): BGR image to save.
        output_dir (str): Folder where the file will be written.

    Returns:
        str: Full path of the saved file.
    """
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"detected_{timestamp}.png"
    filepath = os.path.join(output_dir, filename)

    cv2.imwrite(filepath, frame)
    return filepath


def resize_frame(frame, width=640, height=480):
    """
    Resize a frame to exact dimensions (used for webcam/video display).

    Args:
        frame (numpy.ndarray): Input BGR frame.
        width (int): Target width.
        height (int): Target height.

    Returns:
        numpy.ndarray: Resized frame.
    """
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)


def draw_face_count(frame, count):
    """
    Overlay the detected face count on the top-left corner of a frame.

    Args:
        frame (numpy.ndarray): BGR image to annotate.
        count (int): Number of detected faces.

    Returns:
        numpy.ndarray: Frame with count label drawn.
    """
    label = f"Faces: {count}"
    # Dark semi-transparent background behind text for readability
    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(frame, (8, 8), (16 + text_w, 20 + text_h), (0, 0, 0), -1)
    cv2.putText(
        frame, label,
        (12, 12 + text_h),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7, (0, 255, 0), 2, cv2.LINE_AA
    )
    return frame
