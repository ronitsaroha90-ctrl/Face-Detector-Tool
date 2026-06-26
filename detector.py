"""
detector.py
-----------
Handles all face detection logic using OpenCV's Haar Cascade classifier.
Keeps detection completely separate from GUI code.
"""

import cv2
import numpy as np
import os


class FaceDetector:
    """
    Wraps OpenCV's Haar Cascade face detector.
    Provides methods for detecting faces in images and video frames.
    """

    def __init__(self, cascade_path="haarcascade_frontalface_default.xml"):
        """
        Load the Haar Cascade XML file.

        Args:
            cascade_path (str): Path to the cascade XML file.

        Raises:
            FileNotFoundError: If the cascade file is missing.
        """
        if not os.path.exists(cascade_path):
            # Try OpenCV's built-in data folder as a fallback
            builtin = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            if os.path.exists(builtin):
                cascade_path = builtin
            else:
                raise FileNotFoundError(
                    f"Cascade file not found at '{cascade_path}'. "
                    "Please place haarcascade_frontalface_default.xml in the project folder."
                )

        self.cascade = cv2.CascadeClassifier(cascade_path)

        if self.cascade.empty():
            raise ValueError("Failed to load the cascade classifier. The XML file may be corrupt.")

    def detect(self, frame):
        """
        Detect faces in a single BGR frame.

        Converts the frame to grayscale for detection, then maps
        bounding boxes back to the original colour frame.

        Args:
            frame (numpy.ndarray): BGR image from OpenCV.

        Returns:
            tuple:
                annotated (numpy.ndarray): Frame with bounding boxes drawn.
                count (int): Number of faces detected.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # equalizeHist improves detection in poor lighting
        gray = cv2.equalizeHist(gray)

        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,   # how much the image size is reduced at each scale
            minNeighbors=5,    # higher = fewer false positives
            minSize=(40, 40),  # ignore tiny detections
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        annotated = frame.copy()

        if len(faces) > 0:
            for (x, y, w, h) in faces:
                # Draw a green rectangle around each detected face
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 200, 0), 2)

                # Small label above the bounding box
                label = "Face"
                cv2.putText(
                    annotated, label,
                    (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 200, 0), 1, cv2.LINE_AA
                )

        return annotated, len(faces)

    def detect_from_image_path(self, image_path):
        """
        Load an image file and run face detection on it.

        Args:
            image_path (str): Absolute or relative path to the image.

        Returns:
            tuple: (annotated frame, face count)

        Raises:
            FileNotFoundError: If the image file does not exist.
            ValueError: If the file cannot be read as an image.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        frame = cv2.imread(image_path)

        if frame is None:
            raise ValueError(f"Could not read image: {image_path}")

        return self.detect(frame)
