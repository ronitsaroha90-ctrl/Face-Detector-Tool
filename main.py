"""
Face Detection Tool
-------------------
Entry point for the application.
Initializes the Tkinter root window and launches the GUI.
"""

import tkinter as tk
from gui import FaceDetectionApp


def main():
    root = tk.Tk()
    app = FaceDetectionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
