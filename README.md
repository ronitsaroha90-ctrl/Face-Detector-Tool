# 🎯 Face Detection Tool

A desktop application that detects human faces in real time using Python and OpenCV's Haar Cascade classifier. Built as part of a Computer Vision learning project during my second year of B.Tech CSE.

---

## 📌 Overview

This tool can detect faces from three sources:
- **Live webcam feed** — detects faces in real time
- **Image files** — analyse any JPG, PNG, BMP, or WebP image
- **Pre-recorded videos** — process MP4, AVI, MOV, or MKV files

Detected faces are highlighted with green bounding boxes. The total count is displayed on screen and updated every frame.

---

## ✨ Features

- Real-time webcam face detection
- Static image face detection with file browser
- Video file face detection with frame-by-frame processing
- Green bounding boxes drawn around each detected face
- Live face counter overlay on the video/image feed
- Save the current annotated frame as a PNG
- Clean error handling for missing camera, invalid files, and no-face scenarios
- Start, Stop, Save, and Exit controls
- Neat Tkinter GUI — no web framework required

---

## 🛠 Technologies Used

| Library | Purpose |
|---------|---------|
| Python 3 | Core language |
| OpenCV | Image processing and Haar Cascade detection |
| NumPy | Array operations on frames |
| Pillow (PIL) | Converting OpenCV frames to Tkinter-compatible images |
| Tkinter | GUI (built into Python standard library) |

> No deep learning models, no cloud APIs, no web frameworks.

---

## 📁 Project Structure

```
FaceDetectionTool/
│
├── main.py                              # Entry point — launches the app
├── gui.py                               # Tkinter GUI and button logic
├── detector.py                          # Haar Cascade face detection
├── utils.py                             # Helper functions (conversion, save, resize)
├── requirements.txt                     # Python dependencies
├── README.md                            # This file
├── haarcascade_frontalface_default.xml  # OpenCV Haar Cascade model
└── assets/                              # Saved output images go here
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/FaceDetectionTool.git
cd FaceDetectionTool
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the Haar Cascade file

The cascade XML file is included with OpenCV. If it is not in the project folder, the app will automatically fall back to OpenCV's built-in path. To add it manually:

```bash
# Find the built-in path
python -c "import cv2; print(cv2.data.haarcascades)"
```

Copy `haarcascade_frontalface_default.xml` from that folder into the project root.

---

## ▶️ How to Run

```bash
python main.py
```

The GUI window will open. Use the buttons to:

| Button | Action |
|--------|--------|
| 📷 Open Camera | Start live webcam detection |
| 🖼 Detect from Image | Open an image file and detect faces |
| 🎬 Detect from Video | Open a video file and process it |
| ⏹ Stop | Stop the active feed |
| 💾 Save Output | Save the current frame to `assets/` |
| ✖ Exit | Close the application |

---

## 📸 Screenshots

> *(Add screenshots here after running the application)*

- `screenshots/webcam_detection.png`
- `screenshots/image_detection.png`
- `screenshots/video_detection.png`

---

## 🔍 How It Works

1. Each frame is converted from colour (BGR) to **grayscale** — Haar Cascades work on single-channel images.
2. `cv2.equalizeHist()` normalises brightness to improve detection in dim lighting.
3. `detectMultiScale()` slides a detection window across the image at multiple scales looking for face-like patterns.
4. Detected regions are returned as `(x, y, width, height)` bounding boxes, which are drawn on the original colour frame.
5. The annotated frame is converted to RGB → PIL Image → `ImageTk.PhotoImage` before being displayed on the Tkinter canvas.

Key parameters in `detectMultiScale`:
- `scaleFactor=1.1` — reduces image by 10% at each pyramid level
- `minNeighbors=5` — a face must be confirmed by 5 overlapping detections to count
- `minSize=(40, 40)` — ignores detections smaller than 40×40 pixels

---

## 🔮 Future Improvements

- **Eye detection** — add a second Haar Cascade for eyes
- **Smile detection** — detect smiles within detected face regions
- **Face recognition** — use OpenCV's LBPH (Local Binary Pattern Histogram) recogniser to identify specific individuals
- **Attendance system** — combine face recognition with a SQLite database to mark attendance automatically
- **Emotion detection** — integrate a lightweight pre-trained model to classify expressions (happy, sad, neutral, etc.)
- **Deep learning upgrade** — replace Haar Cascade with a DNN-based detector (e.g. OpenCV's `dnn` module with a SSD model) for higher accuracy on tilted or partially occluded faces
- **Multi-face tracking** — assign IDs to tracked faces across video frames using centroid tracking

---

## 👤 Author

**Ronit Saroha**  
B.Tech CSE — Shri Mata Vaishno Devi University  
📧 ronitsaroha90@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/ronit-saroha-371493381)

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
