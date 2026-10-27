# IRIS RECOGNITION - CNN · Biometric Iris Recognition Studio

[![Launch Live Demo](https://img.shields.io/badge/Launch%20Demo-Live%20Interactive%20Platform-D97706?style=for-the-badge&labelColor=0E131B)](https://mohd-amir-ali.github.io/Iris-Recognition-CNN-Biometrics/)
[![Launch Standalone App](https://img.shields.io/badge/Launch%20App-Zero--Install%20HTML5%20%2B%20Python%20API-0D9488?style=for-the-badge&labelColor=0E131B)](https://mohd-amir-ali.github.io/Iris-Recognition-CNN-Biometrics/Launch_Iris_Dashboard.html)
[![Developer Guide](https://img.shields.io/badge/Developer%20Guide-Interactive%20Handbook-D97706?style=for-the-badge&labelColor=0E131B)](https://mohd-amir-ali.github.io/Iris-Recognition-CNN-Biometrics/Project_Tutorial_Guide.html)
[![Open in GitHub Codespaces](https://img.shields.io/badge/GitHub%20Codespaces-Run%20Python%20Backend-161D2A?style=for-the-badge&logo=github)](https://codespaces.new/mohd-amir-ali/Iris-Recognition-CNN-Biometrics)
[![Author](https://img.shields.io/badge/Author-Mohd%20Amir%20Ali-D97706?style=for-the-badge&labelColor=0E131B)](https://techwithamir.com)

> **B.E. Computer Science & Engineering Biometrics Project**  
> **Author:** Mohd Amir Ali ([techwithamir.com](https://techwithamir.com))  
> **Institution:** Lords Institute of Engineering & Technology (Affiliated to Osmania University, Hyderabad)

> **Live Interactive Links:**
> - **[Launch Live Interactive Demo (GitHub Pages) ↗](https://mohd-amir-ali.github.io/Iris-Recognition-CNN-Biometrics/)** - Explore the full-window Gaze-Tracking Iris & Hough Laser Scanner, 7-Layer CNN Topology Visualizer, Role-Gated Biometric Scanner, and 60-Epoch Analytics.
> - **[Launch Standalone HTML5 Dashboard (`Launch_Iris_Dashboard.html`) ↗](https://mohd-amir-ali.github.io/Iris-Recognition-CNN-Biometrics/Launch_Iris_Dashboard.html)** - Zero-install browser build.
> - **[Read Interactive Developer Handbook (`Project_Tutorial_Guide.html`) ↗](https://mohd-amir-ali.github.io/Iris-Recognition-CNN-Biometrics/Project_Tutorial_Guide.html)** - Complete pipeline architecture and file reference.

---

## 01 - Run Instantly Without Cloning

Anyone can explore and test the entire biometric iris recognition platform directly in their browser without downloading or cloning the repository:

1. **[Launch Live Interactive Web App (GitHub Pages)](https://mohd-amir-ali.github.io/Iris-Recognition-CNN-Biometrics/)**  
   Features the **Full-Window Gaze-Tracking Anatomical Eye & Hough-Circle Laser Scanner** (with autonomous natural blink and pupil light-reflex micro-animation), **Live 7-Layer CNN Neural Network Topology Visualizer**, and the **Role-Gated Admin Biometric Scanner** (Demo Login: `Amir` / `amir123` or **1-Click Demo Admin Unlock**) with preloaded **CASIA-IrisV1** test samples, custom eye image upload, **Synthetic Iris Spoof-Attack Simulator**, **Interactive 7-Layer Architecture Inspector**, and **Live Quality Gate Operator Enrollment**.
2. **[Launch Full Python Backend in GitHub Codespaces](https://codespaces.new/mohd-amir-ali/Iris-Recognition-CNN-Biometrics)**  
   Boots `WebApp.py` (Port `5055`) inside a cloud container in one click.

---

## 02 - System Architecture & Pipeline

Unlike unimodal biometrics that degrade under lighting or age variations, the human iris contains over 240 unique trabecular degrees of freedom (crypts, radial furrows, and collarette rings) that remain stable throughout life. This system processes Near-Infrared (850nm) eye images across **108 subjects (`683` images from CASIA-IrisV1)** through a 4-stage pipeline:

```text
[Raw NIR Eye Scan] ──► [5×5 Median Blur + Hough Circle Transform]
                              │  Isolates pupil & outer iris ring (test.png)
                              ▼
[108-Class Softmax] ◄── [7-Layer Deep Sequential CNN (64×64×3 Tensor)]
 (Person ID #1-#108)     Conv2D(32) → Pool → Conv2D(32) → Pool → Dense(256)
```

1. **Stage 1 - Image Acquisition & Denoising:** Reads grayscale NIR eye images (`cv2.imread`) and applies a `5×5` median filter (`cv2.medianBlur`) to eliminate specular reflection noise.
2. **Stage 2 - Hough Circle Iris Localization:** Applies the Hough Gradient Transform (`cv2.HoughCircles`) with bitwise masking, contour bounding-box extraction, and Daugman polar unwrapping to isolate the circular iris Region of Interest (ROI).
3. **Stage 3 - 7-Layer CNN Feature Extraction:** Normalizes the cropped iris ROI to `64 × 64 × 3` and passes it through two `Conv2D (32 filters, 3×3, ReLU)` + `MaxPooling2D (2×2)` blocks and a `256-unit Dense` embedding layer (`1,643,788` trainable parameters).
4. **Stage 4 - Identity Classification & Liveness Check:** Evaluates the 108-class Softmax output distribution alongside Hamming distance and NIR specular liveness verification, achieving **100.0% training accuracy** across 60 epochs (`final loss: 0.0119`).

---

## 03 - Project Structure

```text
.
├── index.html                        # Live GitHub Pages Interactive Web App
├── Launch_Iris_Dashboard.html        # Standalone Local Browser Dashboard (No Terminal Needed)
├── Project_Tutorial_Guide.html       # Interactive Workflow & Developer Handbook
├── PROJECT_GUIDE_AND_TUTORIAL.md     # Markdown Developer Tutorial & Customization Guide
├── Run_Iris_Project.command          # One-Click macOS Finder Web Server Launcher (Port 5055)
├── run.bat                           # One-Click Windows Launcher
├── start.sh                          # CLI Launcher (Web Server or --desktop Tkinter GUI)
├── WebApp.py                         # Python Web Server (Port 5055) & REST Recognition API
├── Main.py                           # Desktop Tkinter Biometric GUI
├── iris_cnn_engine.py                # Universal CNN Architecture, History & CASIA-108 Engine
├── model/
│   ├── model.json                    # 7-Layer Sequential CNN Architecture
│   ├── model_weights.h5              # Trained CNN Weights (108 CASIA Classes)
│   ├── history.pckl                  # 60-Epoch Accuracy & Crossentropy Loss History
│   └── Y.txt.npy                     # One-Hot Encoded Target Labels (683 × 108)
├── testSamples/                      # Quick-Test NIR Eye Images (a.jpg, b.jpg, c.jpg, d.jpg)
├── CASIA1/                           # 108 Enrolled Subject Folders (1/ through 108/)
└── static/
    └── iris_style.css                # Biometric Optical Studio Master Stylesheet
```

---

## 04 - Quick Start (macOS / Linux / Windows)

### Option A: Instant Standalone Launch (No Terminal Required)
Double-click **`Launch_Iris_Dashboard.html`** (or `index.html`) to open the entire interactive application directly in Chrome, Edge, or Safari.

### Option B: Run the Python Web Server (Port `5055`)
Double-click **`Run_Iris_Project.command`** in macOS Finder, or run:
```bash
chmod +x start.sh
./start.sh
```
Then open **[http://127.0.0.1:5055/index](http://127.0.0.1:5055/index)** in your browser (Demo Admin Login: `Amir` / `amir123`).

### Option C: Launch the Desktop Tkinter GUI
```bash
./start.sh --desktop
# or:
python3 Main.py
```
