# IRIS-CNN // Complete Project Guide, Architecture & Developer Tutorial

> **Project Title:** Iris Recognition by Convolutional Neural Network (CNN) Model (`IRIS-CNN // NEURAL OPTICS`)  
> **Author:** Mohd Amir Ali (`160920733019`) — B.E. CSE, Lords Institute of Engineering & Technology (Osmania University)

---

## 1. How the Whole Project Works (The 4-Stage Biometric Pipeline)

Your mini-project identifies individuals from Near-Infrared (850nm) eye images across **108 subjects (`683` scans from the CASIA-IrisV1 dataset)** using Computer Vision (**OpenCV Hough Circle Transform**) and Deep Learning (**7-Layer Convolutional Neural Network**):

1. **Stage 1 — Image Acquisition & Denoising (`cv2.imread` + `cv2.medianBlur`):**
   * Reads the eye scan in grayscale and applies a `5×5` median blur filter to remove eyelash reflections and sensor noise.
2. **Stage 2 — Circular Iris Segmentation (`cv2.HoughCircles` + Bitwise Masking):**
   * Detects the circular pupil and outer iris ring using the Hough Gradient method (`param1=63, param2=70`).
   * Masks out the sclera and eyelids (`cv2.bitwise_and`), crops the bounding rectangle around the iris contour (`cv2.findContours` + `cv2.boundingRect`), and saves the isolated iris ROI to `test.png`.
3. **Stage 3 — 7-Layer Deep CNN Feature Extraction (`model/model.json` & `model_weights.h5`):**
   * Resizes the cropped iris ROI into a normalized `64 × 64 × 3` tensor (`[0, 1]` float32) and passes it through your 7-layer Keras Sequential CNN:
     * **Layer 1 (`conv2d_1`):** 32 filters (`3×3`, `ReLU`) → Output `62 × 62 × 32` (`896` params)
     * **Layer 2 (`max_pooling2d_1`):** `2×2` MaxPool → Output `31 × 31 × 32`
     * **Layer 3 (`conv2d_2`):** 32 filters (`3×3`, `ReLU`) → Output `29 × 29 × 32` (`9,248` params)
     * **Layer 4 (`max_pooling2d_2`):** `2×2` MaxPool → Output `14 × 14 × 32`
     * **Layer 5 (`flatten_1`):** Flattens spatial feature maps into a 1D vector of `6,272` features
     * **Layer 6 (`dense_1`):** Fully connected layer with `256` units (`ReLU`, `1,605,888` params)
     * **Layer 7 (`dense_2`):** Output layer with `108` units (`Softmax`, `27,756` params)
4. **Stage 4 — Identity Prediction & Liveness Verification:**
   * Takes `argmax(preds) + 1` to predict the exact **Person ID (`1` to `108`)**, computes the Softmax confidence percentage and Hamming distance, and verifies NIR corneal liveness.

---

## 2. File & Folder Map (`/Users/mohdamirali/Desktop/IrisRecognition`)

| File / Folder | Role (`Frontend` vs. `Backend`) | What It Controls |
| :--- | :--- | :--- |
| **`index.html` & `Launch_Iris_Dashboard.html`** | **Standalone Web App (Frontend + Embedded Engine)** | Contains the entire web interface, 3D Eye Scanner canvas, 7-Layer CNN visualizer, Base64 CASIA test samples, and 60-epoch chart in one file for **GitHub Pages** and instant local Chrome launch. |
| **`Run_Iris_Project.command` & `start.sh`** | **macOS / CLI Launchers** | Starts the Python web server (`WebApp.py`) on `http://127.0.0.1:5055/index` (or runs `Main.py` desktop GUI with `./start.sh --desktop`). |
| **`WebApp.py`** | **Python Web Backend (Port 5055)** | Serves the web dashboard on port `5055` and provides `/api/summary` and `/api/recognize` endpoints. |
| **`iris_cnn_engine.py`** | **Core Biometric & CNN Engine (Backend)** | Unpickles `model/history.pckl` (60 epochs), parses `model/model.json`, indexes all 108 folders in `CASIA1/`, and matches uploaded eye images. |
| **`Main.py`** | **Desktop Tkinter GUI** | Upgraded version of your original desktop script with modern styling and a native Tkinter 60-epoch Accuracy & Loss Graph fallback. |
| **`static/iris_style.css`** | **Master Stylesheet (Frontend CSS)** | Controls the **Biometric Neural-Optics** color palette (`#F3F1F8` lavender-alabaster background, `#121024` deep obsidian ink, `#4F46E5` electric iris indigo, `#0891B2` optical cyan). |
| **`model/`** | **Trained CNN Artifacts** | Contains `model.json` (architecture), `model_weights.h5` (trained weights), `history.pckl` (60-epoch accuracy/loss curves), `X.txt.npy`, and `Y.txt.npy`. |
| **`testSamples/` & `CASIA1/`** | **Biometric Iris Dataset** | `testSamples/` has quick test scans (`a.jpg` → ID #1, `b.jpg` → ID #15, `c.jpg` → ID #14, `d.jpg` → ID #6); `CASIA1/` has all 108 subject folders (`1/` through `108/`). |

---

## 3. Where to Make Frontend & Backend Edits

* **Change Colors or Fonts:** Edit `:root` at the top of `static/iris_style.css` (or at the top `<style>` block of `index.html` / `Launch_Iris_Dashboard.html`). Changing `--accent-indigo: #4F46E5;` and `--accent-cyan: #0891B2;` updates all buttons, badges, 3D eye reticles, and neural network highlights.
* **Change Titles, Credits, or Guide Name:** Edit the `.author-credit-box` and `.hero-display` sections inside `index.html` / `Launch_Iris_Dashboard.html`.
* **Add or Edit Preloaded Test Samples:** Edit `get_preloaded_samples()` inside `iris_cnn_engine.py` and re-run `python3 /tmp/gen_iris_app.py` to bundle new samples into `index.html`.
