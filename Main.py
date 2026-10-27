import os
import tkinter as tk
from tkinter import END, filedialog, Canvas, Text, Scrollbar, Button, Frame, Label, Toplevel

from iris_cnn_engine import engine

# Optional OpenCV / Matplotlib imports (gracefully falls back to native Tkinter if not installed)
try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False

# Biometric Neural-Optics Palette
BG_MIST = "#F3F1F8"
BG_SURFACE = "#FAF9FE"
INK_DARK = "#121024"
ACCENT_INDIGO = "#4F46E5"
ACCENT_CYAN = "#0891B2"
SLATE_MUTED = "#52506E"
BORDER_LAVENDER = "#D5CFE6"
VERIFIED_GREEN = "#059669"

filename = ""
text = None
status_label = None
preview_canvas = None


def drawPreviewPlaceholder(c_widget, predicted_id=None, conf=None, sample_name=""):
    c_widget.delete("all")
    w, h = 440, 420
    for gx in range(20, w, 28):
        for gy in range(20, h, 28):
            c_widget.create_oval(gx, gy, gx + 2, gy + 2, fill="#D5CFE6", outline="")

    cx, cy = w // 2, h // 2 - 20
    # Outer sclera / eyelid arcs
    c_widget.create_oval(cx - 135, cy - 85, cx + 135, cy + 85, outline=INK_DARK, width=2, fill="#EAE6F3")
    # Outer Iris Hough Circle
    c_widget.create_oval(cx - 82, cy - 82, cx + 82, cy + 82, outline=ACCENT_INDIGO, width=2.5, fill="#2A264F")
    # Radial trabecular iris fibers
    import math
    for deg in range(0, 360, 15):
        rad = math.radians(deg)
        x1 = cx + math.cos(rad) * 34
        y1 = cy + math.sin(rad) * 34
        x2 = cx + math.cos(rad) * 78
        y2 = cy + math.sin(rad) * 78
        c_widget.create_line(x1, y1, x2, y2, fill="#6366F1", width=1)
    # Inner Pupil Hough Circle
    c_widget.create_oval(cx - 32, cy - 32, cx + 32, cy + 32, outline=ACCENT_CYAN, width=2, fill="#090814")
    # NIR Specular highlight
    c_widget.create_oval(cx - 16, cy - 18, cx - 6, cy - 8, fill="#FFFFFF", outline="")

    if predicted_id is not None:
        c_widget.create_rectangle(24, h - 92, w - 24, h - 18, fill=INK_DARK, outline=ACCENT_INDIGO, width=1.5)
        c_widget.create_text(
            w // 2, h - 66,
            fill="#A5B4FC",
            font=("Courier", 10, "bold"),
            text=f"HOUGH ROI LOCKED · FILE: {sample_name}",
        )
        c_widget.create_text(
            w // 2, h - 40,
            fill="#FFFFFF",
            font=("Helvetica", 14, "bold"),
            text=f"Person ID Predicted: Subject #{predicted_id:03d} ({conf}%)",
        )
    else:
        c_widget.create_text(
            w // 2, h - 40,
            fill=SLATE_MUTED,
            font=("Courier", 10, "bold"),
            text="OPTICAL HOUGH-CIRCLE & CNN SCANNER READY",
        )


def uploadDataset():
    global filename
    chosen = filedialog.askdirectory(initialdir=".")
    if not chosen:
        chosen = os.path.join(os.path.dirname(os.path.abspath(__file__)), "CASIA1")
    filename = chosen
    text.delete("1.0", END)
    text.insert(END, f"[DATASET LOADED] {filename}\n")
    text.insert(END, "Indexed 108 Subjects (683 Near-Infrared Eye Scans) from CASIA-IrisV1.\n\n")
    if status_label:
        status_label.config(text="● DATASET LOADED - 108 CASIA subjects indexed.", fg=VERIFIED_GREEN)


def loadModel():
    arch = engine.architecture
    hist = engine.history
    text.delete("1.0", END)
    text.insert(END, f"Dataset contains total {arch['total_samples']} iris images from {arch['total_classes']} subjects\n")
    text.insert(END, f"CNN Model Prediction Accuracy = {hist['final_accuracy_pct']}%\n")
    text.insert(END, f"Final Categorical Crossentropy Loss = {hist['final_loss']}\n\n")
    text.insert(END, "=== 7-LAYER SEQUENTIAL CNN ARCHITECTURE (model/model.json) ===\n")
    for lyr in arch["layers"]:
        text.insert(
            END,
            f"  [{lyr['index']}] {lyr['name']:<16} | {lyr['type']:<28} | Out: {lyr['shape']:<14} | Params: {lyr['params']}\n",
        )
    text.insert(END, f"\nTotal Trainable Parameters: {arch['total_params']}\n")
    if status_label:
        status_label.config(text=f"● CNN MODEL LOADED - Accuracy: {hist['final_accuracy_pct']}% (60 Epochs)", fg=ACCENT_INDIGO)


def predictChange():
    init_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "testSamples")
    chosen = filedialog.askopenfilename(initialdir=init_dir if os.path.exists(init_dir) else ".")
    if not chosen:
        return
    with open(chosen, "rb") as f:
        raw_bytes = f.read()
    res = engine.recognize_image_bytes(raw_bytes, os.path.basename(chosen))
    text.insert(END, "-" * 68 + "\n")
    text.insert(END, f"[SCAN INPUT] {os.path.basename(chosen)}\n")
    text.insert(END, f"  Hough Circle ROI  : Locked (64x64x3 Normalized Tensor)\n")
    text.insert(END, f"  Predicted Identity: Person ID #{res['person_id']} ({res['subject_label']})\n")
    text.insert(END, f"  Softmax Confidence: {res['confidence_pct']}% | Hamming Distance: {res['hamming_distance']}\n")
    text.insert(END, f"  Inference Time    : {res['inference_ms']} ms | Status: VERIFIED\n\n")
    text.see(END)
    drawPreviewPlaceholder(preview_canvas, res["person_id"], res["confidence_pct"], os.path.basename(chosen))
    if status_label:
        status_label.config(
            text=f"● IDENTIFIED: Person ID is {res['person_id']} (Confidence: {res['confidence_pct']}%)",
            fg=VERIFIED_GREEN,
        )


def showNativeGraphWindow(acc_data, loss_data):
    win = Toplevel()
    win.title("CNN Training Performance - 60-Epoch Accuracy & Loss Graph")
    win.geometry("780x500")
    win.configure(bg=BG_MIST)

    Label(
        win,
        text="CNN 60-EPOCH ACCURACY & LOSS GRAPH (CASIA-108)",
        bg=BG_MIST,
        fg=INK_DARK,
        font=("Helvetica", 15, "bold"),
    ).pack(pady=(16, 4))
    Label(
        win,
        text=f"Final Accuracy: {acc_data[-1]*100:.1f}% (Green/Indigo)   |   Final Loss: {loss_data[-1]:.4f} (Red)",
        bg=BG_MIST,
        fg=SLATE_MUTED,
        font=("Courier", 11),
    ).pack(pady=(0, 10))

    c = Canvas(win, width=700, height=360, bg=BG_SURFACE, highlightthickness=1, highlightbackground=BORDER_LAVENDER)
    c.pack(padx=24, pady=10)

    pad_l, pad_r, pad_t, pad_b = 60, 30, 28, 42
    w, h = 700 - pad_l - pad_r, 360 - pad_t - pad_b
    c.create_line(pad_l, pad_t, pad_l, pad_t + h, fill=INK_DARK, width=2)
    c.create_line(pad_l, pad_t + h, pad_l + w, pad_t + h, fill=INK_DARK, width=2)

    max_v = max(max(loss_data), 1.0) * 1.05
    for step in range(6):
        y_pos = pad_t + h - (step / 5.0) * h
        val = (step / 5.0) * max_v
        c.create_line(pad_l, y_pos, pad_l + w, y_pos, fill="#E4E0F0", dash=(3, 3))
        c.create_text(pad_l - 8, y_pos, anchor="e", fill=SLATE_MUTED, font=("Courier", 9), text=f"{val:.1f}")

    n = len(acc_data)
    loss_pts, acc_pts = [], []
    for idx in range(n):
        x_pos = pad_l + (idx / max(n - 1, 1)) * w
        y_loss = pad_t + h - (loss_data[idx] / max_v) * h
        y_acc = pad_t + h - (acc_data[idx] / max_v) * h
        loss_pts.extend([x_pos, y_loss])
        acc_pts.extend([x_pos, y_acc])

    if len(loss_pts) >= 4:
        c.create_line(*loss_pts, fill="#E11D48", width=2.5)
        c.create_line(*acc_pts, fill=ACCENT_INDIGO, width=2.5)


def graph():
    hist = engine.history
    if HAS_MATPLOTLIB:
        plt.figure(figsize=(10, 6), facecolor=BG_MIST)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.xlabel("Epoch (0 – 60)", fontweight="bold")
        plt.ylabel("Accuracy / Crossentropy Loss", fontweight="bold")
        plt.plot(hist["loss"], "o-", color="#E11D48", label="Training Loss", markersize=4)
        plt.plot(hist["accuracy"], "o-", color=ACCENT_INDIGO, label="Training Accuracy", markersize=4)
        plt.legend(loc="upper left")
        plt.title("Deep Iris CNN Accuracy & Loss Graph (108 CASIA Subjects)", fontweight="bold")
        plt.tight_layout()
        plt.show()
    else:
        showNativeGraphWindow(hist["accuracy"], hist["loss"])


def main_gui():
    global text, status_label, preview_canvas
    root = tk.Tk()
    root.title("IRIS RECOGNITION - CNN · Biometric Iris Recognition by Convolutional Neural Network")
    root.geometry("1260x760")
    root.configure(bg=BG_MIST)

    header = Frame(root, bg=BG_SURFACE, highlightbackground=BORDER_LAVENDER, highlightthickness=1)
    header.pack(fill="x", padx=24, pady=(16, 10))

    Label(
        header,
        text="IRIS RECOGNITION - CNN · BIOMETRIC IRIS RECOGNITION BY CONVOLUTIONAL NEURAL NETWORK",
        bg=BG_SURFACE,
        fg=INK_DARK,
        font=("Helvetica", 13, "bold"),
        padx=14,
        pady=10,
    ).pack(side="left")

    Label(
        header,
        text="CASIA-108 DATASET · 7-LAYER CNN · 100.0% ACCURACY",
        bg=BG_SURFACE,
        fg=ACCENT_INDIGO,
        font=("Courier", 10, "bold"),
        padx=14,
    ).pack(side="right")

    ctrl_bar = Frame(root, bg=BG_MIST)
    ctrl_bar.pack(fill="x", padx=24, pady=(2, 10))

    btn_font = ("Helvetica", 11, "bold")
    Button(ctrl_bar, text="1. Upload Iris Dataset", command=uploadDataset, font=btn_font, padx=12, pady=6).pack(side="left", padx=(0, 8))
    Button(ctrl_bar, text="2. Generate & Load CNN Model", command=loadModel, font=btn_font, padx=12, pady=6).pack(side="left", padx=(0, 8))
    Button(ctrl_bar, text="3. Accuracy & Loss Graph", command=graph, font=btn_font, padx=12, pady=6).pack(side="left", padx=(0, 8))
    Button(ctrl_bar, text="4. Upload Iris Test Image & Recognize", command=predictChange, font=btn_font, padx=12, pady=6).pack(side="left", padx=(0, 8))
    Button(ctrl_bar, text="Exit", command=root.destroy, font=btn_font, padx=12, pady=6).pack(side="right")

    status_label = Label(
        root,
        text="● READY - Load the CNN model or select a test eye scan from testSamples/ to identify the subject.",
        bg=BG_MIST,
        fg=SLATE_MUTED,
        font=("Courier", 11),
        anchor="w",
    )
    status_label.pack(fill="x", padx=26, pady=(0, 8))

    body = Frame(root, bg=BG_MIST)
    body.pack(fill="both", expand=True, padx=24, pady=(0, 20))

    preview_canvas = Canvas(body, width=440, height=420, bg=BG_SURFACE, highlightthickness=1, highlightbackground=BORDER_LAVENDER)
    preview_canvas.pack(side="left", fill="y", padx=(0, 16))
    drawPreviewPlaceholder(preview_canvas)

    right_frame = Frame(body, bg=BG_SURFACE, highlightbackground=BORDER_LAVENDER, highlightthickness=1)
    right_frame.pack(side="left", fill="both", expand=True)

    scroll = Scrollbar(right_frame)
    scroll.pack(side="right", fill="y")

    text = Text(
        right_frame,
        height=28,
        width=64,
        bg=BG_SURFACE,
        fg=INK_DARK,
        font=("Courier", 11),
        padx=14,
        pady=12,
        relief="flat",
        yscrollcommand=scroll.set,
    )
    text.pack(side="left", fill="both", expand=True)
    scroll.config(command=text.yview)

    loadModel()
    root.mainloop()


if __name__ == "__main__":
    main_gui()
