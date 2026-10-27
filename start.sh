#!/bin/bash
# ==============================================================================
# IRIS RECOGNITION - CNN · Biometric Iris Recognition by Convolutional Neural Network
# One-Command macOS / Linux Launcher (Port 5055)
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

echo "========================================================================"
echo "  IRIS RECOGNITION - CNN · Biometric Iris Recognition by Convolutional Neural Network"
echo "========================================================================"

if [ "$1" == "--desktop" ]; then
    echo "Launching Standalone Desktop Tkinter GUI (Main.py)..."
    python3 Main.py
else
    echo "Starting Iris Biometric Web Console on http://127.0.0.1:5055/index ..."
    (sleep 1.2 && open "http://127.0.0.1:5055/index" 2>/dev/null) &
    python3 WebApp.py
fi
