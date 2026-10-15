#!/bin/bash
# Double-click this file in macOS Finder to launch the Iris CNN Biometric Web Console & open your browser!
# Uses port 5055 so it never conflicts with macOS AirPlay (5000) or the Blockchain project (5050).
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

(sleep 1.2 && open "http://127.0.0.1:5055/index") &

python3 WebApp.py
