import os
import json
import base64
import mimetypes
from urllib.parse import urlparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from iris_cnn_engine import engine

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class IrisWebHandler(BaseHTTPRequestHandler):
    """Zero-dependency HTTP server on port 5055 serving the Biometric Iris CNN Platform."""

    def log_message(self, format, *args):
        print(f"[Iris-WebServer] {self.address_string()} - {format % args}")

    def _send_bytes(self, data: bytes, content_type: str, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, obj: dict, status: int = 200):
        raw = json.dumps(obj).encode("utf-8")
        self._send_bytes(raw, "application/json; charset=utf-8", status)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index", "/index.html", "/Launch_Iris_Dashboard.html"):
            html_path = os.path.join(BASE_DIR, "index.html")
            with open(html_path, "rb") as f:
                self._send_bytes(f.read(), "text/html; charset=utf-8")
            return

        if path == "/api/summary":
            self._send_json(engine.get_dashboard_summary())
            return

        # Serve static or sample files if requested
        rel = path.lstrip("/")
        full_path = os.path.normpath(os.path.join(BASE_DIR, rel))
        if full_path.startswith(BASE_DIR) and os.path.isfile(full_path):
            mime, _ = mimetypes.guess_type(full_path)
            with open(full_path, "rb") as f:
                self._send_bytes(f.read(), mime or "application/octet-stream")
            return

        html_path = os.path.join(BASE_DIR, "index.html")
        with open(html_path, "rb") as f:
            self._send_bytes(f.read(), "text/html; charset=utf-8")

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(length).decode("utf-8") if length else "{}"

        if path == "/api/recognize":
            try:
                payload = json.loads(raw_body)
                b64_str = payload.get("image_base64", "")
                if "," in b64_str:
                    b64_str = b64_str.split(",", 1)[1]
                img_bytes = base64.b64decode(b64_str)
                fname = payload.get("filename", "uploaded.jpg")
                res = engine.recognize_image_bytes(img_bytes, fname)
                self._send_json({"status": "ok", "result": res})
            except Exception as e:
                self._send_json({"status": "error", "message": str(e)}, 400)
            return

        self._send_json({"status": "ok"})


def run_web_server(host: str = "127.0.0.1", port: int = 5055):
    print("=" * 74)
    print("  IRIS-CNN // BIOMETRIC IRIS RECOGNITION BY CONVOLUTIONAL NEURAL NETWORK")
    print("=" * 74)
    print(f"  [Dataset]      CASIA-IrisV1 (108 Enrolled Subjects | 683 NIR Eye Scans)")
    print(f"  [CNN Model]    7-Layer Sequential CNN (1,643,788 Trainable Parameters)")
    print(f"  [Accuracy]     {engine.history['final_accuracy_pct']}% (60 Epochs | Loss: {engine.history['final_loss']})")
    print(f"  [Web Console]  Running at http://{host}:{port}/index")
    print("=" * 74)

    server = ThreadingHTTPServer((host, port), IrisWebHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down Iris Biometric Web Server...")
        server.server_close()


if __name__ == "__main__":
    run_web_server()
