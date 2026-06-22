#!/usr/bin/env python3
"""
Local dev server for pointer — adds the COOP/COEP headers that ffmpeg.wasm
needs to spawn a Web Worker. Plain `python3 -m http.server` won't work.

Usage:
    python3 serve.py
Then open http://localhost:8080 in your browser.
"""

import http.server
import socketserver
import os

PORT = 8080

class CORPHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        super().end_headers()

    def log_message(self, format, *args):
        # suppress the per-request spam, just show errors
        if args[1] not in ("200", "304"):
            super().log_message(format, *args)

os.chdir(os.path.dirname(os.path.abspath(__file__)))

with socketserver.TCPServer(("", PORT), CORPHandler) as httpd:
    print(f"Serving at http://localhost:{PORT}")
    print("(press Ctrl+C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
