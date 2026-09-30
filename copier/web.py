from __future__ import annotations

import json
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

from copier.runner import CopierApp

REPO_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = REPO_ROOT / "docs" / "copier"


def demo_status() -> dict[str, Any]:
    return {
        "ok": True,
        "demo": True,
        "connected": False,
        "ready": False,
        "copy_mode": "fills",
        "dry_run": True,
        "environment": "demo",
        "flatten_when_lead_flat": True,
        "lead": {"name": "主帳", "positions": {}},
        "followers": [],
        "events": [
            {
                "ts": "",
                "kind": "log",
                "message": "現在是網頁示範，沒有連 Tradovate。按下面的按鈕看複製怎麼走。",
            }
        ],
    }


class CopierHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args: Any, app: Optional[CopierApp] = None, **kwargs: Any) -> None:
        self.app = app
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        if "/api/" in (args[0] if args else ""):
            return
        super().log_message(format, *args)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path in {"/api/status", "/api/status.json"}:
            payload = self.app.status() if self.app else demo_status()
            payload["ok"] = True
            payload["demo"] = self.app is None
            self._json(payload)
            return
        if parsed.path in {"", "/"}:
            self.path = "/index.html"
        super().do_GET()

    def _json(self, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def serve(host: str, port: int, app: Optional[CopierApp] = None) -> ThreadingHTTPServer:
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    handler = partial(CopierHandler, app=app)
    server = ThreadingHTTPServer((host, port), handler)
    return server
