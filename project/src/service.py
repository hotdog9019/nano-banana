from __future__ import annotations

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .config import DEFAULT_CONFIG_PATH, load_config
from .predict import load_model, predict_ticket


def json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class TicketHandler(BaseHTTPRequestHandler):
    model: Any = None
    model_path: str = ""

    def do_GET(self) -> None:
        if self.path == "/health":
            json_response(
                self,
                200,
                {"status": "ok", "model_loaded": self.model is not None, "model_path": self.model_path},
            )
            return
        json_response(self, 404, {"error": "Not found"})

    def do_POST(self) -> None:
        if self.path != "/predict":
            json_response(self, 404, {"error": "Not found"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            text = payload.get("text")
            if not isinstance(text, str):
                raise ValueError("JSON body must contain string field 'text'.")
            json_response(self, 200, predict_ticket(self.model, text))
        except Exception as exc:
            json_response(self, 400, {"error": str(exc)})

    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.address_string()} - {format % args}")


def run_service(config_path: str | Path = DEFAULT_CONFIG_PATH) -> None:
    config = load_config(config_path)
    service_cfg = config.get("service", {})
    artifacts_cfg = config.get("artifacts", {})
    host = os.getenv("SERVICE_HOST", str(service_cfg.get("host", "127.0.0.1")))
    port = int(os.getenv("SERVICE_PORT", service_cfg.get("port", 8000)))
    model_path = os.getenv("MODEL_PATH", str(artifacts_cfg.get("model_path", "artifacts/ticket_classifier.joblib")))

    TicketHandler.model = load_model(model_path)
    TicketHandler.model_path = model_path
    server = ThreadingHTTPServer((host, port), TicketHandler)
    print(f"Serving support ticket classifier on http://{host}:{port}")
    server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run support ticket classifier API.")
    parser.add_argument("--config", default=os.getenv("PROJECT_CONFIG", str(DEFAULT_CONFIG_PATH)))
    args = parser.parse_args()
    run_service(args.config)


if __name__ == "__main__":
    main()

