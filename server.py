"""An OpenAI-compatible HTTP server for obi-lookout, using only the standard library plus the helper next to it.

    python server.py --port 8000            (or: docker compose -f docker/compose.yaml up)

GET /healthz, GET /v1/models, POST /v1/chat/completions, POST /detect.
It is a span detector, not a chat model: the "assistant" reply is a JSON document listing what was found. Set OBI_API_KEY to require
`Authorization: Bearer <key>`. The text of a request is never logged. Same API as the server in Obiguard's own deployments.
"""
from __future__ import annotations

import argparse
import hmac
import json
import os
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from obi_lookout import ObiLookout, redact

MODEL_NAME = "obi-lookout"
MAX_TEXT_CHARS = 20_000  # about 30 seconds of CPU work (roughly 1.6 s per 1,400 characters)
MAX_BODY_BYTES = 1_000_000
STYLES = ("label", "mask")


class ApiError(Exception):
    def __init__(self, status: int, message: str, kind: str = "invalid_request_error"):
        super().__init__(message)
        self.status, self.message, self.kind = status, message, kind


def error_body(error: ApiError) -> dict:
    return {"error": {"message": error.message, "type": error.kind, "param": None, "code": None}}


def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
        if parts:
            return "\n".join(parts)
    raise ApiError(400, "message content must be a string or a list with text parts")


def _checked(text: str, limit: int) -> str:
    if not text.strip():
        raise ApiError(400, "there is no text to scan")
    if len(text) > limit:
        raise ApiError(413, f"text is {len(text)} characters, the limit is {limit}")
    return text


def _style(value):
    if value is not None and value not in STYLES:
        raise ApiError(400, f"redact must be one of {STYLES}")
    return value


def _result(lookout: ObiLookout, text: str, style) -> dict:
    spans = lookout.detect(text)
    result = {"entities": [{"start": s.start, "end": s.end, "label": s.label, "text": s.text, "score": round(s.score, 4)} for s in spans]}
    if style:
        result["redacted"] = redact(text, spans, style)
    return result


def detect(body, lookout: ObiLookout, limit: int) -> dict:
    if not isinstance(body, dict) or not isinstance(body.get("text"), str):
        raise ApiError(400, "body must be a JSON object with a string field 'text'")
    return _result(lookout, _checked(body["text"], limit), _style(body.get("redact")))


def chat_completion(body, lookout: ObiLookout, limit: int) -> dict:
    if not isinstance(body, dict):
        raise ApiError(400, "body must be a JSON object")
    if body.get("stream"):
        raise ApiError(400, "streaming is not supported: the reply is a single JSON document")
    if body.get("model") not in (None, MODEL_NAME):
        raise ApiError(404, f"the model '{body.get('model')}' does not exist, this server serves '{MODEL_NAME}'")
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ApiError(400, "'messages' must be a non-empty list")
    users = [m for m in messages if isinstance(m, dict) and m.get("role") == "user"]
    if not users:
        raise ApiError(400, "there is no user message to scan")  # the last user message is the one scanned
    text = _checked(_text_of(users[-1].get("content")), limit)
    content = json.dumps(_result(lookout, text, _style(body.get("obi_redact"))), ensure_ascii=False)
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}", "object": "chat.completion", "created": int(time.time()), "model": MODEL_NAME,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def make_server(lookout, host: str, port: int, api_key: str | None = None, limit: int = MAX_TEXT_CHARS) -> ThreadingHTTPServer:
    lock = threading.Lock()  # one model, one request at a time: the CPU is the limit anyway

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # never log request bodies
            pass

        def _send(self, status: int, body: dict) -> None:
            data = json.dumps(body, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _guard(self, work) -> None:
            try:
                header = self.headers.get("Authorization", "")
                if api_key and not (header.startswith("Bearer ") and hmac.compare_digest(header[7:], api_key)):
                    raise ApiError(401, "missing or wrong API key", "authentication_error")
                self._send(200, work())
            except ApiError as error:
                self._send(error.status, error_body(error))
            except Exception:  # a model failure must not leak details or stop the server
                self._send(500, error_body(ApiError(500, "the detector failed on this request", "server_error")))

        def _body(self):
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                raise ApiError(400, "bad Content-Length")
            if length > MAX_BODY_BYTES:
                raise ApiError(413, f"request body is over {MAX_BODY_BYTES} bytes")
            try:
                return json.loads(self.rfile.read(length) or b"null")
            except (json.JSONDecodeError, UnicodeDecodeError):
                raise ApiError(400, "body is not valid JSON")

        def do_GET(self):
            if self.path == "/healthz":
                self._send(200, {"status": "ok"})
            elif self.path == "/v1/models":
                self._guard(lambda: {"object": "list", "data": [{"id": MODEL_NAME, "object": "model", "created": 0, "owned_by": "obiguard"}]})
            else:
                self._send(404, error_body(ApiError(404, "not found")))

        def do_POST(self):
            routes = {"/v1/chat/completions": chat_completion, "/detect": detect}
            fn = routes.get(self.path)
            if fn is None:
                self._send(404, error_body(ApiError(404, "not found")))
                return

            def run():
                body = self._body()
                with lock:
                    return fn(body, lookout, limit)

            self._guard(run)

    return ThreadingHTTPServer((host, port), Handler)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-chars", type=int, default=MAX_TEXT_CHARS, help="longest text accepted, in characters")
    args = parser.parse_args()
    key = os.environ.get("OBI_API_KEY")
    server = make_server(ObiLookout(threshold=args.threshold), args.host, args.port, key, args.max_chars)
    print(f"{MODEL_NAME} listening on {args.host}:{args.port}, API key {'required' if key else 'NOT required'}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
