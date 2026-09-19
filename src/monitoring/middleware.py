"""ASGI Middleware for Prometheus metrics exposition, API latency tracing, and socket tracking."""

import time
from typing import Any, Callable, Dict

from src.monitoring.metrics import (
    PROMETHEUS_CONTENT_TYPE,
    generate_metrics_exposition,
    record_api_latency,
    record_socket_connection,
    record_socket_disconnection,
)


class PrometheusMiddleware:
    """ASGI middleware to handle /metrics scrapes, track HTTP latencies, and monitor WebSockets."""

    def __init__(self, app: Any, metrics_path: str = "/metrics") -> None:
        self.app = app
        self.metrics_path = metrics_path

    async def __call__(
        self, scope: Dict[str, Any], receive: Callable, send: Callable
    ) -> None:
        scope_type = scope.get("type")

        if scope_type == "http":
            path = scope.get("path", "")
            method = scope.get("method", "GET")

            if path == self.metrics_path:
                if method != "GET":
                    await send({
                        "type": "http.response.start",
                        "status": 405,
                        "headers": [(b"content-type", b"text/plain; charset=utf-8")],
                    })
                    await send({
                        "type": "http.response.body",
                        "body": b"Method Not Allowed",
                        "more_body": False,
                    })
                    return

                body = generate_metrics_exposition().encode("utf-8")
                headers = [
                    (b"content-type", PROMETHEUS_CONTENT_TYPE.encode("latin-1")),
                    (b"content-length", str(len(body)).encode("latin-1")),
                ]
                await send({
                    "type": "http.response.start",
                    "status": 200,
                    "headers": headers,
                })
                await send({
                    "type": "http.response.body",
                    "body": body,
                    "more_body": False,
                })
                return

            start_time = time.perf_counter()
            status_code = 500

            async def wrapped_send(message: Dict[str, Any]) -> None:
                nonlocal status_code
                if message.get("type") == "http.response.start":
                    status_code = message.get("status", 200)
                await send(message)

            try:
                await self.app(scope, receive, wrapped_send)
            except BaseException:
                duration = time.perf_counter() - start_time
                record_api_latency(
                    method=method,
                    endpoint=path,
                    status_code=500,
                    duration=duration,
                )
                raise
            else:
                duration = time.perf_counter() - start_time
                record_api_latency(
                    method=method,
                    endpoint=path,
                    status_code=status_code,
                    duration=duration,
                )
            return

        if scope_type == "websocket":
            connected = False

            async def wrapped_send(message: Dict[str, Any]) -> None:
                nonlocal connected
                if message.get("type") == "websocket.accept":
                    record_socket_connection()
                    connected = True
                await send(message)

            async def wrapped_receive() -> Dict[str, Any]:
                nonlocal connected
                message = await receive()
                if message.get("type") == "websocket.disconnect" and connected:
                    record_socket_disconnection()
                    connected = False
                return message

            try:
                await self.app(scope, wrapped_receive, wrapped_send)
            finally:
                if connected:
                    record_socket_disconnection()
                    connected = False
            return

        await self.app(scope, receive, send)