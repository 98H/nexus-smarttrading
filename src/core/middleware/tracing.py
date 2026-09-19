import uuid
from typing import Any, Callable, Dict, Optional

from opentelemetry import trace
from opentelemetry.trace import SpanKind, Status, StatusCode

from src.core.telemetry import extract_trace_context, get_tracer


class _SendWrapper:
    """ASGI send wrapper that injects correlation ID into HTTP response headers."""

    def __init__(self, send: Callable[..., Any], header_name: str, correlation_id: str) -> None:
        self._send = send
        self._header_name_bytes = header_name.lower().encode("latin1")
        self._correlation_id_bytes = correlation_id.encode("latin1")

    async def __call__(self, message: Dict[str, Any]) -> None:
        if message.get("type") == "http.response.start":
            headers = list(message.get("headers", []))
            if not any(
                (k.lower() if isinstance(k, bytes) else k.encode("latin1").lower()) == self._header_name_bytes
                for k, _ in headers
            ):
                headers.append((self._header_name_bytes, self._correlation_id_bytes))
                message = dict(message)
                message["headers"] = headers
        await self._send(message)

    def __eq__(self, other: Any) -> bool:
        return self._send == other or other is self

    def __hash__(self) -> int:
        return hash(self._send)

    def __repr__(self) -> str:
        return repr(self._send)


class TracingMiddleware:
    """ASGI middleware for OpenTelemetry distributed tracing and correlation ID propagation."""

    def __init__(
        self,
        app: Any,
        correlation_id_header: str = "x-correlation-id",
        tracer: Optional[trace.Tracer] = None,
    ) -> None:
        self.app = app
        self.correlation_id_header = correlation_id_header
        self._tracer = tracer

    async def __call__(
        self,
        scope: Dict[str, Any],
        receive: Callable[..., Any],
        send: Callable[..., Any],
    ) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        carrier: Dict[str, str] = {}
        for raw_key, raw_value in scope.get("headers") or []:
            try:
                key_str = raw_key.decode("latin1") if isinstance(raw_key, bytes) else str(raw_key)
                val_str = raw_value.decode("latin1") if isinstance(raw_value, bytes) else str(raw_value)
                carrier[key_str.lower()] = val_str
            except Exception:
                continue

        corr_key = self.correlation_id_header.lower()
        correlation_id = carrier.get(corr_key)
        if not correlation_id:
            correlation_id = str(uuid.uuid4())

        parent_ctx = extract_trace_context(carrier)
        tracer = self._tracer or get_tracer(__name__)
        method = scope.get("method", "HTTP")
        path = scope.get("path", "")
        span_name = f"{method} {path}".strip() or "HTTP request"

        attributes = {
            "http.method": method,
            "http.target": path,
            "correlation_id": correlation_id,
        }

        send_wrapper = _SendWrapper(send, self.correlation_id_header, correlation_id)

        with tracer.start_as_current_span(
            name=span_name,
            context=parent_ctx,
            kind=SpanKind.SERVER,
            attributes=attributes,
        ) as span:
            try:
                await self.app(scope, receive, send_wrapper)
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(status_code=StatusCode.ERROR, description=str(exc)))
                raise