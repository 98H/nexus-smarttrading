"""
Unit tests for OpenTelemetry Distributed Tracing and Tracing Middleware.

Tests cover:
- Story 11.4.2: Implement OpenTelemetry Distributed Tracing across Microservices
- Target modules: src/core/telemetry.py, src/core/middleware/tracing.py
"""

import uuid
from typing import Any, Callable, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from opentelemetry import context, trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind, StatusCode
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

# Target modules under test
from src.core.middleware.tracing import TracingMiddleware
from src.core.telemetry import (
    extract_trace_context,
    get_tracer,
    inject_trace_context,
    setup_telemetry,
)


@pytest.fixture
def memory_exporter() -> InMemorySpanExporter:
    """Fixture providing an in-memory span exporter for deterministic assertions."""
    return InMemorySpanExporter()


@pytest.fixture
def tracer_provider(memory_exporter: InMemorySpanExporter) -> TracerProvider:
    """Fixture providing an isolated TracerProvider with an in-memory exporter."""
    provider = TracerProvider()
    processor = SimpleSpanProcessor(memory_exporter)
    provider.add_span_processor(processor)

    # Save original provider to restore during teardown
    original_provider = trace.get_tracer_provider()
    trace.set_tracer_provider(provider)

    yield provider

    # Teardown
    provider.shutdown()
    trace.set_tracer_provider(original_provider)


@pytest.fixture
def sample_traceparent() -> str:
    """Valid W3C traceparent header value: version-trace_id-parent_id-trace_flags."""
    return "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"


@pytest.fixture
def sample_trace_id_hex() -> str:
    return "4bf92f3577b34da6a3ce929d0e0e4736"


@pytest.fixture
def sample_parent_span_id_hex() -> str:
    return "00f067aa0ba902b7"


# ============================================================================
# Core Telemetry Module Unit Tests (src/core/telemetry.py)
# ============================================================================


def test_setup_telemetry_initializes_tracer_provider_and_propagators() -> None:
    """Test that setup_telemetry configures the provider, resource, and W3C propagator."""
    with patch("opentelemetry.propagate.set_global_textmap") as mock_set_propagator, patch(
        "opentelemetry.trace.set_tracer_provider"
    ) as mock_set_provider:

        provider = setup_telemetry(service_name="order-service", environment="testing")

        assert isinstance(provider, TracerProvider)
        mock_set_provider.assert_called_once_with(provider)
        mock_set_propagator.assert_called_once()
        propagator_arg = mock_set_propagator.call_args[0][0]
        assert isinstance(propagator_arg, TraceContextTextMapPropagator)


def test_get_tracer_returns_configured_tracer(tracer_provider: TracerProvider) -> None:
    """Test that get_tracer retrieves a Tracer from the configured provider."""
    tracer = get_tracer("test_component")
    assert tracer is not None
    assert hasattr(tracer, "start_as_current_span")


def test_inject_trace_context_populates_carrier(
    tracer_provider: TracerProvider,
) -> None:
    """Test that inject_trace_context writes W3C trace headers to a dictionary carrier."""
    tracer = tracer_provider.get_tracer("test")
    carrier: Dict[str, str] = {}

    with tracer.start_as_current_span("parent_operation"):
        inject_trace_context(carrier)

    assert "traceparent" in carrier
    traceparent_value = carrier["traceparent"]
    assert traceparent_value.startswith("00-")
    parts = traceparent_value.split("-")
    assert len(parts) == 4
    assert len(parts[1]) == 32  # 128-bit trace id hex
    assert len(parts[2]) == 16  # 64-bit span id hex


def test_inject_trace_context_with_no_active_span() -> None:
    """Test that inject_trace_context does not fail or inject empty context when no span is active."""
    carrier: Dict[str, str] = {}
    token = context.attach(context.Context())
    try:
        inject_trace_context(carrier)
        assert "traceparent" not in carrier
    finally:
        context.detach(token)


def test_extract_trace_context_extracts_remote_parent(
    sample_traceparent: str, sample_trace_id_hex: str
) -> None:
    """Test that extract_trace_context reconstructs valid context from carrier."""
    carrier = {"traceparent": sample_traceparent}
    extracted_ctx = extract_trace_context(carrier)

    assert extracted_ctx is not None
    span = trace.get_current_span(extracted_ctx)
    span_ctx = span.get_span_context()

    assert span_ctx.is_valid
    assert format(span_ctx.trace_id, "032x") == sample_trace_id_hex


def test_setup_telemetry_raises_type_error_on_invalid_service_name() -> None:
    """Test that setup_telemetry asserts input types strictly."""
    with pytest.raises(TypeError):
        setup_telemetry(service_name=None)  # type: ignore[arg-type]


# ============================================================================
# Tracing Middleware Unit Tests (src/core/middleware/tracing.py)
# ============================================================================


@pytest.mark.asyncio
async def test_middleware_creates_server_span_with_extracted_trace_and_correlation_id(
    tracer_provider: TracerProvider,
    memory_exporter: InMemorySpanExporter,
    sample_traceparent: str,
    sample_trace_id_hex: str,
    sample_parent_span_id_hex: str,
) -> None:
    """
    Given an incoming HTTP request containing W3C traceparent and correlation ID,
    when passed through TracingMiddleware,
    then an OpenTelemetry SERVER span is created with matching trace_id, parent_id,
    and the correlation ID injected as a span attribute.
    """
    correlation_id = "corr-" + str(uuid.uuid4())
    headers = [
        (b"traceparent", sample_traceparent.encode("utf-8")),
        (b"x-correlation-id", correlation_id.encode("utf-8")),
    ]

    scope: Dict[str, Any] = {
        "type": "http",
        "method": "POST",
        "path": "/api/v1/orders",
        "headers": headers,
    }

    mock_app = AsyncMock()
    middleware = TracingMiddleware(app=mock_app)

    receive = AsyncMock()
    send = AsyncMock()

    await middleware(scope, receive, send)

    mock_app.assert_awaited_once_with(scope, receive, send)

    spans = memory_exporter.get_finished_spans()
    assert len(spans) == 1

    server_span = spans[0]
    assert server_span.kind == SpanKind.SERVER
    assert format(server_span.context.trace_id, "032x") == sample_trace_id_hex
    assert server_span.parent is not None
    assert format(server_span.parent.span_id, "016x") == sample_parent_span_id_hex

    # Correlation ID must be an attribute on the span
    assert "correlation_id" in server_span.attributes
    assert server_span.attributes["correlation_id"] == correlation_id
    assert server_span.attributes["http.method"] == "POST"
    assert server_span.attributes["http.target"] == "/api/v1/orders"


@pytest.mark.asyncio
async def test_middleware_propagates_trace_context_across_downstream_client_calls(
    tracer_provider: TracerProvider,
    memory_exporter: InMemorySpanExporter,
    sample_traceparent: str,
    sample_trace_id_hex: str,
) -> None:
    """
    Given an incoming request with a trace context,
    when downstream calls occur within the request scope,
    then the trace context is propagated across downstream client calls.
    """
    downstream_carrier: Dict[str, str] = {}

    async def downstream_app(scope: Any, receive: Any, send: Any) -> None:
        # Simulate downstream client call making an outbound HTTP request
        inject_trace_context(downstream_carrier)

    middleware = TracingMiddleware(app=downstream_app)

    scope: Dict[str, Any] = {
        "type": "http",
        "method": "GET",
        "path": "/api/v1/portfolio",
        "headers": [
            (b"traceparent", sample_traceparent.encode("utf-8")),
            (b"x-correlation-id", b"client-corr-001"),
        ],
    }

    await middleware(scope, AsyncMock(), AsyncMock())

    assert "traceparent" in downstream_carrier
    downstream_traceparent = downstream_carrier["traceparent"]
    parts = downstream_traceparent.split("-")
    # Downstream trace ID must be identical to incoming trace ID
    assert parts[1] == sample_trace_id_hex

    # Downstream parent should match the middleware's newly created server span ID
    server_spans = memory_exporter.get_finished_spans()
    assert len(server_spans) == 1
    assert format(server_spans[0].context.span_id, "016x") == parts[2]


@pytest.mark.asyncio
async def test_middleware_generates_correlation_id_if_missing_in_request(
    tracer_provider: TracerProvider,
    memory_exporter: InMemorySpanExporter,
) -> None:
    """Test that middleware generates a correlation ID when none is provided in headers."""
    scope: Dict[str, Any] = {
        "type": "http",
        "method": "GET",
        "path": "/api/v1/health",
        "headers": [],
    }

    captured_headers: List[Any] = []

    async def dummy_send(message: Dict[str, Any]) -> None:
        if message["type"] == "http.response.start":
            captured_headers.extend(message.get("headers", []))

    async def dummy_app(scope: Any, receive: Any, send: Any) -> None:
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"OK"})

    middleware = TracingMiddleware(app=dummy_app)
    await middleware(scope, AsyncMock(), dummy_send)

    spans = memory_exporter.get_finished_spans()
    assert len(spans) == 1
    server_span = spans[0]

    # Verify a generated correlation ID was attached to span attributes
    assert "correlation_id" in server_span.attributes
    generated_corr_id = server_span.attributes["correlation_id"]
    assert isinstance(generated_corr_id, str)
    assert len(generated_corr_id) > 0

    # Verify correlation ID is injected into HTTP response headers
    header_map = {k.decode("utf-8").lower(): v.decode("utf-8") for k, v in captured_headers}
    assert "x-correlation-id" in header_map
    assert header_map["x-correlation-id"] == generated_corr_id


@pytest.mark.asyncio
async def test_middleware_handles_malformed_traceparent_gracefully(
    tracer_provider: TracerProvider,
    memory_exporter: InMemorySpanExporter,
) -> None:
    """Test that a malformed traceparent does not cause crash; creates a new trace."""
    scope: Dict[str, Any] = {
        "type": "http",
        "method": "GET",
        "path": "/api/v1/items",
        "headers": [
            (b"traceparent", b"invalid-traceparent-format"),
            (b"x-correlation-id", b"valid-corr-id"),
        ],
    }

    mock_app = AsyncMock()
    middleware = TracingMiddleware(app=mock_app)

    await middleware(scope, AsyncMock(), AsyncMock())

    spans = memory_exporter.get_finished_spans()
    assert len(spans) == 1
    server_span = spans[0]

    # New root span should be created (parent is None)
    assert server_span.parent is None
    assert server_span.context.is_valid
    assert server_span.attributes["correlation_id"] == "valid-corr-id"


@pytest.mark.asyncio
async def test_middleware_records_exception_and_sets_error_status(
    tracer_provider: TracerProvider,
    memory_exporter: InMemorySpanExporter,
) -> None:
    """Test that downstream exceptions are recorded on the span and status is set to ERROR."""
    expected_error = RuntimeError("Database connection timeout")

    async def failing_app(scope: Any, receive: Any, send: Any) -> None:
        raise expected_error

    middleware = TracingMiddleware(app=failing_app)
    scope: Dict[str, Any] = {
        "type": "http",
        "method": "DELETE",
        "path": "/api/v1/orders/123",
        "headers": [(b"x-correlation-id", b"err-test-id")],
    }

    with pytest.raises(RuntimeError):
        await middleware(scope, AsyncMock(), AsyncMock())

    spans = memory_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert span.status.status_code == StatusCode.ERROR
    assert len(span.events) > 0
    exception_event = next(e for e in span.events if e.name == "exception")
    assert exception_event.attributes["exception.type"] == "RuntimeError"


@pytest.mark.asyncio
async def test_middleware_supports_custom_correlation_id_header(
    tracer_provider: TracerProvider,
    memory_exporter: InMemorySpanExporter,
) -> None:
    """Test that custom correlation ID header names (e.g. X-Request-ID) can be configured."""
    custom_header = "x-request-id"
    custom_id = "custom-req-456"

    scope: Dict[str, Any] = {
        "type": "http",
        "method": "GET",
        "path": "/api/v1/market-data",
        "headers": [(custom_header.encode("utf-8"), custom_id.encode("utf-8"))],
    }

    mock_app = AsyncMock()
    middleware = TracingMiddleware(app=mock_app, correlation_id_header=custom_header)

    await middleware(scope, AsyncMock(), AsyncMock())

    spans = memory_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes["correlation_id"] == custom_id


@pytest.mark.asyncio
async def test_middleware_bypasses_non_http_scopes(
    tracer_provider: TracerProvider,
    memory_exporter: InMemorySpanExporter,
) -> None:
    """Test that non-HTTP scopes (e.g. lifespan, websocket) pass through without creating server spans."""
    scope: Dict[str, Any] = {"type": "lifespan"}
    mock_app = AsyncMock()
    middleware = TracingMiddleware(app=mock_app)

    receive = AsyncMock()
    send = AsyncMock()

    await middleware(scope, receive, send)

    mock_app.assert_awaited_once_with(scope, receive, send)
    spans = memory_exporter.get_finished_spans()
    assert len(spans) == 0