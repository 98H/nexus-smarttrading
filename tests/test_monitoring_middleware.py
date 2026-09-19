"""
Unit tests for Prometheus metrics collection and ASGI middleware.

Requirements:
- Story 11.4.1: Implement Prometheus Metrics and Grafana System Dashboards
- Acceptance Criteria:
    Given a running service with Prometheus instrumentation enabled, when
    the /metrics endpoint is scraped, then socket counts, frame render times,
    API latencies, and worker memory metrics are returned in Prometheus text
    exposition format.
"""

import asyncio
from typing import Any, Callable, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.monitoring.metrics import (
    PROMETHEUS_CONTENT_TYPE,
    MetricsManager,
    collect_current_worker_memory,
    generate_metrics_exposition,
    record_api_latency,
    record_frame_render_time,
    record_socket_connection,
    record_socket_disconnection,
    record_worker_memory,
)
from src.monitoring.middleware import PrometheusMiddleware


# ============================================================================
# Helpers and Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
def reset_metrics_state():
    """Ensure a clean metrics registry state before and after each test."""
    manager = MetricsManager.get_instance()
    manager.reset()
    yield
    manager.reset()


async def simulate_http_request(
    app: Callable,
    method: str = "GET",
    path: str = "/metrics",
    headers: List[tuple] | None = None,
) -> tuple[int, Dict[bytes, bytes], bytes]:
    """Simulate an ASGI HTTP request and capture status, headers, and body."""
    headers = headers or []
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.0"},
        "http_version": "1.1",
        "method": method,
        "path": path,
        "raw_path": path.encode("latin-1"),
        "query_string": b"",
        "headers": headers,
    }
    sent_messages: List[Dict[str, Any]] = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: Dict[str, Any]):
        sent_messages.append(message)

    await app(scope, receive, send)

    response_start = next(
        m for m in sent_messages if m["type"] == "http.response.start"
    )
    response_body = next(
        m for m in sent_messages if m["type"] == "http.response.body"
    )

    status_code = response_start["status"]
    response_headers = dict(response_start.get("headers", []))
    body = response_body.get("body", b"")

    return status_code, response_headers, body


# ============================================================================
# Unit Tests: src/monitoring/metrics.py
# ============================================================================

class TestMetricsManager:
    """Test the core Prometheus metric manager, collectors, and exposition."""

    def test_singleton_instance(self):
        """MetricsManager should enforce a singleton pattern for shared instrumentation."""
        instance1 = MetricsManager.get_instance()
        instance2 = MetricsManager.get_instance()
        assert instance1 is instance2

    def test_socket_connection_tracking(self):
        """Active socket gauge increments on connection and decrements on disconnection."""
        manager = MetricsManager.get_instance()
        assert manager.get_active_socket_count() == 0

        record_socket_connection()
        record_socket_connection()
        assert manager.get_active_socket_count() == 2

        record_socket_disconnection()
        assert manager.get_active_socket_count() == 1

        record_socket_disconnection()
        assert manager.get_active_socket_count() == 0

    def test_socket_disconnection_underflow_prevention(self):
        """Active socket count should not decrease below zero; should raise ValueError."""
        with pytest.raises(ValueError):
            record_socket_disconnection()

    def test_frame_render_time_metric_valid_observation(self):
        """Valid float frame render times are accepted and recorded in histogram."""
        manager = MetricsManager.get_instance()
        record_frame_render_time(0.016)
        record_frame_render_time(0.033)

        exposition = manager.generate_exposition()
        assert "frame_render_duration_seconds" in exposition
        assert "frame_render_duration_seconds_count 2" in exposition or (
            "frame_render_duration_seconds_count 2.0" in exposition
        )

    def test_frame_render_time_negative_raises_error(self):
        """Negative render time is physically invalid and must raise ValueError."""
        with pytest.raises(ValueError):
            record_frame_render_time(-0.005)

    def test_api_latency_metric_observation(self):
        """API latency histogram accurately tracks method, endpoint, status code, and duration."""
        manager = MetricsManager.get_instance()
        record_api_latency(
            method="GET",
            endpoint="/api/v1/orderbook",
            status_code=200,
            duration=0.045,
        )

        exposition = manager.generate_exposition()
        assert "api_request_duration_seconds" in exposition
        assert 'endpoint="/api/v1/orderbook"' in exposition
        assert 'method="GET"' in exposition
        assert 'status="200"' in exposition

    def test_api_latency_negative_duration_raises_error(self):
        """Negative API duration must raise ValueError."""
        with pytest.raises(ValueError):
            record_api_latency(
                method="POST",
                endpoint="/api/v1/order",
                status_code=201,
                duration=-0.1,
            )

    def test_worker_memory_recording(self):
        """Worker memory gauge tracks memory bytes per worker identifier."""
        manager = MetricsManager.get_instance()
        worker_id = "worker-feed-01"
        memory_bytes = 104857600  # 100 MB

        record_worker_memory(worker_id=worker_id, memory_bytes=memory_bytes)

        exposition = manager.generate_exposition()
        assert "worker_memory_bytes" in exposition
        assert f'worker_id="{worker_id}"' in exposition
        assert "104857600" in exposition

    def test_worker_memory_negative_bytes_raises_error(self):
        """Negative memory bytes value must raise ValueError."""
        with pytest.raises(ValueError):
            record_worker_memory(worker_id="worker-01", memory_bytes=-1024)

    @patch("src.monitoring.metrics.psutil", create=True)
    def test_collect_current_worker_memory_with_system_probe(self, mock_psutil):
        """Worker memory collector probes the OS process RSS and updates the gauge."""
        mock_process = MagicMock()
        mock_process.memory_info.return_value = MagicMock(rss=209715200)
        mock_psutil.Process.return_value = mock_process

        collect_current_worker_memory(worker_id="worker-proc-test")

        exposition = generate_metrics_exposition()
        assert "worker_memory_bytes" in exposition
        assert 'worker_id="worker-proc-test"' in exposition
        assert "209715200" in exposition

    def test_exposition_includes_all_required_metrics_in_openmetrics_format(self):
        """Acceptance Criteria: /metrics exposition returns socket counts, frame render times,
        API latencies, and worker memory metrics in Prometheus format."""
        # Seed all 4 metrics
        record_socket_connection()
        record_frame_render_time(0.012)
        record_api_latency(
            method="GET", endpoint="/api/v1/health", status_code=200, duration=0.002
        )
        record_worker_memory(worker_id="main-worker", memory_bytes=52428800)

        exposition = generate_metrics_exposition()

        # Check Prometheus exposition semantics: HELP, TYPE, and Metric definitions
        assert "# HELP socket_connections_active" in exposition
        assert "# TYPE socket_connections_active gauge" in exposition
        assert "socket_connections_active 1" in exposition or "socket_connections_active 1.0" in exposition

        assert "# HELP frame_render_duration_seconds" in exposition
        assert "# TYPE frame_render_duration_seconds histogram" in exposition

        assert "# HELP api_request_duration_seconds" in exposition
        assert "# TYPE api_request_duration_seconds histogram" in exposition

        assert "# HELP worker_memory_bytes" in exposition
        assert "# TYPE worker_memory_bytes gauge" in exposition
        assert 'worker_id="main-worker"' in exposition


# ============================================================================
# Unit Tests: src/monitoring/middleware.py
# ============================================================================

class TestPrometheusMiddleware:
    """Test ASGI Middleware metric interception, HTTP latency tracing, and socket lifecycle."""

    @pytest.mark.asyncio
    async def test_scrape_endpoint_returns_metrics_exposition(self):
        """Scraping /metrics returns 200, Prometheus exposition content type, and metrics body."""
        record_socket_connection()
        record_frame_render_time(0.015)
        record_api_latency("GET", "/health", 200, 0.003)
        record_worker_memory("worker-01", 67108864)

        downstream_mock = AsyncMock()
        middleware = PrometheusMiddleware(app=downstream_mock, metrics_path="/metrics")

        status_code, headers, body = await simulate_http_request(
            app=middleware, method="GET", path="/metrics"
        )

        assert status_code == 200
        # Check standard Prometheus Content-Type header
        assert b"content-type" in headers
        content_type = headers[b"content-type"].decode("latin-1")
        assert (
            PROMETHEUS_CONTENT_TYPE in content_type
            or "text/plain; version=0.0.4" in content_type
        )

        decoded_body = body.decode("utf-8")
        assert "socket_connections_active" in decoded_body
        assert "frame_render_duration_seconds" in decoded_body
        assert "api_request_duration_seconds" in decoded_body
        assert "worker_memory_bytes" in decoded_body

        # Downstream application should not be called when scraping /metrics
        downstream_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_scrape_endpoint_rejects_non_get_methods(self):
        """POST /metrics should return 405 Method Not Allowed and not leak state."""
        downstream_mock = AsyncMock()
        middleware = PrometheusMiddleware(app=downstream_mock, metrics_path="/metrics")

        status_code, _, _ = await simulate_http_request(
            app=middleware, method="POST", path="/metrics"
        )

        assert status_code == 405
        downstream_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_custom_metrics_endpoint_path(self):
        """Middleware can be initialized with a custom endpoint path."""
        downstream_mock = AsyncMock()
        middleware = PrometheusMiddleware(
            app=downstream_mock, metrics_path="/internal/prometheus"
        )

        status_code, _, body = await simulate_http_request(
            app=middleware, method="GET", path="/internal/prometheus"
        )
        assert status_code == 200
        assert "socket_connections_active" in body.decode("utf-8")
        downstream_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_tracks_api_latencies_on_downstream_requests(self):
        """HTTP requests to regular endpoints invoke downstream app and record latencies."""
        async def dummy_app(scope, receive, send):
            await send({
                "type": "http.response.start",
                "status": 201,
                "headers": [(b"content-type", b"application/json")],
            })
            await send({
                "type": "http.response.body",
                "body": b'{"status": "created"}',
                "more_body": False,
            })

        middleware = PrometheusMiddleware(app=dummy_app)

        status_code, _, _ = await simulate_http_request(
            app=middleware, method="POST", path="/api/v1/orders"
        )

        assert status_code == 201

        exposition = generate_metrics_exposition()
        assert 'endpoint="/api/v1/orders"' in exposition
        assert 'method="POST"' in exposition
        assert 'status="201"' in exposition

    @pytest.mark.asyncio
    async def test_downstream_exception_records_500_and_reraises(self):
        """Unhandled exceptions from downstream app record status 500 and propagate up."""
        async def faulty_app(scope, receive, send):
            raise RuntimeError("Database connection lost")

        middleware = PrometheusMiddleware(app=faulty_app)

        with pytest.raises(RuntimeError):
            await simulate_http_request(
                app=middleware, method="GET", path="/api/v1/portfolio"
            )

        exposition = generate_metrics_exposition()
        assert 'endpoint="/api/v1/portfolio"' in exposition
        assert 'method="GET"' in exposition
        assert 'status="500"' in exposition

    @pytest.mark.asyncio
    async def test_websocket_socket_lifecycle_tracking(self):
        """ASGI WebSocket connect and disconnect events increment and decrement socket counts."""
        manager = MetricsManager.get_instance()
        assert manager.get_active_socket_count() == 0

        async def dummy_ws_app(scope, receive, send):
            msg = await receive()
            if msg["type"] == "websocket.connect":
                await send({"type": "websocket.accept"})
            await receive()  # Wait for disconnect

        middleware = PrometheusMiddleware(app=dummy_ws_app)

        scope = {
            "type": "websocket",
            "asgi": {"version": "3.0"},
            "path": "/ws/stream",
            "headers": [],
        }

        # Event sequence: connect -> receive triggers -> disconnect
        events_queue = [
            {"type": "websocket.connect"},
            {"type": "websocket.disconnect", "code": 1000},
        ]

        async def ws_receive():
            if events_queue:
                return events_queue.pop(0)
            return {"type": "websocket.disconnect", "code": 1000}

        sent_messages: List[Dict[str, Any]] = []

        async def ws_send(message: Dict[str, Any]):
            sent_messages.append(message)
            # When accepted, the socket count should have been incremented
            if message["type"] == "websocket.accept":
                assert manager.get_active_socket_count() == 1

        await middleware(scope, ws_receive, ws_send)

        # After disconnect terminates the scope, active sockets must return to 0
        assert manager.get_active_socket_count() == 0