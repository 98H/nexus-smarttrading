"""Prometheus metrics collection and exposition."""

import math
import threading
from typing import Dict, List, Optional, Tuple

try:
    import psutil
except ImportError:
    psutil = None

PROMETHEUS_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"

HISTOGRAM_BUCKETS: Tuple[float, ...] = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.075,
    0.1,
    0.25,
    0.5,
    0.75,
    1.0,
    2.5,
    5.0,
    7.5,
    10.0,
    float("inf"),
)


class MetricsManager:
    """Singleton manager tracking socket connections, render times, API latencies, and worker memory."""

    _instance: Optional["MetricsManager"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "MetricsManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        self._state_lock = threading.Lock()
        self.reset()

    @classmethod
    def get_instance(cls) -> "MetricsManager":
        return cls()

    def reset(self) -> None:
        """Reset all tracked metrics to their default states."""
        with self._state_lock:
            self._active_sockets: int = 0
            self._frame_render_durations: List[float] = []
            self._api_latencies: Dict[Tuple[str, str, str], List[float]] = {}
            self._worker_memory: Dict[str, int | float] = {}

    def get_active_socket_count(self) -> int:
        """Return the current number of active socket connections."""
        with self._state_lock:
            return self._active_sockets

    def record_socket_connection(self) -> None:
        """Increment active socket count."""
        with self._state_lock:
            self._active_sockets += 1

    def record_socket_disconnection(self) -> None:
        """Decrement active socket count, preventing underflow."""
        with self._state_lock:
            if self._active_sockets <= 0:
                raise ValueError("Active socket count cannot decrease below zero.")
            self._active_sockets -= 1

    def record_frame_render_time(self, duration: float) -> None:
        """Record frame render duration in seconds."""
        if duration < 0:
            raise ValueError("Frame render duration cannot be negative.")
        with self._state_lock:
            self._frame_render_durations.append(duration)

    def record_api_latency(
        self,
        method: str,
        endpoint: str,
        status_code: int | str,
        duration: float,
    ) -> None:
        """Record API request duration in seconds."""
        if duration < 0:
            raise ValueError("API request duration cannot be negative.")
        key = (method, endpoint, str(status_code))
        with self._state_lock:
            self._api_latencies.setdefault(key, []).append(duration)

    def record_worker_memory(self, worker_id: str, memory_bytes: int | float) -> None:
        """Record resident memory usage in bytes for a specific worker."""
        if memory_bytes < 0:
            raise ValueError("Worker memory bytes cannot be negative.")
        with self._state_lock:
            self._worker_memory[worker_id] = memory_bytes

    def generate_exposition(self) -> str:
        """Generate Prometheus text exposition format for all registered metrics."""
        with self._state_lock:
            lines: List[str] = []

            # 1. socket_connections_active
            lines.append(
                "# HELP socket_connections_active Number of active socket connections."
            )
            lines.append("# TYPE socket_connections_active gauge")
            lines.append(f"socket_connections_active {self._active_sockets}")

            # 2. frame_render_duration_seconds
            lines.append(
                "# HELP frame_render_duration_seconds Frame render duration in seconds."
            )
            lines.append("# TYPE frame_render_duration_seconds histogram")
            frame_obs = self._frame_render_durations
            frame_count = len(frame_obs)
            frame_sum = sum(frame_obs) if frame_obs else 0.0
            for b in HISTOGRAM_BUCKETS:
                le_str = "+Inf" if math.isinf(b) else str(b)
                b_count = sum(1 for x in frame_obs if x <= b)
                lines.append(
                    f'frame_render_duration_seconds_bucket{{le="{le_str}"}} {b_count}'
                )
            lines.append(f"frame_render_duration_seconds_count {frame_count}")
            lines.append(f"frame_render_duration_seconds_sum {frame_sum}")

            # 3. api_request_duration_seconds
            lines.append(
                "# HELP api_request_duration_seconds API request duration in seconds."
            )
            lines.append("# TYPE api_request_duration_seconds histogram")
            for (method, endpoint, status) in sorted(self._api_latencies.keys()):
                api_obs = self._api_latencies[(method, endpoint, status)]
                api_count = len(api_obs)
                api_sum = sum(api_obs) if api_obs else 0.0
                for b in HISTOGRAM_BUCKETS:
                    le_str = "+Inf" if math.isinf(b) else str(b)
                    b_count = sum(1 for x in api_obs if x <= b)
                    lines.append(
                        f'api_request_duration_seconds_bucket{{endpoint="{endpoint}",method="{method}",status="{status}",le="{le_str}"}} {b_count}'
                    )
                lines.append(
                    f'api_request_duration_seconds_count{{endpoint="{endpoint}",method="{method}",status="{status}"}} {api_count}'
                )
                lines.append(
                    f'api_request_duration_seconds_sum{{endpoint="{endpoint}",method="{method}",status="{status}"}} {api_sum}'
                )

            # 4. worker_memory_bytes
            lines.append(
                "# HELP worker_memory_bytes Resident memory bytes per worker."
            )
            lines.append("# TYPE worker_memory_bytes gauge")
            for worker_id in sorted(self._worker_memory.keys()):
                mem = self._worker_memory[worker_id]
                lines.append(f'worker_memory_bytes{{worker_id="{worker_id}"}} {mem}')

            lines.append("")
            return "\n".join(lines)


def record_socket_connection() -> None:
    """Record an active socket connection."""
    MetricsManager.get_instance().record_socket_connection()


def record_socket_disconnection() -> None:
    """Record a socket disconnection."""
    MetricsManager.get_instance().record_socket_disconnection()


def record_frame_render_time(duration: float) -> None:
    """Record frame render duration observation."""
    MetricsManager.get_instance().record_frame_render_time(duration)


def record_api_latency(
    method: str,
    endpoint: str,
    status_code: int | str,
    duration: float,
) -> None:
    """Record an API request latency observation."""
    MetricsManager.get_instance().record_api_latency(
        method, endpoint, status_code, duration
    )


def record_worker_memory(worker_id: str, memory_bytes: int | float) -> None:
    """Record worker resident memory in bytes."""
    MetricsManager.get_instance().record_worker_memory(worker_id, memory_bytes)


def collect_current_worker_memory(worker_id: str) -> None:
    """Probe current process RSS memory and record it for the specified worker."""
    global psutil
    if psutil is None:
        try:
            import psutil as _psutil

            psutil = _psutil
        except ImportError:
            raise RuntimeError("psutil is required to collect worker memory.")
    process = psutil.Process()
    rss = process.memory_info().rss
    record_worker_memory(worker_id=worker_id, memory_bytes=rss)


def generate_metrics_exposition() -> str:
    """Generate Prometheus exposition text for all metrics."""
    return MetricsManager.get_instance().generate_exposition()