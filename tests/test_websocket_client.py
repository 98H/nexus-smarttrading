"""
Unit tests for resilient WebSocket client with exponential backoff.

Target Module: src.network.websocket_client
Story 2.1.1: Build Resilient WebSocket Client with Exponential Backoff
"""

import asyncio
import inspect
from typing import Any, Callable, List, Optional
from unittest.mock import AsyncMock, patch

import pytest

from src.network.websocket_client import (
    ConnectionState,
    MaxRetriesExceededError,
    WebSocketClient,
)


# ============================================================================
# Test Fixtures & Helpers
# ============================================================================

class DualModeConnection:
    """A mock connection object compatible with both synchronous and asynchronous usage."""

    def __init__(self) -> None:
        self.is_open: bool = True

    def __await__(self):
        async def _resolve():
            return self
        return _resolve().__await__()

    def close(self) -> None:
        self.is_open = False

    async def aclose(self) -> None:
        self.is_open = False


def _run(target: Any) -> Any:
    """Execute awaitables in an event loop or return sync results directly."""
    if inspect.isawaitable(target):
        return asyncio.run(target)
    return target


def _get_retry_count(client: WebSocketClient) -> int:
    """Extract current retry counter value across supported naming conventions."""
    for attr in ("retry_count", "retry_counter", "attempt_count", "attempts"):
        if hasattr(client, attr):
            return int(getattr(client, attr))
    raise AttributeError("WebSocketClient missing retry counter attribute (e.g. retry_count, retry_counter)")


def _assert_disconnected(client: WebSocketClient) -> None:
    """Assert client is in a disconnected state via boolean flag and/or state enum."""
    if hasattr(client, "is_connected"):
        assert client.is_connected is False
    if hasattr(client, "state"):
        state_name = getattr(client.state, "name", str(client.state)).upper()
        assert state_name == ConnectionState.DISCONNECTED.name


def _assert_connected(client: WebSocketClient) -> None:
    """Assert client is in a connected state via boolean flag and/or state enum."""
    if hasattr(client, "is_connected"):
        assert client.is_connected is True
    if hasattr(client, "state"):
        state_name = getattr(client.state, "name", str(client.state)).upper()
        assert state_name == ConnectionState.CONNECTED.name


def _trigger_reconnect(client: WebSocketClient, error: Optional[Exception] = None) -> Any:
    """Trigger reconnection via client hooks or reconnect method."""
    exc = error or ConnectionResetError("Remote host closed connection")
    if hasattr(client, "handle_disconnect"):
        return _run(client.handle_disconnect(exc))
    if hasattr(client, "on_disconnect"):
        return _run(client.on_disconnect(exc))
    if hasattr(client, "reconnect"):
        return _run(client.reconnect())
    raise AttributeError("WebSocketClient does not expose handle_disconnect, on_disconnect, or reconnect")


@pytest.fixture
def recorded_sleeps():
    """Deterministic sleep interceptor recording durations for both sync and async sleep."""
    delays: List[float] = []

    def sync_sleep(duration: float) -> None:
        delays.append(duration)

    async def async_sleep(duration: float, *args, **kwargs) -> None:
        delays.append(duration)

    with patch("time.sleep", side_effect=sync_sleep), \
         patch("asyncio.sleep", side_effect=async_sleep):
        yield delays


# ============================================================================
# Acceptance Criteria 1: Exponential Backoff Formula & Delay Progression
# ============================================================================

def test_backoff_formula_direct_calculation():
    """
    Given a WebSocket client configured with base_delay=1.0 and max_delay=10.0,
    When calculating delay for successive attempt indices,
    Then the result matches min(base_delay * (2 ** attempt), max_delay).
    """
    client = WebSocketClient(
        url="wss://stream.target.com/v1",
        base_delay=1.0,
        max_delay=10.0,
        max_retries=5,
    )

    expected_progression = [
        (0, 1.0),    # min(1.0 * 1, 10.0) = 1.0
        (1, 2.0),    # min(1.0 * 2, 10.0) = 2.0
        (2, 4.0),    # min(1.0 * 4, 10.0) = 4.0
        (3, 8.0),    # min(1.0 * 8, 10.0) = 8.0
        (4, 10.0),   # min(1.0 * 16, 10.0) = 10.0 (capped)
        (5, 10.0),   # min(1.0 * 32, 10.0) = 10.0 (capped)
        (10, 10.0),  # large attempt clamped safely
    ]

    calc_func: Callable[[int], float]
    if hasattr(client, "compute_backoff_delay"):
        calc_func = client.compute_backoff_delay
    elif hasattr(client, "calculate_backoff_delay"):
        calc_func = client.calculate_backoff_delay
    elif hasattr(client, "get_backoff_delay"):
        calc_func = client.get_backoff_delay
    else:
        pytest.fail("WebSocketClient must expose a backoff calculation method")

    for attempt, expected in expected_progression:
        actual = calc_func(attempt)
        assert actual == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize(
    "base_delay,max_delay,attempt,expected_delay",
    [
        (0.5, 5.0, 0, 0.5),
        (0.5, 5.0, 1, 1.0),
        (0.5, 5.0, 2, 2.0),
        (0.5, 5.0, 3, 4.0),
        (0.5, 5.0, 4, 5.0),
        (0.1, 0.3, 0, 0.1),
        (0.1, 0.3, 1, 0.2),
        (0.1, 0.3, 2, 0.3),
        (2.0, 60.0, 3, 16.0),
    ],
)
def test_backoff_formula_parameterized_inputs(base_delay, max_delay, attempt, expected_delay):
    """
    Given varied base_delay and max_delay configurations,
    When computing backoff delay for an attempt,
    Then the value strictly adheres to min(base_delay * (2 ** attempt), max_delay).
    """
    client = WebSocketClient(
        url="wss://stream.target.com/v1",
        base_delay=base_delay,
        max_delay=max_delay,
        max_retries=5,
    )
    calc_func = getattr(
        client,
        "compute_backoff_delay",
        getattr(client, "calculate_backoff_delay", getattr(client, "get_backoff_delay", None)),
    )
    assert calc_func is not None
    assert calc_func(attempt) == pytest.approx(expected_delay, rel=1e-6)


def test_connection_retries_wait_with_backoff_and_increment_counter(recorded_sleeps):
    """
    Given a connection that fails 3 consecutive times before succeeding,
    When attempting connection,
    Then the client sleeps for [base * 2^0, base * 2^1, base * 2^2] before each retry,
    and increments the attempt count accordingly.
    """
    attempts = 0

    def flaky_connect(url: str):
        nonlocal attempts
        attempts += 1
        if attempts <= 3:
            raise ConnectionRefusedError(f"Handshake failure #{attempts}")
        return DualModeConnection()

    client = WebSocketClient(
        url="wss://stream.target.com/v1",
        base_delay=1.0,
        max_delay=10.0,
        max_retries=5,
        connection_factory=flaky_connect,
    )

    _run(client.connect())

    # Should have retried 3 times, sleeping 1.0, 2.0, 4.0
    assert recorded_sleeps == [1.0, 2.0, 4.0]
    _assert_connected(client)


# ============================================================================
# Acceptance Criteria 2: Unexpected Disconnect, Reconnect & Counter Reset
# ============================================================================

def test_successful_initial_connection_state_and_counter():
    """
    Given a healthy endpoint,
    When connection succeeds on the first attempt,
    Then the client transitions to CONNECTED and retry counter remains 0.
    """
    conn = DualModeConnection()
    client = WebSocketClient(
        url="wss://stream.target.com/v1",
        base_delay=1.0,
        max_delay=10.0,
        max_retries=3,
        connection_factory=lambda url: conn,
    )

    _run(client.connect())

    _assert_connected(client)
    assert _get_retry_count(client) == 0


def test_reconnect_resets_retry_counter_upon_successful_handshake(recorded_sleeps):
    """
    Given an active connection that disconnects unexpectedly,
    When reconnecting with one transient failure before success,
    Then backoff delay is applied and the retry counter is reset to 0 upon handshake.
    """
    active_connection = DualModeConnection()
    connection_attempts = 0

    def mock_factory(url: str):
        nonlocal connection_attempts
        connection_attempts += 1
        if connection_attempts == 1:
            # Initial successful connection
            return active_connection
        if connection_attempts == 2:
            # First reconnect attempt fails
            raise ConnectionResetError("Failed handshake during reconnect")
        # Second reconnect attempt succeeds
        return DualModeConnection()

    client = WebSocketClient(
        url="wss://stream.target.com/v1",
        base_delay=1.0,
        max_delay=10.0,
        max_retries=3,
        connection_factory=mock_factory,
    )

    # 1. Initial connection
    _run(client.connect())
    _assert_connected(client)
    assert _get_retry_count(client) == 0

    # 2. Unexpected drop triggers reconnection
    _trigger_reconnect(client)

    # 3. Verify reconnect backoff was applied and counter was reset to 0
    assert recorded_sleeps == [1.0]  # First reconnect failed (attempt 0 -> 1.0s)
    _assert_connected(client)
    assert _get_retry_count(client) == 0


def test_subsequent_disconnect_starts_backoff_from_zero_after_prior_reset(recorded_sleeps):
    """
    Given a connection that reconnected and reset its retry counter,
    When another unexpected disconnect occurs later,
    Then backoff delay starts from attempt 0 (base_delay), proving the counter was reset.
    """
    connection_attempts = 0

    def mock_factory(url: str):
        nonlocal connection_attempts
        connection_attempts += 1
        # Fails on attempt 2 (first reconnect) and attempt 4 (second reconnect)
        if connection_attempts in (2, 4):
            raise ConnectionError("Transient drop")
        return DualModeConnection()

    client = WebSocketClient(
        url="wss://stream.target.com/v1",
        base_delay=2.0,
        max_delay=20.0,
        max_retries=3,
        connection_factory=mock_factory,
    )

    # Initial connect (attempt 1: success)
    _run(client.connect())

    # First unexpected drop (attempt 2: fail -> sleep 2.0, attempt 3: success)
    _trigger_reconnect(client)
    assert recorded_sleeps == [2.0]
    assert _get_retry_count(client) == 0

    # Second unexpected drop (attempt 4: fail -> must sleep base_delay 2.0, NOT 4.0)
    _trigger_reconnect(client)
    assert recorded_sleeps == [2.0, 2.0]
    assert _get_retry_count(client) == 0


# ============================================================================
# Acceptance Criteria 3: Persistent Failure, MaxRetriesExceeded & State
# ============================================================================

def test_persistent_failure_raises_max_retries_exceeded_and_disconnects(recorded_sleeps):
    """
    Given a persistent network failure exceeding max_retries,
    When reconnection attempts are exhausted,
    Then MaxRetriesExceededError is raised directly and client is DISCONNECTED.
    """
    def always_fails(url: str):
        raise OSError("Host unreachable")

    client = WebSocketClient(
        url="wss://stream.target.com/v1",
        base_delay=0.5,
        max_delay=4.0,
        max_retries=3,
        connection_factory=always_fails,
    )

    with pytest.raises(MaxRetriesExceededError):
        _run(client.connect())

    # Sleeps for attempt 0 (0.5), attempt 1 (1.0), attempt 2 (2.0)
    assert recorded_sleeps == [0.5, 1.0, 2.0]
    _assert_disconnected(client)


def test_reconnection_exhausts_retries_and_raises_max_retries_exceeded(recorded_sleeps):
    """
    Given an established connection that drops,
    When reconnect attempts fail persistently exceeding max_retries,
    Then MaxRetriesExceededError is raised and client transitions to DISCONNECTED.
    """
    connected_once = False

    def drop_and_fail(url: str):
        nonlocal connected_once
        if not connected_once:
            connected_once = True
            return DualModeConnection()
        raise ConnectionResetError("Persistent downstream failure")

    client = WebSocketClient(
        url="wss://stream.target.com/v1",
        base_delay=1.0,
        max_delay=8.0,
        max_retries=2,
        connection_factory=drop_and_fail,
    )

    _run(client.connect())
    _assert_connected(client)

    with pytest.raises(MaxRetriesExceededError):
        _trigger_reconnect(client)

    # 2 retries attempted: attempt 0 (1.0), attempt 1 (2.0)
    assert recorded_sleeps == [1.0, 2.0]
    _assert_disconnected(client)


def test_max_retries_zero_fails_immediately_without_sleeping(recorded_sleeps):
    """
    Given max_retries configured to 0,
    When the initial connection attempt fails,
    Then client raises MaxRetriesExceededError immediately without retrying or sleeping.
    """
    def fail_immediately(url: str):
        raise ConnectionRefusedError("Immediate rejection")

    client = WebSocketClient(
        url="wss://stream.target.com/v1",
        base_delay=1.0,
        max_delay=10.0,
        max_retries=0,
        connection_factory=fail_immediately,
    )

    with pytest.raises(MaxRetriesExceededError):
        _run(client.connect())

    assert recorded_sleeps == []
    _assert_disconnected(client)


# ============================================================================
# Lifecycle, State Transitions & Parameter Validations
# ============================================================================

def test_initial_client_state():
    """
    Given a newly instantiated WebSocketClient,
    Then it starts in a DISCONNECTED state with 0 retries.
    """
    client = WebSocketClient(
        url="wss://stream.target.com/v1",
        base_delay=1.0,
        max_delay=10.0,
        max_retries=3,
    )

    _assert_disconnected(client)
    assert _get_retry_count(client) == 0


def test_clean_disconnect_does_not_trigger_reconnection(recorded_sleeps):
    """
    Given an actively connected client,
    When disconnect() is called intentionally,
    Then client transitions to DISCONNECTED and no reconnection attempts occur.
    """
    conn = DualModeConnection()
    client = WebSocketClient(
        url="wss://stream.target.com/v1",
        base_delay=1.0,
        max_delay=10.0,
        max_retries=3,
        connection_factory=lambda url: conn,
    )

    _run(client.connect())
    _assert_connected(client)

    _run(client.disconnect())

    _assert_disconnected(client)
    assert recorded_sleeps == []
    assert _get_retry_count(client) == 0


@pytest.mark.parametrize(
    "invalid_kwargs",
    [
        {"base_delay": -1.0},
        {"base_delay": 0.0},
        {"max_delay": -5.0},
        {"base_delay": 5.0, "max_delay": 2.0},  # max_delay < base_delay
        {"max_retries": -1},
        {"url": ""},
    ],
)
def test_constructor_parameter_validation(invalid_kwargs):
    """
    Given invalid backoff parameters or invalid URL,
    When instantiating WebSocketClient,
    Then a ValueError is raised directly.
    """
    default_kwargs = {
        "url": "wss://stream.target.com/v1",
        "base_delay": 1.0,
        "max_delay": 10.0,
        "max_retries": 3,
    }
    default_kwargs.update(invalid_kwargs)

    with pytest.raises(ValueError):
        WebSocketClient(**default_kwargs)