"""
Resilient WebSocket Client with Exponential Backoff.

Story 2.1.1: Build Resilient WebSocket Client with Exponential Backoff
"""

import asyncio
from enum import Enum
import inspect
import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """Enumeration of WebSocket client connection states."""

    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    RECONNECTING = "RECONNECTING"


class MaxRetriesExceededError(Exception):
    """Raised when the maximum number of reconnection attempts is exceeded."""


class WebSocketClient:
    """A resilient WebSocket client supporting exponential backoff reconnection."""

    def __init__(
        self,
        url: str,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        max_retries: int = 5,
        connection_factory: Optional[Callable[[str], Any]] = None,
    ) -> None:
        if not isinstance(url, str) or not url.strip():
            raise ValueError("URL must be a non-empty string.")
        if base_delay <= 0:
            raise ValueError("base_delay must be strictly positive.")
        if max_delay <= 0:
            raise ValueError("max_delay must be strictly positive.")
        if max_delay < base_delay:
            raise ValueError("max_delay cannot be less than base_delay.")
        if max_retries < 0:
            raise ValueError("max_retries cannot be negative.")

        self.url: str = url
        self.base_delay: float = float(base_delay)
        self.max_delay: float = float(max_delay)
        self.max_retries: int = int(max_retries)
        self.connection_factory: Optional[Callable[[str], Any]] = connection_factory

        self.state: ConnectionState = ConnectionState.DISCONNECTED
        self.retry_count: int = 0
        self.connection: Any = None

    @property
    def is_connected(self) -> bool:
        """Return True if the client is currently connected."""
        return self.state == ConnectionState.CONNECTED

    @property
    def retry_counter(self) -> int:
        """Alias for retry_count."""
        return self.retry_count

    @property
    def attempt_count(self) -> int:
        """Alias for retry_count."""
        return self.retry_count

    @property
    def attempts(self) -> int:
        """Alias for retry_count."""
        return self.retry_count

    def compute_backoff_delay(self, attempt: int) -> float:
        """Calculate exponential backoff delay: min(base_delay * (2 ** attempt), max_delay)."""
        try:
            delay = self.base_delay * (2 ** max(0, attempt))
            return float(min(delay, self.max_delay))
        except OverflowError:
            return float(self.max_delay)

    calculate_backoff_delay = compute_backoff_delay
    get_backoff_delay = compute_backoff_delay

    async def connect(self) -> None:
        """Establish the initial WebSocket connection using exponential backoff."""
        await self._close_connection()
        self.state = ConnectionState.CONNECTING
        await self._connect_with_retry()

    async def disconnect(self) -> None:
        """Cleanly disconnect the client and transition to DISCONNECTED."""
        await self._close_connection()
        self.state = ConnectionState.DISCONNECTED
        self.retry_count = 0

    async def reconnect(self, error: Optional[Exception] = None) -> None:
        """Reconnect to the target endpoint applying exponential backoff."""
        await self._close_connection()
        self.state = ConnectionState.RECONNECTING
        await self._connect_with_retry()

    async def handle_disconnect(self, error: Optional[Exception] = None) -> None:
        """Hook called when an unexpected disconnect occurs."""
        await self.reconnect(error)

    async def on_disconnect(self, error: Optional[Exception] = None) -> None:
        """Hook called when an unexpected disconnect occurs."""
        await self.reconnect(error)

    async def _connect_with_retry(self) -> None:
        """Internal loop executing connection attempts with backoff and retry tracking."""
        self.retry_count = 0
        factory = self.connection_factory or self._default_connection_factory

        while True:
            try:
                conn = factory(self.url)
                if inspect.isawaitable(conn):
                    conn = await conn
                self.connection = conn
                self.state = ConnectionState.CONNECTED
                self.retry_count = 0
                logger.info("Successfully established connection to %s", self.url)
                return
            except Exception as exc:
                await self._close_connection()
                if self.retry_count < self.max_retries:
                    delay = self.compute_backoff_delay(self.retry_count)
                    logger.warning(
                        "Connection attempt failed (%s). Retrying in %.2fs (attempt %d/%d)...",
                        exc,
                        delay,
                        self.retry_count + 1,
                        self.max_retries,
                    )
                    await asyncio.sleep(delay)
                    self.retry_count += 1
                else:
                    self.state = ConnectionState.DISCONNECTED
                    logger.error(
                        "Persistent connection failure after %d retries: %s",
                        self.retry_count,
                        exc,
                    )
                    raise MaxRetriesExceededError(
                        f"Connection retries exhausted ({self.max_retries}): {exc}"
                    ) from exc

    async def _close_connection(self) -> None:
        """Safely close any active underlying connection object."""
        conn = self.connection
        self.connection = None
        if conn is not None:
            try:
                if hasattr(conn, "aclose"):
                    res = conn.aclose()
                    if inspect.isawaitable(res):
                        await res
                elif hasattr(conn, "close"):
                    res = conn.close()
                    if inspect.isawaitable(res):
                        await res
            except Exception as exc:
                logger.debug("Error while closing underlying connection: %s", exc)

    async def _default_connection_factory(self, url: str) -> Any:
        """Fallback connection factory when none is explicitly provided."""
        try:
            import websockets

            return await websockets.connect(url)
        except ImportError:
            raise NotImplementedError(
                "No connection_factory provided and 'websockets' package is not installed."
            )

    async def __aenter__(self) -> "WebSocketClient":
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.disconnect()