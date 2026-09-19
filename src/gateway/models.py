"""Gateway configuration models for Envoy proxy."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class WebSocketCompressionConfig:
    """Configuration for WebSocket permessage-deflate compression (RFC 7692)."""

    enabled: bool = True
    client_max_window_bits: Optional[int] = None
    server_max_window_bits: Optional[int] = None
    client_no_context_takeover: bool = False
    server_no_context_takeover: bool = False

    def __post_init__(self) -> None:
        self._validate_window_bits("client_max_window_bits", self.client_max_window_bits)
        self._validate_window_bits("server_max_window_bits", self.server_max_window_bits)

    @staticmethod
    def _validate_window_bits(field_name: str, value: Optional[int]) -> None:
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, int) or not (8 <= value <= 15):
                raise ValueError(
                    f"{field_name} must be an integer between 8 and 15 (RFC 7692), got {value}"
                )


@dataclass
class GatewayConfig:
    """Edge gateway configuration parameters."""

    enable_websocket: bool = True
    websocket_compression: Optional[WebSocketCompressionConfig] = None
    listener_address: str = "0.0.0.0"
    listener_port: int = 10000