"""Envoy proxy edge gateway configuration generator."""

from typing import Any, Dict, List, Optional

from src.gateway.models import GatewayConfig


class EnvoyConfigGenerator:
    """Generates Envoy proxy configuration with WebSocket and compression support."""

    def __init__(self, config: Optional[GatewayConfig] = None) -> None:
        self.config = config if config is not None else GatewayConfig()

    def generate_http_connection_manager(self) -> Dict[str, Any]:
        """Generate HTTP Connection Manager (HCM) filter configuration."""
        upgrade_configs: List[Dict[str, Any]] = []
        if self.config.enable_websocket:
            upgrade_configs.append(
                {
                    "upgrade_type": "websocket",
                    "enabled": True,
                }
            )

        http_filters = self._build_http_filters()

        return {
            "@type": "type.googleapis.com/envoy.extensions.filters.network.http_connection_manager.v3.HttpConnectionManager",
            "stat_prefix": "ingress_http",
            "upgrade_configs": upgrade_configs,
            "http_filters": http_filters,
        }

    def _build_http_filters(self) -> List[Dict[str, Any]]:
        """Build the HTTP filter chain in the correct Envoy order."""
        filters: List[Dict[str, Any]] = []

        if (
            self.config.enable_websocket
            and self.config.websocket_compression is not None
            and self.config.websocket_compression.enabled
        ):
            filters.append(self._build_compressor_filter())

        # Envoy requires envoy.filters.http.router as the terminal filter
        filters.append(
            {
                "name": "envoy.filters.http.router",
                "typed_config": {
                    "@type": "type.googleapis.com/envoy.extensions.filters.http.router.v3.Router",
                },
            }
        )

        return filters

    def _build_compressor_filter(self) -> Dict[str, Any]:
        """Build envoy.filters.http.compressor filter with permessage-deflate."""
        ws_comp = self.config.websocket_compression
        deflate_config: Dict[str, Any] = {
            "@type": "type.googleapis.com/envoy.extensions.compression.permessage_deflate.v3.PermessageDeflate",
        }

        if ws_comp is not None:
            if ws_comp.client_max_window_bits is not None:
                deflate_config["client_max_window_bits"] = ws_comp.client_max_window_bits
            if ws_comp.server_max_window_bits is not None:
                deflate_config["server_max_window_bits"] = ws_comp.server_max_window_bits
            deflate_config["client_no_context_takeover"] = ws_comp.client_no_context_takeover
            deflate_config["server_no_context_takeover"] = ws_comp.server_no_context_takeover

        return {
            "name": "envoy.filters.http.compressor",
            "typed_config": {
                "@type": "type.googleapis.com/envoy.extensions.filters.http.compressor.v3.Compressor",
                "compressor_library": {
                    "name": "permessage-deflate",
                    "typed_config": deflate_config,
                },
            },
        }

    def generate_config(self) -> Dict[str, Any]:
        """Generate the full Envoy bootstrap and listener configuration."""
        return {
            "static_resources": {
                "listeners": [
                    {
                        "name": "ingress_listener",
                        "address": {
                            "socket_address": {
                                "address": self.config.listener_address,
                                "port_value": self.config.listener_port,
                            }
                        },
                        "filter_chains": [
                            {
                                "filters": [
                                    {
                                        "name": "envoy.filters.network.http_connection_manager",
                                        "typed_config": self.generate_http_connection_manager(),
                                    }
                                ]
                            }
                        ],
                    }
                ]
            }
        }