import pytest
from typing import Any, Dict, List, Optional

from src.gateway.models import GatewayConfig, WebSocketCompressionConfig
from src.gateway.envoy_config import EnvoyConfigGenerator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_filter_by_name(filters: List[Dict[str, Any]], name: str) -> Optional[Dict[str, Any]]:
    """Helper to locate a filter definition by its Envoy filter name."""
    for flt in filters:
        if flt.get("name") == name:
            return flt
    return None


def _is_permessage_deflate_compressor(filter_cfg: Dict[str, Any]) -> bool:
    """Helper to determine if an HTTP filter is configured for permessage-deflate."""
    if filter_cfg.get("name") != "envoy.filters.http.compressor":
        return False

    typed_config = filter_cfg.get("typed_config", {})
    compressor_lib = typed_config.get("compressor_library", {})
    lib_name = compressor_lib.get("name", "")
    lib_typed_config = compressor_lib.get("typed_config", {})
    lib_type = lib_typed_config.get("@type", "")

    # The library name or type must identify permessage-deflate
    return (
        "permessage-deflate" in lib_name
        or "permessage_deflate" in lib_name
        or "permessage-deflate" in lib_type
        or "permessage_deflate" in lib_type
        or "permessage-deflate" in str(typed_config)
    )


# ---------------------------------------------------------------------------
# Model Tests: src/gateway/models.py
# ---------------------------------------------------------------------------

class TestGatewayModels:
    """Tests for Gateway configuration models."""

    def test_websocket_compression_config_defaults(self):
        """Default instantiation of WebSocketCompressionConfig should be enabled."""
        ws_config = WebSocketCompressionConfig()
        assert ws_config.enabled is True
        assert ws_config.client_max_window_bits is None
        assert ws_config.server_max_window_bits is None
        assert ws_config.client_no_context_takeover is False
        assert ws_config.server_no_context_takeover is False

    def test_websocket_compression_config_custom_parameters(self):
        """Custom window bits and context takeover flags should be stored accurately."""
        ws_config = WebSocketCompressionConfig(
            enabled=True,
            client_max_window_bits=12,
            server_max_window_bits=15,
            client_no_context_takeover=True,
            server_no_context_takeover=True,
        )
        assert ws_config.enabled is True
        assert ws_config.client_max_window_bits == 12
        assert ws_config.server_max_window_bits == 15
        assert ws_config.client_no_context_takeover is True
        assert ws_config.server_no_context_takeover is True

    @pytest.mark.parametrize("invalid_bits", [7, 16, 0, -1])
    def test_websocket_compression_invalid_window_bits(self, invalid_bits: int):
        """Window bits outside RFC 7692 range (8-15) must raise ValueError."""
        with pytest.raises(ValueError):
            WebSocketCompressionConfig(client_max_window_bits=invalid_bits)

        with pytest.raises(ValueError):
            WebSocketCompressionConfig(server_max_window_bits=invalid_bits)

    def test_gateway_config_default_omits_compression(self):
        """When not specified, websocket_compression must default to None."""
        gw_config = GatewayConfig()
        assert gw_config.enable_websocket is True
        assert gw_config.websocket_compression is None

    def test_gateway_config_explicit_compression_assignment(self):
        """Explicit assignment of WebSocketCompressionConfig to GatewayConfig."""
        ws_config = WebSocketCompressionConfig(enabled=True)
        gw_config = GatewayConfig(websocket_compression=ws_config)
        assert gw_config.websocket_compression is not None
        assert gw_config.websocket_compression.enabled is True


# ---------------------------------------------------------------------------
# Generator Tests: AC 1 - WebSocket Upgrade & Permessage-Deflate Compression
# ---------------------------------------------------------------------------

class TestEnvoyConfigGeneratorCompressionEnabled:
    """Acceptance Criteria 1:
    Given an Envoy configuration generator with WebSocket upgrade support,
    When generating the HTTP Connection Manager filter chain with compression enabled,
    Then the output configuration includes the `websocket` entry under `upgrade_configs`
    and the `envoy.filters.http.compressor` filter configured with `permessage-deflate`.
    """

    @pytest.fixture
    def generator_with_compression(self) -> EnvoyConfigGenerator:
        gw_config = GatewayConfig(
            enable_websocket=True,
            websocket_compression=WebSocketCompressionConfig(
                enabled=True,
                client_max_window_bits=15,
                server_max_window_bits=15,
                client_no_context_takeover=False,
                server_no_context_takeover=False,
            ),
        )
        return EnvoyConfigGenerator(gw_config)

    def test_hcm_includes_websocket_upgrade_config(self, generator_with_compression: EnvoyConfigGenerator):
        """HCM must include 'websocket' upgrade configuration."""
        hcm = generator_with_compression.generate_http_connection_manager()
        upgrade_configs = hcm.get("upgrade_configs", [])

        assert isinstance(upgrade_configs, list)
        websocket_entries = [
            cfg for cfg in upgrade_configs if cfg.get("upgrade_type") == "websocket"
        ]
        assert len(websocket_entries) == 1
        assert websocket_entries[0].get("enabled", True) is True

    def test_hcm_includes_permessage_deflate_compressor_filter(self, generator_with_compression: EnvoyConfigGenerator):
        """HCM filter chain must include envoy.filters.http.compressor configured with permessage-deflate."""
        hcm = generator_with_compression.generate_http_connection_manager()
        http_filters = hcm.get("http_filters", [])

        compressor_filter = _find_filter_by_name(http_filters, "envoy.filters.http.compressor")
        assert compressor_filter is not None
        assert _is_permessage_deflate_compressor(compressor_filter) is True

    def test_compressor_filter_typed_config_details(self, generator_with_compression: EnvoyConfigGenerator):
        """Permessage-deflate typed_config must carry window bits and tuning options."""
        hcm = generator_with_compression.generate_http_connection_manager()
        http_filters = hcm.get("http_filters", [])

        compressor_filter = _find_filter_by_name(http_filters, "envoy.filters.http.compressor")
        assert compressor_filter is not None

        typed_config = compressor_filter.get("typed_config", {})
        compressor_lib = typed_config.get("compressor_library", {})
        lib_config = compressor_lib.get("typed_config", {})

        # Assert library configuration contains permessage-deflate parameters
        assert (
            compressor_lib.get("name") == "permessage-deflate"
            or "permessage_deflate" in compressor_lib.get("name", "")
            or "permessage_deflate" in lib_config.get("@type", "")
        )
        # Check window bits mapped into configuration
        if "client_max_window_bits" in lib_config:
            assert lib_config["client_max_window_bits"] == 15

    def test_filter_ordering_compressor_before_router(self, generator_with_compression: EnvoyConfigGenerator):
        """Envoy requires envoy.filters.http.router to be the terminal filter in the HCM filter chain."""
        hcm = generator_with_compression.generate_http_connection_manager()
        http_filters = hcm.get("http_filters", [])

        filter_names = [f.get("name") for f in http_filters]
        assert "envoy.filters.http.compressor" in filter_names
        assert "envoy.filters.http.router" in filter_names

        compressor_index = filter_names.index("envoy.filters.http.compressor")
        router_index = filter_names.index("envoy.filters.http.router")

        assert compressor_index < router_index
        assert router_index == len(filter_names) - 1


# ---------------------------------------------------------------------------
# Generator Tests: AC 2 - Compression Omitted or Disabled
# ---------------------------------------------------------------------------

class TestEnvoyConfigGeneratorCompressionDisabledOrOmitted:
    """Acceptance Criteria 2:
    Given a gateway configuration model defining edge parameters,
    When WebSocket compression settings are omitted or disabled,
    Then the generator produces the HTTP Connection Manager filter chain
    without the permessage-deflate compression filter extension.
    """

    def test_hcm_omits_compressor_filter_when_settings_omitted(self):
        """When websocket_compression is omitted (None), permessage-deflate filter must not be present."""
        gw_config = GatewayConfig(enable_websocket=True, websocket_compression=None)
        generator = EnvoyConfigGenerator(gw_config)

        hcm = generator.generate_http_connection_manager()
        http_filters = hcm.get("http_filters", [])

        # WebSocket upgrade must still be present
        upgrade_configs = hcm.get("upgrade_configs", [])
        assert any(cfg.get("upgrade_type") == "websocket" for cfg in upgrade_configs)

        # Permessage-deflate filter must NOT be present
        deflate_filters = [f for f in http_filters if _is_permessage_deflate_compressor(f)]
        assert len(deflate_filters) == 0

    def test_hcm_omits_compressor_filter_when_settings_disabled(self):
        """When websocket_compression has enabled=False, permessage-deflate filter must not be present."""
        gw_config = GatewayConfig(
            enable_websocket=True,
            websocket_compression=WebSocketCompressionConfig(enabled=False),
        )
        generator = EnvoyConfigGenerator(gw_config)

        hcm = generator.generate_http_connection_manager()
        http_filters = hcm.get("http_filters", [])

        # WebSocket upgrade must still be present
        upgrade_configs = hcm.get("upgrade_configs", [])
        assert any(cfg.get("upgrade_type") == "websocket" for cfg in upgrade_configs)

        # Permessage-deflate filter must NOT be present
        deflate_filters = [f for f in http_filters if _is_permessage_deflate_compressor(f)]
        assert len(deflate_filters) == 0

    def test_terminal_router_filter_retained_when_compression_omitted(self):
        """Router filter must remain present as the terminal filter even when compression is omitted."""
        gw_config = GatewayConfig(websocket_compression=None)
        generator = EnvoyConfigGenerator(gw_config)

        hcm = generator.generate_http_connection_manager()
        http_filters = hcm.get("http_filters", [])

        assert len(http_filters) >= 1
        assert http_filters[-1].get("name") == "envoy.filters.http.router"


# ---------------------------------------------------------------------------
# Generator Tests: Full Config & Edge Cases
# ---------------------------------------------------------------------------

class TestEnvoyConfigGeneratorEdgeCases:
    """Additional edge cases for Gateway configuration and full Envoy listener output."""

    def test_websocket_disabled_entirely(self):
        """When enable_websocket is False, neither websocket upgrade nor permessage-deflate should be added."""
        gw_config = GatewayConfig(
            enable_websocket=False,
            websocket_compression=WebSocketCompressionConfig(enabled=True),
        )
        generator = EnvoyConfigGenerator(gw_config)

        hcm = generator.generate_http_connection_manager()
        upgrade_configs = hcm.get("upgrade_configs", [])

        websocket_entries = [
            cfg for cfg in upgrade_configs if cfg.get("upgrade_type") == "websocket"
        ]
        assert len(websocket_entries) == 0

        http_filters = hcm.get("http_filters", [])
        deflate_filters = [f for f in http_filters if _is_permessage_deflate_compressor(f)]
        assert len(deflate_filters) == 0

    def test_full_envoy_config_generation_structure(self):
        """Generating the full Envoy bootstrap/listener configuration contains valid structure."""
        gw_config = GatewayConfig(
            enable_websocket=True,
            websocket_compression=WebSocketCompressionConfig(enabled=True),
        )
        generator = EnvoyConfigGenerator(gw_config)

        full_config = generator.generate_config()
        assert isinstance(full_config, dict)

        listeners = full_config.get("static_resources", {}).get("listeners", [])
        assert len(listeners) >= 1

        listener = listeners[0]
        filter_chains = listener.get("filter_chains", [])
        assert len(filter_chains) >= 1

        network_filters = filter_chains[0].get("filters", [])
        hcm_filter = _find_filter_by_name(network_filters, "envoy.filters.network.http_connection_manager")
        assert hcm_filter is not None

        typed_config = hcm_filter.get("typed_config", {})
        assert typed_config.get("@type") == "type.googleapis.com/envoy.extensions.filters.network.http_connection_manager.v3.HttpConnectionManager"
        assert any(cfg.get("upgrade_type") == "websocket" for cfg in typed_config.get("upgrade_configs", []))

    def test_independent_generation_calls_do_not_mutate_state(self):
        """Repeated generation calls must produce deterministic and independent results."""
        gw_config = GatewayConfig(
            enable_websocket=True,
            websocket_compression=WebSocketCompressionConfig(enabled=True),
        )
        generator = EnvoyConfigGenerator(gw_config)

        hcm_first = generator.generate_http_connection_manager()
        hcm_second = generator.generate_http_connection_manager()

        assert hcm_first == hcm_second
        assert hcm_first is not hcm_second