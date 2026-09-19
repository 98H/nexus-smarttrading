import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.adapters.config import ExchangeConfig
from src.adapters.ccxt_adapter import CCXTExchangeAdapter, AuthenticationError


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def valid_config() -> ExchangeConfig:
    """Returns a valid ExchangeConfig instance."""
    return ExchangeConfig(
        exchange_id="binance",
        api_key="valid-api-key-12345",
        secret="valid-secret-key-67890",
        password="valid-password",
        sandbox=False,
    )


@pytest.fixture
def mock_rest_client():
    """Mock CCXT REST async client session."""
    client = MagicMock()
    client.id = "binance"
    client.apiKey = "valid-api-key-12345"
    client.secret = "valid-secret-key-67890"
    client.password = "valid-password"
    client.load_markets = AsyncMock(return_value={"BTC/USDT": {}})
    client.fetch_balance = AsyncMock(return_value={"total": {"BTC": 1.0}})
    client.check_required_credentials = MagicMock(return_value=True)
    client.close = AsyncMock()
    client.set_sandbox_mode = MagicMock()
    return client


@pytest.fixture
def mock_ws_client():
    """Mock CCXT Pro WebSocket async client session."""
    client = MagicMock()
    client.id = "binance"
    client.apiKey = "valid-api-key-12345"
    client.secret = "valid-secret-key-67890"
    client.password = "valid-password"
    client.load_markets = AsyncMock(return_value={"BTC/USDT": {}})
    client.check_required_credentials = MagicMock(return_value=True)
    client.watch_ticker = AsyncMock()
    client.watch_order_book = AsyncMock()
    client.watch_trades = AsyncMock()
    client.close = AsyncMock()
    client.set_sandbox_mode = MagicMock()
    return client


# ============================================================================
# Test Suite: ExchangeConfig
# ============================================================================

class TestExchangeConfig:
    """Unit tests for configuration validation in src/adapters/config.py."""

    def test_valid_exchange_config(self, valid_config: ExchangeConfig):
        """Test successful configuration initialization with valid fields."""
        assert valid_config.exchange_id == "binance"
        assert valid_config.api_key == "valid-api-key-12345"
        assert valid_config.secret == "valid-secret-key-67890"
        assert valid_config.password == "valid-password"
        assert valid_config.sandbox is False

    def test_config_empty_exchange_id_raises_value_error(self):
        """Exchange ID cannot be empty or whitespace."""
        with pytest.raises(ValueError):
            ExchangeConfig(
                exchange_id="",
                api_key="valid-key",
                secret="valid-secret",
            )

    def test_config_empty_api_key_raises_value_error(self):
        """API key cannot be empty or whitespace."""
        with pytest.raises(ValueError):
            ExchangeConfig(
                exchange_id="binance",
                api_key="",
                secret="valid-secret",
            )

    def test_config_empty_secret_raises_value_error(self):
        """API secret cannot be empty or whitespace."""
        with pytest.raises(ValueError):
            ExchangeConfig(
                exchange_id="binance",
                api_key="valid-key",
                secret="",
            )

    def test_config_default_optional_values(self):
        """Test that default values are set correctly for optional attributes."""
        config = ExchangeConfig(
            exchange_id="coinbase",
            api_key="test-key",
            secret="test-secret",
        )
        assert config.password is None
        assert config.sandbox is False


# ============================================================================
# Test Suite: CCXTExchangeAdapter Authorization & Client Sessions
# ============================================================================

class TestCCXTExchangeAdapterAuthorization:
    """
    Unit tests for CCXT exchange adapter initialization and authorization
    matching Story 7.2.3 Acceptance Criteria.
    """

    @pytest.mark.asyncio
    @patch("src.adapters.ccxt_adapter.ccxtpro")
    @patch("src.adapters.ccxt_adapter.ccxt")
    async def test_successful_adapter_initialization_and_authorization(
        self,
        mock_ccxt,
        mock_ccxtpro,
        valid_config: ExchangeConfig,
        mock_rest_client,
        mock_ws_client,
    ):
        """
        Acceptance Criteria:
        Given valid API credentials and a supported exchange ID,
        when the CCXT adapter is initialized and authorized,
        then authenticated REST and WebSocket (ccxt.pro) client sessions
        are instantiated and marked as connected.
        """
        # Configure dynamic class retrieval for the exchange
        setattr(mock_ccxt, "binance", MagicMock(return_value=mock_rest_client))
        setattr(mock_ccxtpro, "binance", MagicMock(return_value=mock_ws_client))
        mock_ccxt.exchanges = ["binance", "coinbase", "kraken"]
        mock_ccxtpro.exchanges = ["binance", "coinbase", "kraken"]

        adapter = CCXTExchangeAdapter(config=valid_config)

        # Before authorization, adapter is not connected
        assert adapter.is_connected is False
        assert adapter.rest_client is None
        assert adapter.ws_client is None

        # Execute authorization
        await adapter.authorize()

        # REST client assertions
        assert adapter.rest_client is not None
        mock_rest_client.load_markets.assert_awaited_once()
        mock_rest_client.check_required_credentials.assert_called_once()

        # WebSocket (ccxt.pro) client assertions
        assert adapter.ws_client is not None
        mock_ws_client.load_markets.assert_awaited_once()
        mock_ws_client.check_required_credentials.assert_called_once()

        # Adapter status
        assert adapter.is_connected is True
        assert adapter.is_rest_connected is True
        assert adapter.is_ws_connected is True

    @pytest.mark.asyncio
    @patch("src.adapters.ccxt_adapter.ccxtpro")
    @patch("src.adapters.ccxt_adapter.ccxt")
    async def test_authorization_fails_with_invalid_credentials_raises_authentication_error(
        self,
        mock_ccxt,
        mock_ccxtpro,
        valid_config: ExchangeConfig,
        mock_rest_client,
        mock_ws_client,
    ):
        """
        Acceptance Criteria:
        Given invalid exchange credentials, when authorization is attempted,
        then an AuthenticationError is raised and no WebSocket stream is opened.
        """
        # Simulate authentication failure on credentials verification
        mock_rest_client.check_required_credentials.side_effect = AuthenticationError(
            "Invalid API credentials"
        )
        setattr(mock_ccxt, "binance", MagicMock(return_value=mock_rest_client))
        setattr(mock_ccxtpro, "binance", MagicMock(return_value=mock_ws_client))
        mock_ccxt.exchanges = ["binance"]
        mock_ccxtpro.exchanges = ["binance"]

        adapter = CCXTExchangeAdapter(config=valid_config)

        with pytest.raises(AuthenticationError):
            await adapter.authorize()

        # Assert no WebSocket stream was opened
        mock_ws_client.watch_ticker.assert_not_called()
        mock_ws_client.watch_order_book.assert_not_called()
        mock_ws_client.watch_trades.assert_not_called()

        # Assert adapter is not connected
        assert adapter.is_connected is False
        assert adapter.is_ws_connected is False

    @pytest.mark.asyncio
    @patch("src.adapters.ccxt_adapter.ccxtpro")
    @patch("src.adapters.ccxt_adapter.ccxt")
    async def test_authorization_fails_when_rest_auth_probe_fails(
        self,
        mock_ccxt,
        mock_ccxtpro,
        valid_config: ExchangeConfig,
        mock_rest_client,
        mock_ws_client,
    ):
        """
        Given credentials rejected by the remote exchange during authenticated balance probe,
        then AuthenticationError is raised and no WebSocket stream is opened.
        """
        # Simulate exchange-side 401/Invalid Key via fetch_balance
        mock_rest_client.fetch_balance.side_effect = AuthenticationError("Unauthorized")
        setattr(mock_ccxt, "binance", MagicMock(return_value=mock_rest_client))
        setattr(mock_ccxtpro, "binance", MagicMock(return_value=mock_ws_client))
        mock_ccxt.exchanges = ["binance"]
        mock_ccxtpro.exchanges = ["binance"]

        adapter = CCXTExchangeAdapter(config=valid_config)

        with pytest.raises(AuthenticationError):
            await adapter.authorize()

        # Verify no WebSocket streams were initiated
        mock_ws_client.watch_ticker.assert_not_called()
        mock_ws_client.watch_order_book.assert_not_called()
        assert adapter.is_connected is False

    @pytest.mark.asyncio
    @patch("src.adapters.ccxt_adapter.ccxtpro")
    @patch("src.adapters.ccxt_adapter.ccxt")
    async def test_authorization_fails_with_missing_credentials(
        self,
        mock_ccxt,
        mock_ccxtpro,
    ):
        """
        Acceptance Criteria:
        Given missing exchange credentials when authorization is attempted,
        then an AuthenticationError is raised and no WebSocket stream is opened.
        """
        # Bypassing config validation to test adapter-level defense against missing credentials
        empty_config = MagicMock(spec=ExchangeConfig)
        empty_config.exchange_id = "binance"
        empty_config.api_key = None
        empty_config.secret = None
        empty_config.password = None
        empty_config.sandbox = False

        adapter = CCXTExchangeAdapter(config=empty_config)

        with pytest.raises(AuthenticationError):
            await adapter.authorize()

        # No CCXT instances created and no WS opened
        mock_ccxt.binance.assert_not_called()
        mock_ccxtpro.binance.assert_not_called()
        assert adapter.is_connected is False

    @pytest.mark.asyncio
    @patch("src.adapters.ccxt_adapter.ccxtpro")
    @patch("src.adapters.ccxt_adapter.ccxt")
    async def test_unsupported_exchange_id_raises_value_error(
        self,
        mock_ccxt,
        mock_ccxtpro,
    ):
        """
        Attempting to initialize an adapter with an exchange ID not supported by
        CCXT or CCXT Pro raises a ValueError.
        """
        mock_ccxt.exchanges = ["binance", "coinbase"]
        mock_ccxtpro.exchanges = ["binance", "coinbase"]

        config = ExchangeConfig(
            exchange_id="unsupported_exchange",
            api_key="valid-key",
            secret="valid-secret",
        )
        adapter = CCXTExchangeAdapter(config=config)

        with pytest.raises(ValueError):
            await adapter.authorize()

        assert adapter.is_connected is False

    @pytest.mark.asyncio
    @patch("src.adapters.ccxt_adapter.ccxtpro")
    @patch("src.adapters.ccxt_adapter.ccxt")
    async def test_sandbox_mode_enabled_on_both_clients(
        self,
        mock_ccxt,
        mock_ccxtpro,
        mock_rest_client,
        mock_ws_client,
    ):
        """When sandbox is set to True, both REST and WebSocket clients configure testnet."""
        sandbox_config = ExchangeConfig(
            exchange_id="binance",
            api_key="test-api-key",
            secret="test-secret-key",
            sandbox=True,
        )

        setattr(mock_ccxt, "binance", MagicMock(return_value=mock_rest_client))
        setattr(mock_ccxtpro, "binance", MagicMock(return_value=mock_ws_client))
        mock_ccxt.exchanges = ["binance"]
        mock_ccxtpro.exchanges = ["binance"]

        adapter = CCXTExchangeAdapter(config=sandbox_config)
        await adapter.authorize()

        mock_rest_client.set_sandbox_mode.assert_called_once_with(True)
        mock_ws_client.set_sandbox_mode.assert_called_once_with(True)
        assert adapter.is_connected is True

    @pytest.mark.asyncio
    @patch("src.adapters.ccxt_adapter.ccxtpro")
    @patch("src.adapters.ccxt_adapter.ccxt")
    async def test_disconnect_cleans_up_client_sessions(
        self,
        mock_ccxt,
        mock_ccxtpro,
        valid_config: ExchangeConfig,
        mock_rest_client,
        mock_ws_client,
    ):
        """Disconnecting the adapter safely terminates REST and WebSocket sessions."""
        setattr(mock_ccxt, "binance", MagicMock(return_value=mock_rest_client))
        setattr(mock_ccxtpro, "binance", MagicMock(return_value=mock_ws_client))
        mock_ccxt.exchanges = ["binance"]
        mock_ccxtpro.exchanges = ["binance"]

        adapter = CCXTExchangeAdapter(config=valid_config)
        await adapter.authorize()
        assert adapter.is_connected is True

        await adapter.disconnect()

        mock_rest_client.close.assert_awaited_once()
        mock_ws_client.close.assert_awaited_once()
        assert adapter.is_connected is False
        assert adapter.is_rest_connected is False
        assert adapter.is_ws_connected is False

    @pytest.mark.asyncio
    async def test_streaming_before_authorization_raises_authentication_error(
        self,
        valid_config: ExchangeConfig,
    ):
        """Attempting to stream orderbook/trades before authorization raises AuthenticationError."""
        adapter = CCXTExchangeAdapter(config=valid_config)

        with pytest.raises(AuthenticationError):
            await adapter.subscribe_order_book("BTC/USDT")