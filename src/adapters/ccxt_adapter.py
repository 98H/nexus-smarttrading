"""CCXT and CCXT Pro real-time exchange adapter integration."""

import inspect
from typing import Any

try:
    import ccxt.async_support as ccxt
except ImportError:
    try:
        import ccxt  # type: ignore
    except ImportError:
        ccxt = None  # type: ignore

try:
    import ccxt.pro as ccxtpro
except ImportError:
    try:
        import ccxtpro  # type: ignore
    except ImportError:
        ccxtpro = None  # type: ignore

from src.adapters.config import ExchangeConfig


class AuthenticationError(Exception):
    """Raised when exchange authentication or credentials verification fails."""


class CCXTExchangeAdapter:
    """
    Exchange adapter managing authenticated CCXT REST and CCXT Pro WebSocket sessions.
    """

    def __init__(self, config: ExchangeConfig) -> None:
        self.config = config
        self._rest_client: Any = None
        self._ws_client: Any = None
        self._is_rest_connected: bool = False
        self._is_ws_connected: bool = False

    @property
    def rest_client(self) -> Any:
        """Authenticated CCXT REST async client session."""
        return self._rest_client

    @property
    def ws_client(self) -> Any:
        """Authenticated CCXT Pro WebSocket async client session."""
        return self._ws_client

    @property
    def is_connected(self) -> bool:
        """True if both REST and WebSocket sessions are authorized and connected."""
        return self._is_rest_connected and self._is_ws_connected

    @property
    def is_rest_connected(self) -> bool:
        """True if REST session is authenticated and connected."""
        return self._is_rest_connected

    @property
    def is_ws_connected(self) -> bool:
        """True if WebSocket session is authenticated and connected."""
        return self._is_ws_connected

    async def authorize(self) -> None:
        """
        Authorize and establish authenticated REST and CCXT Pro WebSocket sessions.

        Raises:
            AuthenticationError: If credentials are missing, invalid, or rejected.
            ValueError: If exchange_id is not supported by CCXT or CCXT Pro.
        """
        # Defense against missing or unpopulated credentials
        if (
            not self.config.api_key
            or not str(self.config.api_key).strip()
            or not self.config.secret
            or not str(self.config.secret).strip()
        ):
            raise AuthenticationError("API key and secret must be provided for authorization.")

        exchange_id = self.config.exchange_id

        if ccxt is None or ccxtpro is None:
            raise ValueError("CCXT or CCXT Pro library is not available.")

        ccxt_exchanges = getattr(ccxt, "exchanges", None)
        if ccxt_exchanges is not None and exchange_id not in ccxt_exchanges:
            raise ValueError(f"Exchange '{exchange_id}' is not supported by CCXT.")

        ccxtpro_exchanges = getattr(ccxtpro, "exchanges", None)
        if ccxtpro_exchanges is not None and exchange_id not in ccxtpro_exchanges:
            raise ValueError(f"Exchange '{exchange_id}' is not supported by CCXT Pro.")

        if not hasattr(ccxt, exchange_id) or not hasattr(ccxtpro, exchange_id):
            raise ValueError(f"Exchange '{exchange_id}' is not available in CCXT or CCXT Pro.")

        rest_class = getattr(ccxt, exchange_id)
        ws_class = getattr(ccxtpro, exchange_id)

        client_params: dict[str, Any] = {
            "apiKey": self.config.api_key,
            "secret": self.config.secret,
        }
        if self.config.password is not None:
            client_params["password"] = self.config.password

        rest_client = rest_class(client_params)
        ws_client = ws_class(client_params)

        if self.config.sandbox:
            if hasattr(rest_client, "set_sandbox_mode"):
                rest_client.set_sandbox_mode(True)
            if hasattr(ws_client, "set_sandbox_mode"):
                ws_client.set_sandbox_mode(True)

        try:
            # REST credentials verification and authenticated probe
            if hasattr(rest_client, "check_required_credentials"):
                res = rest_client.check_required_credentials()
                if res is False:
                    raise AuthenticationError("REST credentials validation failed.")

            if hasattr(rest_client, "load_markets"):
                res = rest_client.load_markets()
                if inspect.isawaitable(res):
                    await res

            if hasattr(rest_client, "fetch_balance"):
                res = rest_client.fetch_balance()
                if inspect.isawaitable(res):
                    await res

            # WebSocket credentials verification and market setup
            if hasattr(ws_client, "check_required_credentials"):
                res = ws_client.check_required_credentials()
                if res is False:
                    raise AuthenticationError("WebSocket credentials validation failed.")

            if hasattr(ws_client, "load_markets"):
                res = ws_client.load_markets()
                if inspect.isawaitable(res):
                    await res

            self._rest_client = rest_client
            self._ws_client = ws_client
            self._is_rest_connected = True
            self._is_ws_connected = True

        except AuthenticationError:
            await self._cleanup(rest_client, ws_client)
            raise
        except Exception as exc:
            await self._cleanup(rest_client, ws_client)
            if "AuthenticationError" in exc.__class__.__name__ or any(
                "AuthenticationError" in base.__name__ for base in exc.__class__.__mro__
            ):
                raise AuthenticationError(str(exc)) from exc
            raise

    async def disconnect(self) -> None:
        """Safely close and clean up active REST and WebSocket sessions."""
        rest_client = self._rest_client
        ws_client = self._ws_client

        self._rest_client = None
        self._ws_client = None
        self._is_rest_connected = False
        self._is_ws_connected = False

        errors: list[Exception] = []
        if rest_client is not None and hasattr(rest_client, "close"):
            try:
                res = rest_client.close()
                if inspect.isawaitable(res):
                    await res
            except Exception as exc:
                errors.append(exc)

        if ws_client is not None and hasattr(ws_client, "close"):
            try:
                res = ws_client.close()
                if inspect.isawaitable(res):
                    await res
            except Exception as exc:
                errors.append(exc)

        if errors:
            raise errors[0]

    async def _cleanup(self, rest_client: Any = None, ws_client: Any = None) -> None:
        """Clean up client instances if authorization fails."""
        self._rest_client = None
        self._ws_client = None
        self._is_rest_connected = False
        self._is_ws_connected = False

        if rest_client is not None and hasattr(rest_client, "close"):
            try:
                res = rest_client.close()
                if inspect.isawaitable(res):
                    await res
            except Exception:
                pass

        if ws_client is not None and hasattr(ws_client, "close"):
            try:
                res = ws_client.close()
                if inspect.isawaitable(res):
                    await res
            except Exception:
                pass

    async def subscribe_order_book(self, symbol: str, *args: Any, **kwargs: Any) -> Any:
        """Subscribe to real-time order book WebSocket stream."""
        if not self.is_ws_connected or self._ws_client is None:
            raise AuthenticationError("WebSocket client is not authorized or connected.")
        return await self._ws_client.watch_order_book(symbol, *args, **kwargs)

    async def subscribe_ticker(self, symbol: str, *args: Any, **kwargs: Any) -> Any:
        """Subscribe to real-time ticker WebSocket stream."""
        if not self.is_ws_connected or self._ws_client is None:
            raise AuthenticationError("WebSocket client is not authorized or connected.")
        return await self._ws_client.watch_ticker(symbol, *args, **kwargs)

    async def subscribe_trades(self, symbol: str, *args: Any, **kwargs: Any) -> Any:
        """Subscribe to real-time trades WebSocket stream."""
        if not self.is_ws_connected or self._ws_client is None:
            raise AuthenticationError("WebSocket client is not authorized or connected.")
        return await self._ws_client.watch_trades(symbol, *args, **kwargs)