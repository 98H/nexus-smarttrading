"""Configuration models for exchange adapters."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ExchangeConfig:
    """Configuration parameters for exchange adapter initialization."""

    exchange_id: str
    api_key: str
    secret: str
    password: Optional[str] = None
    sandbox: bool = False

    def __post_init__(self) -> None:
        """Validate required configuration attributes."""
        if not self.exchange_id or not isinstance(self.exchange_id, str) or not self.exchange_id.strip():
            raise ValueError("exchange_id cannot be empty or whitespace")
        if not self.api_key or not isinstance(self.api_key, str) or not self.api_key.strip():
            raise ValueError("api_key cannot be empty or whitespace")
        if not self.secret or not isinstance(self.secret, str) or not self.secret.strip():
            raise ValueError("secret cannot be empty or whitespace")