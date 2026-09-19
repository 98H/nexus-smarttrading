from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class WebhookDeliveryStatus(str, Enum):
    """Lifecycle status of an outbound webhook delivery."""

    PENDING = "PENDING"
    SUCCEEDED = "SUCCEEDED"
    RETRY_SCHEDULED = "RETRY_SCHEDULED"
    FAILED = "FAILED"


@dataclass
class WebhookDelivery:
    """Represents an outbound webhook delivery attempt and its state."""

    id: str
    url: str
    payload: Any
    max_retries: int = 3
    attempt_count: int = 0
    status: WebhookDeliveryStatus = WebhookDeliveryStatus.PENDING
    last_error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_terminal(self) -> bool:
        """Indicates whether the delivery has reached a terminal status."""
        return self.status in (
            WebhookDeliveryStatus.SUCCEEDED,
            WebhookDeliveryStatus.FAILED,
        )