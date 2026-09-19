from __future__ import annotations

from typing import Any, Optional
from urllib.parse import urlparse

from src.webhooks.dlq import DeadLetterQueue
from src.webhooks.models import WebhookDelivery, WebhookDeliveryStatus


class WebhookDispatcher:
    """Dispatches outbound webhooks with retry management and DLQ routing."""

    def __init__(
        self,
        http_client: Any = None,
        dlq: Optional[DeadLetterQueue] = None,
        timeout: float = 10.0,
    ) -> None:
        self.http_client = http_client
        self.dlq = dlq if dlq is not None else DeadLetterQueue()
        self.timeout = timeout

    def dispatch(self, delivery: WebhookDelivery) -> WebhookDelivery:
        """Execute HTTP delivery for a webhook payload, managing retries and DLQ routing."""
        self._validate_delivery(delivery)

        delivery.attempt_count += 1

        try:
            if self.http_client is None:
                raise RuntimeError("HTTP client is not configured for WebhookDispatcher")

            response = self.http_client.post(
                delivery.url,
                json=delivery.payload,
                timeout=self.timeout,
            )
            status_code = getattr(response, "status_code", None)
            if status_code is not None and 200 <= status_code < 300:
                delivery.status = WebhookDeliveryStatus.SUCCEEDED
                delivery.last_error = None
                return delivery

            text = getattr(response, "text", "")
            error_msg = f"HTTP {status_code}: {text}".strip() if text else f"HTTP {status_code}"
        except Exception as exc:
            error_msg = str(exc) if str(exc) else exc.__class__.__name__

        self._handle_failure(delivery, error_msg)
        return delivery

    def _validate_delivery(self, delivery: WebhookDelivery) -> None:
        """Validate delivery status and destination URL before attempting delivery."""
        if delivery.status in (WebhookDeliveryStatus.SUCCEEDED, WebhookDeliveryStatus.FAILED):
            raise ValueError(
                f"Delivery '{delivery.id}' is already in terminal status: {delivery.status.value}"
            )
        self._validate_url(delivery.url)

    def _validate_url(self, url: str) -> None:
        """Ensure destination URL has a valid HTTP/HTTPS scheme and network location."""
        if not url or not isinstance(url, str):
            raise ValueError(f"Invalid URL: {url!r}")
        parsed = urlparse(url)
        if parsed.scheme.lower() not in ("http", "https") or not parsed.netloc:
            raise ValueError(f"Invalid URL: {url!r}. Scheme must be http or https with a host.")

    def _handle_failure(self, delivery: WebhookDelivery, error_msg: str) -> None:
        """Update delivery status on failure and enqueue to DLQ if retries are exhausted."""
        delivery.last_error = error_msg
        if delivery.attempt_count < delivery.max_retries:
            delivery.status = WebhookDeliveryStatus.RETRY_SCHEDULED
            delivery.metadata["retry_pending"] = True
            delivery.metadata["next_attempt"] = delivery.attempt_count + 1
        else:
            delivery.status = WebhookDeliveryStatus.FAILED
            delivery.metadata["retry_pending"] = False
            self.dlq.enqueue(
                delivery_id=delivery.id,
                payload=delivery.payload,
                reason=error_msg,
            )