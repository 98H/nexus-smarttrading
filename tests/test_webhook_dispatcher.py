from unittest.mock import MagicMock
import pytest

from src.webhooks.dispatcher import WebhookDispatcher
from src.webhooks.dlq import DeadLetterQueue, DLQRecord
from src.webhooks.models import WebhookDelivery, WebhookDeliveryStatus


# =====================================================================
# Test Fixtures & Helpers
# =====================================================================

def make_response(status_code: int, text: str = ""):
    """Helper to simulate an HTTP response object."""
    response = MagicMock()
    response.status_code = status_code
    response.text = text or f"Response with status {status_code}"
    response.ok = 200 <= status_code < 300
    return response


@pytest.fixture
def mock_http_client():
    """Mock HTTP client that can simulate network responses and failures."""
    return MagicMock()


@pytest.fixture
def dlq():
    """Real Dead-Letter Queue instance for integration testing."""
    return DeadLetterQueue()


@pytest.fixture
def dispatcher(mock_http_client, dlq):
    """WebhookDispatcher instance configured with mock client and real DLQ."""
    return WebhookDispatcher(http_client=mock_http_client, dlq=dlq)


@pytest.fixture
def valid_delivery():
    """Standard outbound webhook delivery fixture."""
    return WebhookDelivery(
        id="wh-delivery-001",
        url="https://api.partner.com/webhooks",
        payload={"event": "order.completed", "order_id": "ord_98765"},
        max_retries=3,
        attempt_count=0,
        status=WebhookDeliveryStatus.PENDING,
    )


# =====================================================================
# Acceptance Criteria 1: Successful Delivery (2xx)
# =====================================================================

class TestSuccessfulDelivery:
    """Tests for AC: 2xx response marks status SUCCEEDED and does not enqueue to DLQ."""

    @pytest.mark.parametrize("status_code", [200, 201, 202, 204])
    def test_dispatch_2xx_marks_status_succeeded_and_skips_dlq(
        self, dispatcher, mock_http_client, dlq, valid_delivery, status_code
    ):
        mock_http_client.post.return_value = make_response(status_code)

        result = dispatcher.dispatch(valid_delivery)

        assert result.status == WebhookDeliveryStatus.SUCCEEDED
        assert valid_delivery.status == WebhookDeliveryStatus.SUCCEEDED
        assert result.attempt_count == 1
        assert len(dlq) == 0
        assert dlq.size() == 0

    def test_dispatch_success_does_not_alter_original_payload_or_url(
        self, dispatcher, mock_http_client, valid_delivery
    ):
        mock_http_client.post.return_value = make_response(200, "OK")
        original_payload = dict(valid_delivery.payload)
        original_url = valid_delivery.url

        dispatcher.dispatch(valid_delivery)

        assert valid_delivery.payload == original_payload
        assert valid_delivery.url == original_url

    def test_dispatch_success_clears_or_leaves_no_error_recorded(
        self, dispatcher, mock_http_client, valid_delivery
    ):
        mock_http_client.post.return_value = make_response(200, "OK")

        dispatcher.dispatch(valid_delivery)

        assert valid_delivery.last_error is None

    def test_dispatch_invokes_http_post_with_expected_arguments(
        self, dispatcher, mock_http_client, valid_delivery
    ):
        mock_http_client.post.return_value = make_response(200)

        dispatcher.dispatch(valid_delivery)

        mock_http_client.post.assert_called_once_with(
            valid_delivery.url,
            json=valid_delivery.payload,
            timeout=dispatcher.timeout,
        )


# =====================================================================
# Acceptance Criteria 2: Retryable Failure (5xx or Timeout)
# =====================================================================

class TestRetryableFailure:
    """Tests for AC: 5xx response or timeout increments attempt and schedules retry."""

    @pytest.mark.parametrize("status_code", [500, 502, 503, 504])
    def test_dispatch_5xx_increments_attempt_and_schedules_retry(
        self, dispatcher, mock_http_client, dlq, valid_delivery, status_code
    ):
        mock_http_client.post.return_value = make_response(
            status_code, f"Server Error {status_code}"
        )
        valid_delivery.attempt_count = 0
        valid_delivery.max_retries = 3

        result = dispatcher.dispatch(valid_delivery)

        assert result.status == WebhookDeliveryStatus.RETRY_SCHEDULED
        assert valid_delivery.status == WebhookDeliveryStatus.RETRY_SCHEDULED
        assert valid_delivery.attempt_count == 1
        assert str(status_code) in (valid_delivery.last_error or "")
        assert len(dlq) == 0

    def test_dispatch_timeout_increments_attempt_and_schedules_retry(
        self, dispatcher, mock_http_client, dlq, valid_delivery
    ):
        mock_http_client.post.side_effect = TimeoutError("Connection timed out after 5.0s")
        valid_delivery.attempt_count = 0
        valid_delivery.max_retries = 3

        result = dispatcher.dispatch(valid_delivery)

        assert result.status == WebhookDeliveryStatus.RETRY_SCHEDULED
        assert valid_delivery.attempt_count == 1
        assert "timed out" in (valid_delivery.last_error or "").lower()
        assert len(dlq) == 0

    def test_dispatch_intermediate_retry_increments_existing_attempt_count(
        self, dispatcher, mock_http_client, dlq, valid_delivery
    ):
        # Already attempted once, current attempt count is 1, max is 3
        valid_delivery.attempt_count = 1
        valid_delivery.max_retries = 3
        mock_http_client.post.return_value = make_response(503, "Unavailable")

        dispatcher.dispatch(valid_delivery)

        assert valid_delivery.attempt_count == 2
        assert valid_delivery.status == WebhookDeliveryStatus.RETRY_SCHEDULED
        assert len(dlq) == 0

    def test_retry_scheduled_records_next_retry_metadata(
        self, dispatcher, mock_http_client, valid_delivery
    ):
        mock_http_client.post.return_value = make_response(500)
        valid_delivery.attempt_count = 0

        result = dispatcher.dispatch(valid_delivery)

        # Implementation should track that retry is pending/scheduled
        assert result.status == WebhookDeliveryStatus.RETRY_SCHEDULED
        assert result.attempt_count < result.max_retries


# =====================================================================
# Acceptance Criteria 3: Exhausted Retries and DLQ Routing
# =====================================================================

class TestMaxRetriesExhaustedAndDLQ:
    """Tests for AC: Reaching max retries routes to DLQ and marks status FAILED."""

    def test_dispatch_final_5xx_failure_enqueues_to_dlq_and_marks_failed(
        self, dispatcher, mock_http_client, dlq, valid_delivery
    ):
        # Current attempt is 2 with max_retries=3; this failure is the 3rd and final attempt
        valid_delivery.attempt_count = 2
        valid_delivery.max_retries = 3
        mock_http_client.post.return_value = make_response(500, "Internal Server Error")

        result = dispatcher.dispatch(valid_delivery)

        assert result.status == WebhookDeliveryStatus.FAILED
        assert valid_delivery.status == WebhookDeliveryStatus.FAILED
        assert valid_delivery.attempt_count == 3
        assert len(dlq) == 1

        record = dlq.get_records()[0]
        assert isinstance(record, DLQRecord)
        assert record.delivery_id == valid_delivery.id
        assert record.payload == valid_delivery.payload
        assert "500" in record.error_reason

    def test_dispatch_final_timeout_failure_enqueues_to_dlq_and_marks_failed(
        self, dispatcher, mock_http_client, dlq, valid_delivery
    ):
        valid_delivery.attempt_count = 2
        valid_delivery.max_retries = 3
        mock_http_client.post.side_effect = TimeoutError("Network timeout on final attempt")

        result = dispatcher.dispatch(valid_delivery)

        assert result.status == WebhookDeliveryStatus.FAILED
        assert valid_delivery.status == WebhookDeliveryStatus.FAILED
        assert valid_delivery.attempt_count == 3
        assert len(dlq) == 1

        record = dlq.get_records()[0]
        assert record.delivery_id == valid_delivery.id
        assert record.payload == valid_delivery.payload
        assert "timeout" in record.error_reason.lower()

    def test_single_attempt_limit_fails_immediately_to_dlq_on_error(
        self, dispatcher, mock_http_client, dlq, valid_delivery
    ):
        valid_delivery.attempt_count = 0
        valid_delivery.max_retries = 1
        mock_http_client.post.return_value = make_response(503, "Service Down")

        result = dispatcher.dispatch(valid_delivery)

        assert result.status == WebhookDeliveryStatus.FAILED
        assert valid_delivery.attempt_count == 1
        assert len(dlq) == 1
        assert dlq.get_records()[0].payload == valid_delivery.payload

    def test_multiple_failed_webhooks_accumulate_in_dlq(
        self, dispatcher, mock_http_client, dlq
    ):
        mock_http_client.post.return_value = make_response(500, "Persistent failure")

        delivery_1 = WebhookDelivery(
            id="wh-001",
            url="https://api.partner.com/endpoint1",
            payload={"event": "event_1"},
            attempt_count=2,
            max_retries=3,
        )
        delivery_2 = WebhookDelivery(
            id="wh-002",
            url="https://api.partner.com/endpoint2",
            payload={"event": "event_2"},
            attempt_count=2,
            max_retries=3,
        )

        dispatcher.dispatch(delivery_1)
        dispatcher.dispatch(delivery_2)

        assert len(dlq) == 2
        records = dlq.get_records()
        assert records[0].delivery_id == "wh-001"
        assert records[0].payload == {"event": "event_1"}
        assert records[1].delivery_id == "wh-002"
        assert records[1].payload == {"event": "event_2"}


# =====================================================================
# Edge Cases & Validation Tests
# =====================================================================

class TestDispatcherEdgeCases:
    """Edge cases: invalid URLs, already-terminal status, and DLQ operations."""

    @pytest.mark.parametrize("invalid_url", ["", "ftp://invalid-scheme.com", "not-a-valid-url"])
    def test_dispatch_invalid_url_raises_value_error(
        self, dispatcher, valid_delivery, invalid_url
    ):
        valid_delivery.url = invalid_url
        with pytest.raises(ValueError):
            dispatcher.dispatch(valid_delivery)

    def test_dispatch_already_succeeded_delivery_raises_error(
        self, dispatcher, valid_delivery
    ):
        valid_delivery.status = WebhookDeliveryStatus.SUCCEEDED
        with pytest.raises(ValueError):
            dispatcher.dispatch(valid_delivery)

    def test_dispatch_already_failed_delivery_raises_error(
        self, dispatcher, valid_delivery
    ):
        valid_delivery.status = WebhookDeliveryStatus.FAILED
        with pytest.raises(ValueError):
            dispatcher.dispatch(valid_delivery)

    def test_dlq_peek_and_pop_behavior(self, dlq):
        payload = {"test": "data"}
        dlq.enqueue(delivery_id="del-1", payload=payload, reason="Connection Refused")

        assert dlq.size() == 1
        record = dlq.peek()
        assert record.delivery_id == "del-1"
        assert record.payload == payload
        assert record.error_reason == "Connection Refused"
        assert dlq.size() == 1

        popped = dlq.pop()
        assert popped.delivery_id == "del-1"
        assert dlq.size() == 0

    def test_dlq_pop_empty_queue_raises_error(self, dlq):
        with pytest.raises(IndexError):
            dlq.pop()