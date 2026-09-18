import json
from unittest.mock import MagicMock, call
import pytest

from src.workers import RuleEvaluationWorker
from src.workers.rule_evaluation_worker import (
    RuleEvaluationWorker as DirectRuleEvaluationWorker,
    InvalidMessageError,
    MessageParsingError,
)


@pytest.fixture
def mock_redis_client():
    """Mock Redis client and its pubsub interface for external I/O isolation."""
    client = MagicMock()
    pubsub_mock = MagicMock()
    client.pubsub.return_value = pubsub_mock
    return client


@pytest.fixture
def worker(mock_redis_client):
    """Instantiate a RuleEvaluationWorker with default configuration."""
    return RuleEvaluationWorker(redis_client=mock_redis_client)


class TestRuleEvaluationWorkerInitialization:
    """Tests worker instantiation and package exports."""

    def test_module_exports_worker_class(self):
        """Worker class should be importable from both package root and module."""
        assert RuleEvaluationWorker is DirectRuleEvaluationWorker

    def test_default_channel_is_rule_events(self, worker):
        """Worker must default to subscribing to the 'rule_events' channel."""
        assert worker.channel == "rule_events"

    def test_custom_channel_assignment(self, mock_redis_client):
        """Worker should accept custom channel configuration."""
        custom_worker = RuleEvaluationWorker(
            redis_client=mock_redis_client, channel="custom_events"
        )
        assert custom_worker.channel == "custom_events"

    def test_worker_initial_state(self, worker, mock_redis_client):
        """Worker state should be properly initialized before subscription."""
        assert worker.redis_client == mock_redis_client
        assert worker.is_running is False
        assert worker.pubsub is not None


class TestRuleEvaluationWorkerSubscription:
    """Acceptance Criteria 1: Given a running Redis instance and a configured

    RuleEvaluationWorker subscribed to the 'rule_events' channel.
    """

    def test_subscribe_subscribes_to_rule_events_channel(self, worker):
        """Worker.subscribe() should invoke Redis pubsub.subscribe with 'rule_events'."""
        worker.subscribe()

        worker.pubsub.subscribe.assert_called_once_with("rule_events")
        assert worker.is_subscribed is True

    def test_subscribe_twice_idempotent(self, worker):
        """Calling subscribe multiple times should not create duplicate subscriptions."""
        worker.subscribe()
        worker.subscribe()

        worker.pubsub.subscribe.assert_called_once_with("rule_events")

    def test_unsubscribe_cleans_up_subscription(self, worker):
        """Worker should support cleanly unsubscribing from the channel."""
        worker.subscribe()
        worker.unsubscribe()

        worker.pubsub.unsubscribe.assert_called_once_with("rule_events")
        assert worker.is_subscribed is False


class TestRuleEvaluationWorkerMessageParsing:
    """Acceptance Criteria 2: Given an invalid or unparseable message published to

    the 'rule_events' channel.
    """

    def test_parse_message_valid_json_string(self, worker):
        """Valid JSON string should parse into a dictionary."""
        valid_payload = json.dumps({"rule_id": "rule-101", "event": "user_signup"})
        parsed = worker.parse_message(valid_payload)
        assert parsed == {"rule_id": "rule-101", "event": "user_signup"}

    def test_parse_message_valid_json_bytes(self, worker):
        """Valid JSON bytes should parse into a dictionary."""
        valid_payload = json.dumps({"rule_id": "rule-102", "value": 42}).encode("utf-8")
        parsed = worker.parse_message(valid_payload)
        assert parsed == {"rule_id": "rule-102", "value": 42}

    def test_parse_message_malformed_json_raises_message_parsing_error(self, worker):
        """Malformed JSON payload must raise MessageParsingError."""
        malformed_payload = '{"rule_id": "rule-101", missing_closing_bracket'
        with pytest.raises(MessageParsingError):
            worker.parse_message(malformed_payload)

    def test_parse_message_empty_payload_raises_message_parsing_error(self, worker):
        """Empty string or empty bytes payload must raise MessageParsingError."""
        with pytest.raises(MessageParsingError):
            worker.parse_message("")

        with pytest.raises(MessageParsingError):
            worker.parse_message(b"")

    def test_parse_message_invalid_payload_type_raises_message_parsing_error(self, worker):
        """Non-string/non-bytes payloads must raise MessageParsingError."""
        with pytest.raises(MessageParsingError):
            worker.parse_message(None)

        with pytest.raises(MessageParsingError):
            worker.parse_message(12345)

    def test_parse_message_non_dict_json_raises_invalid_message_error(self, worker):
        """Valid JSON that evaluates to a non-dict type (e.g., list or scalar) must raise InvalidMessageError."""
        list_payload = json.dumps(["rule-1", "rule-2"])
        scalar_payload = json.dumps("just a string")

        with pytest.raises(InvalidMessageError):
            worker.parse_message(list_payload)

        with pytest.raises(InvalidMessageError):
            worker.parse_message(scalar_payload)

    def test_validate_message_schema_missing_required_fields_raises_invalid_message_error(self, worker):
        """Message missing mandatory fields like 'rule_id' must raise InvalidMessageError."""
        incomplete_message = {"timestamp": 1699999999}
        with pytest.raises(InvalidMessageError):
            worker.validate_message(incomplete_message)


class TestRuleEvaluationWorkerHandlingAndResilience:
    """Tests worker message dispatching, error tolerance, and recovery on invalid inputs."""

    def test_handle_unparseable_message_does_not_crash_worker(self, worker):
        """Worker handle_message must catch MessageParsingError and return a failure result instead of crashing."""
        raw_pubsub_message = {
            "type": "message",
            "channel": "rule_events",
            "data": "{corrupt_json: true",
        }

        result = worker.handle_message(raw_pubsub_message)

        assert result is not None
        assert result.get("success") is False
        assert "error" in result

    def test_handle_invalid_schema_message_does_not_crash_worker(self, worker):
        """Worker handle_message must catch InvalidMessageError and return failure without crashing."""
        raw_pubsub_message = {
            "type": "message",
            "channel": "rule_events",
            "data": json.dumps({"unrecognized": "payload"}),
        }

        result = worker.handle_message(raw_pubsub_message)

        assert result is not None
        assert result.get("success") is False
        assert "error" in result

    def test_handle_unparseable_message_routes_to_dlq_when_configured(self, mock_redis_client):
        """When DLQ channel is configured, unparseable messages should be published to the dead-letter queue."""
        worker_with_dlq = RuleEvaluationWorker(
            redis_client=mock_redis_client,
            channel="rule_events",
            dlq_channel="rule_events_dlq",
        )
        invalid_data = b"unparseable-data-payload"
        raw_pubsub_message = {
            "type": "message",
            "channel": "rule_events",
            "data": invalid_data,
        }

        worker_with_dlq.handle_message(raw_pubsub_message)

        mock_redis_client.publish.assert_called_once()
        published_channel, published_payload = mock_redis_client.publish.call_args[0]
        assert published_channel == "rule_events_dlq"
        assert "unparseable-data-payload" in str(published_payload)

    def test_ignore_redis_non_message_events(self, worker):
        """Redis system messages (e.g. subscribe acknowledgements) should be safely ignored."""
        system_pubsub_message = {
            "type": "subscribe",
            "channel": "rule_events",
            "data": 1,
        }

        result = worker.handle_message(system_pubsub_message)
        assert result is None

    def test_process_next_message_continues_after_invalid_message(self, worker):
        """Worker processing loop step continues processing subsequent valid messages after an invalid one."""
        invalid_pubsub_message = {
            "type": "message",
            "channel": "rule_events",
            "data": "{bad_payload",
        }
        valid_pubsub_message = {
            "type": "message",
            "channel": "rule_events",
            "data": json.dumps({"rule_id": "rule-404", "event_data": {"active": True}}),
        }

        worker.pubsub.get_message.side_effect = [
            invalid_pubsub_message,
            valid_pubsub_message,
            None,
        ]

        # First cycle: invalid message handled without raising
        result_first = worker.process_next_message()
        assert result_first is not None
        assert result_first.get("success") is False

        # Second cycle: valid message processed successfully
        result_second = worker.process_next_message()
        assert result_second is not None
        assert result_second.get("success") is True
        assert result_second.get("rule_id") == "rule-404"