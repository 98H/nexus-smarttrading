import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class MessageParsingError(Exception):
    """Raised when a message payload cannot be parsed as JSON."""


class InvalidMessageError(Exception):
    """Raised when a message schema or payload structure is invalid."""


class RuleEvaluationWorker:
    """Distributed worker that evaluates rules from Redis Pub/Sub events."""

    def __init__(
        self,
        redis_client: Any,
        channel: str = "rule_events",
        dlq_channel: Optional[str] = None,
    ) -> None:
        self.redis_client = redis_client
        self.channel = channel
        self.dlq_channel = dlq_channel
        self.pubsub = redis_client.pubsub()
        self.is_subscribed: bool = False
        self.is_running: bool = False

    def subscribe(self) -> None:
        """Subscribe to the configured Redis Pub/Sub channel idempotently."""
        if not self.is_subscribed:
            self.pubsub.subscribe(self.channel)
            self.is_subscribed = True
            logger.info("Subscribed to channel: %s", self.channel)

    def unsubscribe(self) -> None:
        """Unsubscribe from the Redis Pub/Sub channel cleanly."""
        if self.is_subscribed:
            self.pubsub.unsubscribe(self.channel)
            self.is_subscribed = False
            logger.info("Unsubscribed from channel: %s", self.channel)

    def close(self) -> None:
        """Clean up subscriptions and close the Redis Pub/Sub connection."""
        self.unsubscribe()
        if hasattr(self.pubsub, "close"):
            try:
                self.pubsub.close()
            except Exception as exc:
                logger.warning("Error closing pubsub connection: %s", exc)

    def parse_message(self, payload: Any) -> dict:
        """Parse raw payload into a dictionary.

        Raises:
            MessageParsingError: If payload is not str/bytes, is empty, or cannot be parsed as JSON.
            InvalidMessageError: If parsed JSON is not a dictionary.
        """
        if not isinstance(payload, (str, bytes)):
            raise MessageParsingError(
                f"Payload must be of type str or bytes, got {type(payload).__name__}"
            )

        if len(payload) == 0:
            raise MessageParsingError("Payload cannot be empty")

        try:
            parsed = json.loads(payload)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
            raise MessageParsingError(f"Failed to parse JSON payload: {exc}") from exc

        if not isinstance(parsed, dict):
            raise InvalidMessageError(
                f"Parsed JSON must be a dictionary, got {type(parsed).__name__}"
            )

        return parsed

    def validate_message(self, message: dict) -> None:
        """Validate message schema against required fields.

        Raises:
            InvalidMessageError: If message is not a dict or missing mandatory fields.
        """
        if not isinstance(message, dict):
            raise InvalidMessageError("Message must be a dictionary")

        if "rule_id" not in message or message["rule_id"] is None or message["rule_id"] == "":
            raise InvalidMessageError("Missing required field 'rule_id'")

    def evaluate_rule(self, message: dict) -> dict:
        """Evaluate the rule associated with the message."""
        rule_id = message["rule_id"]
        logger.info("Evaluating rule: %s", rule_id)
        return {
            "success": True,
            "rule_id": rule_id,
            "data": message,
        }

    def handle_message(self, raw_message: Optional[dict]) -> Optional[dict]:
        """Dispatch and handle a raw Redis Pub/Sub message safely."""
        if not isinstance(raw_message, dict):
            return None

        if raw_message.get("type") != "message":
            return None

        raw_data = raw_message.get("data")
        try:
            parsed = self.parse_message(raw_data)
            self.validate_message(parsed)
            return self.evaluate_rule(parsed)
        except (MessageParsingError, InvalidMessageError) as exc:
            logger.warning("Invalid message received: %s", exc)
            self._route_to_dlq(raw_data, exc)
            return {
                "success": False,
                "error": str(exc),
            }
        except Exception as exc:
            logger.exception("Unexpected error processing message: %s", exc)
            self._route_to_dlq(raw_data, exc)
            return {
                "success": False,
                "error": str(exc),
            }

    def process_next_message(self, **kwargs: Any) -> Optional[dict]:
        """Retrieve and process the next available message from Pub/Sub."""
        raw_message = self.pubsub.get_message(**kwargs)
        if raw_message is None:
            return None
        return self.handle_message(raw_message)

    def _route_to_dlq(self, raw_data: Any, error: Exception) -> None:
        """Route unparseable or invalid message to dead-letter queue if configured."""
        if not self.dlq_channel:
            return

        try:
            if isinstance(raw_data, bytes):
                payload_str = raw_data.decode("utf-8", errors="replace")
            elif isinstance(raw_data, str):
                payload_str = raw_data
            else:
                payload_str = str(raw_data)

            dlq_payload = json.dumps({
                "error": str(error),
                "payload": payload_str,
                "data": payload_str,
            })
            self.redis_client.publish(self.dlq_channel, dlq_payload)
            logger.info("Routed failed message to DLQ: %s", self.dlq_channel)
        except Exception as exc:
            logger.error("Failed to route message to DLQ channel '%s': %s", self.dlq_channel, exc)

    def run(self, timeout: float = 1.0) -> None:
        """Run worker processing loop with non-zero polling timeout to prevent busy-waiting."""
        self.is_running = True
        self.subscribe()
        try:
            while self.is_running:
                self.process_next_message(timeout=timeout)
        finally:
            self.is_running = False
            self.close()

    def stop(self) -> None:
        """Signal worker processing loop to stop."""
        self.is_running = False