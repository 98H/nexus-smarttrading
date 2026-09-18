import pytest
from typing import Any, List
from src.client.subscription_mux import SubscriptionMultiplexer, DataFrame


class MessageListener:
    """Helper callable object to record received payloads for testing."""

    def __init__(self) -> None:
        self.received_payloads: List[Any] = []

    def __call__(self, payload: Any) -> None:
        self.received_payloads.append(payload)

    @property
    def call_count(self) -> int:
        return len(self.received_payloads)

    @property
    def last_payload(self) -> Any:
        return self.received_payloads[-1] if self.received_payloads else None


@pytest.fixture
def mux() -> SubscriptionMultiplexer:
    """Provides a fresh SubscriptionMultiplexer instance for each test."""
    return SubscriptionMultiplexer()


# ==============================================================================
# AC 1: Subscribing & ID Registration & Listener Mapping
# ==============================================================================

def test_subscribe_returns_valid_unique_subscription_ids(mux: SubscriptionMultiplexer) -> None:
    """
    Given an active SubscriptionMultiplexer instance,
    When multiple client handlers subscribe to distinct channels/topics,
    Then distinct subscription IDs are registered.
    """
    listener_orders = MessageListener()
    listener_trades = MessageListener()
    listener_quotes = MessageListener()

    sub_id_1 = mux.subscribe("market.orders", listener_orders)
    sub_id_2 = mux.subscribe("market.trades", listener_trades)
    sub_id_3 = mux.subscribe("market.quotes", listener_quotes)

    # Validate IDs are returned and non-empty
    assert isinstance(sub_id_1, str) and len(sub_id_1.strip()) > 0
    assert isinstance(sub_id_2, str) and len(sub_id_2.strip()) > 0
    assert isinstance(sub_id_3, str) and len(sub_id_3.strip()) > 0

    # Validate all generated subscription IDs are strictly unique
    registered_ids = {sub_id_1, sub_id_2, sub_id_3}
    assert len(registered_ids) == 3


def test_subscribe_multiple_handlers_same_channel_yields_distinct_ids(
    mux: SubscriptionMultiplexer,
) -> None:
    """
    When multiple listeners subscribe to the exact same channel,
    Then each subscription must still receive a distinct subscription ID.
    """
    listener_a = MessageListener()
    listener_b = MessageListener()

    sub_id_a = mux.subscribe("telemetry.system", listener_a)
    sub_id_b = mux.subscribe("telemetry.system", listener_b)

    assert sub_id_a != sub_id_b
    assert mux.is_subscribed(sub_id_a) is True
    assert mux.is_subscribed(sub_id_b) is True


def test_subscribe_registers_listener_mapping(mux: SubscriptionMultiplexer) -> None:
    """
    Verifies that the multiplexer correctly registers the listener mapping
    and reflects active subscription state.
    """
    listener = MessageListener()
    sub_id = mux.subscribe("sensor.temperature", listener)

    assert mux.is_subscribed(sub_id) is True


def test_subscribe_with_empty_or_invalid_channel_raises_error(
    mux: SubscriptionMultiplexer,
) -> None:
    """
    Subscribing with an empty or non-string channel identifier must raise a ValueError.
    """
    listener = MessageListener()

    with pytest.raises(ValueError):
        mux.subscribe("", listener)

    with pytest.raises(ValueError):
        mux.subscribe("   ", listener)


def test_subscribe_with_non_callable_callback_raises_error(
    mux: SubscriptionMultiplexer,
) -> None:
    """
    Subscribing with an uncallable object instead of a callback must raise a TypeError.
    """
    with pytest.raises(TypeError):
        mux.subscribe("test.channel", None)  # type: ignore[arg-type]

    with pytest.raises(TypeError):
        mux.subscribe("test.channel", "not_a_callable")  # type: ignore[arg-type]


# ==============================================================================
# AC 2: Demultiplexing & Sole Payload Routing
# ==============================================================================

def test_demultiplex_routes_payload_solely_to_matching_listener(
    mux: SubscriptionMultiplexer,
) -> None:
    """
    Given an incoming multiplexed data frame with a specific subscription ID,
    When processed by the demultiplexer,
    Then the payload is routed solely to the listener callback registered for that ID.
    """
    listener_a = MessageListener()
    listener_b = MessageListener()
    listener_c = MessageListener()

    sub_id_a = mux.subscribe("channel.alpha", listener_a)
    sub_id_b = mux.subscribe("channel.beta", listener_b)
    sub_id_c = mux.subscribe("channel.gamma", listener_c)

    payload_b = {"symbol": "BTC/USD", "rate": 50000}
    frame = DataFrame(subscription_id=sub_id_b, payload=payload_b)

    mux.demultiplex(frame)

    # Only listener B should have received the payload
    assert listener_b.call_count == 1
    assert listener_b.last_payload == payload_b

    # Listeners A and C must not have been invoked
    assert listener_a.call_count == 0
    assert listener_c.call_count == 0


def test_demultiplex_handles_sequential_messages_to_different_subscribers(
    mux: SubscriptionMultiplexer,
) -> None:
    """
    When consecutive frames for different subscriptions are processed,
    Each payload is accurately delivered to its designated listener in arrival order.
    """
    listener_1 = MessageListener()
    listener_2 = MessageListener()

    sub_id_1 = mux.subscribe("topic.1", listener_1)
    sub_id_2 = mux.subscribe("topic.2", listener_2)

    frames = [
        DataFrame(subscription_id=sub_id_1, payload="payload-1A"),
        DataFrame(subscription_id=sub_id_2, payload="payload-2A"),
        DataFrame(subscription_id=sub_id_1, payload="payload-1B"),
        DataFrame(subscription_id=sub_id_2, payload="payload-2B"),
    ]

    for frame in frames:
        mux.demultiplex(frame)

    assert listener_1.received_payloads == ["payload-1A", "payload-1B"]
    assert listener_2.received_payloads == ["payload-2A", "payload-2B"]


@pytest.mark.parametrize(
    "payload",
    [
        {"dict_key": "dict_val", "nested": [1, 2, 3]},
        "string_payload",
        12345,
        3.14159,
        b"raw_bytes_payload",
        [True, False, None],
        None,
    ],
)
def test_demultiplex_preserves_payload_types_and_values(
    mux: SubscriptionMultiplexer, payload: Any
) -> None:
    """
    The demultiplexer must route payloads unchanged without mutation or data loss,
    including None and arbitrary data types.
    """
    listener = MessageListener()
    sub_id = mux.subscribe("channel.echo", listener)

    frame = DataFrame(subscription_id=sub_id, payload=payload)
    mux.demultiplex(frame)

    assert listener.call_count == 1
    assert listener.last_payload == payload


def test_demultiplex_unknown_subscription_id_is_dropped_safely(
    mux: SubscriptionMultiplexer,
) -> None:
    """
    Given an incoming frame with a subscription ID that does not exist,
    When processed by the demultiplexer,
    Then the message is safely dropped and does not affect other registered listeners.
    """
    listener = MessageListener()
    sub_id = mux.subscribe("active.channel", listener)

    unknown_frame = DataFrame(
        subscription_id="unregistered-sub-id-9999",
        payload={"message": "ghost payload"},
    )

    # Should not raise exception and should safely drop
    mux.demultiplex(unknown_frame)

    assert listener.call_count == 0
    assert mux.is_subscribed(sub_id) is True


# ==============================================================================
# AC 3: Unsubscribing & Dropping Subsequent Incoming Messages
# ==============================================================================

def test_unsubscribe_removes_listener_and_drops_subsequent_messages(
    mux: SubscriptionMultiplexer,
) -> None:
    """
    Given an existing subscription ID,
    When the client unsubscribes,
    Then the multiplexer removes the listener and drops subsequent incoming
    messages intended for that subscription ID.
    """
    listener = MessageListener()
    sub_id = mux.subscribe("alerts.security", listener)

    # First frame arrives while subscribed
    initial_frame = DataFrame(subscription_id=sub_id, payload={"level": "warning"})
    mux.demultiplex(initial_frame)
    assert listener.call_count == 1
    assert listener.last_payload == {"level": "warning"}

    # Client unsubscribes
    mux.unsubscribe(sub_id)

    assert mux.is_subscribed(sub_id) is False

    # Second frame arrives after unsubscribe
    subsequent_frame = DataFrame(subscription_id=sub_id, payload={"level": "critical"})
    mux.demultiplex(subsequent_frame)

    # Listener must NOT have received the subsequent payload
    assert listener.call_count == 1


def test_unsubscribe_one_subscription_leaves_other_subscriptions_intact(
    mux: SubscriptionMultiplexer,
) -> None:
    """
    Unsubscribing one ID must strictly remove only that listener,
    leaving any other active subscriptions functioning normally.
    """
    listener_retain = MessageListener()
    listener_remove = MessageListener()

    sub_id_retain = mux.subscribe("stream.retained", listener_retain)
    sub_id_remove = mux.subscribe("stream.removed", listener_remove)

    mux.unsubscribe(sub_id_remove)

    assert mux.is_subscribed(sub_id_remove) is False
    assert mux.is_subscribed(sub_id_retain) is True

    # Frame for unsubscribed ID is dropped
    mux.demultiplex(DataFrame(subscription_id=sub_id_remove, payload="dropped"))
    assert listener_remove.call_count == 0

    # Frame for retained ID is delivered
    mux.demultiplex(DataFrame(subscription_id=sub_id_retain, payload="delivered"))
    assert listener_retain.call_count == 1
    assert listener_retain.last_payload == "delivered"


def test_unsubscribe_nonexistent_subscription_raises_error(
    mux: SubscriptionMultiplexer,
) -> None:
    """
    Attempting to unsubscribe an unknown or invalid subscription ID
    must raise a KeyError or ValueError.
    """
    with pytest.raises((KeyError, ValueError)):
        mux.unsubscribe("non-existent-sub-id")


def test_unsubscribe_already_unsubscribed_id_raises_error(
    mux: SubscriptionMultiplexer,
) -> None:
    """
    Unsubscribing an ID that was already unsubscribed must raise
    a KeyError or ValueError.
    """
    listener = MessageListener()
    sub_id = mux.subscribe("temp.stream", listener)

    mux.unsubscribe(sub_id)

    with pytest.raises((KeyError, ValueError)):
        mux.unsubscribe(sub_id)


def test_resubscribe_to_same_channel_creates_independent_subscription(
    mux: SubscriptionMultiplexer,
) -> None:
    """
    Unsubscribing from a channel and subsequently subscribing again
    must allocate a new subscription ID, and messages sent to the old ID
    must not route to the new listener.
    """
    listener_first = MessageListener()
    listener_second = MessageListener()

    sub_id_first = mux.subscribe("events.lifecycle", listener_first)
    mux.unsubscribe(sub_id_first)

    sub_id_second = mux.subscribe("events.lifecycle", listener_second)
    assert sub_id_second != sub_id_first

    # Send message to obsolete sub_id_first -> must be dropped
    mux.demultiplex(DataFrame(subscription_id=sub_id_first, payload="old_msg"))
    assert listener_first.call_count == 0
    assert listener_second.call_count == 0

    # Send message to new sub_id_second -> must be delivered
    mux.demultiplex(DataFrame(subscription_id=sub_id_second, payload="new_msg"))
    assert listener_first.call_count == 0
    assert listener_second.call_count == 1
    assert listener_second.last_payload == "new_msg"


# ==============================================================================
# Edge Cases & Robustness
# ==============================================================================

def test_demultiplex_on_empty_multiplexer_does_not_crash(
    mux: SubscriptionMultiplexer,
) -> None:
    """
    An active multiplexer with zero subscriptions must gracefully handle
    demultiplex calls without raising exceptions.
    """
    frame = DataFrame(subscription_id="any-id", payload={"data": 123})
    mux.demultiplex(frame)


def test_listener_exception_does_not_corrupt_subsequent_routing(
    mux: SubscriptionMultiplexer,
) -> None:
    """
    If a listener callback raises an exception during execution,
    the exception should propagate or be handled without corrupting
    the multiplexer's internal state for other subscribers.
    """
    def faulty_callback(payload: Any) -> None:
        raise RuntimeError("Callback failure")

    listener_healthy = MessageListener()

    sub_faulty = mux.subscribe("channel.faulty", faulty_callback)
    sub_healthy = mux.subscribe("channel.healthy", listener_healthy)

    frame_faulty = DataFrame(subscription_id=sub_faulty, payload="error")
    with pytest.raises(RuntimeError):
        mux.demultiplex(frame_faulty)

    # Next frame to healthy listener must still route cleanly
    frame_healthy = DataFrame(subscription_id=sub_healthy, payload="success")
    mux.demultiplex(frame_healthy)

    assert listener_healthy.call_count == 1
    assert listener_healthy.last_payload == "success"