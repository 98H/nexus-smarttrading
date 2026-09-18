"""Subscription multiplexer and demultiplexer for client-side messaging."""

from dataclasses import dataclass
from typing import Any, Callable, Dict
import uuid


@dataclass
class DataFrame:
    """Represents an incoming multiplexed message frame."""

    subscription_id: str
    payload: Any


class SubscriptionMultiplexer:
    """Manages subscriptions and routes demultiplexed payloads to designated listeners."""

    def __init__(self) -> None:
        self._subscriptions: Dict[str, Callable[[Any], None]] = {}

    def subscribe(self, channel: str, callback: Callable[[Any], None]) -> str:
        """
        Registers a listener callback for a specific channel and returns a unique subscription ID.

        :param channel: Channel identifier string.
        :param callback: Callable invoked with the frame payload when demultiplexed.
        :return: Unique subscription ID string.
        """
        if not isinstance(channel, str) or not channel.strip():
            raise ValueError("Channel must be a non-empty string.")
        if not callable(callback):
            raise TypeError("Callback must be callable.")

        subscription_id = uuid.uuid4().hex
        self._subscriptions[subscription_id] = callback
        return subscription_id

    def unsubscribe(self, subscription_id: str) -> None:
        """
        Removes an existing subscription by ID.

        :param subscription_id: Unique subscription ID to remove.
        :raises KeyError: If the subscription ID does not exist.
        """
        if subscription_id not in self._subscriptions:
            raise KeyError(f"Subscription ID '{subscription_id}' is not registered.")
        del self._subscriptions[subscription_id]

    def is_subscribed(self, subscription_id: str) -> bool:
        """Checks if a subscription ID is currently active."""
        return subscription_id in self._subscriptions

    def demultiplex(self, frame: DataFrame) -> None:
        """
        Routes incoming frame payload to the corresponding registered listener.
        Safely drops frames with unknown subscription IDs.

        :param frame: DataFrame containing subscription_id and payload.
        """
        callback = self._subscriptions.get(frame.subscription_id)
        if callback is not None:
            callback(frame.payload)