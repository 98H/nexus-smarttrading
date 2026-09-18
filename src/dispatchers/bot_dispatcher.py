"""Real-time bot dispatcher for Telegram, Discord, and Slack notifications."""

from typing import Any, Callable, Dict, Optional, Tuple

from src.dispatchers.models import Notification


class UnsupportedPlatformError(Exception):
    """Raised when a target platform is unknown, invalid, or unconfigured."""
    pass


def format_telegram_payload(notification: Notification) -> Dict[str, Any]:
    """Format notification into a Telegram Bot API sendMessage payload."""
    text = (
        f"{notification.title}\n\n{notification.message}"
        if notification.title
        else notification.message
    )
    payload: Dict[str, Any] = {
        "chat_id": notification.recipient_id,
        "text": text,
    }
    if notification.metadata is not None:
        payload["metadata"] = notification.metadata
    return payload


def format_discord_payload(notification: Notification) -> Dict[str, Any]:
    """Format notification into a Discord message payload."""
    content = (
        f"**{notification.title}**\n\n{notification.message}"
        if notification.title
        else notification.message
    )
    payload: Dict[str, Any] = {
        "channel_id": notification.recipient_id,
        "content": content,
    }
    if notification.metadata is not None:
        payload["metadata"] = notification.metadata
    return payload


def format_slack_payload(notification: Notification) -> Dict[str, Any]:
    """Format notification into a Slack chat.postMessage payload."""
    text = (
        f"*{notification.title}*\n\n{notification.message}"
        if notification.title
        else notification.message
    )
    payload: Dict[str, Any] = {
        "channel": notification.recipient_id,
        "text": text,
    }
    if notification.metadata is not None:
        payload["metadata"] = notification.metadata
    return payload


class BotDispatcher:
    """Dispatches notifications to Telegram, Discord, and Slack bot clients."""

    SUPPORTED_PLATFORMS = {"telegram", "discord", "slack"}

    def __init__(
        self,
        telegram_client: Any = None,
        discord_client: Any = None,
        slack_client: Any = None,
    ) -> None:
        self.telegram_client = telegram_client
        self.discord_client = discord_client
        self.slack_client = slack_client

    def _get_client_and_formatter(
        self, platform: str
    ) -> Tuple[Any, Callable[[Notification], Dict[str, Any]]]:
        normalized = platform.strip().lower() if isinstance(platform, str) else ""
        if normalized not in self.SUPPORTED_PLATFORMS:
            raise UnsupportedPlatformError(f"Unsupported target platform: {platform!r}")

        if normalized == "telegram":
            client = self.telegram_client
            formatter = format_telegram_payload
        elif normalized == "discord":
            client = self.discord_client
            formatter = format_discord_payload
        elif normalized == "slack":
            client = self.slack_client
            formatter = format_slack_payload
        else:
            raise UnsupportedPlatformError(f"Unsupported target platform: {platform!r}")

        if client is None:
            raise UnsupportedPlatformError(
                f"Client for platform '{platform}' is not configured on this dispatcher."
            )

        return client, formatter

    @staticmethod
    def _send_to_client(client: Any, payload: Dict[str, Any]) -> Any:
        """Invoke platform client handler method."""
        if hasattr(client, "send") and callable(client.send):
            return client.send(payload)
        if hasattr(client, "send_message") and callable(client.send_message):
            return client.send_message(payload)
        if callable(client):
            return client(payload)
        raise AttributeError(f"Platform client {client!r} has no send or send_message method.")

    def dispatch(self, notification: Notification) -> Any:
        """
        Route and format notification to the corresponding platform client handler.

        Raises:
            UnsupportedPlatformError: If target platform is unsupported or client unconfigured.
        """
        client, formatter = self._get_client_and_formatter(notification.platform)
        payload = formatter(notification)
        return self._send_to_client(client, payload)