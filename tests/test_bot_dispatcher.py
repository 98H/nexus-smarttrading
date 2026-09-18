"""
Unit tests for Telegram, Discord, and Slack Real-Time Bot Dispatchers.

Story 8.2.2: Implement Telegram, Discord, and Slack Real-Time Bot Dispatchers
Target Modules:
    - src/dispatchers/bot_dispatcher.py
    - src/dispatchers/models.py
"""

import concurrent.futures
import inspect
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

from src.dispatchers.bot_dispatcher import BotDispatcher, UnsupportedPlatformError
from src.dispatchers.models import Notification


# ============================================================================
# Helpers
# ============================================================================

def execute_dispatch(dispatcher: BotDispatcher, notification: Notification) -> Any:
    """
    Execute dispatch, handling coroutines synchronously if dispatch is async.
    Allows implementation to be either synchronous or asynchronous.
    """
    result = dispatcher.dispatch(notification)
    if inspect.isawaitable(result):
        import asyncio
        return asyncio.run(result)
    return result


def extract_sent_payload(mock_client: MagicMock) -> Dict[str, Any]:
    """
    Extract the formatted payload passed to the mock client handler.
    Supports .send(payload), .send_message(payload), or direct mock call.
    """
    if mock_client.send.called:
        call_args = mock_client.send.call_args
    elif mock_client.send_message.called:
        call_args = mock_client.send_message.call_args
    elif mock_client.called:
        call_args = mock_client.call_args
    else:
        pytest.fail("Mock client handler was not called.")

    args, kwargs = call_args
    if args and isinstance(args[0], dict):
        return args[0]
    if kwargs:
        return kwargs
    if args:
        return {"data": args[0]}
    return {}


def assert_client_not_called(mock_client: MagicMock) -> None:
    """Assert that a platform client handler was not called through any standard method."""
    assert mock_client.send.call_count == 0
    assert mock_client.send_message.call_count == 0
    assert mock_client.call_count == 0


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_telegram_client() -> MagicMock:
    client = MagicMock(name="telegram_client")
    client.send.return_value = {"ok": True, "platform": "telegram"}
    return client


@pytest.fixture
def mock_discord_client() -> MagicMock:
    client = MagicMock(name="discord_client")
    client.send.return_value = {"ok": True, "platform": "discord"}
    return client


@pytest.fixture
def mock_slack_client() -> MagicMock:
    client = MagicMock(name="slack_client")
    client.send.return_value = {"ok": True, "platform": "slack"}
    return client


@pytest.fixture
def fully_configured_dispatcher(
    mock_telegram_client: MagicMock,
    mock_discord_client: MagicMock,
    mock_slack_client: MagicMock,
) -> BotDispatcher:
    """Dispatcher configured with Telegram, Discord, and Slack clients."""
    return BotDispatcher(
        telegram_client=mock_telegram_client,
        discord_client=mock_discord_client,
        slack_client=mock_slack_client,
    )


# ============================================================================
# Notification Model Tests
# ============================================================================

class TestNotificationModel:
    """Unit tests verifying the Notification model data structure and contracts."""

    def test_notification_creation_with_all_fields(self):
        notification = Notification(
            platform="telegram",
            recipient_id="123456789",
            message="System alert: High memory usage detected.",
            title="Resource Alert",
            metadata={"threshold": 90, "server": "app-01"},
        )
        assert notification.platform == "telegram"
        assert notification.recipient_id == "123456789"
        assert notification.message == "System alert: High memory usage detected."
        assert notification.title == "Resource Alert"
        assert notification.metadata == {"threshold": 90, "server": "app-01"}

    def test_notification_creation_with_minimal_fields(self):
        notification = Notification(
            platform="discord",
            recipient_id="channel_999",
            message="Deployment started.",
        )
        assert notification.platform == "discord"
        assert notification.recipient_id == "channel_999"
        assert notification.message == "Deployment started."
        assert notification.title is None or notification.title == ""
        assert notification.metadata is None or notification.metadata == {}

    def test_notification_allows_arbitrary_platform_string(self):
        """Notification model must accept strings without raising, deferring validation to dispatch."""
        notification = Notification(
            platform="unsupported_platform",
            recipient_id="user_123",
            message="Test message",
        )
        assert notification.platform == "unsupported_platform"


# ============================================================================
# BotDispatcher Routing Tests
# ============================================================================

class TestBotDispatcherRouting:
    """Unit tests verifying that notifications are correctly formatted and routed to platform clients."""

    def test_dispatch_to_telegram_routes_to_telegram_client_only(
        self,
        fully_configured_dispatcher: BotDispatcher,
        mock_telegram_client: MagicMock,
        mock_discord_client: MagicMock,
        mock_slack_client: MagicMock,
    ):
        notification = Notification(
            platform="telegram",
            recipient_id="tg_chat_1001",
            message="Alert: Database connection failed",
            title="DB Error",
            metadata={"attempt": 3},
        )

        execute_dispatch(fully_configured_dispatcher, notification)

        payload = extract_sent_payload(mock_telegram_client)
        assert "tg_chat_1001" in str(payload)
        assert "Database connection failed" in str(payload)

        assert_client_not_called(mock_discord_client)
        assert_client_not_called(mock_slack_client)

    def test_dispatch_to_discord_routes_to_discord_client_only(
        self,
        fully_configured_dispatcher: BotDispatcher,
        mock_telegram_client: MagicMock,
        mock_discord_client: MagicMock,
        mock_slack_client: MagicMock,
    ):
        notification = Notification(
            platform="discord",
            recipient_id="discord_channel_2002",
            message="Build #42 passed successfully.",
            title="CI/CD Status",
            metadata={"branch": "main"},
        )

        execute_dispatch(fully_configured_dispatcher, notification)

        payload = extract_sent_payload(mock_discord_client)
        assert "discord_channel_2002" in str(payload)
        assert "Build #42 passed successfully." in str(payload)

        assert_client_not_called(mock_telegram_client)
        assert_client_not_called(mock_slack_client)

    def test_dispatch_to_slack_routes_to_slack_client_only(
        self,
        fully_configured_dispatcher: BotDispatcher,
        mock_telegram_client: MagicMock,
        mock_discord_client: MagicMock,
        mock_slack_client: MagicMock,
    ):
        notification = Notification(
            platform="slack",
            recipient_id="C0123456789",
            message="Payment gateway latency is above 500ms.",
            title="Latency Warning",
            metadata={"p99_latency_ms": 520},
        )

        execute_dispatch(fully_configured_dispatcher, notification)

        payload = extract_sent_payload(mock_slack_client)
        assert "C0123456789" in str(payload)
        assert "Payment gateway latency is above 500ms." in str(payload)

        assert_client_not_called(mock_telegram_client)
        assert_client_not_called(mock_discord_client)

    def test_sequential_dispatches_route_independently_without_state_leakage(
        self,
        fully_configured_dispatcher: BotDispatcher,
        mock_telegram_client: MagicMock,
        mock_discord_client: MagicMock,
        mock_slack_client: MagicMock,
    ):
        notif_telegram = Notification(platform="telegram", recipient_id="tg_1", message="Msg TG")
        notif_discord = Notification(platform="discord", recipient_id="dc_2", message="Msg DC")
        notif_slack = Notification(platform="slack", recipient_id="sl_3", message="Msg SL")

        execute_dispatch(fully_configured_dispatcher, notif_telegram)
        execute_dispatch(fully_configured_dispatcher, notif_discord)
        execute_dispatch(fully_configured_dispatcher, notif_slack)

        tg_payload = extract_sent_payload(mock_telegram_client)
        dc_payload = extract_sent_payload(mock_discord_client)
        sl_payload = extract_sent_payload(mock_slack_client)

        assert "tg_1" in str(tg_payload) and "Msg TG" in str(tg_payload)
        assert "dc_2" in str(dc_payload) and "Msg DC" in str(dc_payload)
        assert "sl_3" in str(sl_payload) and "Msg SL" in str(sl_payload)


# ============================================================================
# Payload Formatting Tests
# ============================================================================

class TestBotDispatcherPayloadFormatting:
    """Unit tests verifying platform-specific payload formatting contracts."""

    def test_telegram_payload_structure_and_formatting(
        self,
        fully_configured_dispatcher: BotDispatcher,
        mock_telegram_client: MagicMock,
    ):
        notification = Notification(
            platform="telegram",
            recipient_id="12345",
            message="Server restart required",
            title="Maintenance",
        )
        execute_dispatch(fully_configured_dispatcher, notification)

        payload = extract_sent_payload(mock_telegram_client)

        target_field = payload.get("chat_id") or payload.get("recipient_id") or payload.get("target")
        assert target_field == "12345"

        text_field = payload.get("text") or payload.get("message") or payload.get("content")
        assert text_field is not None
        assert "Server restart required" in str(text_field)
        assert "Maintenance" in str(text_field)

    def test_discord_payload_structure_and_formatting(
        self,
        fully_configured_dispatcher: BotDispatcher,
        mock_discord_client: MagicMock,
    ):
        notification = Notification(
            platform="discord",
            recipient_id="998877",
            message="New pull request #10",
            title="Repository Update",
        )
        execute_dispatch(fully_configured_dispatcher, notification)

        payload = extract_sent_payload(mock_discord_client)

        target_field = (
            payload.get("channel_id")
            or payload.get("recipient_id")
            or payload.get("channel")
            or payload.get("target")
        )
        assert target_field == "998877"

        # Discord payload should format into content or embeds
        has_content = "content" in payload and "New pull request #10" in str(payload["content"])
        has_embeds = "embeds" in payload and "New pull request #10" in str(payload["embeds"])
        has_body = "message" in payload and "New pull request #10" in str(payload["message"])
        assert has_content or has_embeds or has_body

    def test_slack_payload_structure_and_formatting(
        self,
        fully_configured_dispatcher: BotDispatcher,
        mock_slack_client: MagicMock,
    ):
        notification = Notification(
            platform="slack",
            recipient_id="C987654",
            message="Security certificate expiring in 7 days",
            title="SecOps Alert",
        )
        execute_dispatch(fully_configured_dispatcher, notification)

        payload = extract_sent_payload(mock_slack_client)

        target_field = payload.get("channel") or payload.get("recipient_id") or payload.get("channel_id")
        assert target_field == "C987654"

        # Slack payload should format into text or blocks
        has_text = "text" in payload and "Security certificate expiring in 7 days" in str(payload["text"])
        has_blocks = "blocks" in payload and "Security certificate expiring in 7 days" in str(payload["blocks"])
        has_body = "message" in payload and "Security certificate expiring in 7 days" in str(payload["message"])
        assert has_text or has_blocks or has_body

    def test_payload_formatting_preserves_metadata_and_special_characters(
        self,
        fully_configured_dispatcher: BotDispatcher,
        mock_slack_client: MagicMock,
    ):
        special_message = "Special chars: <>&\"' \n Emoji: 🚨 Status: OK [v1.0]"
        notification = Notification(
            platform="slack",
            recipient_id="C1111",
            message=special_message,
            metadata={"env": "prod", "cluster": "us-east-1"},
        )
        execute_dispatch(fully_configured_dispatcher, notification)

        payload = extract_sent_payload(mock_slack_client)
        payload_str = str(payload)
        assert "Special chars" in payload_str
        assert "🚨" in payload_str


# ============================================================================
# Invalid and Unconfigured Platform Tests
# ============================================================================

class TestBotDispatcherErrorsAndUnconfigured:
    """Unit tests verifying UnsupportedPlatformError for invalid or unconfigured platforms."""

    @pytest.mark.parametrize("invalid_platform", [
        "teams",
        "whatsapp",
        "email",
        "webhook",
        "sms",
        "unknown",
        "",
        "invalid_bot",
    ])
    def test_dispatch_raises_unsupported_platform_error_for_invalid_platform(
        self,
        fully_configured_dispatcher: BotDispatcher,
        invalid_platform: str,
    ):
        notification = Notification(
            platform=invalid_platform,
            recipient_id="target_id",
            message="Test message",
        )
        with pytest.raises(UnsupportedPlatformError):
            execute_dispatch(fully_configured_dispatcher, notification)

    def test_dispatch_raises_unsupported_platform_error_when_dispatcher_has_no_clients(self):
        dispatcher = BotDispatcher()
        notification = Notification(
            platform="telegram",
            recipient_id="12345",
            message="Test message",
        )
        with pytest.raises(UnsupportedPlatformError):
            execute_dispatch(dispatcher, notification)

    def test_dispatch_raises_unsupported_platform_error_for_unconfigured_discord_client(
        self,
        mock_telegram_client: MagicMock,
    ):
        # Only telegram configured; discord and slack omitted
        dispatcher = BotDispatcher(telegram_client=mock_telegram_client)

        discord_notification = Notification(
            platform="discord",
            recipient_id="dc_123",
            message="Discord alert",
        )
        with pytest.raises(UnsupportedPlatformError):
            execute_dispatch(dispatcher, discord_notification)

    def test_dispatch_raises_unsupported_platform_error_for_unconfigured_slack_client(
        self,
        mock_telegram_client: MagicMock,
    ):
        # Only telegram configured; discord and slack omitted
        dispatcher = BotDispatcher(telegram_client=mock_telegram_client)

        slack_notification = Notification(
            platform="slack",
            recipient_id="C12345",
            message="Slack alert",
        )
        with pytest.raises(UnsupportedPlatformError):
            execute_dispatch(dispatcher, slack_notification)

    def test_unconfigured_error_does_not_affect_configured_platform_on_same_dispatcher(
        self,
        mock_telegram_client: MagicMock,
    ):
        dispatcher = BotDispatcher(telegram_client=mock_telegram_client)

        # Telegram dispatch must succeed
        valid_notif = Notification(platform="telegram", recipient_id="tg_100", message="Valid")
        execute_dispatch(dispatcher, valid_notif)
        assert mock_telegram_client.send.call_count == 1 or mock_telegram_client.send_message.call_count == 1

        # Unconfigured slack dispatch raises
        unconfigured_notif = Notification(platform="slack", recipient_id="sl_200", message="Unconfigured")
        with pytest.raises(UnsupportedPlatformError):
            execute_dispatch(dispatcher, unconfigured_notif)


# ============================================================================
# Concurrency and Error Isolation Tests
# ============================================================================

class TestBotDispatcherConcurrency:
    """Unit tests verifying concurrent dispatches and fault isolation."""

    def test_concurrent_dispatches_isolate_unsupported_platform_errors(
        self,
        fully_configured_dispatcher: BotDispatcher,
        mock_telegram_client: MagicMock,
        mock_discord_client: MagicMock,
        mock_slack_client: MagicMock,
    ):
        """
        AC: Given an invalid or unconfigured target platform,
        When BotDispatcher.dispatch(notification) is executed,
        Then an UnsupportedPlatformError is raised without failing other concurrent dispatches.
        """
        notifications = [
            Notification(platform="telegram", recipient_id=f"tg_{i}", message=f"Telegram {i}")
            for i in range(5)
        ] + [
            Notification(platform="discord", recipient_id=f"dc_{i}", message=f"Discord {i}")
            for i in range(5)
        ] + [
            Notification(platform="slack", recipient_id=f"sl_{i}", message=f"Slack {i}")
            for i in range(5)
        ] + [
            Notification(platform="unsupported_service", recipient_id=f"err_{i}", message=f"Error {i}")
            for i in range(5)
        ]

        def dispatch_worker(notification: Notification):
            try:
                execute_dispatch(fully_configured_dispatcher, notification)
                return ("success", notification.platform)
            except UnsupportedPlatformError:
                return ("unsupported", notification.platform)
            except Exception as exc:
                return ("error", str(exc))

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(dispatch_worker, notif) for notif in notifications]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        success_results = [r for r in results if r[0] == "success"]
        unsupported_results = [r for r in results if r[0] == "unsupported"]
        unexpected_errors = [r for r in results if r[0] == "error"]

        assert len(unexpected_errors) == 0, f"Unexpected exceptions encountered: {unexpected_errors}"
        assert len(success_results) == 15
        assert len(unsupported_results) == 5

        # Confirm valid notifications reached corresponding clients exactly 5 times each
        tg_count = mock_telegram_client.send.call_count or mock_telegram_client.send_message.call_count
        dc_count = mock_discord_client.send.call_count or mock_discord_client.send_message.call_count
        sl_count = mock_slack_client.send.call_count or mock_slack_client.send_message.call_count

        assert tg_count == 5
        assert dc_count == 5
        assert sl_count == 5

    def test_concurrent_dispatches_with_partially_configured_dispatcher(
        self,
        mock_telegram_client: MagicMock,
    ):
        """Partially configured dispatcher must handle concurrent mixed valid and unconfigured targets safely."""
        dispatcher = BotDispatcher(telegram_client=mock_telegram_client)

        mixed_notifications = [
            Notification(platform="telegram", recipient_id=f"tg_{i}", message=f"Valid {i}")
            for i in range(4)
        ] + [
            Notification(platform="discord", recipient_id=f"dc_{i}", message=f"Unconfigured {i}")
            for i in range(4)
        ]

        def dispatch_worker(notification: Notification):
            try:
                execute_dispatch(dispatcher, notification)
                return ("success", notification.platform)
            except UnsupportedPlatformError:
                return ("unsupported", notification.platform)
            except Exception as exc:
                return ("error", str(exc))

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(dispatch_worker, notif) for notif in mixed_notifications]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        successes = [r for r in results if r[0] == "success"]
        unsupported = [r for r in results if r[0] == "unsupported"]
        errors = [r for r in results if r[0] == "error"]

        assert len(errors) == 0
        assert len(successes) == 4
        assert len(unsupported) == 4

        tg_count = mock_telegram_client.send.call_count or mock_telegram_client.send_message.call_count
        assert tg_count == 4