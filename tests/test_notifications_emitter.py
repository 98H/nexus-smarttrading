import pytest
from typing import Dict, Any

from src.notifications.schemas import (
    AudioCue,
    ToastPresentation,
    WebPushKeys,
    WebPushSubscription,
    WebPushEnvelope,
    AlertEvent,
    UnifiedNotificationPayload,
)
from src.notifications.emitter import NotificationEmitter


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def valid_web_push_keys() -> WebPushKeys:
    return WebPushKeys(
        p256dh="BCVxsr7N_eNgVRqvDjV6Kw9Yr32b7UP4",
        auth="s8fgh20al1kld8f7",
    )


@pytest.fixture
def valid_web_push_subscription(valid_web_push_keys: WebPushKeys) -> WebPushSubscription:
    return WebPushSubscription(
        endpoint="https://fcm.googleapis.com/fcm/send/sample-subscription-token-123",
        keys=valid_web_push_keys,
    )


@pytest.fixture
def sample_audio_cue() -> AudioCue:
    return AudioCue(
        sound="critical_alarm.mp3",
        volume=0.8,
        loop=True,
    )


@pytest.fixture
def sample_toast() -> ToastPresentation:
    return ToastPresentation(
        title="High Severity Alert",
        message="CPU utilization exceeded 95% on worker-node-04",
        duration_ms=8000,
        variant="error",
        dismissible=True,
    )


@pytest.fixture
def vapid_config() -> Dict[str, str]:
    return {
        "vapid_private_key": "sample_private_key_pem_or_base64",
        "vapid_public_key": "sample_public_key_pem_or_base64",
        "vapid_claims": {"sub": "mailto:alerts@ops-platform.io"},
    }


@pytest.fixture
def emitter(vapid_config: Dict[str, str]) -> NotificationEmitter:
    return NotificationEmitter(
        vapid_private_key=vapid_config["vapid_private_key"],
        vapid_claims=vapid_config["vapid_claims"],
    )


# ============================================================================
# 1. Schema Validation Unit Tests (`src/notifications/schemas.py`)
# ============================================================================


class TestNotificationSchemas:
    def test_audio_cue_valid_attributes(self, sample_audio_cue: AudioCue):
        assert sample_audio_cue.sound == "critical_alarm.mp3"
        assert sample_audio_cue.volume == 0.8
        assert sample_audio_cue.loop is True

    def test_audio_cue_volume_bounds(self):
        # Audio cue volume must be bounded within [0.0, 1.0]
        with pytest.raises(ValueError):
            AudioCue(sound="ping.wav", volume=1.5)

        with pytest.raises(ValueError):
            AudioCue(sound="ping.wav", volume=-0.1)

    def test_toast_presentation_defaults(self):
        toast = ToastPresentation(
            title="System Notice",
            message="Cache refreshed",
        )
        assert toast.duration_ms == 5000
        assert toast.variant == "info"
        assert toast.dismissible is True

    def test_toast_presentation_invalid_variant(self):
        with pytest.raises(ValueError):
            ToastPresentation(
                title="Notice",
                message="Body",
                variant="unsupported_variant",
            )

    def test_web_push_subscription_requires_valid_endpoint(self, valid_web_push_keys: WebPushKeys):
        with pytest.raises(ValueError):
            WebPushSubscription(
                endpoint="",
                keys=valid_web_push_keys,
            )

    def test_web_push_keys_missing_fields(self):
        with pytest.raises(ValueError):
            WebPushKeys(p256dh="", auth="key123")

        with pytest.raises(ValueError):
            WebPushKeys(p256dh="key123", auth="")


# ============================================================================
# 2. Unified Notification Payload Serialization (`Story 8.2.3 - AC 1`)
# ============================================================================


class TestEmitterUnifiedSerialization:
    def test_dispatch_full_event_serializes_unified_schema(
        self,
        emitter: NotificationEmitter,
        sample_audio_cue: AudioCue,
        sample_toast: ToastPresentation,
        valid_web_push_subscription: WebPushSubscription,
    ):
        event = AlertEvent(
            event_id="alert-001",
            title="Database Latency Spike",
            message="p99 read latency > 250ms",
            audio=sample_audio_cue,
            toast=sample_toast,
            push_subscriptions=[valid_web_push_subscription],
            urgency="high",
            ttl_seconds=3600,
        )

        unified_payload = emitter.dispatch(event)

        assert isinstance(unified_payload, UnifiedNotificationPayload)
        assert unified_payload.event_id == "alert-001"

        # Verify Audio Trigger
        assert unified_payload.audio is not None
        assert unified_payload.audio.sound == "critical_alarm.mp3"
        assert unified_payload.audio.volume == 0.8
        assert unified_payload.audio.loop is True

        # Verify Toast Presentation
        assert unified_payload.toast is not None
        assert unified_payload.toast.title == "High Severity Alert"
        assert unified_payload.toast.message == "CPU utilization exceeded 95% on worker-node-04"
        assert unified_payload.toast.variant == "error"
        assert unified_payload.toast.duration_ms == 8000

        # Verify Web Push Envelope
        assert len(unified_payload.web_push_envelopes) == 1
        push_envelope = unified_payload.web_push_envelopes[0]
        assert isinstance(push_envelope, WebPushEnvelope)
        assert push_envelope.endpoint == valid_web_push_subscription.endpoint
        assert "Urgency" in push_envelope.headers
        assert push_envelope.headers["Urgency"] == "high"
        assert "TTL" in push_envelope.headers
        assert push_envelope.headers["TTL"] == "3600"

    def test_unified_payload_serialization_output_contains_expected_keys(
        self,
        emitter: NotificationEmitter,
        sample_audio_cue: AudioCue,
        sample_toast: ToastPresentation,
        valid_web_push_subscription: WebPushSubscription,
    ):
        event = AlertEvent(
            event_id="alert-002",
            title="Pod Evicted",
            message="Node out of memory",
            audio=sample_audio_cue,
            toast=sample_toast,
            push_subscriptions=[valid_web_push_subscription],
        )

        payload = emitter.dispatch(event)
        data: Dict[str, Any] = payload.to_dict()

        assert "event_id" in data
        assert "audio" in data
        assert "toast" in data
        assert "web_push_envelopes" in data
        assert isinstance(data["web_push_envelopes"], list)
        assert len(data["web_push_envelopes"]) == 1


# ============================================================================
# 3. Fallback Handling for Missing Configurations (`Story 8.2.3 - AC 2`)
# ============================================================================


class TestEmitterSafeFallbacks:
    def test_missing_audio_provides_safe_fallback(
        self,
        emitter: NotificationEmitter,
        sample_toast: ToastPresentation,
        valid_web_push_subscription: WebPushSubscription,
    ):
        # Audio configuration is explicitly None
        event = AlertEvent(
            event_id="alert-fallback-01",
            title="Warning: High Disk Usage",
            message="Disk /dev/sda1 at 82%",
            audio=None,
            toast=sample_toast,
            push_subscriptions=[valid_web_push_subscription],
        )

        payload = emitter.dispatch(event)

        assert payload.audio is not None
        assert payload.audio.sound == "default_chime.mp3"
        assert payload.audio.volume == 0.5
        assert payload.audio.loop is False
        # Web push must remain intact and delivered
        assert len(payload.web_push_envelopes) == 1

    def test_missing_toast_provides_safe_fallback(
        self,
        emitter: NotificationEmitter,
        sample_audio_cue: AudioCue,
        valid_web_push_subscription: WebPushSubscription,
    ):
        # Toast configuration is explicitly None
        event = AlertEvent(
            event_id="alert-fallback-02",
            title="Service Degraded",
            message="Payments service response delayed",
            audio=sample_audio_cue,
            toast=None,
            push_subscriptions=[valid_web_push_subscription],
        )

        payload = emitter.dispatch(event)

        assert payload.toast is not None
        assert payload.toast.title == "Service Degraded"
        assert payload.toast.message == "Payments service response delayed"
        assert payload.toast.variant == "info"
        assert payload.toast.duration_ms == 5000
        # Web push must remain intact and delivered
        assert len(payload.web_push_envelopes) == 1

    def test_missing_both_audio_and_toast_provides_fallbacks_without_push_disruption(
        self,
        emitter: NotificationEmitter,
        valid_web_push_subscription: WebPushSubscription,
    ):
        event = AlertEvent(
            event_id="alert-fallback-03",
            title="Cluster Scaling Up",
            message="Auto-scaler provisioned 2 additional nodes",
            audio=None,
            toast=None,
            push_subscriptions=[valid_web_push_subscription],
        )

        payload = emitter.dispatch(event)

        # Both fallbacks safely assigned
        assert payload.audio is not None
        assert payload.audio.sound == "default_chime.mp3"
        assert payload.toast is not None
        assert payload.toast.title == "Cluster Scaling Up"
        assert payload.toast.message == "Auto-scaler provisioned 2 additional nodes"

        # Web push envelope must still be correctly constructed
        assert len(payload.web_push_envelopes) == 1
        assert payload.web_push_envelopes[0].endpoint == valid_web_push_subscription.endpoint


# ============================================================================
# 4. Web Push RFC-Compliant Payload Envelope & VAPID (`Story 8.2.3 - AC 3`)
# ============================================================================


class TestWebPushEnvelopeFormatting:
    def test_build_web_push_envelope_rfc_compliance(
        self,
        emitter: NotificationEmitter,
        valid_web_push_subscription: WebPushSubscription,
    ):
        envelope = emitter.format_web_push_envelope(
            subscription=valid_web_push_subscription,
            title="Security Alert",
            body="Suspicious login detected",
            urgency="very-low",
            ttl_seconds=120,
        )

        assert isinstance(envelope, WebPushEnvelope)
        assert envelope.endpoint == valid_web_push_subscription.endpoint

        # RFC 8030 / 8292 headers validation
        headers = envelope.headers
        assert "TTL" in headers
        assert headers["TTL"] == "120"
        assert "Urgency" in headers
        assert headers["Urgency"] == "very-low"
        assert "Authorization" in headers
        assert headers["Authorization"].startswith("vapid t=") or headers["Authorization"].startswith("WebPush ")

        # Payload envelope must contain notification data
        assert envelope.body is not None
        assert isinstance(envelope.body, (str, bytes))

    def test_web_push_default_ttl_and_urgency(
        self,
        emitter: NotificationEmitter,
        valid_web_push_subscription: WebPushSubscription,
    ):
        envelope = emitter.format_web_push_envelope(
            subscription=valid_web_push_subscription,
            title="Heartbeat",
            body="Routine status check",
        )

        # Defaults should align with standard RFC Web Push recommendations
        assert envelope.headers["TTL"] == "86400"
        assert envelope.headers["Urgency"] == "normal"

    def test_web_push_invalid_urgency_raises(
        self,
        emitter: NotificationEmitter,
        valid_web_push_subscription: WebPushSubscription,
    ):
        with pytest.raises(ValueError):
            emitter.format_web_push_envelope(
                subscription=valid_web_push_subscription,
                title="Invalid Urgency",
                body="Test",
                urgency="invalid-urgency-level",
            )

    def test_dispatch_multiple_web_push_subscriptions(
        self,
        emitter: NotificationEmitter,
        valid_web_push_keys: WebPushKeys,
    ):
        sub_1 = WebPushSubscription(
            endpoint="https://push.example.com/client-1",
            keys=valid_web_push_keys,
        )
        sub_2 = WebPushSubscription(
            endpoint="https://push.example.com/client-2",
            keys=valid_web_push_keys,
        )

        event = AlertEvent(
            event_id="alert-multi-01",
            title="Broadcast Alert",
            message="Maintenance starting in 10 minutes",
            push_subscriptions=[sub_1, sub_2],
        )

        payload = emitter.dispatch(event)

        assert len(payload.web_push_envelopes) == 2
        endpoints = [env.endpoint for env in payload.web_push_envelopes]
        assert "https://push.example.com/client-1" in endpoints
        assert "https://push.example.com/client-2" in endpoints

    def test_web_push_empty_subscriptions_dispatches_with_empty_envelopes(
        self,
        emitter: NotificationEmitter,
        sample_audio_cue: AudioCue,
        sample_toast: ToastPresentation,
    ):
        event = AlertEvent(
            event_id="alert-no-push",
            title="Local Console Event",
            message="No push targets specified",
            audio=sample_audio_cue,
            toast=sample_toast,
            push_subscriptions=[],
        )

        payload = emitter.dispatch(event)

        assert len(payload.web_push_envelopes) == 0
        assert payload.audio == sample_audio_cue
        assert payload.toast == sample_toast