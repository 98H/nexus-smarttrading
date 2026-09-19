"""Schemas for audio cues, toast presentations, and Web Push notifications."""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Union

VALID_TOAST_VARIANTS = {"info", "warning", "error", "success"}
VALID_URGENCIES = {"very-low", "low", "normal", "high"}


@dataclass
class AudioCue:
    """Audio cue presentation payload."""

    sound: str
    volume: float = 0.5
    loop: bool = False

    def __post_init__(self) -> None:
        if not (0.0 <= self.volume <= 1.0):
            raise ValueError(f"Volume must be within [0.0, 1.0], got: {self.volume}")


@dataclass
class ToastPresentation:
    """In-browser visual toast presentation payload."""

    title: str
    message: str
    duration_ms: int = 5000
    variant: str = "info"
    dismissible: bool = True

    def __post_init__(self) -> None:
        if self.variant not in VALID_TOAST_VARIANTS:
            raise ValueError(
                f"Invalid toast variant: '{self.variant}'. Expected one of: {sorted(VALID_TOAST_VARIANTS)}"
            )


@dataclass
class WebPushKeys:
    """Cryptographic keys for Web Push subscription."""

    p256dh: str
    auth: str

    def __post_init__(self) -> None:
        if not self.p256dh or not self.p256dh.strip():
            raise ValueError("p256dh key cannot be empty")
        if not self.auth or not self.auth.strip():
            raise ValueError("auth secret cannot be empty")


@dataclass
class WebPushSubscription:
    """Registered client subscription for Web Push delivery."""

    endpoint: str
    keys: WebPushKeys

    def __post_init__(self) -> None:
        if not self.endpoint or not self.endpoint.strip():
            raise ValueError("Web push subscription endpoint cannot be empty")


@dataclass
class WebPushEnvelope:
    """RFC-compliant Web Push payload envelope."""

    endpoint: str
    headers: Dict[str, str]
    body: Union[str, bytes]


@dataclass
class AlertEvent:
    """Incoming alert event requiring notification dispatch."""

    event_id: str
    title: str
    message: str
    audio: Optional[AudioCue] = None
    toast: Optional[ToastPresentation] = None
    push_subscriptions: List[WebPushSubscription] = field(default_factory=list)
    urgency: str = "normal"
    ttl_seconds: int = 86400

    def __post_init__(self) -> None:
        if self.push_subscriptions is None:
            self.push_subscriptions = []
        if self.urgency not in VALID_URGENCIES:
            raise ValueError(
                f"Invalid urgency: '{self.urgency}'. Expected one of: {sorted(VALID_URGENCIES)}"
            )
        if self.ttl_seconds < 0:
            raise ValueError(f"TTL must be non-negative, got: {self.ttl_seconds}")


@dataclass
class UnifiedNotificationPayload:
    """Unified notification payload containing audio, toast, and push envelopes."""

    event_id: str
    audio: AudioCue
    toast: ToastPresentation
    web_push_envelopes: List[WebPushEnvelope] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the unified payload to a dictionary."""
        return asdict(self)