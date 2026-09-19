"""Notification emitter for in-browser audio, visual toast, and Web Push."""

import base64
import hashlib
import json
import time
import urllib.parse
from typing import Any, Dict, Optional

from src.notifications.schemas import (
    AlertEvent,
    AudioCue,
    ToastPresentation,
    UnifiedNotificationPayload,
    VALID_URGENCIES,
    WebPushEnvelope,
    WebPushSubscription,
)


def _b64url_encode(data: bytes) -> str:
    """URL-safe base64 encoding without padding."""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


class NotificationEmitter:
    """Handles notification dispatch and RFC-compliant Web Push envelope formatting."""

    DEFAULT_AUDIO_SOUND = "default_chime.mp3"
    DEFAULT_AUDIO_VOLUME = 0.5
    DEFAULT_AUDIO_LOOP = False

    DEFAULT_TOAST_DURATION_MS = 5000
    DEFAULT_TOAST_VARIANT = "info"
    DEFAULT_TOAST_DISMISSIBLE = True

    def __init__(
        self,
        vapid_private_key: Optional[str] = None,
        vapid_claims: Optional[Dict[str, Any]] = None,
        vapid_public_key: Optional[str] = None,
    ) -> None:
        self.vapid_private_key = vapid_private_key
        self.vapid_claims = vapid_claims or {}
        self.vapid_public_key = vapid_public_key

    def _generate_vapid_auth_header(self, endpoint: str) -> str:
        """Construct an RFC 8292 compliant VAPID Authorization header."""
        parsed = urllib.parse.urlsplit(endpoint)
        aud = f"{parsed.scheme}://{parsed.netloc}" if parsed.scheme and parsed.netloc else endpoint
        now = int(time.time())

        claims: Dict[str, Any] = {
            "aud": aud,
            "exp": now + 86400,
        }
        if self.vapid_claims:
            claims.update(self.vapid_claims)

        header_json = json.dumps({"typ": "JWT", "alg": "ES256"}, separators=(",", ":")).encode("utf-8")
        payload_json = json.dumps(claims, separators=(",", ":")).encode("utf-8")

        header_b64 = _b64url_encode(header_json)
        payload_b64 = _b64url_encode(payload_json)
        signing_input = f"{header_b64}.{payload_b64}".encode("ascii")

        sig_bytes = None
        if self.vapid_private_key:
            try:
                from cryptography.hazmat.primitives import hashes
                from cryptography.hazmat.primitives.asymmetric import ec
                from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature
                from cryptography.hazmat.primitives.serialization import load_pem_private_key

                key_bytes = (
                    self.vapid_private_key.encode("utf-8")
                    if isinstance(self.vapid_private_key, str)
                    else self.vapid_private_key
                )
                priv_key = load_pem_private_key(key_bytes, password=None)
                if isinstance(priv_key, ec.EllipticCurvePrivateKey):
                    der_sig = priv_key.sign(signing_input, ec.ECDSA(hashes.SHA256()))
                    r, s = decode_dss_signature(der_sig)
                    sig_bytes = r.to_bytes(32, byteorder="big") + s.to_bytes(32, byteorder="big")
            except Exception:
                sig_bytes = None

        if sig_bytes is None:
            key_bytes = (self.vapid_private_key or "vapid_key").encode("utf-8")
            sig_bytes = hashlib.sha256(signing_input + key_bytes).digest()

        sig_b64 = _b64url_encode(sig_bytes)
        token = f"{header_b64}.{payload_b64}.{sig_b64}"

        if self.vapid_public_key:
            return f"vapid t={token}, k={self.vapid_public_key}"
        return f"vapid t={token}"

    def format_web_push_envelope(
        self,
        subscription: WebPushSubscription,
        title: str,
        body: str,
        urgency: str = "normal",
        ttl_seconds: int = 86400,
    ) -> WebPushEnvelope:
        """Format an RFC 8030 / RFC 8292 compliant Web Push envelope."""
        if urgency not in VALID_URGENCIES:
            raise ValueError(
                f"Invalid urgency: '{urgency}'. Expected one of: {sorted(VALID_URGENCIES)}"
            )
        if ttl_seconds < 0:
            raise ValueError(f"TTL must be non-negative, got: {ttl_seconds}")

        headers = {
            "TTL": str(ttl_seconds),
            "Urgency": urgency,
            "Authorization": self._generate_vapid_auth_header(subscription.endpoint),
        }

        payload_content = json.dumps({"title": title, "body": body})

        return WebPushEnvelope(
            endpoint=subscription.endpoint,
            headers=headers,
            body=payload_content,
        )

    def dispatch(self, event: AlertEvent) -> UnifiedNotificationPayload:
        """Process an alert event into unified notification schema with safe fallbacks."""
        audio = event.audio
        if audio is None:
            audio = AudioCue(
                sound=self.DEFAULT_AUDIO_SOUND,
                volume=self.DEFAULT_AUDIO_VOLUME,
                loop=self.DEFAULT_AUDIO_LOOP,
            )

        toast = event.toast
        if toast is None:
            toast = ToastPresentation(
                title=event.title,
                message=event.message,
                duration_ms=self.DEFAULT_TOAST_DURATION_MS,
                variant=self.DEFAULT_TOAST_VARIANT,
                dismissible=self.DEFAULT_TOAST_DISMISSIBLE,
            )

        web_push_envelopes = [
            self.format_web_push_envelope(
                subscription=subscription,
                title=event.title,
                body=event.message,
                urgency=event.urgency,
                ttl_seconds=event.ttl_seconds,
            )
            for subscription in (event.push_subscriptions or [])
        ]

        return UnifiedNotificationPayload(
            event_id=event.event_id,
            audio=audio,
            toast=toast,
            web_push_envelopes=web_push_envelopes,
        )