from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from src.security.secret_rotator import SecretRotator, SecretStatus


@dataclass(frozen=True)
class NetworkRequest:
    source_service: str
    target_service: str
    secret_id: str
    secret_value: str


@dataclass(frozen=True)
class PolicyDecision:
    is_allowed: bool
    reason: str = ""


@dataclass(frozen=True)
class SecurityEvent:
    event_type: str
    source_service: str
    target_service: str
    secret_id: str
    reason: str
    timestamp: datetime


class SecurityEventLogger:
    """In-memory security audit log for tracking policy evaluation events."""

    def __init__(self) -> None:
        self._events: list[SecurityEvent] = []

    def log_event(self, event: SecurityEvent) -> None:
        self._events.append(event)

    def get_unauthorized_events(self) -> list[SecurityEvent]:
        return [e for e in self._events if e.event_type == "UNAUTHORIZED_ACCESS"]

    def get_events_by_source(self, source_service: str) -> list[SecurityEvent]:
        return [e for e in self._events if e.source_service == source_service]


class ZeroTrustPolicyEngine:
    """Evaluates service-to-service communication under Zero-Trust principles."""

    def __init__(self, rotator: SecretRotator, event_logger: SecurityEventLogger) -> None:
        self._rotator = rotator
        self._event_logger = event_logger

    def evaluate_request(self, request: NetworkRequest) -> PolicyDecision:
        # Check if secret ID is registered
        if not self._rotator.contains_secret_id(request.secret_id):
            reason = f"Unverified secret: secret ID '{request.secret_id}' not found."
            self._log_unauthorized_event(request, reason)
            return PolicyDecision(is_allowed=False, reason=reason)

        # Check if provided secret value matches any known version
        matched_secret = self._rotator.get_secret_by_value(
            request.secret_id, request.secret_value
        )
        if matched_secret is None:
            reason = "Unverified secret: provided credential value is invalid."
            self._log_unauthorized_event(request, reason)
            return PolicyDecision(is_allowed=False, reason=reason)

        # Check if secret was retired
        if matched_secret.status == SecretStatus.RETIRED:
            reason = f"Secret '{request.secret_id}' (v{matched_secret.version}) is retired."
            self._log_unauthorized_event(request, reason)
            return PolicyDecision(is_allowed=False, reason=reason)

        # Check if secret was revoked
        if matched_secret.status == SecretStatus.REVOKED:
            reason = f"Secret '{request.secret_id}' (v{matched_secret.version}) is revoked."
            self._log_unauthorized_event(request, reason)
            return PolicyDecision(is_allowed=False, reason=reason)

        # Check if secret validity lease has expired
        if self._is_expired(matched_secret.expires_at):
            reason = f"Secret '{request.secret_id}' (v{matched_secret.version}) validity lease has expired."
            self._log_unauthorized_event(request, reason)
            return PolicyDecision(is_allowed=False, reason=reason)

        # Ensure secret is active
        if matched_secret.status != SecretStatus.ACTIVE:
            reason = f"Secret '{request.secret_id}' is not active."
            self._log_unauthorized_event(request, reason)
            return PolicyDecision(is_allowed=False, reason=reason)

        return PolicyDecision(is_allowed=True, reason="Access granted.")

    def _log_unauthorized_event(self, request: NetworkRequest, reason: str) -> None:
        event = SecurityEvent(
            event_type="UNAUTHORIZED_ACCESS",
            source_service=request.source_service,
            target_service=request.target_service,
            secret_id=request.secret_id,
            reason=reason,
            timestamp=datetime.now(timezone.utc),
        )
        self._event_logger.log_event(event)

    @staticmethod
    def _is_expired(expires_at: datetime) -> bool:
        if expires_at.tzinfo is None:
            now = datetime.now(timezone.utc).replace(tzinfo=None)
        else:
            now = datetime.now(timezone.utc)
        return expires_at <= now