from datetime import datetime, timedelta, timezone
import pytest

from src.security.secret_rotator import (
    Secret,
    SecretNotFoundError,
    SecretRotationError,
    SecretRotator,
    SecretStatus,
)
from src.security.network_policy import (
    NetworkRequest,
    PolicyDecision,
    SecurityEvent,
    SecurityEventLogger,
    ZeroTrustPolicyEngine,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def secret_rotator() -> SecretRotator:
    """Provides a fresh SecretRotator instance for testing."""
    return SecretRotator()


@pytest.fixture
def event_logger() -> SecurityEventLogger:
    """Provides an in-memory SecurityEventLogger instance."""
    return SecurityEventLogger()


@pytest.fixture
def policy_engine(
    secret_rotator: SecretRotator, event_logger: SecurityEventLogger
) -> ZeroTrustPolicyEngine:
    """Provides a ZeroTrustPolicyEngine connected to the rotator and event logger."""
    return ZeroTrustPolicyEngine(rotator=secret_rotator, event_logger=event_logger)


# ============================================================================
# SecretRotator Unit Tests
# ============================================================================

class TestSecretRotator:
    """Tests automated secret rotation and lease lifecycle."""

    def test_rotate_expired_secret_generates_new_and_retires_old(
        self, secret_rotator: SecretRotator
    ) -> None:
        """
        Acceptance Criteria:
        Given an active credential with an expired validity lease,
        When `SecretRotator.rotate_secret` is executed,
        Then a new secret is generated and activated while the expired secret is safely retired.
        """
        secret_id = "payment-gateway-auth"
        initial_value = "initial-token-12345"
        expired_time = datetime.now(timezone.utc) - timedelta(minutes=5)

        # Register an active credential whose validity lease is in the past
        old_secret = secret_rotator.register_secret(
            secret_id=secret_id,
            value=initial_value,
            expires_at=expired_time,
            status=SecretStatus.ACTIVE,
        )

        assert old_secret.status == SecretStatus.ACTIVE
        assert old_secret.expires_at < datetime.now(timezone.utc)

        # Execute secret rotation
        new_secret = secret_rotator.rotate_secret(secret_id=secret_id)

        # Verify new secret generation and activation
        assert new_secret.secret_id == secret_id
        assert new_secret.status == SecretStatus.ACTIVE
        assert new_secret.value != initial_value
        assert new_secret.version == old_secret.version + 1
        assert new_secret.expires_at > datetime.now(timezone.utc)

        # Verify old secret is safely retired
        retired_secret = secret_rotator.get_secret_version(
            secret_id=secret_id, version=old_secret.version
        )
        assert retired_secret.status == SecretStatus.RETIRED
        assert retired_secret.value == initial_value

    def test_rotate_generates_distinct_cryptographic_values(
        self, secret_rotator: SecretRotator
    ) -> None:
        """Ensures consecutive rotations produce unique values with incrementing versions."""
        secret_id = "service-mesh-token"
        secret_rotator.register_secret(
            secret_id=secret_id,
            value="token-v1",
            expires_at=datetime.now(timezone.utc) - timedelta(seconds=1),
            status=SecretStatus.ACTIVE,
        )

        v2_secret = secret_rotator.rotate_secret(secret_id)
        v3_secret = secret_rotator.rotate_secret(secret_id)

        assert v2_secret.version == 2
        assert v3_secret.version == 3
        assert v2_secret.value != v3_secret.value
        assert v3_secret.status == SecretStatus.ACTIVE

        # Prior version should be retired
        assert secret_rotator.get_secret_version(secret_id, 2).status == SecretStatus.RETIRED

    def test_rotate_non_existent_secret_raises_error(
        self, secret_rotator: SecretRotator
    ) -> None:
        """Attempting to rotate a non-existent secret must fail explicitly."""
        with pytest.raises(SecretNotFoundError):
            secret_rotator.rotate_secret(secret_id="unregistered-secret-id")

    def test_rotate_revoked_secret_raises_error(
        self, secret_rotator: SecretRotator
    ) -> None:
        """Attempting to rotate a revoked secret must be rejected."""
        secret_id = "revoked-service-key"
        secret_rotator.register_secret(
            secret_id=secret_id,
            value="revoked-value",
            expires_at=datetime.now(timezone.utc) - timedelta(days=1),
            status=SecretStatus.REVOKED,
        )

        with pytest.raises(SecretRotationError):
            secret_rotator.rotate_secret(secret_id=secret_id)

    def test_get_active_secret_returns_latest_active_only(
        self, secret_rotator: SecretRotator
    ) -> None:
        """Calling get_active_secret returns the current active secret instance."""
        secret_id = "database-master-pw"
        secret_rotator.register_secret(
            secret_id=secret_id,
            value="pwd-v1",
            expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
            status=SecretStatus.ACTIVE,
        )
        rotated_secret = secret_rotator.rotate_secret(secret_id)

        current_active = secret_rotator.get_active_secret(secret_id)
        assert current_active.version == rotated_secret.version
        assert current_active.value == rotated_secret.value
        assert current_active.status == SecretStatus.ACTIVE


# ============================================================================
# ZeroTrustPolicyEngine Unit Tests
# ============================================================================

class TestZeroTrustPolicyEngine:
    """Tests Zero-Trust network policy evaluation and unauthorized event recording."""

    def test_evaluate_request_with_valid_active_secret_allows_access(
        self,
        policy_engine: ZeroTrustPolicyEngine,
        secret_rotator: SecretRotator,
        event_logger: SecurityEventLogger,
    ) -> None:
        """Valid and active secret token allows access and records no unauthorized events."""
        secret_id = "ingress-to-orders"
        secret_value = "valid-active-crypto-token"
        secret_rotator.register_secret(
            secret_id=secret_id,
            value=secret_value,
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
            status=SecretStatus.ACTIVE,
        )

        request = NetworkRequest(
            source_service="ingress-api",
            target_service="orders-service",
            secret_id=secret_id,
            secret_value=secret_value,
        )

        decision: PolicyDecision = policy_engine.evaluate_request(request)

        assert decision.is_allowed is True
        assert len(event_logger.get_unauthorized_events()) == 0

    def test_evaluate_request_with_expired_secret_denies_and_records_event(
        self,
        policy_engine: ZeroTrustPolicyEngine,
        secret_rotator: SecretRotator,
        event_logger: SecurityEventLogger,
    ) -> None:
        """
        Acceptance Criteria:
        Given an incoming service-to-service communication,
        When evaluated against the zero-trust policy engine using an expired secret,
        Then access is denied and an unauthorized security event is recorded.
        """
        secret_id = "orders-to-billing"
        secret_value = "expired-token-val"
        secret_rotator.register_secret(
            secret_id=secret_id,
            value=secret_value,
            expires_at=datetime.now(timezone.utc) - timedelta(minutes=10),
            status=SecretStatus.ACTIVE,
        )

        request = NetworkRequest(
            source_service="orders-service",
            target_service="billing-service",
            secret_id=secret_id,
            secret_value=secret_value,
        )

        decision: PolicyDecision = policy_engine.evaluate_request(request)

        assert decision.is_allowed is False

        # Verify unauthorized security event is recorded
        events = event_logger.get_unauthorized_events()
        assert len(events) == 1

        security_event: SecurityEvent = events[0]
        assert security_event.source_service == "orders-service"
        assert security_event.target_service == "billing-service"
        assert security_event.secret_id == secret_id
        assert security_event.event_type == "UNAUTHORIZED_ACCESS"
        assert "expired" in security_event.reason.lower()
        assert isinstance(security_event.timestamp, datetime)

    def test_evaluate_request_with_unverified_secret_denies_and_records_event(
        self,
        policy_engine: ZeroTrustPolicyEngine,
        secret_rotator: SecretRotator,
        event_logger: SecurityEventLogger,
    ) -> None:
        """
        Acceptance Criteria:
        Given an incoming service-to-service communication,
        When evaluated against the zero-trust policy engine using an unverified secret,
        Then access is denied and an unauthorized security event is recorded.
        """
        secret_id = "inventory-to-warehouse"
        secret_rotator.register_secret(
            secret_id=secret_id,
            value="expected-strong-token",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=2),
            status=SecretStatus.ACTIVE,
        )

        # Incoming request provides a forged or tampered secret
        request = NetworkRequest(
            source_service="inventory-service",
            target_service="warehouse-service",
            secret_id=secret_id,
            secret_value="forged-or-unverified-token",
        )

        decision: PolicyDecision = policy_engine.evaluate_request(request)

        assert decision.is_allowed is False

        events = event_logger.get_unauthorized_events()
        assert len(events) == 1

        security_event: SecurityEvent = events[0]
        assert security_event.source_service == "inventory-service"
        assert security_event.target_service == "warehouse-service"
        assert security_event.secret_id == secret_id
        assert security_event.event_type == "UNAUTHORIZED_ACCESS"
        assert "unverified" in security_event.reason.lower() or "invalid" in security_event.reason.lower()

    def test_evaluate_request_with_retired_secret_denies_and_records_event(
        self,
        policy_engine: ZeroTrustPolicyEngine,
        secret_rotator: SecretRotator,
        event_logger: SecurityEventLogger,
    ) -> None:
        """Incoming communication using a retired secret must be denied and audited."""
        secret_id = "analytics-to-data-lake"
        initial_value = "legacy-token-v1"
        secret_rotator.register_secret(
            secret_id=secret_id,
            value=initial_value,
            expires_at=datetime.now(timezone.utc) - timedelta(seconds=1),
            status=SecretStatus.ACTIVE,
        )

        # Rotate secret, causing initial_value to become RETIRED
        secret_rotator.rotate_secret(secret_id=secret_id)

        request = NetworkRequest(
            source_service="analytics-service",
            target_service="data-lake-service",
            secret_id=secret_id,
            secret_value=initial_value,
        )

        decision: PolicyDecision = policy_engine.evaluate_request(request)

        assert decision.is_allowed is False

        events = event_logger.get_unauthorized_events()
        assert len(events) == 1
        assert events[0].event_type == "UNAUTHORIZED_ACCESS"
        assert events[0].source_service == "analytics-service"
        assert "retired" in events[0].reason.lower()

    def test_evaluate_request_with_nonexistent_secret_id_denies_and_records_event(
        self,
        policy_engine: ZeroTrustPolicyEngine,
        event_logger: SecurityEventLogger,
    ) -> None:
        """Communication using an unknown secret ID is treated as unverified and denied."""
        request = NetworkRequest(
            source_service="rogue-service",
            target_service="auth-service",
            secret_id="unknown-secret-id",
            secret_value="some-token",
        )

        decision: PolicyDecision = policy_engine.evaluate_request(request)

        assert decision.is_allowed is False

        events = event_logger.get_unauthorized_events()
        assert len(events) == 1
        assert events[0].event_type == "UNAUTHORIZED_ACCESS"
        assert events[0].source_service == "rogue-service"
        assert events[0].secret_id == "unknown-secret-id"


# ============================================================================
# SecurityEventLogger Unit Tests
# ============================================================================

class TestSecurityEventLogger:
    """Verifies that security audit records maintain structural integrity."""

    def test_record_and_retrieve_unauthorized_event(
        self, event_logger: SecurityEventLogger
    ) -> None:
        """Event logger captures details immutably."""
        now = datetime.now(timezone.utc)
        event = SecurityEvent(
            event_type="UNAUTHORIZED_ACCESS",
            source_service="service-a",
            target_service="service-b",
            secret_id="secret-abc",
            reason="Validity lease expired",
            timestamp=now,
        )

        event_logger.log_event(event)

        logged_events = event_logger.get_unauthorized_events()
        assert len(logged_events) == 1
        logged = logged_events[0]
        assert logged.event_type == "UNAUTHORIZED_ACCESS"
        assert logged.source_service == "service-a"
        assert logged.target_service == "service-b"
        assert logged.secret_id == "secret-abc"
        assert logged.timestamp == now

    def test_filter_events_by_service(
        self, event_logger: SecurityEventLogger
    ) -> None:
        """Event logger allows filtering unauthorized events by source service."""
        event_logger.log_event(
            SecurityEvent(
                event_type="UNAUTHORIZED_ACCESS",
                source_service="service-alpha",
                target_service="service-omega",
                secret_id="sec-1",
                reason="Invalid token",
                timestamp=datetime.now(timezone.utc),
            )
        )
        event_logger.log_event(
            SecurityEvent(
                event_type="UNAUTHORIZED_ACCESS",
                source_service="service-beta",
                target_service="service-omega",
                secret_id="sec-2",
                reason="Invalid token",
                timestamp=datetime.now(timezone.utc),
            )
        )

        alpha_events = event_logger.get_events_by_source("service-alpha")
        assert len(alpha_events) == 1
        assert alpha_events[0].source_service == "service-alpha"