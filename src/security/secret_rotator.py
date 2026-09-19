from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
import secrets


class SecretStatus(str, Enum):
    ACTIVE = "ACTIVE"
    RETIRED = "RETIRED"
    REVOKED = "REVOKED"


class SecretNotFoundError(Exception):
    """Raised when a requested secret or secret version is not found."""


class SecretRotationError(Exception):
    """Raised when a secret cannot be safely rotated."""


@dataclass
class Secret:
    secret_id: str
    value: str
    version: int
    expires_at: datetime
    status: SecretStatus = SecretStatus.ACTIVE


class SecretRotator:
    """Manages secret credentials, rotation lifecycles, and lease expirations."""

    def __init__(self, default_lease_duration: timedelta = timedelta(hours=1)) -> None:
        self._default_lease_duration = default_lease_duration
        self._secrets: dict[str, dict[int, Secret]] = {}

    def register_secret(
        self,
        secret_id: str,
        value: str,
        expires_at: datetime,
        status: SecretStatus = SecretStatus.ACTIVE,
        version: int | None = None,
    ) -> Secret:
        if secret_id not in self._secrets:
            self._secrets[secret_id] = {}

        if version is None:
            version = (
                max(self._secrets[secret_id].keys()) + 1
                if self._secrets[secret_id]
                else 1
            )

        secret = Secret(
            secret_id=secret_id,
            value=value,
            version=version,
            expires_at=expires_at,
            status=status,
        )
        self._secrets[secret_id][version] = secret
        return secret

    def rotate_secret(
        self,
        secret_id: str,
        lease_duration: timedelta | None = None,
    ) -> Secret:
        if secret_id not in self._secrets or not self._secrets[secret_id]:
            raise SecretNotFoundError(f"Secret '{secret_id}' not found.")

        versions = self._secrets[secret_id]
        latest_version = max(versions.keys())
        latest_secret = versions[latest_version]

        if latest_secret.status == SecretStatus.REVOKED:
            raise SecretRotationError(f"Cannot rotate revoked secret '{secret_id}'.")

        # Safely retire currently active secrets
        for secret in versions.values():
            if secret.status == SecretStatus.ACTIVE:
                secret.status = SecretStatus.RETIRED

        duration = lease_duration or self._default_lease_duration
        new_version = latest_version + 1

        existing_values = {s.value for s in versions.values()}
        new_value = secrets.token_urlsafe(32)
        while new_value in existing_values:
            new_value = secrets.token_urlsafe(32)

        new_secret = Secret(
            secret_id=secret_id,
            value=new_value,
            version=new_version,
            expires_at=datetime.now(timezone.utc) + duration,
            status=SecretStatus.ACTIVE,
        )
        versions[new_version] = new_secret
        return new_secret

    def get_secret_version(self, secret_id: str, version: int) -> Secret:
        if secret_id not in self._secrets or version not in self._secrets[secret_id]:
            raise SecretNotFoundError(
                f"Version {version} of secret '{secret_id}' not found."
            )
        return self._secrets[secret_id][version]

    def get_active_secret(self, secret_id: str) -> Secret:
        if secret_id not in self._secrets:
            raise SecretNotFoundError(f"Secret '{secret_id}' not found.")

        active_secrets = [
            s for s in self._secrets[secret_id].values()
            if s.status == SecretStatus.ACTIVE
        ]
        if not active_secrets:
            raise SecretNotFoundError(f"No active secret found for '{secret_id}'.")

        return max(active_secrets, key=lambda s: s.version)

    def contains_secret_id(self, secret_id: str) -> bool:
        return secret_id in self._secrets and bool(self._secrets[secret_id])

    def get_secret_by_value(self, secret_id: str, value: str) -> Secret | None:
        if secret_id not in self._secrets:
            return None
        for secret in self._secrets[secret_id].values():
            if secrets.compare_digest(secret.value, value):
                return secret
        return None