"""Domain models for script licensing."""

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum

__all__ = ["LicenseType", "User", "Script", "AccessDecision"]


class LicenseType(str, Enum):
    """Supported license types for scripts."""

    OPEN_SOURCE = "OPEN_SOURCE"
    PROTECTED = "PROTECTED"
    INVITE_ONLY = "INVITE_ONLY"


@dataclass(frozen=True)
class User:
    """User account entity."""

    id: str
    username: str


@dataclass
class Script:
    """Script entity representing executable code and its licensing configuration."""

    id: str
    name: str
    author_id: str
    license_type: LicenseType
    invited_user_ids: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        if self.invited_user_ids is None:
            self.invited_user_ids = set()
        elif isinstance(self.invited_user_ids, (str, bytes)):
            raise TypeError("invited_user_ids must be an iterable of strings, not a string or bytes")
        elif isinstance(self.invited_user_ids, Iterable):
            validated_ids: set[str] = set()
            for uid in self.invited_user_ids:
                if not isinstance(uid, str):
                    raise TypeError(f"All invited user IDs must be strings, got {type(uid).__name__}")
                validated_ids.add(uid)
            self.invited_user_ids = validated_ids
        else:
            raise TypeError(
                f"invited_user_ids must be an iterable of strings, got {type(self.invited_user_ids).__name__}"
            )


@dataclass(frozen=True)
class AccessDecision:
    """Permissions decision for a specific user and script."""

    can_view_source: bool
    can_execute: bool