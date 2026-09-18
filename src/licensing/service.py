"""Licensing service for evaluating access permissions and managing script invites."""

from src.licensing.models import AccessDecision, LicenseType, Script, User

__all__ = ["LicensingService"]


class LicensingService:
    """Service responsible for evaluating script access rights and invite lifecycle."""

    def evaluate_access(self, user: User, script: Script) -> AccessDecision:
        """Evaluate source code and execution access rights for a user and script.

        Args:
            user: The user requesting access.
            script: The script to evaluate access against.

        Returns:
            AccessDecision indicating whether execution and source viewing are granted.

        Raises:
            ValueError: If user or script is None, or if script has an invalid license type.
        """
        if user is None:
            raise ValueError("User cannot be None")
        if script is None:
            raise ValueError("Script cannot be None")
        if not isinstance(script.license_type, LicenseType):
            raise ValueError(f"Invalid license type: {script.license_type}")

        # Script authors retain full access regardless of license type
        if user.id == script.author_id:
            return AccessDecision(can_view_source=True, can_execute=True)

        if script.license_type == LicenseType.OPEN_SOURCE:
            return AccessDecision(can_view_source=True, can_execute=True)

        if script.license_type == LicenseType.PROTECTED:
            return AccessDecision(can_view_source=False, can_execute=True)

        if script.license_type == LicenseType.INVITE_ONLY:
            can_execute = script.invited_user_ids is not None and user.id in script.invited_user_ids
            return AccessDecision(can_view_source=False, can_execute=can_execute)

        raise ValueError(f"Unsupported license type: {script.license_type}")

    def invite_user(self, script: Script, user_id: str) -> None:
        """Grant a user execution access to an invite-only script.

        Args:
            script: The target script.
            user_id: The ID of the user to invite.

        Raises:
            ValueError: If script is None or user_id is empty.
        """
        if script is None:
            raise ValueError("Script cannot be None")
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("User ID must be a non-empty string")

        if script.invited_user_ids is None:
            script.invited_user_ids = set()
        script.invited_user_ids.add(user_id)

    def revoke_invite(self, script: Script, user_id: str) -> None:
        """Revoke a user's execution access from an invite-only script.

        Args:
            script: The target script.
            user_id: The ID of the user whose invite is being revoked.

        Raises:
            ValueError: If script is None or user_id is empty.
        """
        if script is None:
            raise ValueError("Script cannot be None")
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("User ID must be a non-empty string")

        if script.invited_user_ids is not None:
            script.invited_user_ids.discard(user_id)