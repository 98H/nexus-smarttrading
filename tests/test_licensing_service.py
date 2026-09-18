from typing import Set
import pytest

from src.licensing.models import AccessDecision, LicenseType, Script, User
from src.licensing.service import LicensingService


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def licensing_service() -> LicensingService:
    """Provides an instance of the LicensingService."""
    return LicensingService()


@pytest.fixture
def author() -> User:
    """Script creator user."""
    return User(id="usr_author", username="script_creator")


@pytest.fixture
def non_author_user() -> User:
    """A registered user who is not the author of the script."""
    return User(id="usr_regular", username="regular_dev")


@pytest.fixture
def invited_user() -> User:
    """A user who has been invited to access an invite-only script."""
    return User(id="usr_invited", username="trusted_collaborator")


@pytest.fixture
def unauthorized_user() -> User:
    """A user who has not been invited to access an invite-only script."""
    return User(id="usr_unauthorized", username="random_user")


@pytest.fixture
def open_source_script(author: User) -> Script:
    """Script configured with OPEN_SOURCE license."""
    return Script(
        id="scr_open_source",
        name="Utility Helper",
        author_id=author.id,
        license_type=LicenseType.OPEN_SOURCE,
    )


@pytest.fixture
def protected_script(author: User) -> Script:
    """Script configured with PROTECTED license."""
    return Script(
        id="scr_protected",
        name="Proprietary Indicator",
        author_id=author.id,
        license_type=LicenseType.PROTECTED,
    )


@pytest.fixture
def invite_only_script(author: User, invited_user: User) -> Script:
    """Script configured with INVITE_ONLY license and one invited user."""
    return Script(
        id="scr_invite_only",
        name="VIP Strategy",
        author_id=author.id,
        license_type=LicenseType.INVITE_ONLY,
        invited_user_ids={invited_user.id},
    )


# ============================================================================
# AC 1: Open-Source License Tests
# ============================================================================


class TestOpenSourceLicensing:
    """Tests for Open-Source script access rules.

    Given a script with an Open-Source license:
    When any user requests access,
    Then both source code access and execution access are granted.
    """

    def test_open_source_grants_full_access_to_non_author(
        self,
        licensing_service: LicensingService,
        open_source_script: Script,
        non_author_user: User,
    ) -> None:
        decision: AccessDecision = licensing_service.evaluate_access(
            user=non_author_user, script=open_source_script
        )

        assert decision.can_view_source is True
        assert decision.can_execute is True

    def test_open_source_grants_full_access_to_author(
        self,
        licensing_service: LicensingService,
        open_source_script: Script,
        author: User,
    ) -> None:
        decision: AccessDecision = licensing_service.evaluate_access(
            user=author, script=open_source_script
        )

        assert decision.can_view_source is True
        assert decision.can_execute is True

    @pytest.mark.parametrize(
        "user_id,username",
        [
            ("usr_1", "trader_alice"),
            ("usr_2", "trader_bob"),
            ("usr_3", "guest_99"),
        ],
    )
    def test_open_source_grants_full_access_to_arbitrary_users(
        self,
        licensing_service: LicensingService,
        open_source_script: Script,
        user_id: str,
        username: str,
    ) -> None:
        user = User(id=user_id, username=username)
        decision: AccessDecision = licensing_service.evaluate_access(
            user=user, script=open_source_script
        )

        assert decision.can_view_source is True
        assert decision.can_execute is True


# ============================================================================
# AC 2: Protected License Tests
# ============================================================================


class TestProtectedLicensing:
    """Tests for Protected script access rules.

    Given a script with a Protected license:
    When a non-author user requests access,
    Then execution access is granted but source code access is denied.
    """

    def test_protected_grants_execution_and_denies_source_to_non_author(
        self,
        licensing_service: LicensingService,
        protected_script: Script,
        non_author_user: User,
    ) -> None:
        decision: AccessDecision = licensing_service.evaluate_access(
            user=non_author_user, script=protected_script
        )

        assert decision.can_execute is True
        assert decision.can_view_source is False

    def test_protected_grants_both_execution_and_source_to_author(
        self,
        licensing_service: LicensingService,
        protected_script: Script,
        author: User,
    ) -> None:
        """The author of a Protected script must retain full access."""
        decision: AccessDecision = licensing_service.evaluate_access(
            user=author, script=protected_script
        )

        assert decision.can_execute is True
        assert decision.can_view_source is True


# ============================================================================
# AC 3 & 4: Invite-Only License Tests
# ============================================================================


class TestInviteOnlyLicensing:
    """Tests for Invite-Only script access rules.

    Given a script with an Invite-Only license:
    - When an unauthorized user requests access, both source code access and
      execution access are denied.
    - When an authorized (invited) user requests execution access, execution
      access is granted and source code access is denied.
    """

    def test_invite_only_denies_all_access_to_unauthorized_user(
        self,
        licensing_service: LicensingService,
        invite_only_script: Script,
        unauthorized_user: User,
    ) -> None:
        decision: AccessDecision = licensing_service.evaluate_access(
            user=unauthorized_user, script=invite_only_script
        )

        assert decision.can_execute is False
        assert decision.can_view_source is False

    def test_invite_only_grants_execution_denies_source_to_invited_user(
        self,
        licensing_service: LicensingService,
        invite_only_script: Script,
        invited_user: User,
    ) -> None:
        decision: AccessDecision = licensing_service.evaluate_access(
            user=invited_user, script=invite_only_script
        )

        assert decision.can_execute is True
        assert decision.can_view_source is False

    def test_invite_only_grants_full_access_to_author(
        self,
        licensing_service: LicensingService,
        invite_only_script: Script,
        author: User,
    ) -> None:
        """The author of an Invite-Only script must retain full access."""
        decision: AccessDecision = licensing_service.evaluate_access(
            user=author, script=invite_only_script
        )

        assert decision.can_execute is True
        assert decision.can_view_source is True


# ============================================================================
# Dynamic Invite Management Tests
# ============================================================================


class TestInviteLifecycle:
    """Tests for dynamically inviting and revoking users."""

    def test_invite_user_grants_execution_access_dynamically(
        self,
        licensing_service: LicensingService,
        invite_only_script: Script,
        unauthorized_user: User,
    ) -> None:
        initial_decision = licensing_service.evaluate_access(
            user=unauthorized_user, script=invite_only_script
        )
        assert initial_decision.can_execute is False
        assert initial_decision.can_view_source is False

        licensing_service.invite_user(
            script=invite_only_script, user_id=unauthorized_user.id
        )

        updated_decision = licensing_service.evaluate_access(
            user=unauthorized_user, script=invite_only_script
        )
        assert updated_decision.can_execute is True
        assert updated_decision.can_view_source is False

    def test_revoke_user_invite_denies_execution_access(
        self,
        licensing_service: LicensingService,
        invite_only_script: Script,
        invited_user: User,
    ) -> None:
        prior_decision = licensing_service.evaluate_access(
            user=invited_user, script=invite_only_script
        )
        assert prior_decision.can_execute is True

        licensing_service.revoke_invite(
            script=invite_only_script, user_id=invited_user.id
        )

        revoked_decision = licensing_service.evaluate_access(
            user=invited_user, script=invite_only_script
        )
        assert revoked_decision.can_execute is False
        assert revoked_decision.can_view_source is False

    def test_invite_user_is_idempotent(
        self,
        licensing_service: LicensingService,
        invite_only_script: Script,
        invited_user: User,
    ) -> None:
        licensing_service.invite_user(
            script=invite_only_script, user_id=invited_user.id
        )
        decision = licensing_service.evaluate_access(
            user=invited_user, script=invite_only_script
        )
        assert decision.can_execute is True
        assert decision.can_view_source is False


# ============================================================================
# Input Validation & Boundary Conditions
# ============================================================================


class TestLicensingValidation:
    """Tests input validation and error handling for licensing operations."""

    def test_evaluate_access_raises_value_error_for_none_user(
        self,
        licensing_service: LicensingService,
        open_source_script: Script,
    ) -> None:
        with pytest.raises(ValueError):
            licensing_service.evaluate_access(user=None, script=open_source_script)  # type: ignore[arg-type]

    def test_evaluate_access_raises_value_error_for_none_script(
        self,
        licensing_service: LicensingService,
        author: User,
    ) -> None:
        with pytest.raises(ValueError):
            licensing_service.evaluate_access(user=author, script=None)  # type: ignore[arg-type]

    def test_evaluate_access_raises_value_error_for_invalid_license_type(
        self,
        licensing_service: LicensingService,
        author: User,
    ) -> None:
        invalid_script = Script(
            id="scr_invalid",
            name="Corrupt Script",
            author_id=author.id,
            license_type="UNSUPPORTED_TYPE",  # type: ignore[arg-type]
        )
        with pytest.raises(ValueError):
            licensing_service.evaluate_access(user=author, script=invalid_script)

    def test_invite_user_raises_value_error_for_empty_user_id(
        self,
        licensing_service: LicensingService,
        invite_only_script: Script,
    ) -> None:
        with pytest.raises(ValueError):
            licensing_service.invite_user(script=invite_only_script, user_id="")

    def test_revoke_invite_raises_value_error_for_empty_user_id(
        self,
        licensing_service: LicensingService,
        invite_only_script: Script,
    ) -> None:
        with pytest.raises(ValueError):
            licensing_service.revoke_invite(script=invite_only_script, user_id="")

    def test_script_model_defaults_invited_user_ids_to_empty_set(
        self,
        author: User,
    ) -> None:
        script = Script(
            id="scr_new",
            name="Fresh Script",
            author_id=author.id,
            license_type=LicenseType.INVITE_ONLY,
        )
        assert isinstance(script.invited_user_ids, set)
        assert len(script.invited_user_ids) == 0