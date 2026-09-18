"""
Unit tests for ArgoCD GitOps Blue-Green Zero-Downtime Deployment.

Story 11.3.2: Implement ArgoCD GitOps Blue-Green Zero-Downtime Deployment
Target Modules:
    - src.gitops.blue_green_service
    - src.gitops.models
"""

from unittest.mock import MagicMock, call
import pytest

from src.gitops.models import (
    ActiveDeploymentUnhealthyError,
    BlueGreenConfig,
    BlueGreenDeploymentError,
    DeploymentAlreadyInProgressError,
    DeploymentColor,
    DeploymentHealthStatus,
    DeploymentResult,
    DeploymentRevision,
    PreviewHealthCheckFailedError,
)
from src.gitops.blue_green_service import BlueGreenDeploymentManager


@pytest.fixture
def mock_gitops_client():
    """Mocks external GitOps/Kubernetes client I/O."""
    client = MagicMock()
    # Default behavior for a healthy blue deployment
    client.get_active_color.return_value = DeploymentColor.BLUE
    client.get_deployment.return_value = DeploymentRevision(
        revision_id="sha-blue-v1",
        color=DeploymentColor.BLUE,
        image="app:1.0.0",
        replicas=3,
        health_status=DeploymentHealthStatus.HEALTHY,
    )
    client.get_in_flight_requests.return_value = 0
    return client


@pytest.fixture
def default_config():
    """Default Blue-Green configuration for testing."""
    return BlueGreenConfig(
        app_name="payments-service",
        active_service_name="payments-active-svc",
        preview_service_name="payments-preview-svc",
        scale_down_delay_seconds=0,
        max_health_retry_attempts=3,
        health_check_interval_seconds=0.0,
    )


@pytest.fixture
def deployment_manager(mock_gitops_client):
    """Instantiates BlueGreenDeploymentManager with mocked external client."""
    return BlueGreenDeploymentManager(gitops_client=mock_gitops_client)


class TestBlueGreenDeploymentSuccess:
    """Tests covering successful zero-downtime blue-green rollouts."""

    def test_deploy_promotes_green_when_blue_is_active(
        self, deployment_manager, mock_gitops_client, default_config
    ):
        """
        Verify that when Blue is active and healthy:
        1. Green preview deployment is provisioned.
        2. Green passes health checks.
        3. Active service traffic selector is routed to Green.
        4. In-flight connections drain.
        5. Old Blue deployment is scaled down.
        """
        # Blue active
        mock_gitops_client.get_active_color.return_value = DeploymentColor.BLUE
        mock_gitops_client.get_deployment.side_effect = [
            # Initial active check
            DeploymentRevision(
                revision_id="sha-v1",
                color=DeploymentColor.BLUE,
                image="app:v1",
                replicas=3,
                health_status=DeploymentHealthStatus.HEALTHY,
            ),
            # Green preview health check
            DeploymentRevision(
                revision_id="sha-v2",
                color=DeploymentColor.GREEN,
                image="app:v2",
                replicas=3,
                health_status=DeploymentHealthStatus.HEALTHY,
            ),
        ]
        mock_gitops_client.get_in_flight_requests.side_effect = [15, 0]

        result = deployment_manager.deploy(
            config=default_config,
            target_revision="sha-v2",
            target_image="app:v2",
        )

        assert isinstance(result, DeploymentResult)
        assert result.success is True
        assert result.active_color == DeploymentColor.GREEN
        assert result.active_revision == "sha-v2"
        assert result.previous_revision == "sha-v1"
        assert result.drained_in_flight_requests == 15

        # Verify order of operations
        mock_gitops_client.create_or_update_preview.assert_called_once_with(
            app_name=default_config.app_name,
            color=DeploymentColor.GREEN,
            revision_id="sha-v2",
            image="app:v2",
        )
        mock_gitops_client.route_traffic.assert_called_once_with(
            service_name=default_config.active_service_name,
            target_color=DeploymentColor.GREEN,
        )
        mock_gitops_client.scale_down.assert_called_once_with(
            app_name=default_config.app_name,
            color=DeploymentColor.BLUE,
        )

    def test_deploy_promotes_blue_when_green_is_active(
        self, deployment_manager, mock_gitops_client, default_config
    ):
        """Verify the alternating flip: promote to Blue when Green is currently active."""
        mock_gitops_client.get_active_color.return_value = DeploymentColor.GREEN
        mock_gitops_client.get_deployment.side_effect = [
            # Initial active check
            DeploymentRevision(
                revision_id="sha-v2",
                color=DeploymentColor.GREEN,
                image="app:v2",
                replicas=3,
                health_status=DeploymentHealthStatus.HEALTHY,
            ),
            # Blue preview health check
            DeploymentRevision(
                revision_id="sha-v3",
                color=DeploymentColor.BLUE,
                image="app:v3",
                replicas=3,
                health_status=DeploymentHealthStatus.HEALTHY,
            ),
        ]

        result = deployment_manager.deploy(
            config=default_config,
            target_revision="sha-v3",
            target_image="app:v3",
        )

        assert result.success is True
        assert result.active_color == DeploymentColor.BLUE
        assert result.active_revision == "sha-v3"
        mock_gitops_client.create_or_update_preview.assert_called_once_with(
            app_name=default_config.app_name,
            color=DeploymentColor.BLUE,
            revision_id="sha-v3",
            image="app:v3",
        )
        mock_gitops_client.route_traffic.assert_called_once_with(
            service_name=default_config.active_service_name,
            target_color=DeploymentColor.BLUE,
        )
        mock_gitops_client.scale_down.assert_called_once_with(
            app_name=default_config.app_name,
            color=DeploymentColor.GREEN,
        )

    def test_health_check_retries_until_healthy(
        self, deployment_manager, mock_gitops_client, default_config
    ):
        """Preview deployment starts PROGRESSING and becomes HEALTHY on subsequent check."""
        mock_gitops_client.get_deployment.side_effect = [
            # Active blue
            DeploymentRevision(
                revision_id="sha-v1",
                color=DeploymentColor.BLUE,
                image="app:v1",
                replicas=3,
                health_status=DeploymentHealthStatus.HEALTHY,
            ),
            # Preview green check 1: PROGRESSING
            DeploymentRevision(
                revision_id="sha-v2",
                color=DeploymentColor.GREEN,
                image="app:v2",
                replicas=3,
                health_status=DeploymentHealthStatus.PROGRESSING,
            ),
            # Preview green check 2: HEALTHY
            DeploymentRevision(
                revision_id="sha-v2",
                color=DeploymentColor.GREEN,
                image="app:v2",
                replicas=3,
                health_status=DeploymentHealthStatus.HEALTHY,
            ),
        ]

        result = deployment_manager.deploy(
            config=default_config,
            target_revision="sha-v2",
            target_image="app:v2",
        )

        assert result.success is True
        assert mock_gitops_client.route_traffic.call_count == 1


class TestZeroDowntimeDraining:
    """Tests ensuring in-flight requests are not dropped during switchover."""

    def test_in_flight_requests_drain_before_scale_down(
        self, deployment_manager, mock_gitops_client, default_config
    ):
        """Scale down must not occur while in-flight requests are still non-zero."""
        mock_gitops_client.get_deployment.side_effect = [
            DeploymentRevision(
                revision_id="sha-v1",
                color=DeploymentColor.BLUE,
                image="app:v1",
                replicas=3,
                health_status=DeploymentHealthStatus.HEALTHY,
            ),
            DeploymentRevision(
                revision_id="sha-v2",
                color=DeploymentColor.GREEN,
                image="app:v2",
                replicas=3,
                health_status=DeploymentHealthStatus.HEALTHY,
            ),
        ]
        # In-flight counts: 40 -> 12 -> 0
        mock_gitops_client.get_in_flight_requests.side_effect = [40, 12, 0]

        result = deployment_manager.deploy(
            config=default_config,
            target_revision="sha-v2",
            target_image="app:v2",
        )

        assert result.drained_in_flight_requests == 40
        assert mock_gitops_client.get_in_flight_requests.call_count == 3
        # Assert scale_down is called only after in_flight becomes 0
        mock_gitops_client.scale_down.assert_called_once_with(
            app_name=default_config.app_name,
            color=DeploymentColor.BLUE,
        )

    def test_traffic_routing_precedes_scale_down(
        self, deployment_manager, mock_gitops_client, default_config
    ):
        """Active traffic routing must execute strictly prior to scaling down the retired color."""
        mock_gitops_client.get_deployment.side_effect = [
            DeploymentRevision(
                revision_id="sha-v1",
                color=DeploymentColor.BLUE,
                image="app:v1",
                replicas=3,
                health_status=DeploymentHealthStatus.HEALTHY,
            ),
            DeploymentRevision(
                revision_id="sha-v2",
                color=DeploymentColor.GREEN,
                image="app:v2",
                replicas=3,
                health_status=DeploymentHealthStatus.HEALTHY,
            ),
        ]
        call_tracker = MagicMock()
        mock_gitops_client.route_traffic.side_effect = lambda *a, **kw: call_tracker("route_traffic")
        mock_gitops_client.scale_down.side_effect = lambda *a, **kw: call_tracker("scale_down")

        deployment_manager.deploy(
            config=default_config,
            target_revision="sha-v2",
            target_image="app:v2",
        )

        expected_order = [call("route_traffic"), call("scale_down")]
        assert call_tracker.mock_calls == expected_order


class TestDeploymentFailureModes:
    """Tests covering error cases, rollbacks, and aborted workflows."""

    def test_deploy_aborts_when_active_blue_is_unhealthy(
        self, deployment_manager, mock_gitops_client, default_config
    ):
        """If active deployment is currently degraded/unhealthy, abort before creating preview."""
        mock_gitops_client.get_deployment.return_value = DeploymentRevision(
            revision_id="sha-v1",
            color=DeploymentColor.BLUE,
            image="app:v1",
            replicas=3,
            health_status=DeploymentHealthStatus.DEGRADED,
        )

        with pytest.raises(ActiveDeploymentUnhealthyError):
            deployment_manager.deploy(
                config=default_config,
                target_revision="sha-v2",
                target_image="app:v2",
            )

        mock_gitops_client.create_or_update_preview.assert_not_called()
        mock_gitops_client.route_traffic.assert_not_called()
        mock_gitops_client.scale_down.assert_not_called()

    def test_deploy_fails_and_cleans_preview_when_health_check_fails(
        self, deployment_manager, mock_gitops_client, default_config
    ):
        """
        If green preview fails health check:
        1. Traffic route is never switched.
        2. Active Blue remains active.
        3. Preview deployment is purged.
        4. PreviewHealthCheckFailedError is raised.
        """
        mock_gitops_client.get_deployment.side_effect = [
            # Active Blue healthy
            DeploymentRevision(
                revision_id="sha-v1",
                color=DeploymentColor.BLUE,
                image="app:v1",
                replicas=3,
                health_status=DeploymentHealthStatus.HEALTHY,
            ),
            # Preview Green DEGRADED on all attempts
            DeploymentRevision(
                revision_id="sha-v2",
                color=DeploymentColor.GREEN,
                image="app:v2",
                replicas=3,
                health_status=DeploymentHealthStatus.DEGRADED,
            ),
            DeploymentRevision(
                revision_id="sha-v2",
                color=DeploymentColor.GREEN,
                image="app:v2",
                replicas=3,
                health_status=DeploymentHealthStatus.DEGRADED,
            ),
            DeploymentRevision(
                revision_id="sha-v2",
                color=DeploymentColor.GREEN,
                image="app:v2",
                replicas=3,
                health_status=DeploymentHealthStatus.DEGRADED,
            ),
        ]

        with pytest.raises(PreviewHealthCheckFailedError):
            deployment_manager.deploy(
                config=default_config,
                target_revision="sha-v2",
                target_image="app:v2",
            )

        # Traffic should never be routed to the degraded green
        mock_gitops_client.route_traffic.assert_not_called()
        mock_gitops_client.scale_down.assert_not_called()
        # Preview must be torn down
        mock_gitops_client.delete_preview.assert_called_once_with(
            app_name=default_config.app_name,
            color=DeploymentColor.GREEN,
        )

    def test_deploy_fails_when_same_revision_is_already_active(
        self, deployment_manager, mock_gitops_client, default_config
    ):
        """Attempting to deploy identical revision to healthy active should raise ValueError."""
        mock_gitops_client.get_deployment.return_value = DeploymentRevision(
            revision_id="sha-v1",
            color=DeploymentColor.BLUE,
            image="app:v1",
            replicas=3,
            health_status=DeploymentHealthStatus.HEALTHY,
        )

        with pytest.raises(ValueError):
            deployment_manager.deploy(
                config=default_config,
                target_revision="sha-v1",
                target_image="app:v1",
            )

        mock_gitops_client.create_or_update_preview.assert_not_called()

    def test_concurrent_deployment_raises_conflict(
        self, deployment_manager, mock_gitops_client, default_config
    ):
        """Deploying while an existing blue-green process is active raises conflict error."""
        mock_gitops_client.is_deployment_locked.return_value = True

        with pytest.raises(DeploymentAlreadyInProgressError):
            deployment_manager.deploy(
                config=default_config,
                target_revision="sha-v2",
                target_image="app:v2",
            )

        mock_gitops_client.create_or_update_preview.assert_not_called()

    def test_abort_deployment_cleans_preview_and_releases_lock(
        self, deployment_manager, mock_gitops_client, default_config
    ):
        """Manual abort cleans the preview environment and releases any gitops synchronization locks."""
        mock_gitops_client.get_preview_color.return_value = DeploymentColor.GREEN

        deployment_manager.abort(config=default_config)

        mock_gitops_client.delete_preview.assert_called_once_with(
            app_name=default_config.app_name,
            color=DeploymentColor.GREEN,
        )
        mock_gitops_client.unlock_deployment.assert_called_once_with(
            app_name=default_config.app_name
        )


class TestModelsAndValidation:
    """Unit tests for validation models in src.gitops.models."""

    def test_blue_green_config_validation(self):
        """Ensure BlueGreenConfig rejects invalid timing parameters."""
        with pytest.raises(ValueError):
            BlueGreenConfig(
                app_name="invalid-svc",
                active_service_name="active",
                preview_service_name="preview",
                scale_down_delay_seconds=-1,
            )

        with pytest.raises(ValueError):
            BlueGreenConfig(
                app_name="invalid-svc",
                active_service_name="active",
                preview_service_name="preview",
                max_health_retry_attempts=0,
            )

    def test_deployment_color_opposite(self):
        """Ensure DeploymentColor correctly determines alternate color."""
        assert DeploymentColor.BLUE.opposite == DeploymentColor.GREEN
        assert DeploymentColor.GREEN.opposite == DeploymentColor.BLUE

    def test_deployment_revision_is_healthy(self):
        """Ensure DeploymentRevision correctly reflects health states."""
        healthy_rev = DeploymentRevision(
            revision_id="sha-1",
            color=DeploymentColor.BLUE,
            image="app:1",
            replicas=3,
            health_status=DeploymentHealthStatus.HEALTHY,
        )
        degraded_rev = DeploymentRevision(
            revision_id="sha-1",
            color=DeploymentColor.BLUE,
            image="app:1",
            replicas=3,
            health_status=DeploymentHealthStatus.DEGRADED,
        )
        assert healthy_rev.is_healthy is True
        assert degraded_rev.is_healthy is False