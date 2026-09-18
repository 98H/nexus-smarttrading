"""Service orchestrating ArgoCD GitOps Blue-Green zero-downtime deployments."""

import time
from typing import Optional

from src.gitops.models import (
    ActiveDeploymentUnhealthyError,
    BlueGreenConfig,
    DeploymentAlreadyInProgressError,
    DeploymentColor,
    DeploymentResult,
    DeploymentRevision,
    GitOpsClientProtocol,
    PreviewHealthCheckFailedError,
    RequestDrainTimeoutError,
)

MIN_DRAIN_POLL_INTERVAL_SECONDS: float = 0.01


class BlueGreenDeploymentManager:
    """Manages zero-downtime blue-green rollouts via GitOps / ArgoCD."""

    def __init__(self, gitops_client: GitOpsClientProtocol) -> None:
        """Initialize the manager with an external GitOps client."""
        self.gitops_client = gitops_client

    def deploy(
        self,
        config: BlueGreenConfig,
        target_revision: str,
        target_image: str,
    ) -> DeploymentResult:
        """
        Executes a zero-downtime blue-green rollout.

        Flow:
        1. Validates deployment synchronization lock and acquires it.
        2. Checks health and state of currently active color.
        3. Provisions or updates preview deployment for alternate color.
        4. Validates preview health via polling retry loop.
        5. Switches active service traffic selector to preview color.
        6. Drains in-flight requests without connection dropping.
        7. Scales down the retired previous deployment.
        8. Releases synchronization lock in finally block.
        """
        self._acquire_lock(config.app_name)

        try:
            active_color = self._get_active_color(config.app_name)
            active_deployment = self._get_deployment(config.app_name, active_color)

            if not active_deployment.is_healthy:
                raise ActiveDeploymentUnhealthyError(
                    f"Active deployment for {config.app_name} is unhealthy: "
                    f"{active_deployment.health_status.value}"
                )

            if active_deployment.revision_id == target_revision:
                raise ValueError(
                    f"Target revision '{target_revision}' is already active on {config.app_name}"
                )

            preview_color = active_color.opposite

            self.gitops_client.create_or_update_preview(
                app_name=config.app_name,
                color=preview_color,
                revision_id=target_revision,
                image=target_image,
            )

            self._validate_preview_health(config, preview_color)

            self.gitops_client.route_traffic(
                service_name=config.active_service_name,
                target_color=preview_color,
            )

            try:
                drained_requests = self._drain_in_flight_requests(config)

                if config.scale_down_delay_seconds > 0:
                    time.sleep(config.scale_down_delay_seconds)

                self.gitops_client.scale_down(
                    app_name=config.app_name,
                    color=active_color,
                )
            except Exception:
                # Rollback active traffic to original deployment if draining or scale-down fails
                self.gitops_client.route_traffic(
                    service_name=config.active_service_name,
                    target_color=active_color,
                )
                raise

            return DeploymentResult(
                success=True,
                active_color=preview_color,
                active_revision=target_revision,
                previous_revision=active_deployment.revision_id,
                drained_in_flight_requests=drained_requests,
            )
        finally:
            self.gitops_client.unlock_deployment(app_name=config.app_name)

    def abort(self, config: BlueGreenConfig) -> None:
        """Cleans up preview resources and releases GitOps locks on abort."""
        try:
            active_color = self._get_active_color(config.app_name)
            preview_color = self._get_preview_color(config.app_name)
            if preview_color:
                # If active traffic is currently pointing to preview, restore to alternate color
                if active_color == preview_color:
                    self.gitops_client.route_traffic(
                        service_name=config.active_service_name,
                        target_color=preview_color.opposite,
                    )
                self.gitops_client.delete_preview(
                    app_name=config.app_name,
                    color=preview_color,
                )
        finally:
            self.gitops_client.unlock_deployment(app_name=config.app_name)

    def _acquire_lock(self, app_name: str) -> None:
        """Acquires deployment synchronization lock, raising error if already locked."""
        if self.gitops_client.is_deployment_locked(app_name=app_name) is True:
            raise DeploymentAlreadyInProgressError(
                f"Deployment already in progress for application '{app_name}'"
            )
        self.gitops_client.lock_deployment(app_name=app_name)

    def _get_active_color(self, app_name: str) -> DeploymentColor:
        """Resolves the current active traffic color."""
        color = self.gitops_client.get_active_color(app_name=app_name)
        if isinstance(color, str):
            return DeploymentColor(color)
        return color

    def _get_preview_color(self, app_name: str) -> Optional[DeploymentColor]:
        """Resolves current preview color if provisioned."""
        color = self.gitops_client.get_preview_color(app_name=app_name)
        if color is None:
            return None
        if isinstance(color, str):
            return DeploymentColor(color)
        return color

    def _get_deployment(self, app_name: str, color: DeploymentColor) -> DeploymentRevision:
        """Fetches deployment state for the requested color."""
        return self.gitops_client.get_deployment(app_name=app_name, color=color)

    def _validate_preview_health(
        self, config: BlueGreenConfig, preview_color: DeploymentColor
    ) -> None:
        """Polls preview deployment until it becomes healthy or exceeds retry limit."""
        healthy = False
        for attempt in range(config.max_health_retry_attempts):
            deployment = self._get_deployment(config.app_name, preview_color)
            if deployment.is_healthy:
                healthy = True
                break

            if (
                attempt < config.max_health_retry_attempts - 1
                and config.health_check_interval_seconds > 0
            ):
                time.sleep(config.health_check_interval_seconds)

        if not healthy:
            self.gitops_client.delete_preview(
                app_name=config.app_name,
                color=preview_color,
            )
            raise PreviewHealthCheckFailedError(
                f"Preview deployment ({preview_color.value}) failed health check after "
                f"{config.max_health_retry_attempts} attempts"
            )

    def _drain_in_flight_requests(self, config: BlueGreenConfig) -> int:
        """Polls in-flight request count until it drains down to zero or times out."""
        in_flight = self.gitops_client.get_in_flight_requests(app_name=config.app_name)
        initial_count = in_flight

        start_time = time.monotonic()
        poll_interval = max(config.drain_poll_interval_seconds, MIN_DRAIN_POLL_INTERVAL_SECONDS)

        while in_flight > 0:
            if time.monotonic() - start_time >= config.drain_timeout_seconds:
                raise RequestDrainTimeoutError(
                    f"Timed out after {config.drain_timeout_seconds}s waiting for {config.app_name} "
                    f"in-flight requests to drain (remaining: {in_flight})"
                )
            if poll_interval > 0:
                time.sleep(poll_interval)
            in_flight = self.gitops_client.get_in_flight_requests(app_name=config.app_name)

        return initial_count