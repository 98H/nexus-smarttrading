"""Data models, protocols, and exception definitions for ArgoCD GitOps Blue-Green deployments."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Protocol, runtime_checkable


class DeploymentColor(str, Enum):
    """Enumeration of active and preview deployment environments."""

    BLUE = "blue"
    GREEN = "green"

    @property
    def opposite(self) -> "DeploymentColor":
        """Returns the alternate color for blue-green switching."""
        if self == DeploymentColor.BLUE:
            return DeploymentColor.GREEN
        return DeploymentColor.BLUE


class DeploymentHealthStatus(str, Enum):
    """Kubernetes / ArgoCD deployment health states."""

    HEALTHY = "Healthy"
    PROGRESSING = "Progressing"
    DEGRADED = "Degraded"
    SUSPENDED = "Suspended"
    MISSING = "Missing"
    UNKNOWN = "Unknown"


class BlueGreenDeploymentError(Exception):
    """Base exception for all GitOps blue-green deployment failures."""


class ActiveDeploymentUnhealthyError(BlueGreenDeploymentError):
    """Raised when the currently active deployment is not in a healthy state."""


class PreviewHealthCheckFailedError(BlueGreenDeploymentError):
    """Raised when the preview deployment fails validation checks."""


class DeploymentAlreadyInProgressError(BlueGreenDeploymentError):
    """Raised when a concurrent blue-green deployment operation is locked."""


class RequestDrainTimeoutError(BlueGreenDeploymentError):
    """Raised when in-flight requests fail to drain within the allocated timeout."""


@dataclass
class BlueGreenConfig:
    """Configuration parameters governing the blue-green rollout lifecycle."""

    app_name: str
    active_service_name: str
    preview_service_name: str
    scale_down_delay_seconds: int = 0
    max_health_retry_attempts: int = 3
    health_check_interval_seconds: float = 0.0
    drain_timeout_seconds: float = 30.0
    drain_poll_interval_seconds: float = 0.1

    def __post_init__(self) -> None:
        """Validates configuration parameters."""
        if self.scale_down_delay_seconds < 0:
            raise ValueError("scale_down_delay_seconds must be non-negative")
        if self.max_health_retry_attempts <= 0:
            raise ValueError("max_health_retry_attempts must be greater than 0")
        if self.health_check_interval_seconds < 0:
            raise ValueError("health_check_interval_seconds must be non-negative")
        if self.drain_timeout_seconds <= 0:
            raise ValueError("drain_timeout_seconds must be greater than 0")
        if self.drain_poll_interval_seconds < 0:
            raise ValueError("drain_poll_interval_seconds must be non-negative")


@dataclass
class DeploymentRevision:
    """Represents a specific revision of a color deployment."""

    revision_id: str
    color: DeploymentColor
    image: str
    replicas: int
    health_status: DeploymentHealthStatus

    @property
    def is_healthy(self) -> bool:
        """Determines if the deployment revision is completely healthy."""
        return self.health_status == DeploymentHealthStatus.HEALTHY


@dataclass
class DeploymentResult:
    """Outcome report for a completed blue-green rollout."""

    success: bool
    active_color: DeploymentColor
    active_revision: str
    previous_revision: Optional[str] = None
    drained_in_flight_requests: int = 0


@runtime_checkable
class GitOpsClientProtocol(Protocol):
    """Formal interface contract for GitOps and Kubernetes client operations."""

    def is_deployment_locked(self, app_name: str) -> bool:
        """Checks whether a deployment lock exists for the specified application."""
        ...

    def lock_deployment(self, app_name: str) -> None:
        """Acquires a deployment synchronization lock."""
        ...

    def unlock_deployment(self, app_name: str) -> None:
        """Releases the deployment synchronization lock."""
        ...

    def get_active_color(self, app_name: str) -> DeploymentColor:
        """Returns the active deployment color serving live traffic."""
        ...

    def get_preview_color(self, app_name: str) -> Optional[DeploymentColor]:
        """Returns the preview deployment color if present."""
        ...

    def get_deployment(self, app_name: str, color: DeploymentColor) -> DeploymentRevision:
        """Retrieves deployment revision metadata for a specific color."""
        ...

    def get_in_flight_requests(self, app_name: str) -> int:
        """Returns the current number of in-flight requests on the active service."""
        ...

    def create_or_update_preview(
        self,
        app_name: str,
        color: DeploymentColor,
        revision_id: str,
        image: str,
    ) -> None:
        """Creates or updates the preview deployment for validation."""
        ...

    def route_traffic(self, service_name: str, target_color: DeploymentColor) -> None:
        """Switches the active traffic selector to route live traffic to target color."""
        ...

    def scale_down(self, app_name: str, color: DeploymentColor) -> None:
        """Scales down the retired deployment color to zero replicas."""
        ...

    def delete_preview(self, app_name: str, color: DeploymentColor) -> None:
        """Tears down the preview deployment after failure or abort."""
        ...