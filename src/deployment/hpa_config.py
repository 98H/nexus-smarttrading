"""Domain model and validation for Kubernetes Horizontal Pod Autoscaler (HPA) configurations."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class HPAConfig:
    """Configuration domain model for Kubernetes Horizontal Pod Autoscaler policies.

    Attributes:
        min_replicas: Lower replica bound (must be >= 1).
        max_replicas: Upper replica bound (must be >= min_replicas).
        cpu_utilization_percentage: Target average CPU utilization percentage (1-100).
        memory_utilization_percentage: Target average memory utilization percentage (1-100).
    """

    min_replicas: int
    max_replicas: int
    cpu_utilization_percentage: Optional[int] = None
    memory_utilization_percentage: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate parameter types and domain constraints."""
        self._validate_types()
        self._validate_values()

    def _validate_types(self) -> None:
        """Ensure all fields have valid integer or None types."""
        if not isinstance(self.min_replicas, int) or isinstance(self.min_replicas, bool):
            raise TypeError("min_replicas must be an integer")

        if not isinstance(self.max_replicas, int) or isinstance(self.max_replicas, bool):
            raise TypeError("max_replicas must be an integer")

        if self.cpu_utilization_percentage is not None:
            if not isinstance(self.cpu_utilization_percentage, int) or isinstance(
                self.cpu_utilization_percentage, bool
            ):
                raise TypeError("cpu_utilization_percentage must be an integer")

        if self.memory_utilization_percentage is not None:
            if not isinstance(self.memory_utilization_percentage, int) or isinstance(
                self.memory_utilization_percentage, bool
            ):
                raise TypeError("memory_utilization_percentage must be an integer")

    def _validate_values(self) -> None:
        """Ensure values comply with Kubernetes HPA constraints."""
        if self.min_replicas < 1:
            raise ValueError("min_replicas must be at least 1")

        if self.max_replicas < self.min_replicas:
            raise ValueError("max_replicas cannot be strictly less than min_replicas")

        if self.cpu_utilization_percentage is not None:
            if not (1 <= self.cpu_utilization_percentage <= 100):
                raise ValueError("cpu_utilization_percentage must be between 1 and 100")

        if self.memory_utilization_percentage is not None:
            if not (1 <= self.memory_utilization_percentage <= 100):
                raise ValueError("memory_utilization_percentage must be between 1 and 100")

        if self.cpu_utilization_percentage is None and self.memory_utilization_percentage is None:
            raise ValueError("At least one metric (CPU or memory) must be specified for autoscaling")