"""Helm values generation for Kubernetes workloads and HPA policies."""

from typing import Any, Dict

from src.deployment.hpa_config import HPAConfig


def generate_helm_hpa_values(config: HPAConfig) -> Dict[str, Any]:
    """Generate declarative Helm values for Kubernetes Horizontal Pod Autoscaler.

    Args:
        config: Validated HPAConfig instance.

    Returns:
        A dictionary containing the declarative Helm autoscaling configuration.

    Raises:
        TypeError: If config is not an instance of HPAConfig.
    """
    if not isinstance(config, HPAConfig):
        raise TypeError(f"Expected HPAConfig instance, got {type(config).__name__}")

    autoscaling: Dict[str, Any] = {
        "enabled": True,
        "minReplicas": config.min_replicas,
        "maxReplicas": config.max_replicas,
    }

    if config.cpu_utilization_percentage is not None:
        autoscaling["targetCPUUtilizationPercentage"] = config.cpu_utilization_percentage

    if config.memory_utilization_percentage is not None:
        autoscaling["targetMemoryUtilizationPercentage"] = config.memory_utilization_percentage

    return {"autoscaling": autoscaling}