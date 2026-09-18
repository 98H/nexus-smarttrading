"""
Unit tests for Kubernetes (K8s) Declarative Helm Charts and HPA Policies.

Story: 11.1.2 - Construct Kubernetes (K8s) Declarative Helm Charts and HPA Policies
Target Modules:
    - src/deployment/hpa_config.py
    - src/deployment/helm_generator.py
"""

import json
from typing import Any, Dict

import pytest

from src.deployment.helm_generator import generate_helm_hpa_values
from src.deployment.hpa_config import HPAConfig


class TestHPAConfigValidation:
    """Unit tests for HPAConfig domain model and parameter validation."""

    def test_valid_dual_metric_configuration(self):
        """Test configuration with both CPU and memory thresholds passes validation."""
        config = HPAConfig(
            min_replicas=2,
            max_replicas=10,
            cpu_utilization_percentage=80,
            memory_utilization_percentage=75,
        )

        assert config.min_replicas == 2
        assert config.max_replicas == 10
        assert config.cpu_utilization_percentage == 80
        assert config.memory_utilization_percentage == 75

    def test_valid_cpu_only_configuration(self):
        """Test configuration with only CPU utilization threshold."""
        config = HPAConfig(
            min_replicas=1,
            max_replicas=5,
            cpu_utilization_percentage=60,
        )

        assert config.min_replicas == 1
        assert config.max_replicas == 5
        assert config.cpu_utilization_percentage == 60
        assert config.memory_utilization_percentage is None

    def test_valid_memory_only_configuration(self):
        """Test configuration with only memory utilization threshold."""
        config = HPAConfig(
            min_replicas=3,
            max_replicas=8,
            memory_utilization_percentage=70,
        )

        assert config.min_replicas == 3
        assert config.max_replicas == 8
        assert config.cpu_utilization_percentage is None
        assert config.memory_utilization_percentage == 70

    def test_boundary_min_replicas_equals_max_replicas(self):
        """Test boundary condition where min_replicas equals max_replicas (valid static bound)."""
        config = HPAConfig(
            min_replicas=4,
            max_replicas=4,
            cpu_utilization_percentage=80,
        )

        assert config.min_replicas == 4
        assert config.max_replicas == 4

    def test_boundary_minimum_allowed_values(self):
        """Test lower boundary limits: min_replicas=1 and thresholds=1%."""
        config = HPAConfig(
            min_replicas=1,
            max_replicas=1,
            cpu_utilization_percentage=1,
            memory_utilization_percentage=1,
        )

        assert config.min_replicas == 1
        assert config.cpu_utilization_percentage == 1
        assert config.memory_utilization_percentage == 1

    def test_boundary_maximum_allowed_thresholds(self):
        """Test upper boundary limits: thresholds=100%."""
        config = HPAConfig(
            min_replicas=1,
            max_replicas=10,
            cpu_utilization_percentage=100,
            memory_utilization_percentage=100,
        )

        assert config.cpu_utilization_percentage == 100
        assert config.memory_utilization_percentage == 100

    @pytest.mark.parametrize("invalid_min", [0, -1, -10])
    def test_raises_for_min_replicas_less_than_one(self, invalid_min: int):
        """Min replicas must be at least 1."""
        with pytest.raises(ValueError):
            HPAConfig(
                min_replicas=invalid_min,
                max_replicas=5,
                cpu_utilization_percentage=80,
            )

    @pytest.mark.parametrize(
        "min_rep, max_rep",
        [
            (5, 4),
            (10, 1),
            (2, 1),
        ],
    )
    def test_raises_when_max_replicas_less_than_min_replicas(
        self, min_rep: int, max_rep: int
    ):
        """Max replicas cannot be strictly less than min replicas."""
        with pytest.raises(ValueError):
            HPAConfig(
                min_replicas=min_rep,
                max_replicas=max_rep,
                cpu_utilization_percentage=80,
            )

    @pytest.mark.parametrize("invalid_cpu", [0, -1, -50, 101, 150, 200])
    def test_raises_for_out_of_bound_cpu_threshold(self, invalid_cpu: int):
        """CPU threshold must be an integer between 1 and 100."""
        with pytest.raises(ValueError):
            HPAConfig(
                min_replicas=1,
                max_replicas=5,
                cpu_utilization_percentage=invalid_cpu,
            )

    @pytest.mark.parametrize("invalid_memory", [0, -5, 101, 250])
    def test_raises_for_out_of_bound_memory_threshold(self, invalid_memory: int):
        """Memory threshold must be an integer between 1 and 100."""
        with pytest.raises(ValueError):
            HPAConfig(
                min_replicas=1,
                max_replicas=5,
                memory_utilization_percentage=invalid_memory,
            )

    def test_raises_when_no_metrics_specified(self):
        """At least one metric (CPU or memory) must be supplied for autoscaling."""
        with pytest.raises(ValueError):
            HPAConfig(
                min_replicas=1,
                max_replicas=5,
                cpu_utilization_percentage=None,
                memory_utilization_percentage=None,
            )

    @pytest.mark.parametrize(
        "invalid_kwargs",
        [
            {"min_replicas": "one", "max_replicas": 5, "cpu_utilization_percentage": 80},
            {"min_replicas": 1, "max_replicas": "five", "cpu_utilization_percentage": 80},
            {"min_replicas": 1.5, "max_replicas": 5, "cpu_utilization_percentage": 80},
            {"min_replicas": 1, "max_replicas": 5.0, "cpu_utilization_percentage": 80},
            {"min_replicas": 1, "max_replicas": 5, "cpu_utilization_percentage": "80"},
            {"min_replicas": 1, "max_replicas": 5, "memory_utilization_percentage": "70"},
        ],
    )
    def test_raises_for_invalid_parameter_types(self, invalid_kwargs: Dict[str, Any]):
        """Non-integer types for replicas or thresholds must raise TypeError or ValueError."""
        with pytest.raises((TypeError, ValueError)):
            HPAConfig(**invalid_kwargs)


class TestGenerateHelmHPAValues:
    """Unit tests for generate_helm_hpa_values Helm declarative generator."""

    def test_generate_helm_hpa_values_enabled_flag(self):
        """Generated values must have autoscaling block enabled."""
        config = HPAConfig(
            min_replicas=2,
            max_replicas=10,
            cpu_utilization_percentage=80,
        )

        values = generate_helm_hpa_values(config)

        assert isinstance(values, dict)
        assert "autoscaling" in values
        assert isinstance(values["autoscaling"], dict)
        assert values["autoscaling"].get("enabled") is True

    def test_generate_helm_hpa_values_maps_replica_bounds(self):
        """Generated values must correctly map replica bounds."""
        config = HPAConfig(
            min_replicas=3,
            max_replicas=12,
            cpu_utilization_percentage=75,
        )

        values = generate_helm_hpa_values(config)
        autoscaling = values["autoscaling"]

        assert autoscaling["minReplicas"] == 3
        assert autoscaling["maxReplicas"] == 12

    def test_generate_helm_hpa_values_dual_metrics_mapping(self):
        """Generated values must map both CPU and memory targets when provided."""
        config = HPAConfig(
            min_replicas=2,
            max_replicas=8,
            cpu_utilization_percentage=80,
            memory_utilization_percentage=70,
        )

        values = generate_helm_hpa_values(config)
        autoscaling = values["autoscaling"]

        assert autoscaling["targetCPUUtilizationPercentage"] == 80
        assert autoscaling["targetMemoryUtilizationPercentage"] == 70

    def test_generate_helm_hpa_values_cpu_only_mapping(self):
        """When memory is omitted, targetCPUUtilizationPercentage is present and memory is omitted or None."""
        config = HPAConfig(
            min_replicas=1,
            max_replicas=4,
            cpu_utilization_percentage=85,
            memory_utilization_percentage=None,
        )

        values = generate_helm_hpa_values(config)
        autoscaling = values["autoscaling"]

        assert autoscaling["targetCPUUtilizationPercentage"] == 85
        assert autoscaling.get("targetMemoryUtilizationPercentage") is None

    def test_generate_helm_hpa_values_memory_only_mapping(self):
        """When CPU is omitted, targetMemoryUtilizationPercentage is present and CPU is omitted or None."""
        config = HPAConfig(
            min_replicas=2,
            max_replicas=6,
            cpu_utilization_percentage=None,
            memory_utilization_percentage=65,
        )

        values = generate_helm_hpa_values(config)
        autoscaling = values["autoscaling"]

        assert autoscaling["targetMemoryUtilizationPercentage"] == 65
        assert autoscaling.get("targetCPUUtilizationPercentage") is None

    def test_generate_helm_hpa_values_declarative_structure(self):
        """Root dictionary must only contain expected Helm keys for autoscaling."""
        config = HPAConfig(
            min_replicas=1,
            max_replicas=5,
            cpu_utilization_percentage=80,
        )

        values = generate_helm_hpa_values(config)

        assert set(values.keys()) == {"autoscaling"}
        autoscaling = values["autoscaling"]
        assert "enabled" in autoscaling
        assert "minReplicas" in autoscaling
        assert "maxReplicas" in autoscaling

    def test_generate_helm_hpa_values_json_serializable(self):
        """Declarative values must be serializable to JSON/YAML for Helm rendering."""
        config = HPAConfig(
            min_replicas=2,
            max_replicas=10,
            cpu_utilization_percentage=75,
            memory_utilization_percentage=80,
        )

        values = generate_helm_hpa_values(config)
        serialized = json.dumps(values)

        assert isinstance(serialized, str)
        deserialized = json.loads(serialized)
        assert deserialized == values

    def test_generate_helm_hpa_values_immutability_and_isolation(self):
        """Modifying the returned dictionary must not affect subsequent calls."""
        config = HPAConfig(
            min_replicas=2,
            max_replicas=10,
            cpu_utilization_percentage=80,
        )

        values_first = generate_helm_hpa_values(config)
        values_first["autoscaling"]["minReplicas"] = 999
        values_first["autoscaling"]["enabled"] = False

        values_second = generate_helm_hpa_values(config)

        assert values_second["autoscaling"]["minReplicas"] == 2
        assert values_second["autoscaling"]["enabled"] is True

    @pytest.mark.parametrize("invalid_input", [None, "invalid_config", 123, [], {}])
    def test_generate_helm_hpa_values_raises_for_invalid_config_type(
        self, invalid_input: Any
    ):
        """Passing non-HPAConfig instances must raise a TypeError or ValueError."""
        with pytest.raises((TypeError, ValueError)):
            generate_helm_hpa_values(invalid_input)