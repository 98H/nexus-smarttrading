"""
Unit tests for scripts/build_production_images.py.

Feature: Build Multi-Stage Production Dockerfiles for Core Services
Requirement: Story 11.1.1: Build Multi-Stage Production Dockerfiles for Core Services
Acceptance Criteria:
- Given core service Docker configurations requiring minimal production footprints
- When the container build runner executes the multi-stage build process for core services
- Then it outputs slim production images excluding build-time dependencies and verifies a non-root execution user
"""

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Module to be implemented: scripts/build_production_images.py
# The imports below will fail until the target module is created.
from scripts.build_production_images import (
    CORE_SERVICES,
    DISALLOWED_BUILD_DEPENDENCIES,
    BuildResult,
    DependencyVerificationError,
    DockerBuildError,
    NonRootUserError,
    ProductionImageBuilder,
    build_core_service_image,
    main,
    verify_minimal_dependencies,
    verify_non_root_user,
)


@pytest.fixture
def mock_service_dir(tmp_path: Path) -> Path:
    """Fixture providing a mock core service directory with Dockerfile."""
    service_dir = tmp_path / "services" / "auth-service"
    service_dir.mkdir(parents=True)
    dockerfile = service_dir / "Dockerfile"
    dockerfile.write_text(
        """
        FROM python:3.11-slim AS builder
        RUN apt-get update && apt-get install -y gcc build-essential
        WORKDIR /app
        COPY . .

        FROM python:3.11-slim AS production
        RUN groupadd -r appgroup && useradd -r -g appgroup -u 10001 appuser
        WORKDIR /app
        COPY --from=builder /app /app
        USER appuser
        CMD ["python", "main.py"]
        """
    )
    return service_dir


@pytest.fixture
def mock_docker_inspect_data():
    """Fixture providing valid inspect metadata for a production container image."""
    return {
        "Config": {
            "User": "10001:10001",
            "Env": ["PATH=/usr/local/bin:/usr/bin"],
        },
        "Size": 150_000_000,  # ~150MB
    }


# ==============================================================================
# Non-Root User Verification Tests
# ==============================================================================

@pytest.mark.parametrize("valid_user", ["appuser", "10001", "appuser:appgroup", "1001:1001", "nobody", "65534"])
def test_verify_non_root_user_accepts_valid_non_root_users(valid_user: str):
    """Verifies that non-root usernames and UIDs pass validation."""
    assert verify_non_root_user(valid_user) is True


@pytest.mark.parametrize("invalid_user", ["", "root", "0", "0:0", "root:root", "ROOT"])
def test_verify_non_root_user_rejects_root_or_empty(invalid_user: str):
    """Verifies that root username, UID 0, or omitted users raise NonRootUserError."""
    with pytest.raises(NonRootUserError):
        verify_non_root_user(invalid_user)


def test_verify_non_root_user_rejects_none_type():
    """Verifies that None as user specification raises NonRootUserError."""
    with pytest.raises(NonRootUserError):
        verify_non_root_user(None)


# ==============================================================================
# Excluded Build Dependencies & Footprint Tests
# ==============================================================================

def test_verify_minimal_dependencies_passes_when_clean():
    """Ensures production images pass verification when no build dependencies exist."""
    runtime_packages = ["ca-certificates", "curl", "libssl3", "python3-minimal"]
    assert verify_minimal_dependencies(runtime_packages) is True


@pytest.mark.parametrize(
    "leaked_dependency",
    ["gcc", "g++", "make", "build-essential", "git", "python3-dev", "libc6-dev"],
)
def test_verify_minimal_dependencies_detects_build_artifacts(leaked_dependency: str):
    """Ensures presence of compiler/build-time tools raises DependencyVerificationError."""
    runtime_packages = ["ca-certificates", "curl", leaked_dependency, "python3-minimal"]
    with pytest.raises(DependencyVerificationError):
        verify_minimal_dependencies(runtime_packages)


def test_verify_minimal_dependencies_respects_custom_disallowed_list():
    """Ensures custom disallowed dependencies are enforced when provided."""
    custom_disallowed = ["valgrind", "strace"]
    packages = ["curl", "valgrind"]
    with pytest.raises(DependencyVerificationError):
        verify_minimal_dependencies(packages, disallowed=custom_disallowed)


# ==============================================================================
# Docker Multi-Stage Build Runner Tests
# ==============================================================================

@patch("subprocess.run")
def test_build_core_service_image_executes_multistage_docker_command(
    mock_subproc: MagicMock, mock_service_dir: Path, mock_docker_inspect_data: dict
):
    """Tests that build runner targets 'production' stage and builds successfully."""
    # Mock successful docker build
    build_success = subprocess.CompletedProcess(
        args=["docker", "build"], returncode=0, stdout="Successfully built", stderr=""
    )
    # Mock successful docker inspect returning non-root user metadata
    inspect_success = subprocess.CompletedProcess(
        args=["docker", "inspect"],
        returncode=0,
        stdout=json.dumps([mock_docker_inspect_data]),
        stderr="",
    )
    # Mock dpkg/package query inside container returning clean runtime packages
    pkg_query_success = subprocess.CompletedProcess(
        args=["docker", "run"],
        returncode=0,
        stdout="ca-certificates\ncurl\npython3-minimal\n",
        stderr="",
    )

    mock_subproc.side_effect = [build_success, inspect_success, pkg_query_success]

    result: BuildResult = build_core_service_image(
        service_name="auth-service",
        context_path=mock_service_dir,
        target_stage="production",
        tag="auth-service:prod-test",
    )

    assert result.success is True
    assert result.service_name == "auth-service"
    assert result.image_tag == "auth-service:prod-test"
    assert result.user == "10001:10001"

    # Assert build command targeted the multi-stage production target
    build_cmd = mock_subproc.call_args_list[0][0][0]
    assert "--target" in build_cmd
    assert build_cmd[build_cmd.index("--target") + 1] == "production"


@patch("subprocess.run")
def test_build_core_service_image_raises_on_docker_build_failure(
    mock_subproc: MagicMock, mock_service_dir: Path
):
    """Ensures DockerBuildError is raised if the underlying docker build fails."""
    mock_subproc.return_value = subprocess.CompletedProcess(
        args=["docker", "build"], returncode=1, stdout="", stderr="Step failed"
    )

    with pytest.raises(DockerBuildError):
        build_core_service_image(
            service_name="auth-service",
            context_path=mock_service_dir,
            target_stage="production",
            tag="auth-service:prod-test",
        )


@patch("subprocess.run")
def test_build_core_service_image_fails_if_built_image_runs_as_root(
    mock_subproc: MagicMock, mock_service_dir: Path
):
    """Ensures failure when the resulting production image executes as root."""
    build_success = subprocess.CompletedProcess(
        args=["docker", "build"], returncode=0, stdout="", stderr=""
    )
    root_inspect = {
        "Config": {"User": "root"},
        "Size": 150_000_000,
    }
    inspect_success = subprocess.CompletedProcess(
        args=["docker", "inspect"],
        returncode=0,
        stdout=json.dumps([root_inspect]),
        stderr="",
    )
    mock_subproc.side_effect = [build_success, inspect_success]

    with pytest.raises(NonRootUserError):
        build_core_service_image(
            service_name="auth-service",
            context_path=mock_service_dir,
            target_stage="production",
            tag="auth-service:prod-test",
        )


@patch("subprocess.run")
def test_build_core_service_image_fails_if_build_dependencies_leak(
    mock_subproc: MagicMock, mock_service_dir: Path, mock_docker_inspect_data: dict
):
    """Ensures build fails if compiler/build tools are detected in the production image."""
    build_success = subprocess.CompletedProcess(
        args=["docker", "build"], returncode=0, stdout="", stderr=""
    )
    inspect_success = subprocess.CompletedProcess(
        args=["docker", "inspect"],
        returncode=0,
        stdout=json.dumps([mock_docker_inspect_data]),
        stderr="",
    )
    leaked_pkg_query = subprocess.CompletedProcess(
        args=["docker", "run"],
        returncode=0,
        stdout="ca-certificates\ngcc\nmake\npython3-minimal\n",
        stderr="",
    )
    mock_subproc.side_effect = [build_success, inspect_success, leaked_pkg_query]

    with pytest.raises(DependencyVerificationError):
        build_core_service_image(
            service_name="auth-service",
            context_path=mock_service_dir,
            target_stage="production",
            tag="auth-service:prod-test",
        )


# ==============================================================================
# ProductionImageBuilder Class & Batch Execution Tests
# ==============================================================================

def test_core_services_constant_configured():
    """Validates that CORE_SERVICES is defined and contains core platform services."""
    assert isinstance(CORE_SERVICES, (list, tuple))
    assert len(CORE_SERVICES) > 0
    assert "auth-service" in CORE_SERVICES or "api-gateway" in CORE_SERVICES


def test_disallowed_build_dependencies_constant_configured():
    """Validates default disallowed build dependencies includes standard build tools."""
    for tool in ["gcc", "make", "build-essential"]:
        assert tool in DISALLOWED_BUILD_DEPENDENCIES


def test_builder_initialization_defaults(tmp_path: Path):
    """Validates default state of ProductionImageBuilder."""
    builder = ProductionImageBuilder(base_dir=tmp_path)
    assert builder.target_stage == "production"
    assert builder.base_dir == tmp_path
    assert builder.services == CORE_SERVICES


@patch("scripts.build_production_images.build_core_service_image")
def test_builder_build_all_processes_all_configured_services(
    mock_build_fn: MagicMock, tmp_path: Path
):
    """Ensures build_all invokes build_core_service_image for every configured service."""
    services = ["auth-service", "billing-service", "order-service"]
    for svc in services:
        (tmp_path / svc).mkdir(parents=True)

    mock_build_fn.side_effect = [
        BuildResult(service_name=s, image_tag=f"{s}:latest", user="appuser", size_bytes=100, success=True)
        for s in services
    ]

    builder = ProductionImageBuilder(base_dir=tmp_path, services=services)
    results = builder.build_all()

    assert len(results) == 3
    assert mock_build_fn.call_count == 3
    called_services = [call.kwargs["service_name"] for call in mock_build_fn.call_args_list]
    assert called_services == services


@patch("scripts.build_production_images.build_core_service_image")
def test_builder_build_all_aborts_or_collects_errors(
    mock_build_fn: MagicMock, tmp_path: Path
):
    """Ensures builder halts or captures failure when an individual service build fails."""
    services = ["auth-service", "billing-service"]
    for svc in services:
        (tmp_path / svc).mkdir(parents=True)

    mock_build_fn.side_effect = DockerBuildError()

    builder = ProductionImageBuilder(base_dir=tmp_path, services=services)
    with pytest.raises(DockerBuildError):
        builder.build_all()


# ==============================================================================
# CLI Entrypoint Tests
# ==============================================================================

@patch("scripts.build_production_images.ProductionImageBuilder.build_all")
def test_main_cli_success(mock_build_all: MagicMock):
    """Verifies main() returns exit code 0 when all service builds pass."""
    mock_build_all.return_value = [
        BuildResult(service_name="auth-service", image_tag="auth:prod", user="appuser", size_bytes=100, success=True)
    ]
    exit_code = main(["--target-stage", "production"])
    assert exit_code == 0


@patch("scripts.build_production_images.ProductionImageBuilder.build_all")
def test_main_cli_failure_returns_non_zero(mock_build_all: MagicMock):
    """Verifies main() returns non-zero exit code when an exception occurs."""
    mock_build_all.side_effect = NonRootUserError()
    exit_code = main(["--target-stage", "production"])
    assert exit_code != 0