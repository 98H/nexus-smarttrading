"""Production Docker image build and verification runner for core services.

Feature: Build Multi-Stage Production Dockerfiles for Core Services
Ensures slim production images exclude build-time dependencies and enforce non-root user execution.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

CORE_SERVICES: tuple[str, ...] = (
    "auth-service",
    "api-gateway",
    "billing-service",
    "order-service",
)

DISALLOWED_BUILD_DEPENDENCIES: tuple[str, ...] = (
    "gcc",
    "g++",
    "make",
    "build-essential",
    "git",
    "python3-dev",
    "libc6-dev",
)


class DockerBuildError(Exception):
    """Raised when docker build or inspection commands fail."""


class NonRootUserError(Exception):
    """Raised when container image runs as root user or user is omitted."""


class DependencyVerificationError(Exception):
    """Raised when disallowed build dependencies are present in production image."""


@dataclass
class BuildResult:
    """Represents the outcome of a Docker production image build."""

    service_name: str
    image_tag: str
    user: str
    size_bytes: int = 0
    success: bool = True


def verify_non_root_user(user: str | None) -> bool:
    """Verify that the container execution user is non-root.

    Args:
        user: The user string from Docker inspect metadata (e.g., '10001', 'appuser', '1001:1001').

    Returns:
        True if the user is non-root.

    Raises:
        NonRootUserError: If user is None, empty, or root (username or UID 0).
    """
    if user is None:
        raise NonRootUserError("Execution user must be specified (cannot be None).")

    if not isinstance(user, str):
        raise NonRootUserError(f"User must be a string, got {type(user).__name__}")

    user_str = user.strip()
    if not user_str:
        raise NonRootUserError("Execution user cannot be empty.")

    parts = user_str.split(":")
    user_part = parts[0].strip().lower()

    if not user_part or user_part in ("root", "0"):
        raise NonRootUserError(f"Container image runs as root user: '{user}'")

    return True


def verify_minimal_dependencies(
    packages: Iterable[str],
    disallowed: Iterable[str] | None = None,
) -> bool:
    """Verify that installed packages do not include disallowed build dependencies.

    Args:
        packages: Package names found inside the container image.
        disallowed: Disallowed package names. Defaults to DISALLOWED_BUILD_DEPENDENCIES.

    Returns:
        True if no disallowed dependencies are found.

    Raises:
        DependencyVerificationError: If any disallowed package is found.
    """
    disallowed_list = disallowed if disallowed is not None else DISALLOWED_BUILD_DEPENDENCIES
    disallowed_set = {d.strip().lower() for d in disallowed_list}

    leaked = []
    for pkg in packages:
        normalized = pkg.strip().lower()
        if normalized in disallowed_set:
            leaked.append(normalized)

    if leaked:
        unique_leaked = sorted(set(leaked))
        raise DependencyVerificationError(
            f"Disallowed build dependencies detected in production image: {', '.join(unique_leaked)}"
        )

    return True


def build_core_service_image(
    service_name: str,
    context_path: Path | str,
    target_stage: str = "production",
    tag: str | None = None,
    disallowed: Iterable[str] | None = None,
) -> BuildResult:
    """Build a multi-stage Docker image and verify non-root user and minimal dependencies.

    Args:
        service_name: Name of the core service.
        context_path: Path to the service build context directory containing Dockerfile.
        target_stage: Multi-stage target to build (default: 'production').
        tag: Image tag to assign. Defaults to '{service_name}:{target_stage}'.
        disallowed: Optional list of disallowed package names to verify against.

    Returns:
        BuildResult metadata.

    Raises:
        DockerBuildError: If underlying docker commands fail.
        NonRootUserError: If resulting image executes as root.
        DependencyVerificationError: If build-time dependencies leak into the image.
    """
    image_tag = tag if tag is not None else f"{service_name}:{target_stage}"
    context_str = str(context_path)

    # 1. Execute multi-stage Docker build targeting the production stage
    build_cmd = [
        "docker",
        "build",
        "--target",
        target_stage,
        "-t",
        image_tag,
        context_str,
    ]
    build_proc = subprocess.run(build_cmd, capture_output=True, text=True)
    if build_proc.returncode != 0:
        raise DockerBuildError(f"Docker build failed for {service_name}: {build_proc.stderr}")

    # 2. Inspect image metadata for execution user and footprint
    inspect_cmd = ["docker", "inspect", image_tag]
    inspect_proc = subprocess.run(inspect_cmd, capture_output=True, text=True)
    if inspect_proc.returncode != 0:
        raise DockerBuildError(f"Docker inspect failed for {image_tag}: {inspect_proc.stderr}")

    try:
        inspect_data = json.loads(inspect_proc.stdout)
        metadata = inspect_data[0] if isinstance(inspect_data, list) else inspect_data
    except (json.JSONDecodeError, IndexError) as err:
        raise DockerBuildError(f"Failed to parse inspect output for {image_tag}: {err}")

    user = metadata.get("Config", {}).get("User")
    size_bytes = metadata.get("Size", 0)

    # 3. Verify non-root execution user
    verify_non_root_user(user)

    # 4. Query container packages to ensure no build dependencies leaked
    pkg_cmd = ["docker", "run", "--rm", image_tag, "dpkg-query", "-f", "${Package}\\n", "-W"]
    pkg_proc = subprocess.run(pkg_cmd, capture_output=True, text=True)
    if pkg_proc.returncode != 0:
        raise DockerBuildError(f"Package query failed for {image_tag}: {pkg_proc.stderr}")

    packages = [line.strip() for line in pkg_proc.stdout.splitlines() if line.strip()]
    verify_minimal_dependencies(packages, disallowed=disallowed)

    return BuildResult(
        service_name=service_name,
        image_tag=image_tag,
        user=user,
        size_bytes=size_bytes,
        success=True,
    )


class ProductionImageBuilder:
    """Orchestrates building production images across configured core services."""

    def __init__(
        self,
        base_dir: Path | str = Path("."),
        target_stage: str = "production",
        services: Sequence[str] | None = None,
        disallowed_dependencies: Iterable[str] | None = None,
    ):
        self.base_dir = Path(base_dir)
        self.target_stage = target_stage
        self.services = services if services is not None else CORE_SERVICES
        self.disallowed_dependencies = disallowed_dependencies

    def build_all(self) -> list[BuildResult]:
        """Build production images for all configured services.

        Returns:
            List of BuildResult instances for each service.
        """
        results: list[BuildResult] = []
        for service_name in self.services:
            service_dir = self.base_dir / service_name
            if not service_dir.exists() and (self.base_dir / "services" / service_name).exists():
                service_dir = self.base_dir / "services" / service_name

            tag = f"{service_name}:{self.target_stage}"
            result = build_core_service_image(
                service_name=service_name,
                context_path=service_dir,
                target_stage=self.target_stage,
                tag=tag,
                disallowed=self.disallowed_dependencies,
            )
            results.append(result)
        return results


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for building production images."""
    parser = argparse.ArgumentParser(
        description="Build multi-stage production Docker images for core services."
    )
    parser.add_argument(
        "--target-stage",
        default="production",
        help="Docker build target stage (default: production)",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("."),
        help="Base directory containing core services",
    )
    parser.add_argument(
        "--services",
        nargs="*",
        default=None,
        help="Core services to build (defaults to CORE_SERVICES)",
    )

    parsed = parser.parse_args(argv)

    try:
        builder = ProductionImageBuilder(
            base_dir=parsed.base_dir,
            target_stage=parsed.target_stage,
            services=parsed.services,
        )
        builder.build_all()
        return 0
    except Exception as exc:
        print(f"Build failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())