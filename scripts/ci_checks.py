"""CI checks automation utility for code formatting, linting, and testing."""

from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List, Optional, Union

# Optional PyYAML import to support standalone execution without yaml installed
try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

# Exit code constants indicating specific check failures
EXIT_CODE_SUCCESS: int = 0
EXIT_CODE_FORMATTING_FAILED: int = 1
EXIT_CODE_LINT_FAILED: int = 2
EXIT_CODE_TESTS_FAILED: int = 3

# Active Python interpreter to ensure isolated virtualenv execution
PYTHON_EXECUTABLE: str = sys.executable or "python"

# Default commands executed during CI checks using active Python environment
DEFAULT_FORMAT_COMMAND: List[str] = [
    PYTHON_EXECUTABLE,
    "-m",
    "black",
    "--check",
    ".",
]
DEFAULT_LINT_COMMAND: List[str] = [
    PYTHON_EXECUTABLE,
    "-m",
    "ruff",
    "check",
    ".",
]
DEFAULT_TEST_COMMAND: List[str] = [PYTHON_EXECUTABLE, "-m", "pytest"]

# Expected path to GitHub Actions CI workflow definition
DEFAULT_WORKFLOW_PATH: Path = (
    Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml"
)


def run_step(step_name: str, cmd: List[str]) -> int:
    """
    Execute a single CI command step streaming output to stdout and stderr.

    Returns the exit code of the executed command. Propagates environment errors
    such as FileNotFoundError if the executable binary does not exist.
    """
    print(f"\n--- Running: {step_name} ({' '.join(cmd)}) ---", flush=True)
    result = subprocess.run(cmd)

    # Forward stdout and stderr if captured (e.g. in mocked test environments)
    stdout = getattr(result, "stdout", None)
    if stdout:
        text_out = (
            stdout.decode("utf-8", errors="replace")
            if isinstance(stdout, bytes)
            else str(stdout)
        )
        print(text_out, file=sys.stdout, flush=True)

    stderr = getattr(result, "stderr", None)
    if stderr:
        text_err = (
            stderr.decode("utf-8", errors="replace")
            if isinstance(stderr, bytes)
            else str(stderr)
        )
        print(text_err, file=sys.stderr, flush=True)

    if result.returncode != 0:
        print(
            f"Step '{step_name}' failed with exit code {result.returncode}.",
            file=sys.stderr,
            flush=True,
        )

    return result.returncode


def run_formatting(cmd: Optional[List[str]] = None) -> int:
    """Execute code formatting checks."""
    command = cmd if cmd is not None else DEFAULT_FORMAT_COMMAND
    code = run_step("formatting", command)
    if code != 0:
        print("Code formatting check failed.", file=sys.stderr, flush=True)
        return EXIT_CODE_FORMATTING_FAILED
    return EXIT_CODE_SUCCESS


def run_static_analysis(cmd: Optional[List[str]] = None) -> int:
    """Execute static analysis and linting checks."""
    command = cmd if cmd is not None else DEFAULT_LINT_COMMAND
    code = run_step("static analysis / lint", command)
    if code != 0:
        print(
            "Static analysis / lint check failed.",
            file=sys.stderr,
            flush=True,
        )
        return EXIT_CODE_LINT_FAILED
    return EXIT_CODE_SUCCESS


def run_tests(cmd: Optional[List[str]] = None) -> int:
    """Execute test suite using pytest."""
    command = cmd if cmd is not None else DEFAULT_TEST_COMMAND
    code = run_step("pytest test suite", command)
    if code != 0:
        print("Pytest test execution failed.", file=sys.stderr, flush=True)
        return EXIT_CODE_TESTS_FAILED
    return EXIT_CODE_SUCCESS


def run_all_checks() -> int:
    """
    Run formatting, static analysis, and pytest test suite in order.

    Idempotent and read-only. Terminates early and returns a non-zero status
    code indicating the specific failed step.
    """
    formatting_code = run_formatting()
    if formatting_code != 0:
        return formatting_code

    lint_code = run_static_analysis()
    if lint_code != 0:
        return lint_code

    test_code = run_tests()
    if test_code != 0:
        return test_code

    print("All CI checks passed successfully.", file=sys.stdout, flush=True)
    return EXIT_CODE_SUCCESS


def validate_workflow_schema(
    workflow: Union[Dict[str, Any], Path, str]
) -> bool:
    """
    Validate that a GitHub Actions CI workflow config adheres to requirements.

    Ensures triggers for push and pull_request on main/master branches and jobs
    executing linting, testing, or ci_checks.py are present. Accepts a parsed
    dictionary, a Path object, or a raw YAML string / file path.
    """
    if isinstance(workflow, dict):
        data = workflow
    elif isinstance(workflow, Path):
        if not workflow.is_file():
            raise ValueError(f"Workflow file does not exist: {workflow}")
        if yaml is None:
            raise RuntimeError(
                "PyYAML is required to parse YAML workflow files."
            )
        try:
            parsed = yaml.safe_load(workflow.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ValueError(f"Failed to parse YAML workflow: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("Workflow file must parse into a dictionary.")
        data = parsed
    elif isinstance(workflow, str):
        is_inline_content = (
            "\n" in workflow
            or "\r" in workflow
            or workflow.strip().startswith("{")
        )
        if is_inline_content:
            raw_content = workflow
        else:
            path_candidate = Path(workflow)
            if path_candidate.is_file():
                raw_content = path_candidate.read_text(encoding="utf-8")
            else:
                raise ValueError(f"Workflow file does not exist: {workflow}")

        if yaml is None:
            raise RuntimeError(
                "PyYAML is required to parse YAML workflow content."
            )
        try:
            parsed = yaml.safe_load(raw_content)
        except yaml.YAMLError as exc:
            raise ValueError(f"Failed to parse YAML workflow: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("Workflow content must parse into a dictionary.")
        data = parsed
    else:
        raise ValueError("Workflow must be a dictionary, Path, or YAML string.")

    # In YAML 1.1, unquoted 'on' can parse as boolean True or string 'on'
    triggers = data.get("on") if "on" in data else data.get(True)
    if triggers is None:
        raise ValueError("Workflow must define an 'on' trigger section.")

    if isinstance(triggers, list):
        if "push" not in triggers or "pull_request" not in triggers:
            raise ValueError(
                "Workflow triggers must include 'push' and 'pull_request'."
            )
    elif isinstance(triggers, dict):
        if "push" not in triggers:
            raise ValueError("Trigger 'push' must be defined in workflow.")
        if "pull_request" not in triggers:
            raise ValueError(
                "Trigger 'pull_request' must be defined in workflow."
            )

        for event in ("push", "pull_request"):
            event_config = triggers.get(event)
            if isinstance(event_config, dict) and "branches" in event_config:
                branches = event_config["branches"]
                if isinstance(branches, str):
                    branches = [branches]
                if not isinstance(branches, (list, tuple)) or not any(
                    b in ("main", "master") for b in branches
                ):
                    raise ValueError(
                        f"'{event}' trigger must target 'main' or 'master'."
                    )
    else:
        raise ValueError("Invalid format for 'on' trigger section.")

    if (
        "jobs" not in data
        or not isinstance(data["jobs"], dict)
        or len(data["jobs"]) == 0
    ):
        raise ValueError("Workflow must define a non-empty 'jobs' section.")

    all_steps: List[Any] = []
    for job_id, job_config in data["jobs"].items():
        if not isinstance(job_config, dict) or "steps" not in job_config:
            raise ValueError(f"Job '{job_id}' must contain 'steps'.")
        steps = job_config["steps"]
        if not isinstance(steps, list) or len(steps) == 0:
            raise ValueError(f"Job '{job_id}' steps must be a non-empty list.")
        all_steps.extend(steps)

    step_commands: List[str] = [
        step.get("run", "")
        for step in all_steps
        if isinstance(step, dict) and "run" in step
    ]
    combined_commands = " ".join(step_commands)

    has_ci_script = "ci_checks.py" in combined_commands
    has_lint_and_test = (
        any(
            tool in combined_commands
            for tool in ("black", "ruff", "flake8", "lint")
        )
        and "pytest" in combined_commands
    )

    if not (has_ci_script or has_lint_and_test):
        raise ValueError(
            "Workflow jobs must execute scripts/ci_checks.py or explicit "
            "Python lint and test steps."
        )

    return True


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI execution flow returning integer exit code."""
    return run_all_checks()


def cli_entrypoint() -> None:
    """CLI entrypoint that exits the process with the step status code."""
    sys.exit(main())


if __name__ == "__main__":
    cli_entrypoint()