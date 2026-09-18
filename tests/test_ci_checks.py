from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List
from unittest.mock import MagicMock, call, patch
import pytest
import yaml

# Core module under test
import scripts.ci_checks as ci_checks


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def workflow_file_path() -> Path:
    """Returns the expected path to the GitHub Actions CI workflow definition."""
    repo_root = Path(__file__).resolve().parents[1]
    workflow_path = repo_root / ".github" / "workflows" / "ci.yml"
    return workflow_path


@pytest.fixture
def mock_subprocess_run(monkeypatch: pytest.MonkeyPatch):
    """
    Mocks subprocess.run for external command execution.
    Defaults to returning a successful CompletedProcess.
    """
    mock_run = MagicMock()
    mock_run.return_value = subprocess.CompletedProcess(
        args=["mock"], returncode=0, stdout="Success", stderr=""
    )
    monkeypatch.setattr(ci_checks.subprocess, "run", mock_run)
    return mock_run


# ============================================================================
# Acceptance Criteria 1: Success Execution of Lint, Static Analysis, and Tests
# ============================================================================

class TestCiChecksSuccessFlow:
    """Tests confirming formatting, static analysis, and pytest run and succeed."""

    def test_run_all_checks_executes_required_phases_in_order(
        self, mock_subprocess_run: MagicMock
    ):
        """
        Verify that ci_checks executes the required phases:
        1. Code formatting check
        2. Static analysis / linting
        3. Pytest test suite
        And returns exit code 0 when all succeed.
        """
        exit_code = ci_checks.run_all_checks()

        assert exit_code == 0
        assert mock_subprocess_run.call_count == 3

        executed_commands: List[List[str]] = [
            call_args[0][0] for call_args in mock_subprocess_run.call_args_list
        ]

        # Verify Stage 1: Formatting check
        formatting_cmd_str = " ".join(executed_commands[0])
        assert any(
            tool in formatting_cmd_str for tool in ("black", "ruff format", "flake8")
        )
        assert any(flag in formatting_cmd_str for flag in ("--check", "check"))

        # Verify Stage 2: Static analysis / Linting
        lint_cmd_str = " ".join(executed_commands[1])
        assert any(
            tool in lint_cmd_str for tool in ("ruff", "flake8", "mypy", "pylint")
        )

        # Verify Stage 3: Pytest
        test_cmd_str = " ".join(executed_commands[2])
        assert "pytest" in test_cmd_str

    def test_run_individual_steps_pass_with_exit_code_zero(
        self, mock_subprocess_run: MagicMock
    ):
        """Verify each step can be executed individually and returns 0 on success."""
        assert ci_checks.run_formatting() == 0
        assert ci_checks.run_static_analysis() == 0
        assert ci_checks.run_tests() == 0

    def test_main_cli_returns_zero_when_all_checks_pass(
        self, mock_subprocess_run: MagicMock, monkeypatch: pytest.MonkeyPatch
    ):
        """Verify main() entrypoint returns 0 and does not raise an unhandled error."""
        monkeypatch.setattr(sys, "argv", ["ci_checks.py"])
        exit_code = ci_checks.main()
        assert exit_code == 0


# ============================================================================
# Acceptance Criteria 2: Step Failures and Non-Zero Exit Code Indication
# ============================================================================

class TestCiChecksFailureFlow:
    """Tests confirming failures terminate execution and indicate the failed step."""

    def test_formatting_failure_terminates_and_indicates_step(
        self, mock_subprocess_run: MagicMock, capsys: pytest.CaptureFixture
    ):
        """
        When the formatting step fails, ci_checks must return a non-zero exit code
        distinctly indicating a formatting failure, and must not proceed to lint/test.
        """
        mock_subprocess_run.side_effect = [
            subprocess.CompletedProcess(
                args=["formatting"], returncode=1, stdout="", stderr="Format issues"
            )
        ]

        exit_code = ci_checks.run_all_checks()

        assert exit_code != 0
        assert exit_code == ci_checks.EXIT_CODE_FORMATTING_FAILED
        assert mock_subprocess_run.call_count == 1

        captured = capsys.readouterr()
        assert "format" in captured.err.lower() or "format" in captured.out.lower()

    def test_static_analysis_failure_terminates_and_indicates_step(
        self, mock_subprocess_run: MagicMock, capsys: pytest.CaptureFixture
    ):
        """
        When static analysis fails, ci_checks must return a non-zero exit code
        distinctly indicating a linting failure, and must not execute pytest.
        """
        mock_subprocess_run.side_effect = [
            # Formatting passes
            subprocess.CompletedProcess(args=["format"], returncode=0, stdout="OK"),
            # Static analysis fails
            subprocess.CompletedProcess(
                args=["lint"], returncode=2, stdout="", stderr="Lint errors found"
            ),
        ]

        exit_code = ci_checks.run_all_checks()

        assert exit_code != 0
        assert exit_code == ci_checks.EXIT_CODE_LINT_FAILED
        assert mock_subprocess_run.call_count == 2

        captured = capsys.readouterr()
        output = (captured.err + captured.out).lower()
        assert "lint" in output or "static analysis" in output

    def test_pytest_failure_terminates_and_indicates_step(
        self, mock_subprocess_run: MagicMock, capsys: pytest.CaptureFixture
    ):
        """
        When pytest execution fails, ci_checks must terminate with a non-zero
        exit code indicating a test suite failure.
        """
        mock_subprocess_run.side_effect = [
            # Formatting passes
            subprocess.CompletedProcess(args=["format"], returncode=0, stdout="OK"),
            # Lint passes
            subprocess.CompletedProcess(args=["lint"], returncode=0, stdout="OK"),
            # Pytest fails
            subprocess.CompletedProcess(
                args=["pytest"], returncode=1, stdout="", stderr="3 tests failed"
            ),
        ]

        exit_code = ci_checks.run_all_checks()

        assert exit_code != 0
        assert exit_code == ci_checks.EXIT_CODE_TESTS_FAILED
        assert mock_subprocess_run.call_count == 3

        captured = capsys.readouterr()
        output = (captured.err + captured.out).lower()
        assert "test" in output or "pytest" in output

    def test_main_cli_raises_system_exit_with_non_zero_on_failure(
        self, mock_subprocess_run: MagicMock, monkeypatch: pytest.MonkeyPatch
    ):
        """Verify cli_entrypoint invokes sys.exit with the non-zero code on failure."""
        mock_subprocess_run.return_value = subprocess.CompletedProcess(
            args=["format"], returncode=1, stdout="", stderr="Failed"
        )
        monkeypatch.setattr(sys, "argv", ["ci_checks.py"])

        with pytest.raises(SystemExit) as exc_info:
            ci_checks.cli_entrypoint()

        assert exc_info.value.code != 0
        assert exc_info.value.code == ci_checks.EXIT_CODE_FORMATTING_FAILED

    def test_missing_binary_raises_environment_error(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Verify behavior when a required binary/command cannot be found."""
        def raise_file_not_found(*args, **kwargs):
            raise FileNotFoundError("Executable not found")

        monkeypatch.setattr(ci_checks.subprocess, "run", raise_file_not_found)

        with pytest.raises(FileNotFoundError):
            ci_checks.run_step("formatting", ["non_existent_binary"])


# ============================================================================
# Acceptance Criteria 3: GitHub Actions Workflow Definition Verification
# ============================================================================

class TestWorkflowDefinition:
    """Verifies schema, triggers, and jobs defined in GitHub Actions CI workflow."""

    def _load_workflow_yaml(self, path: Path) -> Dict[str, Any]:
        """Helper to parse the YAML workflow file safely."""
        if not path.exists():
            pytest.fail(f"Workflow file does not exist at expected location: {path}")

        content = path.read_text(encoding="utf-8")
        parsed = yaml.safe_load(content)
        assert isinstance(
            parsed, dict
        ), "Workflow file must parse into a YAML dictionary."
        return parsed

    def test_workflow_file_exists(self, workflow_file_path: Path):
        """The GitHub Actions CI workflow file must be present in the repository."""
        assert (
            workflow_file_path.is_file()
        ), f"Missing CI workflow file at {workflow_file_path}"

    def test_workflow_defines_push_and_pull_request_triggers_on_main(
        self, workflow_file_path: Path
    ):
        """
        Given the workflow definition,
        Then it defines triggers for 'push' and 'pull_request' targeting main branches.
        """
        workflow = self._load_workflow_yaml(workflow_file_path)

        # In YAML 1.1, 'on' unquoted can parse as boolean True or string 'on'
        triggers = workflow.get("on") or workflow.get(True)
        assert triggers is not None, "Workflow must define an 'on' trigger section."

        if isinstance(triggers, list):
            assert "push" in triggers
            assert "pull_request" in triggers
        elif isinstance(triggers, dict):
            assert "push" in triggers, "Trigger 'push' must be defined in workflow."
            assert (
                "pull_request" in triggers
            ), "Trigger 'pull_request' must be defined in workflow."

            # Check branch targets if specified under triggers
            for event in ("push", "pull_request"):
                event_config = triggers[event]
                if isinstance(event_config, dict) and "branches" in event_config:
                    branches = event_config["branches"]
                    assert any(
                        branch in ["main", "master"] for branch in branches
                    ), f"{event} must target main or master branch."

    def test_workflow_defines_python_lint_and_test_jobs(
        self, workflow_file_path: Path
    ):
        """
        Given the workflow definition,
        Then it targets Python lint and test jobs (or a unified CI runner job).
        """
        workflow = self._load_workflow_yaml(workflow_file_path)

        assert "jobs" in workflow, "Workflow must specify 'jobs'."
        jobs = workflow["jobs"]
        assert isinstance(jobs, dict) and len(jobs) > 0

        # Scan jobs for Python setup and execution of linting/tests or ci_checks.py
        all_steps = []
        for job_id, job_config in jobs.items():
            assert "steps" in job_config, f"Job '{job_id}' must contain 'steps'."
            all_steps.extend(job_config["steps"])

        step_commands: List[str] = [
            step.get("run", "") for step in all_steps if "run" in step
        ]
        combined_commands = " ".join(step_commands)

        # Either runs `scripts/ci_checks.py` or directly runs lint and test commands
        has_ci_script = "ci_checks.py" in combined_commands
        has_lint_and_test = (
            any(
                tool in combined_commands
                for tool in ("black", "ruff", "flake8", "lint")
            )
            and "pytest" in combined_commands
        )

        assert has_ci_script or has_lint_and_test, (
            "Workflow jobs must execute either scripts/ci_checks.py or explicit "
            "Python linting and testing steps."
        )

    def test_workflow_validator_function_rejects_invalid_schema(self):
        """
        Validates that ci_checks.validate_workflow_schema correctly validates
        compliant workflows and rejects invalid configurations.
        """
        invalid_workflow: Dict[str, Any] = {
            "name": "Invalid CI",
            "on": {"push": {"branches": ["feature-branch"]}},  # Missing pull_request
            "jobs": {},
        }

        valid_workflow: Dict[str, Any] = {
            "name": "Valid CI",
            "on": {
                "push": {"branches": ["main"]},
                "pull_request": {"branches": ["main"]},
            },
            "jobs": {
                "ci": {
                    "runs-on": "ubuntu-latest",
                    "steps": [{"run": "python scripts/ci_checks.py"}],
                }
            },
        }

        assert ci_checks.validate_workflow_schema(valid_workflow) is True

        with pytest.raises(ValueError):
            ci_checks.validate_workflow_schema(invalid_workflow)