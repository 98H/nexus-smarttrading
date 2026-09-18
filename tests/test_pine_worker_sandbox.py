"""
Unit tests for Pine Script Sandboxed WebWorker Execution Sandbox.

Story 3.2.3: Build Pine Script Sandboxed WebWorker Execution Sandbox
Target Modules:
    - src/services/pine_worker_sandbox.py
    - src/models/pine_execution.py
"""

import os
import time
from typing import Dict, List
import pytest

from src.models.pine_execution import (
    MarketSeriesData,
    PineCompiledPayload,
    PineExecutionResult,
    PineExecutionStatus,
    PineIndicatorOutput,
    PineSignal,
    PineSignalType,
    SandboxExecutionLimits,
    SandboxExecutionError,
    SandboxMemoryLimitError,
    SandboxSecurityViolationError,
    SandboxTimeoutError,
)
from src.services.pine_worker_sandbox import PineWorkerSandbox


# =====================================================================
# Fixtures
# =====================================================================


@pytest.fixture
def sample_market_series() -> MarketSeriesData:
    """Provides a deterministic 10-bar OHLCV market data series."""
    base_time = 1700000000
    timestamps = [base_time + (i * 60) for i in range(10)]
    return MarketSeriesData(
        timestamps=timestamps,
        open=[100.0 + i for i in range(10)],
        high=[105.0 + i for i in range(10)],
        low=[95.0 + i for i in range(10)],
        close=[102.0 + i for i in range(10)],
        volume=[1000.0 * (i + 1) for i in range(10)],
    )


@pytest.fixture
def sample_compiled_payload() -> PineCompiledPayload:
    """Provides a valid compiled Pine Script payload (SMA Crossover)."""
    return PineCompiledPayload(
        script_id="pine_script_crossover_v1",
        version="5",
        bytecode="""
        // Mock IR/bytecode for Pine Script Worker execution
        const fast = ta.sma(close, 2);
        const slow = ta.sma(close, 5);
        if (ta.crossover(fast, slow)) {
            strategy.entry("Long", strategy.long);
        }
        plot(fast, "FastSMA");
        plot(slow, "SlowSMA");
        """,
        inputs={"fast_length": 2, "slow_length": 5},
    )


@pytest.fixture
def default_limits() -> SandboxExecutionLimits:
    """Provides standard execution limits for tests."""
    return SandboxExecutionLimits(
        timeout_ms=1000,
        memory_limit_mb=64,
        max_output_size_bytes=1024 * 1024,
    )


@pytest.fixture
def sandbox() -> PineWorkerSandbox:
    """Provides an initialized PineWorkerSandbox instance."""
    return PineWorkerSandbox()


# =====================================================================
# Model Validation Tests
# =====================================================================


class TestPineExecutionModels:
    """Tests data models and validation constraints."""

    def test_market_series_length_consistency(self):
        """OHLCV arrays must all have matching dimensions."""
        timestamps = [1700000000, 1700000060]
        valid_open = [100.0, 101.0]
        valid_high = [105.0, 106.0]
        valid_low = [95.0, 96.0]
        mismatched_close = [102.0]  # Missing one element
        valid_volume = [1000.0, 1100.0]

        with pytest.raises(ValueError):
            MarketSeriesData(
                timestamps=timestamps,
                open=valid_open,
                high=valid_high,
                low=valid_low,
                close=mismatched_close,
                volume=valid_volume,
            )

    def test_market_series_requires_non_empty_data(self):
        """OHLCV data cannot be empty."""
        with pytest.raises(ValueError):
            MarketSeriesData(
                timestamps=[],
                open=[],
                high=[],
                low=[],
                close=[],
                volume=[],
            )

    def test_sandbox_execution_limits_bounds(self):
        """Execution limits must be strictly positive."""
        with pytest.raises(ValueError):
            SandboxExecutionLimits(timeout_ms=0, memory_limit_mb=64)

        with pytest.raises(ValueError):
            SandboxExecutionLimits(timeout_ms=500, memory_limit_mb=-10)

        with pytest.raises(ValueError):
            SandboxExecutionLimits(timeout_ms=500, memory_limit_mb=64, max_output_size_bytes=0)

    def test_pine_signal_attributes(self):
        """PineSignal correctly holds signal semantics and attributes."""
        signal = PineSignal(
            timestamp=1700000060,
            signal_type=PineSignalType.BUY,
            price=103.5,
            label="Long Entry",
            metadata={"bar_index": 1, "order_id": "Long"},
        )
        assert signal.timestamp == 1700000060
        assert signal.signal_type == PineSignalType.BUY
        assert signal.price == 103.5
        assert signal.label == "Long Entry"
        assert signal.metadata["order_id"] == "Long"


# =====================================================================
# Happy Path Sandbox Execution Tests
# =====================================================================


class TestPineWorkerSandboxExecution:
    """Tests normal execution lifecycle and output normalization."""

    def test_successful_execution_returns_normalized_outputs(
        self,
        sandbox: PineWorkerSandbox,
        sample_compiled_payload: PineCompiledPayload,
        sample_market_series: MarketSeriesData,
        default_limits: SandboxExecutionLimits,
    ):
        """Valid script and series must return indicators and signals with execution metrics."""
        result: PineExecutionResult = sandbox.execute(
            payload=sample_compiled_payload,
            series=sample_market_series,
            limits=default_limits,
        )

        assert result.status == PineExecutionStatus.SUCCESS
        assert isinstance(result.signals, list)
        assert isinstance(result.indicator_outputs, dict)
        assert result.execution_time_ms > 0
        assert result.memory_used_bytes > 0
        assert result.execution_time_ms <= default_limits.timeout_ms

        # Check indicator outputs
        assert "FastSMA" in result.indicator_outputs
        assert "SlowSMA" in result.indicator_outputs
        fast_sma: PineIndicatorOutput = result.indicator_outputs["FastSMA"]
        assert len(fast_sma.values) == len(sample_market_series.timestamps)
        assert fast_sma.name == "FastSMA"

        # Check signals
        for sig in result.signals:
            assert isinstance(sig, PineSignal)
            assert sig.signal_type in [PineSignalType.BUY, PineSignalType.SELL, PineSignalType.CLOSE]
            assert sig.timestamp in sample_market_series.timestamps
            assert sig.price > 0

    def test_script_with_no_signals_returns_empty_signal_list(
        self,
        sandbox: PineWorkerSandbox,
        sample_market_series: MarketSeriesData,
        default_limits: SandboxExecutionLimits,
    ):
        """Script that only plots indicators returns empty signals list, not None."""
        payload = PineCompiledPayload(
            script_id="indicator_only",
            version="5",
            bytecode="""
            const hl2 = (high + low) / 2;
            plot(hl2, "MedianPrice");
            """,
        )

        result = sandbox.execute(
            payload=payload,
            series=sample_market_series,
            limits=default_limits,
        )

        assert result.status == PineExecutionStatus.SUCCESS
        assert result.signals == []
        assert "MedianPrice" in result.indicator_outputs
        assert len(result.indicator_outputs["MedianPrice"].values) == len(sample_market_series.timestamps)


# =====================================================================
# Resource Boundary & Limit Enforcement Tests
# =====================================================================


class TestPineWorkerSandboxResourceLimits:
    """Tests strict enforcement of execution time, memory, and output boundaries."""

    def test_timeout_execution_terminates_and_raises(
        self,
        sandbox: PineWorkerSandbox,
        sample_market_series: MarketSeriesData,
    ):
        """Scripts entering infinite loops must abort within limits and raise SandboxTimeoutError."""
        infinite_loop_payload = PineCompiledPayload(
            script_id="infinite_loop",
            version="5",
            bytecode="""
            let counter = 0;
            while (true) {
                counter++;
            }
            """,
        )
        strict_timeout = SandboxExecutionLimits(timeout_ms=100, memory_limit_mb=64)

        start_time = time.monotonic()
        with pytest.raises(SandboxTimeoutError):
            sandbox.execute(
                payload=infinite_loop_payload,
                series=sample_market_series,
                limits=strict_timeout,
            )
        elapsed_time_ms = (time.monotonic() - start_time) * 1000

        # Worker must have been killed promptly, with tolerance buffer
        assert elapsed_time_ms < 1500

    def test_memory_limit_exceeded_raises_error(
        self,
        sandbox: PineWorkerSandbox,
        sample_market_series: MarketSeriesData,
    ):
        """Scripts allocating excessive memory must be terminated with SandboxMemoryLimitError."""
        memory_hog_payload = PineCompiledPayload(
            script_id="memory_hog",
            version="5",
            bytecode="""
            // Attempt to allocate a large array exceeding limit
            const arrays = [];
            for (let i = 0; i < 100000; i++) {
                arrays.push(new Array(100000).fill(1.0));
            }
            """,
        )
        low_memory_limit = SandboxExecutionLimits(timeout_ms=2000, memory_limit_mb=16)

        with pytest.raises(SandboxMemoryLimitError):
            sandbox.execute(
                payload=memory_hog_payload,
                series=sample_market_series,
                limits=low_memory_limit,
            )

    def test_max_output_size_exceeded_raises_error(
        self,
        sandbox: PineWorkerSandbox,
        sample_market_series: MarketSeriesData,
    ):
        """Payload generating excessive indicator/signal payload output must be blocked."""
        bloated_output_payload = PineCompiledPayload(
            script_id="bloated_output",
            version="5",
            bytecode="""
            // Generate thousands of massive indicator plot structures
            for (let i = 0; i < 500; i++) {
                plot(close, "Plot_" + i.toString());
            }
            """,
        )
        strict_output_limit = SandboxExecutionLimits(
            timeout_ms=2000,
            memory_limit_mb=64,
            max_output_size_bytes=512,  # Artificially small limit
        )

        with pytest.raises(SandboxExecutionError):
            sandbox.execute(
                payload=bloated_output_payload,
                series=sample_market_series,
                limits=strict_output_limit,
            )


# =====================================================================
# Host Isolation & Security Tests
# =====================================================================


class TestPineWorkerSandboxSecurityIsolation:
    """Tests that WebWorker context does not expose host primitives or sensitive environment info."""

    @pytest.mark.parametrize(
        "exploit_code",
        [
            "const fs = require('fs'); fs.readFileSync('/etc/passwd');",
            "process.exit(1);",
            "const env = process.env;",
            "eval('console.log(process)');",
            "Function('return this')().process.mainModule.require('child_process').execSync('whoami');",
            "globalThis.constructor.constructor('return process')();",
        ],
    )
    def test_sandboxed_environment_blocks_host_globals(
        self,
        sandbox: PineWorkerSandbox,
        sample_market_series: MarketSeriesData,
        default_limits: SandboxExecutionLimits,
        exploit_code: str,
    ):
        """Worker environment must reject access to Node/Host primitives without compromising host."""
        malicious_payload = PineCompiledPayload(
            script_id="security_exploit_probe",
            version="5",
            bytecode=exploit_code,
        )

        with pytest.raises((SandboxSecurityViolationError, SandboxExecutionError)):
            sandbox.execute(
                payload=malicious_payload,
                series=sample_market_series,
                limits=default_limits,
            )

    def test_sandbox_error_sanitization_does_not_leak_host_paths(
        self,
        sandbox: PineWorkerSandbox,
        sample_market_series: MarketSeriesData,
        default_limits: SandboxExecutionLimits,
    ):
        """Errors originating from worker execution must sanitize host filesystem paths and env."""
        failing_payload = PineCompiledPayload(
            script_id="path_leak_probe",
            version="5",
            bytecode="throw new Error('Explosion at /home/user/app/runner/worker.js');",
        )

        try:
            sandbox.execute(
                payload=failing_payload,
                series=sample_market_series,
                limits=default_limits,
            )
            pytest.fail("Expected SandboxExecutionError was not raised")
        except SandboxExecutionError as exc:
            err_message = str(exc)
            err_details = getattr(exc, "details", "")

            # Ensure host system paths are redacted/not leaked
            assert "/home/user" not in err_message
            assert "/home/user" not in str(err_details)
            assert os.getcwd() not in err_message
            assert os.getcwd() not in str(err_details)


# =====================================================================
# Error Handling and Edge Cases
# =====================================================================


class TestPineWorkerSandboxErrorHandling:
    """Tests runtime error capture and worker crash recovery."""

    def test_runtime_syntax_or_math_error_raises_structured_error(
        self,
        sandbox: PineWorkerSandbox,
        sample_market_series: MarketSeriesData,
        default_limits: SandboxExecutionLimits,
    ):
        """Script runtime errors (e.g. referencing undefined variables) raise structured SandboxExecutionError."""
        invalid_script = PineCompiledPayload(
            script_id="runtime_error_script",
            version="5",
            bytecode="nonExistentFunctionCall(close);",
        )

        with pytest.raises(SandboxExecutionError) as exc_info:
            sandbox.execute(
                payload=invalid_script,
                series=sample_market_series,
                limits=default_limits,
            )

        assert exc_info.value.script_id == "runtime_error_script"
        assert exc_info.value.status == PineExecutionStatus.ERROR

    def test_sandbox_cleans_up_worker_process_on_failure(
        self,
        sandbox: PineWorkerSandbox,
        sample_market_series: MarketSeriesData,
        default_limits: SandboxExecutionLimits,
    ):
        """Sandbox worker processes must be terminated and cleaned up even when executions fail."""
        crash_script = PineCompiledPayload(
            script_id="crash_worker",
            version="5",
            bytecode="throw new Error('Fatal crash');",
        )

        initial_active_workers = sandbox.get_active_worker_count()

        with pytest.raises(SandboxExecutionError):
            sandbox.execute(
                payload=crash_script,
                series=sample_market_series,
                limits=default_limits,
            )

        # Worker count must not leak or grow after crash
        assert sandbox.get_active_worker_count() == initial_active_workers

    def test_sandbox_sequential_execution_isolation(
        self,
        sandbox: PineWorkerSandbox,
        sample_market_series: MarketSeriesData,
        default_limits: SandboxExecutionLimits,
    ):
        """State from a prior script run must not pollute a subsequent run."""
        state_leak_script_1 = PineCompiledPayload(
            script_id="leak_state",
            version="5",
            bytecode="""
            globalThis.pollutedState = 9999;
            plot(close, "Run1");
            """,
        )
        state_check_script_2 = PineCompiledPayload(
            script_id="check_state",
            version="5",
            bytecode="""
            if (typeof globalThis.pollutedState !== 'undefined') {
                throw new Error("State polluted from previous execution");
            }
            plot(close, "Run2");
            """,
        )

        result_1 = sandbox.execute(
            payload=state_leak_script_1,
            series=sample_market_series,
            limits=default_limits,
        )
        assert result_1.status == PineExecutionStatus.SUCCESS

        # Second script must not see pollutedState from run 1
        result_2 = sandbox.execute(
            payload=state_check_script_2,
            series=sample_market_series,
            limits=default_limits,
        )
        assert result_2.status == PineExecutionStatus.SUCCESS