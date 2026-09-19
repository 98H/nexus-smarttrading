"""
Unit tests for the Auto-Save Engine with Conflict Resolution.

Specification:
- Feature: Implement Auto-Save Engine with Conflict Resolution
- Story 9.2.2:
  - Given a chart with current version N, when an auto-save delta is processed
    with expected version N, then the changes are persisted and the stored
    version increments to N + 1.
  - Given a chart with current version N + 1, when an auto-save delta arrives
    from a concurrent session specifying expected version N, then a
    ConcurrencyConflictError is raised and existing persisted state remains unaltered.

Target Modules:
- src/models/chart_state.py
- src/services/autosave_engine.py
"""

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, call

import pytest

from src.models.chart_state import ChartDelta, ChartState
from src.services.autosave_engine import (
    AutoSaveEngine,
    ChartNotFoundError,
    ConcurrencyConflictError,
)


@pytest.fixture
def sample_chart_state() -> ChartState:
    """Provides a baseline ChartState fixture at version 1."""
    return ChartState(
        chart_id="chart-100",
        version=1,
        data={
            "drawings": [{"id": "line_1", "type": "trendline"}],
            "indicators": [{"id": "ind_1", "type": "RSI", "period": 14}],
        },
        user_id="user-42",
        updated_at=datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
    )


@pytest.fixture
def mock_storage():
    """Mock storage adapter for chart state persistence."""
    storage = MagicMock()
    storage.get_chart = AsyncMock()
    storage.save_chart = AsyncMock()
    return storage


@pytest.fixture
def engine(mock_storage) -> AutoSaveEngine:
    """Provides an instance of AutoSaveEngine configured with mock storage."""
    return AutoSaveEngine(storage=mock_storage)


# ==============================================================================
# Model Tests: src/models/chart_state.py
# ==============================================================================


class TestChartStateModel:
    """Unit tests validating domain models and immutability/integrity invariants."""

    def test_chart_state_creation_valid(self):
        timestamp = datetime.now(timezone.utc)
        state = ChartState(
            chart_id="chart-001",
            version=0,
            data={"symbols": ["BTC/USD"]},
            user_id="trader-1",
            updated_at=timestamp,
        )
        assert state.chart_id == "chart-001"
        assert state.version == 0
        assert state.data == {"symbols": ["BTC/USD"]}
        assert state.user_id == "trader-1"
        assert state.updated_at == timestamp

    def test_chart_state_rejects_negative_version(self):
        with pytest.raises(ValueError):
            ChartState(
                chart_id="chart-001",
                version=-1,
                data={},
                user_id="trader-1",
            )

    def test_chart_delta_creation_valid(self):
        delta = ChartDelta(
            chart_id="chart-001",
            expected_version=1,
            changes={"theme": "dark"},
            session_id="sess-alpha",
        )
        assert delta.chart_id == "chart-001"
        assert delta.expected_version == 1
        assert delta.changes == {"theme": "dark"}
        assert delta.session_id == "sess-alpha"

    def test_chart_delta_rejects_negative_expected_version(self):
        with pytest.raises(ValueError):
            ChartDelta(
                chart_id="chart-001",
                expected_version=-1,
                changes={"theme": "dark"},
                session_id="sess-alpha",
            )


# ==============================================================================
# Service Tests: Acceptance Criteria 1 (Version N -> N + 1 Success)
# ==============================================================================


class TestAutoSaveEngineSuccess:
    """Tests verifying successful delta application when expected version matches N."""

    @pytest.mark.asyncio
    async def test_autosave_delta_increments_version_from_n_to_n_plus_one(
        self, engine, mock_storage, sample_chart_state
    ):
        """
        AC: Given a chart with current version N, when an auto-save delta is processed
        with expected version N, then the changes are persisted and the stored version
        increments to N + 1.
        """
        initial_version = 5
        sample_chart_state.version = initial_version
        mock_storage.get_chart.return_value = sample_chart_state

        delta = ChartDelta(
            chart_id="chart-100",
            expected_version=initial_version,
            changes={"theme": "dark", "zoom_level": 1.5},
            session_id="sess-test-1",
        )

        result_state = await engine.process_delta(delta)

        # Verify version incremented to N + 1
        assert result_state.version == initial_version + 1
        assert result_state.chart_id == "chart-100"

        # Verify storage was called to persist changes with version N + 1
        mock_storage.save_chart.assert_awaited_once()
        saved_state = mock_storage.save_chart.call_args[0][0]
        assert saved_state.version == initial_version + 1
        assert saved_state.data["theme"] == "dark"
        assert saved_state.data["zoom_level"] == 1.5

    @pytest.mark.asyncio
    async def test_autosave_delta_merges_data_without_loss(
        self, engine, mock_storage, sample_chart_state
    ):
        """Ensures that existing chart state fields are preserved when delta changes are applied."""
        sample_chart_state.version = 10
        sample_chart_state.data = {
            "timeframe": "1h",
            "drawings": ["line_1"],
        }
        mock_storage.get_chart.return_value = sample_chart_state

        delta = ChartDelta(
            chart_id="chart-100",
            expected_version=10,
            changes={"drawings": ["line_1", "line_2"], "grid": True},
            session_id="sess-test-1",
        )

        result_state = await engine.process_delta(delta)

        assert result_state.version == 11
        # Existing untouched key preserved
        assert result_state.data["timeframe"] == "1h"
        # Modified key updated
        assert result_state.data["drawings"] == ["line_1", "line_2"]
        # Added key appended
        assert result_state.data["grid"] is True

    @pytest.mark.asyncio
    async def test_autosave_sequential_deltas_increment_monotonically(
        self, engine, mock_storage, sample_chart_state
    ):
        """Verifies that multiple sequential deltas advance version sequentially: N -> N+1 -> N+2."""
        current_state = sample_chart_state
        current_state.version = 1

        async def fake_save(state: ChartState):
            nonlocal current_state
            current_state = state
            return current_state

        async def fake_get(chart_id: str):
            return current_state

        mock_storage.get_chart.side_effect = fake_get
        mock_storage.save_chart.side_effect = fake_save

        delta_1 = ChartDelta(
            chart_id="chart-100",
            expected_version=1,
            changes={"step": 1},
            session_id="sess-1",
        )
        res_1 = await engine.process_delta(delta_1)
        assert res_1.version == 2
        assert res_1.data["step"] == 1

        delta_2 = ChartDelta(
            chart_id="chart-100",
            expected_version=2,
            changes={"step": 2},
            session_id="sess-1",
        )
        res_2 = await engine.process_delta(delta_2)
        assert res_2.version == 3
        assert res_2.data["step"] == 2

        assert mock_storage.save_chart.await_count == 2


# ==============================================================================
# Service Tests: Acceptance Criteria 2 (Concurrency Conflict Resolution)
# ==============================================================================


class TestAutoSaveEngineConflict:
    """Tests verifying rejection and safety when expected version does not match current state."""

    @pytest.mark.asyncio
    async def test_autosave_delta_stale_expected_version_raises_concurrency_conflict(
        self, engine, mock_storage, sample_chart_state
    ):
        """
        AC: Given a chart with current version N + 1, when an auto-save delta arrives
        from a concurrent session specifying expected version N, then a ConcurrencyConflictError
        is raised and existing persisted state remains unaltered.
        """
        # Current version is N + 1 (version 3)
        sample_chart_state.version = 3
        sample_chart_state.data = {"active_layer": "main"}
        mock_storage.get_chart.return_value = sample_chart_state

        # Incoming delta expects stale version N (version 2)
        stale_delta = ChartDelta(
            chart_id="chart-100",
            expected_version=2,
            changes={"active_layer": "stale_overwrite"},
            session_id="concurrent-sess-b",
        )

        with pytest.raises(ConcurrencyConflictError):
            await engine.process_delta(stale_delta)

        # Existing persisted state MUST remain unaltered
        mock_storage.save_chart.assert_not_called()
        assert sample_chart_state.version == 3
        assert sample_chart_state.data == {"active_layer": "main"}

    @pytest.mark.asyncio
    async def test_autosave_delta_future_expected_version_raises_concurrency_conflict(
        self, engine, mock_storage, sample_chart_state
    ):
        """A delta specifying expected version ahead of persisted version (N + 2 vs N) is rejected."""
        sample_chart_state.version = 5
        mock_storage.get_chart.return_value = sample_chart_state

        future_delta = ChartDelta(
            chart_id="chart-100",
            expected_version=7,
            changes={"bogus": "data"},
            session_id="sess-out-of-order",
        )

        with pytest.raises(ConcurrencyConflictError):
            await engine.process_delta(future_delta)

        mock_storage.save_chart.assert_not_called()
        assert sample_chart_state.version == 5

    @pytest.mark.asyncio
    async def test_concurrency_conflict_error_exposes_conflict_diagnostics(
        self, engine, mock_storage, sample_chart_state
    ):
        """Verifies ConcurrencyConflictError provides structured version context for recovery."""
        sample_chart_state.version = 4
        mock_storage.get_chart.return_value = sample_chart_state

        delta = ChartDelta(
            chart_id="chart-100",
            expected_version=3,
            changes={"color": "red"},
            session_id="sess-conflict",
        )

        try:
            await engine.process_delta(delta)
            pytest.fail("Expected ConcurrencyConflictError was not raised.")
        except ConcurrencyConflictError as exc:
            assert hasattr(exc, "current_version")
            assert exc.current_version == 4
            assert hasattr(exc, "expected_version")
            assert exc.expected_version == 3
            assert hasattr(exc, "chart_id")
            assert exc.chart_id == "chart-100"


# ==============================================================================
# Service Tests: Race Condition & Parallel Session Execution
# ==============================================================================


class TestAutoSaveEngineConcurrentRace:
    """Tests simulating actual race conditions with concurrent async tasks."""

    @pytest.mark.asyncio
    async def test_simultaneous_concurrent_deltas_only_one_succeeds(
        self, mock_storage, sample_chart_state
    ):
        """
        Simulate two concurrent sessions dispatching deltas for the same base version N=1.
        Using an internal lock or CAS mechanism, exactly one must succeed (becoming N=2),
        and the other must fail with ConcurrencyConflictError.
        """
        current_state = sample_chart_state
        current_state.version = 1
        lock = asyncio.Lock()

        async def synchronized_get_chart(chart_id: str):
            return current_state

        async def synchronized_save_chart(state: ChartState):
            nonlocal current_state
            # Simulate a slight async yield to trigger interleaved task scheduling
            await asyncio.sleep(0.01)
            current_state = state
            return current_state

        mock_storage.get_chart.side_effect = synchronized_get_chart
        mock_storage.save_chart.side_effect = synchronized_save_chart

        engine = AutoSaveEngine(storage=mock_storage)

        delta_sess_a = ChartDelta(
            chart_id="chart-100",
            expected_version=1,
            changes={"editor": "session_A"},
            session_id="session_A",
        )
        delta_sess_b = ChartDelta(
            chart_id="chart-100",
            expected_version=1,
            changes={"editor": "session_B"},
            session_id="session_B",
        )

        # Run both deltas concurrently
        results = await asyncio.gather(
            engine.process_delta(delta_sess_a),
            engine.process_delta(delta_sess_b),
            return_exceptions=True,
        )

        successes = [r for r in results if isinstance(r, ChartState)]
        conflicts = [r for r in results if isinstance(r, ConcurrencyConflictError)]

        assert len(successes) == 1, "Exactly one delta should have succeeded"
        assert len(conflicts) == 1, "Exactly one delta should have encountered ConcurrencyConflictError"
        assert successes[0].version == 2
        assert mock_storage.save_chart.await_count == 1


# ==============================================================================
# Service Tests: Boundary Conditions & Error Handling
# ==============================================================================


class TestAutoSaveEngineBoundaries:
    """Tests verifying boundary conditions, non-existent charts, and storage validation."""

    @pytest.mark.asyncio
    async def test_autosave_nonexistent_chart_raises_not_found(self, engine, mock_storage):
        """Attempting to apply a delta to a chart that doesn't exist raises ChartNotFoundError."""
        mock_storage.get_chart.return_value = None

        delta = ChartDelta(
            chart_id="non-existent-chart",
            expected_version=0,
            changes={"new": "chart"},
            session_id="sess-orphan",
        )

        with pytest.raises(ChartNotFoundError):
            await engine.process_delta(delta)

        mock_storage.save_chart.assert_not_called()

    @pytest.mark.asyncio
    async def test_autosave_propagates_storage_persistence_errors(
        self, engine, mock_storage, sample_chart_state
    ):
        """Storage failures during persist bubble up and do not leave state inconsistent."""
        sample_chart_state.version = 1
        mock_storage.get_chart.return_value = sample_chart_state
        mock_storage.save_chart.side_effect = RuntimeError("Storage connection failed")

        delta = ChartDelta(
            chart_id="chart-100",
            expected_version=1,
            changes={"theme": "light"},
            session_id="sess-fail",
        )

        with pytest.raises(RuntimeError):
            await engine.process_delta(delta)

        mock_storage.save_chart.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_autosave_updates_timestamp_on_successful_save(
        self, engine, mock_storage, sample_chart_state
    ):
        """Verifies that updated_at timestamp is refreshed when a delta is saved."""
        past_time = datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        sample_chart_state.version = 2
        sample_chart_state.updated_at = past_time
        mock_storage.get_chart.return_value = sample_chart_state

        delta = ChartDelta(
            chart_id="chart-100",
            expected_version=2,
            changes={"indicator_added": "MACD"},
            session_id="sess-ts",
        )

        saved = await engine.process_delta(delta)

        assert saved.updated_at > past_time