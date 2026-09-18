import math
from typing import Any
import pytest

from src.engine.series import Series
from src.engine.evaluator import Evaluator


def is_nan_or_none(value: Any) -> bool:
    """Helper to verify if a returned value represents an undefined / missing historic bar."""
    if value is None:
        return True
    if isinstance(value, (int, float)):
        return math.isnan(value)
    return False


class TestSeriesHistoricReferenceOperator:
    """Unit tests verifying the Series historic reference operator `[ ]` contract."""

    def test_historic_reference_offset_one_returns_previous_value(self) -> None:
        """AC-1: Given a series with [10.0, 12.0, 15.0] where 15.0 is current (offset 0),

        when series[1] is evaluated, then it returns 12.0.
        """
        series = Series([10.0, 12.0, 15.0])

        result = series[1]

        assert result == 12.0

    def test_historic_reference_offset_zero_returns_current_value(self) -> None:
        """Offset 0 must reference the most recent / current data point."""
        series = Series([10.0, 12.0, 15.0])

        result = series[0]

        assert result == 15.0

    def test_historic_reference_oldest_available_offset(self) -> None:
        """Offset equal to (depth - 1) must return the oldest recorded historical data point."""
        series = Series([10.0, 12.0, 15.0])

        result = series[2]

        assert result == 10.0

    @pytest.mark.parametrize(
        "offset, expected",
        [
            (0, 15.0),
            (1, 12.0),
            (2, 10.0),
        ],
    )
    def test_historic_reference_parameterized_valid_offsets(
        self, offset: int, expected: float
    ) -> None:
        """Verifies deterministic mapping of offsets to historical positions."""
        series = Series([10.0, 12.0, 15.0])

        assert series[offset] == expected

    def test_historic_reference_offset_exceeding_depth_returns_nan_or_none(self) -> None:
        """AC-2: Offset greater than available depth returns NaN or None without throwing."""
        series = Series([10.0, 12.0, 15.0])

        # Depth is 3 (valid offsets: 0, 1, 2)
        at_limit = series[3]
        beyond_limit = series[10]
        far_beyond_limit = series[1000]

        assert is_nan_or_none(at_limit)
        assert is_nan_or_none(beyond_limit)
        assert is_nan_or_none(far_beyond_limit)

    def test_historic_reference_empty_series_returns_nan_or_none(self) -> None:
        """Evaluation on an empty series should return NaN or None without exception."""
        empty_series = Series()

        result = empty_series[0]

        assert is_nan_or_none(result)

    def test_historic_reference_single_element_series(self) -> None:
        """Evaluation on a single-element series handles current bar and exceeds depth gracefully."""
        series = Series([42.0])

        assert series[0] == 42.0
        assert is_nan_or_none(series[1])

    @pytest.mark.parametrize("negative_offset", [-1, -2, -10, -999])
    def test_historic_reference_negative_offset_raises_value_error(
        self, negative_offset: int
    ) -> None:
        """AC-3: Negative integer offset raises ValueError to reject lookaheads."""
        series = Series([10.0, 12.0, 15.0])

        with pytest.raises(ValueError):
            _ = series[negative_offset]

    def test_historic_reference_empty_series_negative_offset_raises_value_error(self) -> None:
        """Future referencing is rejected with ValueError even if history is empty."""
        empty_series = Series()

        with pytest.raises(ValueError):
            _ = empty_series[-1]

    @pytest.mark.parametrize("invalid_key", ["1", 1.5, None, [1]])
    def test_historic_reference_non_integer_offset_raises_type_error(
        self, invalid_key: Any
    ) -> None:
        """Non-integer indexing must raise TypeError."""
        series = Series([10.0, 12.0, 15.0])

        with pytest.raises(TypeError):
            _ = series[invalid_key]

    def test_historic_reference_dynamically_appended_data(self) -> None:
        """Appending a new current value shifts historic offsets correctly."""
        series = Series([10.0, 12.0, 15.0])

        series.append(18.0)

        assert series[0] == 18.0
        assert series[1] == 15.0
        assert series[2] == 12.0
        assert series[3] == 10.0
        assert is_nan_or_none(series[4])


class TestEvaluatorHistoricReference:
    """Unit tests verifying evaluator-driven historic operator evaluation."""

    @pytest.fixture
    def evaluator(self) -> Evaluator:
        return Evaluator()

    def test_evaluator_evaluates_historic_reference_offset_one(
        self, evaluator: Evaluator
    ) -> None:
        """AC-1: Evaluator returns 12.0 when resolving series[1] on [10.0, 12.0, 15.0]."""
        series = Series([10.0, 12.0, 15.0])

        result = evaluator.evaluate_historic_reference(series, 1)

        assert result == 12.0

    def test_evaluator_evaluates_historic_reference_offset_zero(
        self, evaluator: Evaluator
    ) -> None:
        """Evaluator returns the current value for offset 0."""
        series = Series([10.0, 12.0, 15.0])

        result = evaluator.evaluate_historic_reference(series, 0)

        assert result == 15.0

    def test_evaluator_out_of_bounds_offset_returns_nan_or_none(
        self, evaluator: Evaluator
    ) -> None:
        """AC-2: Evaluator returns NaN or None when offset exceeds depth."""
        series = Series([10.0, 12.0, 15.0])

        result = evaluator.evaluate_historic_reference(series, 5)

        assert is_nan_or_none(result)

    def test_evaluator_empty_series_returns_nan_or_none(
        self, evaluator: Evaluator
    ) -> None:
        """Evaluator handles empty series safely returning NaN or None."""
        series = Series([])

        result = evaluator.evaluate_historic_reference(series, 0)

        assert is_nan_or_none(result)

    @pytest.mark.parametrize("negative_offset", [-1, -3, -100])
    def test_evaluator_negative_offset_raises_value_error(
        self, evaluator: Evaluator, negative_offset: int
    ) -> None:
        """AC-3: Evaluator rejects negative offsets with ValueError."""
        series = Series([10.0, 12.0, 15.0])

        with pytest.raises(ValueError):
            evaluator.evaluate_historic_reference(series, negative_offset)

    @pytest.mark.parametrize("invalid_offset", [1.2, "0", None])
    def test_evaluator_invalid_offset_type_raises_type_error(
        self, evaluator: Evaluator, invalid_offset: Any
    ) -> None:
        """Evaluator enforces integer offset type safety."""
        series = Series([10.0, 12.0, 15.0])

        with pytest.raises(TypeError):
            evaluator.evaluate_historic_reference(series, invalid_offset)