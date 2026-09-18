import math
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
import pytest

from src import ta
from src.ta.moving_averages import ema, rma, sma, wma


# ============================================================================
# Helpers
# ============================================================================

def _to_float_list(series_like: Any) -> list[float]:
    """Convert input or output sequence to a standard list of floats/NaNs."""
    if isinstance(series_like, (pd.Series, np.ndarray)):
        return [float(x) if pd.notna(x) else float("nan") for x in series_like]
    if isinstance(series_like, Sequence):
        return [float(x) if x is not None and not math.isnan(x) else float("nan") for x in series_like]
    raise TypeError(f"Unsupported sequence type: {type(series_like)}")


def _assert_series_equal(actual: Any, expected: Sequence[float | None], rel_tol: float = 1e-9) -> None:
    """Assert actual output matches expected values with exact NaN warmup handling."""
    actual_list = _to_float_list(actual)
    assert len(actual_list) == len(
        expected
    ), f"Length mismatch: got {len(actual_list)}, expected {len(expected)}"

    for idx, (act_val, exp_val) in enumerate(zip(actual_list, expected)):
        if exp_val is None or math.isnan(exp_val):
            assert math.isnan(
                act_val
            ), f"Expected NaN/None at index {idx}, got {act_val}"
        else:
            assert not math.isnan(
                act_val
            ), f"Expected {exp_val} at index {idx}, got NaN"
            assert act_val == pytest.approx(
                exp_val, rel=rel_tol
            ), f"Value mismatch at index {idx}: got {act_val}, expected {exp_val}"


# ============================================================================
# Test Data Fixtures
# ============================================================================

@pytest.fixture
def sample_prices() -> list[float]:
    """Distinct non-linear price sequence to prevent accidental MA equality."""
    return [10.0, 11.0, 12.0, 14.0, 18.0]


# ============================================================================
# Module Exports & Namespace Verification
# ============================================================================

def test_ta_namespace_exports() -> None:
    """Verify ta package root exposes moving average functions."""
    assert hasattr(ta, "sma")
    assert hasattr(ta, "ema")
    assert hasattr(ta, "wma")
    assert hasattr(ta, "rma")
    assert callable(ta.sma)
    assert callable(ta.ema)
    assert callable(ta.wma)
    assert callable(ta.rma)


def test_moving_averages_module_exports() -> None:
    """Verify src.ta.moving_averages exports the functions directly."""
    import src.ta.moving_averages as ma

    assert hasattr(ma, "sma")
    assert hasattr(ma, "ema")
    assert hasattr(ma, "wma")
    assert hasattr(ma, "rma")


# ============================================================================
# SMA (Simple Moving Average) Tests
# ============================================================================

class TestSMA:
    def test_sma_arithmetic_mean(self, sample_prices: list[float]) -> None:
        """ta.sma returns arithmetic mean: sum(prices[i-length+1..i]) / length."""
        length = 3
        # index 0, 1: NaN
        # index 2: (10 + 11 + 12) / 3 = 11.0
        # index 3: (11 + 12 + 14) / 3 = 37 / 3 = 12.3333333333
        # index 4: (12 + 14 + 18) / 3 = 44 / 3 = 14.6666666667
        expected = [float("nan"), float("nan"), 11.0, 37.0 / 3.0, 44.0 / 3.0]
        result = sma(sample_prices, length)
        _assert_series_equal(result, expected)

    def test_sma_warmup_elements(self) -> None:
        """Warmup elements where index < length - 1 must evaluate to NaN."""
        prices = [100.0, 102.0, 104.0, 106.0, 108.0]
        length = 4
        result = _to_float_list(sma(prices, length))

        # indices 0, 1, 2 must be NaN
        for i in range(length - 1):
            assert math.isnan(result[i]), f"Expected warmup NaN at index {i}"
        # index 3 must be valid numeric value
        assert not math.isnan(result[3])

    def test_sma_length_one(self, sample_prices: list[float]) -> None:
        """SMA with length 1 has no warmup elements and equals the original series."""
        result = sma(sample_prices, length=1)
        _assert_series_equal(result, sample_prices)

    def test_sma_constant_series(self) -> None:
        """Constant series should yield the constant value after warmup."""
        prices = [42.0] * 6
        expected = [float("nan"), float("nan"), 42.0, 42.0, 42.0, 42.0]
        result = sma(prices, length=3)
        _assert_series_equal(result, expected)


# ============================================================================
# WMA (Weighted Moving Average) Tests
# ============================================================================

class TestWMA:
    def test_wma_linear_weighting(self, sample_prices: list[float]) -> None:
        """ta.wma returns linearly weighted moving average with weights 1 to length."""
        length = 3
        # weights = [1, 2, 3], sum(weights) = 6
        # index 0, 1: NaN
        # index 2: (10*1 + 11*2 + 12*3) / 6 = 68 / 6 = 11.3333333333
        # index 3: (11*1 + 12*2 + 14*3) / 6 = 77 / 6 = 12.8333333333
        # index 4: (12*1 + 14*2 + 18*3) / 6 = 94 / 6 = 15.6666666667
        expected = [float("nan"), float("nan"), 68.0 / 6.0, 77.0 / 6.0, 94.0 / 6.0]
        result = wma(sample_prices, length)
        _assert_series_equal(result, expected)

    def test_wma_warmup_elements(self) -> None:
        """Warmup elements where index < length - 1 evaluate to NaN."""
        prices = [1.0, 2.0, 3.0, 4.0, 5.0]
        length = 4
        result = _to_float_list(wma(prices, length))

        for i in range(length - 1):
            assert math.isnan(result[i]), f"Expected warmup NaN at index {i}"
        assert not math.isnan(result[3])

    def test_wma_length_one(self, sample_prices: list[float]) -> None:
        """WMA with length 1 has no warmup elements and equals the original series."""
        result = wma(sample_prices, length=1)
        _assert_series_equal(result, sample_prices)

    def test_wma_weight_distribution_formula(self) -> None:
        """Explicitly test weight distribution over a window of length 4."""
        prices = [1.0, 3.0, 5.0, 7.0, 11.0]
        length = 4
        # weights = [1, 2, 3, 4], sum = 10
        # index 3: (1*1 + 2*3 + 3*5 + 4*7) / 10 = (1 + 6 + 15 + 28) / 10 = 50 / 10 = 5.0
        # index 4: (1*3 + 2*5 + 3*7 + 4*11) / 10 = (3 + 10 + 21 + 44) / 10 = 78 / 10 = 7.8
        expected = [float("nan"), float("nan"), float("nan"), 5.0, 7.8]
        result = wma(prices, length)
        _assert_series_equal(result, expected)

    def test_wma_constant_series(self) -> None:
        """Constant series should yield constant value after warmup."""
        prices = [15.0] * 5
        expected = [float("nan"), float("nan"), 15.0, 15.0, 15.0]
        result = wma(prices, length=3)
        _assert_series_equal(result, expected)


# ============================================================================
# EMA (Exponential Moving Average) Tests
# ============================================================================

class TestEMA:
    def test_ema_smoothing_factor(self, sample_prices: list[float]) -> None:
        """ta.ema uses alpha = 2 / (length + 1) with SMA seed at index length - 1."""
        length = 3
        # alpha = 2 / (3 + 1) = 0.5
        # index 0, 1: NaN
        # index 2 (seed SMA): (10 + 11 + 12) / 3 = 11.0
        # index 3: 0.5 * 14.0 + (1 - 0.5) * 11.0 = 7.0 + 5.5 = 12.5
        # index 4: 0.5 * 18.0 + (1 - 0.5) * 12.5 = 9.0 + 6.25 = 15.25
        expected = [float("nan"), float("nan"), 11.0, 12.5, 15.25]
        result = ema(sample_prices, length)
        _assert_series_equal(result, expected)

    def test_ema_warmup_elements(self) -> None:
        """Warmup elements where index < length - 1 evaluate to NaN."""
        prices = [5.0, 10.0, 15.0, 20.0, 25.0]
        length = 4
        result = _to_float_list(ema(prices, length))

        for i in range(length - 1):
            assert math.isnan(result[i]), f"Expected warmup NaN at index {i}"
        assert not math.isnan(result[3])

    def test_ema_length_one(self, sample_prices: list[float]) -> None:
        """EMA with length 1 (alpha = 2/(1+1) = 1.0) equals original series."""
        result = ema(sample_prices, length=1)
        _assert_series_equal(result, sample_prices)

    def test_ema_longer_sequence_recursion(self) -> None:
        """Verify EMA recursive property over length 4 window."""
        prices = [1.0, 3.0, 5.0, 7.0, 11.0]
        length = 4
        # alpha = 2 / (4 + 1) = 0.4
        # seed at index 3: (1 + 3 + 5 + 7) / 4 = 4.0
        # index 4: 0.4 * 11.0 + 0.6 * 4.0 = 4.4 + 2.4 = 6.8
        expected = [float("nan"), float("nan"), float("nan"), 4.0, 6.8]
        result = ema(prices, length)
        _assert_series_equal(result, expected)

    def test_ema_constant_series(self) -> None:
        """Constant series should yield constant value after warmup."""
        prices = [25.0] * 5
        expected = [float("nan"), float("nan"), 25.0, 25.0, 25.0]
        result = ema(prices, length=3)
        _assert_series_equal(result, expected)


# ============================================================================
# RMA (Wilder's Smoothing Moving Average) Tests
# ============================================================================

class TestRMA:
    def test_rma_wilders_smoothing_factor(self, sample_prices: list[float]) -> None:
        """ta.rma uses alpha = 1 / length with SMA seed at index length - 1."""
        length = 3
        # alpha = 1 / 3
        # index 0, 1: NaN
        # index 2 (seed SMA): (10 + 11 + 12) / 3 = 11.0
        # index 3: (1/3) * 14.0 + (2/3) * 11.0 = (14 + 22) / 3 = 12.0
        # index 4: (1/3) * 18.0 + (2/3) * 12.0 = 6.0 + 8.0 = 14.0
        expected = [float("nan"), float("nan"), 11.0, 12.0, 14.0]
        result = rma(sample_prices, length)
        _assert_series_equal(result, expected)

    def test_rma_warmup_elements(self) -> None:
        """Warmup elements where index < length - 1 evaluate to NaN."""
        prices = [2.0, 4.0, 6.0, 8.0, 10.0]
        length = 4
        result = _to_float_list(rma(prices, length))

        for i in range(length - 1):
            assert math.isnan(result[i]), f"Expected warmup NaN at index {i}"
        assert not math.isnan(result[3])

    def test_rma_length_one(self, sample_prices: list[float]) -> None:
        """RMA with length 1 (alpha = 1 / 1 = 1.0) equals original series."""
        result = rma(sample_prices, length=1)
        _assert_series_equal(result, sample_prices)

    def test_rma_longer_sequence_recursion(self) -> None:
        """Verify RMA recursive property over length 4 window."""
        prices = [1.0, 3.0, 5.0, 7.0, 11.0]
        length = 4
        # alpha = 1 / 4 = 0.25
        # seed at index 3: (1 + 3 + 5 + 7) / 4 = 4.0
        # index 4: 0.25 * 11.0 + 0.75 * 4.0 = 2.75 + 3.0 = 5.75
        expected = [float("nan"), float("nan"), float("nan"), 4.0, 5.75]
        result = rma(prices, length)
        _assert_series_equal(result, expected)

    def test_rma_constant_series(self) -> None:
        """Constant series should yield constant value after warmup."""
        prices = [10.0] * 5
        expected = [float("nan"), float("nan"), 10.0, 10.0, 10.0]
        result = rma(prices, length=3)
        _assert_series_equal(result, expected)


# ============================================================================
# Edge Cases & Validation for all Moving Averages
# ============================================================================

@pytest.mark.parametrize("ma_func", [sma, ema, wma, rma])
class TestCommonMovingAverageProperties:
    def test_invalid_length_zero_or_negative(self, ma_func: Callable) -> None:
        """Window length must be a positive integer (> 0)."""
        prices = [1.0, 2.0, 3.0, 4.0]
        with pytest.raises(ValueError):
            ma_func(prices, length=0)
        with pytest.raises(ValueError):
            ma_func(prices, length=-3)

    @pytest.mark.parametrize("invalid_type_length", [1.5, "3", None])
    def test_invalid_length_type(self, ma_func: Callable, invalid_type_length: Any) -> None:
        """Window length must be strictly an integer."""
        prices = [1.0, 2.0, 3.0, 4.0]
        with pytest.raises((TypeError, ValueError)):
            ma_func(prices, length=invalid_type_length)

    def test_series_length_equals_window_length(self, ma_func: Callable) -> None:
        """When series length == window length, only the final element is valid."""
        prices = [10.0, 20.0, 30.0]
        length = 3
        result = _to_float_list(ma_func(prices, length))
        assert len(result) == 3
        assert math.isnan(result[0])
        assert math.isnan(result[1])
        assert not math.isnan(result[2])

    def test_series_shorter_than_window_length(self, ma_func: Callable) -> None:
        """When series length < window length, all items evaluate to NaN."""
        prices = [10.0, 20.0]
        length = 5
        result = _to_float_list(ma_func(prices, length))
        assert len(result) == len(prices)
        assert all(math.isnan(x) for x in result)

    def test_empty_series(self, ma_func: Callable) -> None:
        """Empty series input should produce empty output sequence."""
        prices: list[float] = []
        result = _to_float_list(ma_func(prices, length=3))
        assert len(result) == 0

    def test_preserves_input_length(self, ma_func: Callable) -> None:
        """Output series length must strictly equal input series length."""
        prices = [float(x) for x in range(50)]
        result = _to_float_list(ma_func(prices, length=14))
        assert len(result) == len(prices)

    @pytest.mark.parametrize("input_container", [list, np.array, pd.Series])
    def test_supports_different_container_types(
        self, ma_func: Callable, input_container: Callable
    ) -> None:
        """Moving average functions must support list, np.ndarray, and pd.Series inputs."""
        raw_prices = [10.0, 20.0, 30.0, 40.0, 50.0]
        container_input = input_container(raw_prices)
        result = ma_func(container_input, length=3)
        result_list = _to_float_list(result)

        assert len(result_list) == len(raw_prices)
        assert math.isnan(result_list[0])
        assert math.isnan(result_list[1])
        assert not math.isnan(result_list[2])
        assert not math.isnan(result_list[3])
        assert not math.isnan(result_list[4])