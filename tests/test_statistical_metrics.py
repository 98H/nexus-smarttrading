import math
from typing import Sequence

import numpy as np
import pandas as pd
import pytest

from src.analytics import PerformanceMetrics as AnalyticsPerformanceMetrics
from src.analytics import (
    calculate_performance_metrics as analytics_calculate_performance_metrics,
)
from src.analytics.statistical_metrics import (
    PerformanceMetrics,
    calculate_performance_metrics,
)


class TestPackageExports:
    """Verify that required classes and functions are properly exposed."""

    def test_package_level_exports(self):
        """Ensure src.analytics exports PerformanceMetrics and calculate_performance_metrics."""
        assert AnalyticsPerformanceMetrics is PerformanceMetrics
        assert (
            analytics_calculate_performance_metrics
            is calculate_performance_metrics
        )


class TestPerformanceMetricsStructure:
    """Verify the structure, types, and fields of the PerformanceMetrics object."""

    def test_metrics_object_attributes(self):
        """Ensure PerformanceMetrics has the specified attributes with float types."""
        metrics = calculate_performance_metrics([0.05, -0.02, 0.03], risk_free_rate=0.01)

        assert isinstance(metrics, PerformanceMetrics)
        assert hasattr(metrics, "sharpe_ratio")
        assert hasattr(metrics, "sortino_ratio")
        assert hasattr(metrics, "max_drawdown")

        assert isinstance(metrics.sharpe_ratio, float)
        assert isinstance(metrics.sortino_ratio, float)
        assert isinstance(metrics.max_drawdown, float)

    def test_default_risk_free_rate(self):
        """Ensure risk_free_rate defaults to 0.0 when not explicitly provided."""
        metrics_default = calculate_performance_metrics([0.02, 0.04, -0.01])
        metrics_explicit_zero = calculate_performance_metrics(
            [0.02, 0.04, -0.01], risk_free_rate=0.0
        )

        assert metrics_default.sharpe_ratio == pytest.approx(
            metrics_explicit_zero.sharpe_ratio
        )
        assert metrics_default.sortino_ratio == pytest.approx(
            metrics_explicit_zero.sortino_ratio
        )
        assert metrics_default.max_drawdown == pytest.approx(
            metrics_explicit_zero.max_drawdown
        )


class TestZeroDivisionAndEdgeCases:
    """Verify zero division safety and default 0.0 behavior for empty and single-element inputs."""

    @pytest.mark.parametrize("empty_series", [[], (), np.array([]), pd.Series([], dtype=float)])
    def test_empty_returns_series_returns_zero_metrics(self, empty_series: Sequence[float]):
        """Empty series must return 0.0 for ratios and drawdown without raising an exception."""
        metrics = calculate_performance_metrics(empty_series, risk_free_rate=0.02)

        assert metrics.sharpe_ratio == 0.0
        assert metrics.sortino_ratio == 0.0
        assert metrics.max_drawdown == 0.0

    @pytest.mark.parametrize(
        "single_element_series",
        [
            [0.05],
            [-0.05],
            [0.0],
            (0.10,),
            np.array([-0.02]),
            pd.Series([0.03]),
        ],
    )
    def test_single_element_returns_series_returns_zero_metrics(
        self, single_element_series: Sequence[float]
    ):
        """Single-element series must return 0.0 for ratios and drawdown without raising an exception."""
        metrics = calculate_performance_metrics(single_element_series, risk_free_rate=0.01)

        assert metrics.sharpe_ratio == 0.0
        assert metrics.sortino_ratio == 0.0
        assert metrics.max_drawdown == 0.0

    def test_identical_returns_zero_volatility_handles_zero_division(self):
        """Flat return series (standard deviation == 0.0) must return 0.0 for Sharpe and Sortino."""
        flat_returns = [0.03, 0.03, 0.03, 0.03]
        metrics = calculate_performance_metrics(flat_returns, risk_free_rate=0.01)

        assert metrics.sharpe_ratio == 0.0
        assert metrics.sortino_ratio == 0.0
        assert metrics.max_drawdown == 0.0

    def test_no_downside_returns_handles_zero_division_for_sortino(self):
        """When all returns are strictly above the risk-free rate, downside risk is 0.0."""
        positive_returns = [0.05, 0.08, 0.06, 0.07]
        metrics = calculate_performance_metrics(positive_returns, risk_free_rate=0.02)

        assert not math.isnan(metrics.sortino_ratio)
        assert not math.isinf(metrics.sortino_ratio)
        assert metrics.sortino_ratio == 0.0
        assert metrics.sharpe_ratio > 0.0
        assert metrics.max_drawdown == 0.0


class TestMaximumDrawdownCalculations:
    """Verify maximum drawdown calculations across various compounding return profiles."""

    def test_monotonic_positive_returns_has_zero_drawdown(self):
        """Consistently positive returns should never experience a drawdown."""
        returns = [0.01, 0.02, 0.03, 0.015]
        metrics = calculate_performance_metrics(returns, risk_free_rate=0.0)

        assert metrics.max_drawdown == 0.0

    def test_known_maximum_drawdown_profile(self):
        """
        Test a known peak-to-trough path:
        T0: Wealth = 1.0
        T1: +10% -> Wealth = 1.10 (Peak)
        T2: -20% -> Wealth = 1.10 * 0.80 = 0.88 (Drawdown = (1.10 - 0.88) / 1.10 = 0.20)
        T3: +5%  -> Wealth = 0.88 * 1.05 = 0.924 (Drawdown = (1.10 - 0.924) / 1.10 = 0.16)
        Maximum Drawdown should be 0.20 (20%).
        """
        returns = [0.10, -0.20, 0.05]
        metrics = calculate_performance_metrics(returns, risk_free_rate=0.0)

        assert metrics.max_drawdown == pytest.approx(0.20, rel=1e-5)

    def test_multiple_drawdowns_tracks_global_maximum(self):
        """
        Ensure the calculator tracks the deepest drawdown across multiple drops:
        T1: +20% -> Wealth = 1.20 (Peak = 1.20)
        T2: -10% -> Wealth = 1.08 (DD = 0.10)
        T3: +30% -> Wealth = 1.404 (Peak = 1.404)
        T4: -25% -> Wealth = 1.053 (DD = 0.25)
        T5: -10% -> Wealth = 0.9477 (DD = (1.404 - 0.9477) / 1.404 = 0.325)
        Global Maximum Drawdown should be 0.325.
        """
        returns = [0.20, -0.10, 0.30, -0.25, -0.10]
        metrics = calculate_performance_metrics(returns, risk_free_rate=0.0)

        assert metrics.max_drawdown == pytest.approx(0.325, rel=1e-5)

    def test_consecutive_losses_accumulate_drawdown(self):
        """Two consecutive -10% drops: (1 - 0.9 * 0.9) = 19% drawdown."""
        returns = [-0.10, -0.10]
        metrics = calculate_performance_metrics(returns, risk_free_rate=0.0)

        assert metrics.max_drawdown == pytest.approx(0.19, rel=1e-5)


class TestSharpeAndSortinoCalculations:
    """Verify deterministic statistical metrics against expected mathematical formulas."""

    def test_zero_excess_return_yields_zero_sharpe_and_sortino(self):
        """When mean return equals the risk-free rate, Sharpe and Sortino ratios must be 0.0."""
        # Mean of [0.04, 0.00, 0.02, 0.02] is 0.02
        returns = [0.04, 0.00, 0.02, 0.02]
        metrics = calculate_performance_metrics(returns, risk_free_rate=0.02)

        assert metrics.sharpe_ratio == pytest.approx(0.0, abs=1e-7)
        assert metrics.sortino_ratio == pytest.approx(0.0, abs=1e-7)

    def test_sharpe_ratio_deterministic_value(self):
        """
        Returns: [0.08, -0.04]
        risk_free_rate = 0.01
        Mean return = 0.02
        Mean excess return = 0.02 - 0.01 = 0.01
        Sample std (ddof=1) = std([0.08, -0.04]) = sqrt(((0.08 - 0.02)^2 + (-0.04 - 0.02)^2) / 1)
                            = sqrt(0.0036 + 0.0036) = sqrt(0.0072) ≈ 0.0848528
        Population std (ddof=0) = sqrt(0.0072 / 2) = sqrt(0.0036) = 0.06
        Sharpe ratio is expected to be positive and match the analytical standard deviation.
        """
        returns = [0.08, -0.04]
        metrics = calculate_performance_metrics(returns, risk_free_rate=0.01)

        excess = np.array(returns) - 0.01
        mean_excess = np.mean(excess)
        sample_sharpe = mean_excess / np.std(returns, ddof=1)
        pop_sharpe = mean_excess / np.std(returns, ddof=0)

        assert metrics.sharpe_ratio == pytest.approx(
            sample_sharpe, rel=1e-4
        ) or metrics.sharpe_ratio == pytest.approx(pop_sharpe, rel=1e-4)

    def test_sortino_ratio_deterministic_value(self):
        """
        Ensure Sortino ratio penalizes only downside deviations below the risk-free rate.
        Returns: [0.10, 0.04, -0.02], risk_free_rate = 0.02
        Excess: [0.08, 0.02, -0.04]
        Downside difference: [0.0, 0.0, -0.04]
        """
        returns = [0.10, 0.04, -0.02]
        rf = 0.02
        metrics = calculate_performance_metrics(returns, risk_free_rate=rf)

        assert metrics.sortino_ratio > 0.0
        # When positive outliers exist, Sortino ratio is typically higher than Sharpe ratio
        assert metrics.sortino_ratio > metrics.sharpe_ratio

    def test_risk_free_rate_impact_on_ratios(self):
        """Higher risk-free rate must decrease both Sharpe and Sortino ratios."""
        returns = [0.06, 0.02, 0.08, -0.01]

        metrics_low_rf = calculate_performance_metrics(returns, risk_free_rate=0.01)
        metrics_high_rf = calculate_performance_metrics(returns, risk_free_rate=0.03)

        assert metrics_low_rf.sharpe_ratio > metrics_high_rf.sharpe_ratio
        assert metrics_low_rf.sortino_ratio > metrics_high_rf.sortino_ratio
        # Maximum drawdown depends only on return sequence, not the risk-free rate
        assert metrics_low_rf.max_drawdown == pytest.approx(metrics_high_rf.max_drawdown)


class TestInputTypeCompatibilityAndValidation:
    """Verify support for standard Python and numerical data structures, as well as input validation."""

    @pytest.mark.parametrize(
        "converter",
        [
            list,
            tuple,
            np.array,
            pd.Series,
        ],
    )
    def test_supported_sequence_containers(self, converter):
        """Function must accept list, tuple, numpy.ndarray, and pandas.Series identically."""
        raw_data = [0.03, -0.01, 0.02, 0.05, -0.02]
        container = converter(raw_data)

        metrics = calculate_performance_metrics(container, risk_free_rate=0.01)

        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.sharpe_ratio > 0.0
        assert metrics.sortino_ratio > 0.0
        assert metrics.max_drawdown > 0.0

    @pytest.mark.parametrize("invalid_input", [None, "invalid_string", 12345])
    def test_invalid_input_type_raises_exception(self, invalid_input):
        """Non-sequence inputs must raise TypeError or ValueError."""
        with pytest.raises((TypeError, ValueError)):
            calculate_performance_metrics(invalid_input)

    def test_non_numeric_elements_raises_exception(self):
        """Series containing non-numeric elements must raise TypeError or ValueError."""
        with pytest.raises((TypeError, ValueError)):
            calculate_performance_metrics([0.01, "bad_data", 0.05])