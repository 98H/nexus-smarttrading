import math
from datetime import datetime
import numpy as np
import pandas as pd
import pytest

from src.indicators.adaptive_trend import (
    AdaptiveTrendFilter,
    AdaptiveTrendResult,
    TrendDirection,
)
from src.signals.confirmation import (
    ConfirmedSignal,
    RawSignal,
    SignalConfirmationFilter,
    SignalDirection,
    SignalStatus,
)


# =====================================================================
# Unit Tests: Adaptive Trend Filter (src/indicators/adaptive_trend.py)
# =====================================================================

class TestAdaptiveTrendFilterCalculation:
    """Tests for computing the Efficiency Ratio and dynamic adaptive filter values."""

    @pytest.fixture
    def default_filter(self) -> AdaptiveTrendFilter:
        return AdaptiveTrendFilter(period=10, fast_period=2, slow_period=30)

    def test_efficiency_ratio_pure_trend_equals_one(self, default_filter: AdaptiveTrendFilter):
        """A strictly monotonic price series should have an efficiency ratio of 1.0."""
        # 11 data points so that the 10-period lookback has 10 differences
        prices = pd.Series([100.0 + i * 2.0 for i in range(11)])
        result: AdaptiveTrendResult = default_filter.calculate(prices)

        latest_er = result.efficiency_ratio.iloc[-1]
        assert math.isclose(latest_er, 1.0, rel_tol=1e-5)

    def test_efficiency_ratio_perfect_oscillation_equals_zero(self, default_filter: AdaptiveTrendFilter):
        """A price series oscillating back to start over the period should have ER = 0.0."""
        # Oscillates between 100.0 and 105.0, ending at 100.0 after 10 intervals
        prices = pd.Series([100.0 if i % 2 == 0 else 105.0 for i in range(11)])
        result: AdaptiveTrendResult = default_filter.calculate(prices)

        latest_er = result.efficiency_ratio.iloc[-1]
        assert math.isclose(latest_er, 0.0, abs_tol=1e-5)

    def test_efficiency_ratio_zero_volatility_handles_division_by_zero(self, default_filter: AdaptiveTrendFilter):
        """When prices are entirely flat, volatility is 0; ER should evaluate to 0.0 without crashing."""
        prices = pd.Series([100.0] * 15)
        result: AdaptiveTrendResult = default_filter.calculate(prices)

        assert not result.efficiency_ratio.isna().all()
        # For flat price after lookback, ER should be safely 0.0
        assert math.isclose(result.efficiency_ratio.iloc[-1], 0.0, abs_tol=1e-5)

    def test_efficiency_ratio_bounded_between_zero_and_one(self, default_filter: AdaptiveTrendFilter):
        """ER must strictly remain within [0.0, 1.0] across noisy market series."""
        np.random.seed(42)
        random_prices = pd.Series(100.0 + np.cumsum(np.random.normal(0, 1, 100)))
        result: AdaptiveTrendResult = default_filter.calculate(random_prices)

        valid_er = result.efficiency_ratio.dropna()
        assert (valid_er >= 0.0).all()
        assert (valid_er <= 1.0).all()

    def test_dynamic_filter_adjusts_smoothing_based_on_noise(self, default_filter: AdaptiveTrendFilter):
        """High-efficiency data must produce faster filter adaptation than noisy data."""
        # 1. Noisy flat segment with slight fluctuation
        noisy_segment = [100.0 if i % 2 == 0 else 101.0 for i in range(20)]
        # 2. Sudden trending breakout
        trending_segment = [100.0 + i * 5.0 for i in range(1, 21)]

        full_series = pd.Series(noisy_segment + trending_segment)
        result = default_filter.calculate(full_series)

        # In the noisy segment, filter updates should be minimal (smooth)
        noise_change = abs(result.filter_values.iloc[19] - result.filter_values.iloc[10])
        # In the trending segment, filter updates should be noticeably higher
        trend_change = abs(result.filter_values.iloc[39] - result.filter_values.iloc[30])

        assert trend_change > noise_change

    def test_filter_output_structure_and_dimensions(self, default_filter: AdaptiveTrendFilter):
        """AdaptiveTrendResult must contain aligned series with equal lengths."""
        prices = pd.Series(np.linspace(100.0, 200.0, 30))
        result = default_filter.calculate(prices)

        assert isinstance(result, AdaptiveTrendResult)
        assert len(result.filter_values) == len(prices)
        assert len(result.efficiency_ratio) == len(prices)
        assert len(result.trend_direction) == len(prices)


class TestAdaptiveTrendDirection:
    """Tests for direction classification based on adaptive filter state."""

    @pytest.fixture
    def default_filter(self) -> AdaptiveTrendFilter:
        return AdaptiveTrendFilter(period=5, fast_period=2, slow_period=30)

    def test_trend_direction_upward(self, default_filter: AdaptiveTrendFilter):
        """Monotonically rising prices produce TrendDirection.UPWARD."""
        prices = pd.Series([10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0])
        result = default_filter.calculate(prices)

        assert result.current_direction == TrendDirection.UPWARD
        assert result.trend_direction.iloc[-1] == TrendDirection.UPWARD

    def test_trend_direction_downward(self, default_filter: AdaptiveTrendFilter):
        """Monotonically falling prices produce TrendDirection.DOWNWARD."""
        prices = pd.Series([50.0, 45.0, 40.0, 35.0, 30.0, 25.0, 20.0])
        result = default_filter.calculate(prices)

        assert result.current_direction == TrendDirection.DOWNWARD
        assert result.trend_direction.iloc[-1] == TrendDirection.DOWNWARD

    def test_trend_direction_flat_when_constant_prices(self, default_filter: AdaptiveTrendFilter):
        """Constant prices produce TrendDirection.FLAT."""
        prices = pd.Series([100.0] * 15)
        result = default_filter.calculate(prices)

        assert result.current_direction == TrendDirection.FLAT
        assert result.trend_direction.iloc[-1] == TrendDirection.FLAT


class TestAdaptiveTrendFilterValidation:
    """Validation and error handling for AdaptiveTrendFilter inputs."""

    @pytest.mark.parametrize("invalid_period", [0, -1, -10])
    def test_invalid_period_raises_value_error(self, invalid_period: int):
        with pytest.raises(ValueError):
            AdaptiveTrendFilter(period=invalid_period)

    @pytest.mark.parametrize(
        "fast_p,slow_p",
        [
            (30, 2),   # fast_period > slow_period
            (10, 10),  # fast_period == slow_period
            (0, 30),   # fast_period <= 0
            (2, 0),    # slow_period <= 0
        ],
    )
    def test_invalid_period_relationships_raise_value_error(self, fast_p: int, slow_p: int):
        with pytest.raises(ValueError):
            AdaptiveTrendFilter(period=10, fast_period=fast_p, slow_period=slow_p)

    def test_empty_prices_series_raises_value_error(self):
        adaptive_filter = AdaptiveTrendFilter(period=10)
        with pytest.raises(ValueError):
            adaptive_filter.calculate(pd.Series([], dtype=float))

    def test_insufficient_data_points_raises_value_error(self):
        adaptive_filter = AdaptiveTrendFilter(period=10)
        short_series = pd.Series([10.0, 11.0, 12.0])  # Less than period=10
        with pytest.raises(ValueError):
            adaptive_filter.calculate(short_series)


# =====================================================================
# Unit Tests: Signal Confirmation (src/signals/confirmation.py)
# =====================================================================

class TestSignalConfirmationFilter:
    """Tests for confirming or rejecting entry signals against adaptive trend states."""

    @pytest.fixture
    def confirmation_filter(self) -> SignalConfirmationFilter:
        return SignalConfirmationFilter()

    @pytest.fixture
    def sample_timestamp(self) -> datetime:
        return datetime(2023, 10, 15, 12, 0, 0)

    def test_long_signal_with_upward_trend_is_approved(
        self, confirmation_filter: SignalConfirmationFilter, sample_timestamp: datetime
    ):
        """A raw LONG signal matching an UPWARD trend must be APPROVED."""
        raw_signal = RawSignal(
            signal_id="sig-001",
            direction=SignalDirection.LONG,
            timestamp=sample_timestamp,
            price=150.0,
        )

        confirmed: ConfirmedSignal = confirmation_filter.confirm(
            signal=raw_signal, trend_direction=TrendDirection.UPWARD
        )

        assert confirmed.status == SignalStatus.APPROVED
        assert confirmed.raw_signal == raw_signal
        assert confirmed.trend_direction == TrendDirection.UPWARD

    def test_short_signal_with_downward_trend_is_approved(
        self, confirmation_filter: SignalConfirmationFilter, sample_timestamp: datetime
    ):
        """A raw SHORT signal matching a DOWNWARD trend must be APPROVED."""
        raw_signal = RawSignal(
            signal_id="sig-002",
            direction=SignalDirection.SHORT,
            timestamp=sample_timestamp,
            price=140.0,
        )

        confirmed: ConfirmedSignal = confirmation_filter.confirm(
            signal=raw_signal, trend_direction=TrendDirection.DOWNWARD
        )

        assert confirmed.status == SignalStatus.APPROVED
        assert confirmed.raw_signal == raw_signal
        assert confirmed.trend_direction == TrendDirection.DOWNWARD

    def test_long_signal_with_downward_trend_is_rejected(
        self, confirmation_filter: SignalConfirmationFilter, sample_timestamp: datetime
    ):
        """A raw LONG signal conflicting with a DOWNWARD trend must be REJECTED."""
        raw_signal = RawSignal(
            signal_id="sig-003",
            direction=SignalDirection.LONG,
            timestamp=sample_timestamp,
            price=130.0,
        )

        confirmed: ConfirmedSignal = confirmation_filter.confirm(
            signal=raw_signal, trend_direction=TrendDirection.DOWNWARD
        )

        assert confirmed.status == SignalStatus.REJECTED
        assert confirmed.raw_signal == raw_signal
        assert confirmed.trend_direction == TrendDirection.DOWNWARD

    def test_short_signal_with_upward_trend_is_rejected(
        self, confirmation_filter: SignalConfirmationFilter, sample_timestamp: datetime
    ):
        """A raw SHORT signal conflicting with an UPWARD trend must be REJECTED."""
        raw_signal = RawSignal(
            signal_id="sig-004",
            direction=SignalDirection.SHORT,
            timestamp=sample_timestamp,
            price=160.0,
        )

        confirmed: ConfirmedSignal = confirmation_filter.confirm(
            signal=raw_signal, trend_direction=TrendDirection.UPWARD
        )

        assert confirmed.status == SignalStatus.REJECTED
        assert confirmed.raw_signal == raw_signal
        assert confirmed.trend_direction == TrendDirection.UPWARD

    @pytest.mark.parametrize("signal_direction", [SignalDirection.LONG, SignalDirection.SHORT])
    def test_signal_with_flat_trend_is_rejected(
        self,
        confirmation_filter: SignalConfirmationFilter,
        sample_timestamp: datetime,
        signal_direction: SignalDirection,
    ):
        """Any entry signal received during a FLAT trend must be REJECTED."""
        raw_signal = RawSignal(
            signal_id="sig-flat",
            direction=signal_direction,
            timestamp=sample_timestamp,
            price=100.0,
        )

        confirmed: ConfirmedSignal = confirmation_filter.confirm(
            signal=raw_signal, trend_direction=TrendDirection.FLAT
        )

        assert confirmed.status == SignalStatus.REJECTED
        assert confirmed.trend_direction == TrendDirection.FLAT


class TestSignalConfirmationBatchAndEdgeCases:
    """Tests batch processing and edge condition validation."""

    def test_batch_confirmation_filtering(self):
        """Batch processing correctly categorizes approved and rejected signals."""
        now = datetime.now()
        signals = [
            RawSignal("1", SignalDirection.LONG, now, 100.0),
            RawSignal("2", SignalDirection.SHORT, now, 99.0),
            RawSignal("3", SignalDirection.LONG, now, 101.0),
            RawSignal("4", SignalDirection.SHORT, now, 98.0),
        ]
        trend_states = [
            TrendDirection.UPWARD,    # 1: Approved
            TrendDirection.UPWARD,    # 2: Rejected
            TrendDirection.DOWNWARD,  # 3: Rejected
            TrendDirection.DOWNWARD,  # 4: Approved
        ]

        filter_instance = SignalConfirmationFilter()
        results = [
            filter_instance.confirm(sig, state)
            for sig, state in zip(signals, trend_states)
        ]

        statuses = [r.status for r in results]
        expected_statuses = [
            SignalStatus.APPROVED,
            SignalStatus.REJECTED,
            SignalStatus.REJECTED,
            SignalStatus.APPROVED,
        ]
        assert statuses == expected_statuses

    def test_confirmed_signal_carries_rejection_reason_when_rejected(self):
        """Rejected signals should include an informative reason attribute."""
        raw_signal = RawSignal("rej-1", SignalDirection.LONG, datetime.now(), 50.0)
        filter_instance = SignalConfirmationFilter()

        confirmed = filter_instance.confirm(raw_signal, TrendDirection.DOWNWARD)
        assert confirmed.status == SignalStatus.REJECTED
        assert confirmed.reason is not None
        assert isinstance(confirmed.reason, str)

    def test_invalid_arguments_raise_type_or_value_error(self):
        """Passing None or invalid arguments should raise an exception."""
        filter_instance = SignalConfirmationFilter()
        raw_signal = RawSignal("err-1", SignalDirection.LONG, datetime.now(), 50.0)

        with pytest.raises((ValueError, TypeError)):
            filter_instance.confirm(signal=None, trend_direction=TrendDirection.UPWARD)  # type: ignore

        with pytest.raises((ValueError, TypeError)):
            filter_instance.confirm(signal=raw_signal, trend_direction=None)  # type: ignore