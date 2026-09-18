import numpy as np
import pandas as pd
import pytest

# Module exports under test
import src.ta as ta
from src.ta.momentum import macd, rsi
from src.ta.volatility import atr, bb


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def sample_close_series() -> pd.Series:
    """Deterministic close price series of 60 periods."""
    np.random.seed(42)
    base = 100.0
    changes = np.sin(np.linspace(0, 4 * np.pi, 60)) * 5.0
    prices = base + changes + np.linspace(0, 10, 60)
    index = pd.date_range("2023-01-01", periods=60, freq="D")
    return pd.Series(prices, index=index, name="close")


@pytest.fixture
def monotonic_increasing_close() -> pd.Series:
    """Monotonically increasing close prices."""
    index = pd.date_range("2023-01-01", periods=30, freq="D")
    return pd.Series(np.linspace(10.0, 40.0, 30), index=index, name="close")


@pytest.fixture
def monotonic_decreasing_close() -> pd.Series:
    """Monotonically decreasing close prices."""
    index = pd.date_range("2023-01-01", periods=30, freq="D")
    return pd.Series(np.linspace(40.0, 10.0, 30), index=index, name="close")


@pytest.fixture
def sample_ohlc_data() -> pd.DataFrame:
    """Deterministic OHLC DataFrame with 60 periods."""
    index = pd.date_range("2023-01-01", periods=60, freq="D")
    close = 100.0 + np.sin(np.linspace(0, 4 * np.pi, 60)) * 5.0
    high = close + 2.0
    low = close - 2.0
    return pd.DataFrame({"high": high, "low": low, "close": close}, index=index)


# =====================================================================
# Helpers
# =====================================================================

def extract_macd_components(result):
    """Extract macd, signal, and histogram from DataFrame or namedtuple."""
    if isinstance(result, pd.DataFrame):
        return result["macd"], result["signal"], result["histogram"]
    return result.macd, result.signal, result.histogram


def extract_bb_components(result):
    """Extract upper, middle, lower from DataFrame, namedtuple, or tuple."""
    if isinstance(result, pd.DataFrame):
        return result["upper"], result["middle"], result["lower"]
    if hasattr(result, "upper") and hasattr(result, "middle") and hasattr(result, "lower"):
        return result.upper, result.middle, result.lower
    if isinstance(result, tuple) and len(result) == 3:
        return result[0], result[1], result[2]
    raise TypeError(f"Unexpected Bollinger Bands return structure: {type(result)}")


# =====================================================================
# Package / API Exports Verification
# =====================================================================

class TestTaExports:
    """Verify package level accessibility per requirements."""

    def test_package_exports(self):
        assert hasattr(ta, "rsi")
        assert callable(ta.rsi)
        assert hasattr(ta, "macd")
        assert callable(ta.macd)
        assert hasattr(ta, "bb")
        assert callable(ta.bb)
        assert hasattr(ta, "atr")
        assert callable(ta.atr)

    def test_direct_momentum_module_exports(self):
        assert callable(rsi)
        assert callable(macd)

    def test_direct_volatility_module_exports(self):
        assert callable(bb)
        assert callable(atr)


# =====================================================================
# RSI Indicator Tests (src/ta/momentum.py)
# =====================================================================

class TestRSI:
    """
    Acceptance Criteria:
    - Given a series of closing prices,
      When ta.rsi is computed with a period of 14,
      Then return the Relative Strength Index values bounded between 0 and 100
      with matching initial NaN padding.
    """

    def test_rsi_bounds_and_nan_padding_default_period(self, sample_close_series):
        period = 14
        result = ta.rsi(sample_close_series, period=period)

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_close_series)
        assert result.index.equals(sample_close_series.index)

        # Initial NaN padding: first `period` entries must be NaN
        assert result.iloc[:period].isna().all()

        # Remaining non-NaN values must be bounded [0, 100]
        valid_values = result.dropna()
        assert not valid_values.empty
        assert (valid_values >= 0.0).all()
        assert (valid_values <= 100.0).all()

    def test_rsi_custom_period(self, sample_close_series):
        period = 7
        result = rsi(sample_close_series, period=period)

        assert result.iloc[:period].isna().all()
        assert not result.iloc[period:].isna().any()
        assert (result.dropna() >= 0.0).all()
        assert (result.dropna() <= 100.0).all()

    def test_rsi_monotonic_increase_approaches_100(self, monotonic_increasing_close):
        period = 14
        result = ta.rsi(monotonic_increasing_close, period=period)

        valid = result.dropna()
        assert not valid.empty
        # With zero downward price changes, RSI should equal 100
        np.testing.assert_allclose(valid.values, 100.0, atol=1e-5)

    def test_rsi_monotonic_decrease_approaches_0(self, monotonic_decreasing_close):
        period = 14
        result = ta.rsi(monotonic_decreasing_close, period=period)

        valid = result.dropna()
        assert not valid.empty
        # With zero upward price changes, RSI should equal 0
        np.testing.assert_allclose(valid.values, 0.0, atol=1e-5)

    def test_rsi_constant_series_handles_zero_change(self):
        index = pd.date_range("2023-01-01", periods=20, freq="D")
        constant_close = pd.Series([50.0] * 20, index=index)
        result = ta.rsi(constant_close, period=14)

        valid = result.dropna()
        # When price does not move, RSI is typically defined as 50 or bounded [0, 100]
        assert not valid.empty
        assert (valid >= 0.0).all()
        assert (valid <= 100.0).all()

    def test_rsi_insufficient_data_raises_value_error(self):
        short_series = pd.Series([10.0, 11.0, 12.0])
        with pytest.raises(ValueError):
            ta.rsi(short_series, period=14)

    @pytest.mark.parametrize("invalid_period", [0, -5])
    def test_rsi_invalid_period_raises_value_error(self, sample_close_series, invalid_period):
        with pytest.raises(ValueError):
            ta.rsi(sample_close_series, period=invalid_period)


# =====================================================================
# MACD Indicator Tests (src/ta/momentum.py)
# =====================================================================

class TestMACD:
    """
    Acceptance Criteria:
    - Given a series of closing prices,
      When ta.macd is invoked with fast (12), slow (26), and signal (9) parameters,
      Then return a named tuple or DataFrame containing macd, signal, and histogram series.
    """

    def test_macd_returns_valid_structure(self, sample_close_series):
        result = ta.macd(sample_close_series, fast=12, slow=26, signal=9)

        # Must return DataFrame or namedtuple with macd, signal, histogram
        macd_line, signal_line, hist_line = extract_macd_components(result)

        assert isinstance(macd_line, pd.Series)
        assert isinstance(signal_line, pd.Series)
        assert isinstance(hist_line, pd.Series)

        assert len(macd_line) == len(sample_close_series)
        assert len(signal_line) == len(sample_close_series)
        assert len(hist_line) == len(sample_close_series)

        assert macd_line.index.equals(sample_close_series.index)
        assert signal_line.index.equals(sample_close_series.index)
        assert hist_line.index.equals(sample_close_series.index)

    def test_macd_histogram_equals_difference(self, sample_close_series):
        result = ta.macd(sample_close_series, fast=12, slow=26, signal=9)
        macd_line, signal_line, hist_line = extract_macd_components(result)

        # histogram = macd - signal
        diff = macd_line - signal_line
        valid_mask = ~(macd_line.isna() | signal_line.isna() | hist_line.isna())
        assert valid_mask.sum() > 0

        np.testing.assert_allclose(
            hist_line[valid_mask].values,
            diff[valid_mask].values,
            rtol=1e-7,
            atol=1e-7
        )

    def test_macd_initial_nan_padding(self, sample_close_series):
        fast, slow, signal = 12, 26, 9
        result = macd(sample_close_series, fast=fast, slow=slow, signal=signal)
        macd_line, signal_line, hist_line = extract_macd_components(result)

        # Slow period is required before first valid MACD value can be formed
        assert macd_line.iloc[: slow - 1].isna().all()

        # Signal line requires slow + signal - 1 before valid value
        warmup_required = slow + signal - 2
        assert signal_line.iloc[:warmup_required].isna().all()
        assert hist_line.iloc[:warmup_required].isna().all()

    def test_macd_custom_parameters(self, sample_close_series):
        result = ta.macd(sample_close_series, fast=5, slow=10, signal=3)
        macd_line, signal_line, hist_line = extract_macd_components(result)

        valid_hist = hist_line.dropna()
        assert not valid_hist.empty

    @pytest.mark.parametrize(
        "fast,slow,signal",
        [
            (26, 12, 9),  # fast >= slow
            (12, 12, 9),  # fast == slow
            (0, 26, 9),   # fast <= 0
            (12, 0, 9),   # slow <= 0
            (12, 26, 0),  # signal <= 0
            (-1, 26, 9),
        ],
    )
    def test_macd_invalid_parameters_raise_value_error(self, sample_close_series, fast, slow, signal):
        with pytest.raises(ValueError):
            ta.macd(sample_close_series, fast=fast, slow=slow, signal=signal)

    def test_macd_insufficient_data_raises_value_error(self):
        short_series = pd.Series([100.0] * 15)
        with pytest.raises(ValueError):
            ta.macd(short_series, fast=12, slow=26, signal=9)


# =====================================================================
# Bollinger Bands Tests (src/ta/volatility.py)
# =====================================================================

class TestBollingerBands:
    """
    Acceptance Criteria:
    - Given a series of closing prices,
      When ta.bb is called with period 20 and std multiplier 2,
      Then return the upper band, middle band (SMA), and lower band series.
    """

    def test_bb_default_structure_and_index(self, sample_close_series):
        period, std_mult = 20, 2
        result = ta.bb(sample_close_series, period=period, std_multiplier=std_mult)

        upper, middle, lower = extract_bb_components(result)

        assert isinstance(upper, pd.Series)
        assert isinstance(middle, pd.Series)
        assert isinstance(lower, pd.Series)

        assert len(upper) == len(sample_close_series)
        assert len(middle) == len(sample_close_series)
        assert len(lower) == len(sample_close_series)

        assert upper.index.equals(sample_close_series.index)
        assert middle.index.equals(sample_close_series.index)
        assert lower.index.equals(sample_close_series.index)

    def test_bb_middle_band_matches_sma(self, sample_close_series):
        period = 20
        result = ta.bb(sample_close_series, period=period, std_multiplier=2)
        _, middle, _ = extract_bb_components(result)

        expected_sma = sample_close_series.rolling(window=period).mean()
        pd.testing.assert_series_equal(middle, expected_sma, check_names=False)

    def test_bb_band_order_and_symmetry(self, sample_close_series):
        period, std_mult = 20, 2
        result = bb(sample_close_series, period=period, std_multiplier=std_mult)
        upper, middle, lower = extract_bb_components(result)

        valid_idx = ~(upper.isna() | middle.isna() | lower.isna())
        u = upper[valid_idx]
        m = middle[valid_idx]
        l = lower[valid_idx]

        # Upper >= Middle >= Lower
        assert (u >= m).all()
        assert (m >= l).all()

        # Symmetry: (Upper - Middle) == (Middle - Lower)
        diff_upper = u - m
        diff_lower = m - l
        np.testing.assert_allclose(diff_upper.values, diff_lower.values, rtol=1e-7, atol=1e-7)

    def test_bb_initial_nan_padding(self, sample_close_series):
        period = 20
        result = ta.bb(sample_close_series, period=period, std_multiplier=2)
        upper, middle, lower = extract_bb_components(result)

        # First period - 1 values must be NaN
        assert upper.iloc[: period - 1].isna().all()
        assert middle.iloc[: period - 1].isna().all()
        assert lower.iloc[: period - 1].isna().all()

        # Period-th element onward must be numeric
        assert not middle.iloc[period - 1 :].isna().any()

    def test_bb_zero_volatility(self):
        index = pd.date_range("2023-01-01", periods=25, freq="D")
        constant_close = pd.Series([100.0] * 25, index=index)
        result = ta.bb(constant_close, period=20, std_multiplier=2)
        upper, middle, lower = extract_bb_components(result)

        valid_idx = ~middle.isna()
        # When std is zero: upper == middle == lower
        np.testing.assert_allclose(upper[valid_idx], 100.0)
        np.testing.assert_allclose(middle[valid_idx], 100.0)
        np.testing.assert_allclose(lower[valid_idx], 100.0)

    @pytest.mark.parametrize("period,std_mult", [(0, 2), (-10, 2), (20, 0), (20, -1)])
    def test_bb_invalid_parameters_raise_value_error(self, sample_close_series, period, std_mult):
        with pytest.raises(ValueError):
            ta.bb(sample_close_series, period=period, std_multiplier=std_mult)

    def test_bb_insufficient_data_raises_value_error(self):
        short_series = pd.Series([10.0] * 10)
        with pytest.raises(ValueError):
            ta.bb(short_series, period=20, std_multiplier=2)


# =====================================================================
# Average True Range (ATR) Tests (src/ta/volatility.py)
# =====================================================================

class TestATR:
    """
    Acceptance Criteria:
    - Given high, low, and close price series,
      When ta.atr is computed with period 14,
      Then return the Average True Range reflecting moving average of true ranges.
    """

    def test_atr_output_series_and_nan_padding(self, sample_ohlc_data):
        period = 14
        result = ta.atr(
            high=sample_ohlc_data["high"],
            low=sample_ohlc_data["low"],
            close=sample_ohlc_data["close"],
            period=period
        )

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_ohlc_data)
        assert result.index.equals(sample_ohlc_data.index)

        # Initial NaN padding matching warmup
        assert result.iloc[: period - 1].isna().all()
        assert not result.iloc[period :].isna().any()

        # ATR must be non-negative
        valid_atr = result.dropna()
        assert (valid_atr >= 0.0).all()

    def test_atr_manual_calculation_with_gaps(self):
        """Verify True Range captures high-low, abs(high-prev_close), abs(low-prev_close)."""
        high = pd.Series([10.0, 12.0, 16.0, 14.0])
        low = pd.Series([8.0, 9.0, 13.0, 11.0])
        close = pd.Series([9.0, 11.0, 15.0, 12.0])

        # Manual TR calculation:
        # bar 0: TR = 10 - 8 = 2.0
        # bar 1: TR = max(12-9, |12-9|, |9-9|) = max(3, 3, 0) = 3.0
        # bar 2: TR = max(16-13, |16-11|, |13-11|) = max(3, 5, 2) = 5.0 (gap up impact)
        # bar 3: TR = max(14-11, |14-15|, |11-15|) = max(3, 1, 4) = 4.0 (gap down impact)

        # With period=3, index 2 is first valid ATR (SMA or RMA initial mean of first 3 TRs)
        # (2.0 + 3.0 + 5.0) / 3 = 10.0 / 3 = 3.3333333333333335
        result = atr(high=high, low=low, close=close, period=3)

        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        np.testing.assert_allclose(result.iloc[2], 10.0 / 3.0, rtol=1e-5)

    def test_atr_zero_volatility(self):
        """When high == low == close everywhere, ATR must be 0."""
        n = 20
        high = pd.Series([50.0] * n)
        low = pd.Series([50.0] * n)
        close = pd.Series([50.0] * n)

        result = ta.atr(high=high, low=low, close=close, period=14)
        valid = result.dropna()
        assert not valid.empty
        np.testing.assert_allclose(valid.values, 0.0)

    def test_atr_invalid_inputs_high_less_than_low(self):
        """High must always be >= Low."""
        high = pd.Series([10.0, 8.0, 12.0] * 5)
        low = pd.Series([11.0, 9.0, 10.0] * 5)  # low > high at index 0 and 1
        close = pd.Series([10.5, 8.5, 11.0] * 5)

        with pytest.raises(ValueError):
            ta.atr(high=high, low=low, close=close, period=5)

    def test_atr_mismatched_series_lengths_raise_value_error(self):
        high = pd.Series([12.0] * 20)
        low = pd.Series([10.0] * 20)
        close = pd.Series([11.0] * 15)  # Mismatched length

        with pytest.raises(ValueError):
            ta.atr(high=high, low=low, close=close, period=14)

    @pytest.mark.parametrize("invalid_period", [0, -1])
    def test_atr_invalid_period_raises_value_error(self, sample_ohlc_data, invalid_period):
        with pytest.raises(ValueError):
            ta.atr(
                high=sample_ohlc_data["high"],
                low=sample_ohlc_data["low"],
                close=sample_ohlc_data["close"],
                period=invalid_period,
            )

    def test_atr_insufficient_data_raises_value_error(self):
        high = pd.Series([12.0, 13.0])
        low = pd.Series([10.0, 11.0])
        close = pd.Series([11.0, 12.0])

        with pytest.raises(ValueError):
            ta.atr(high=high, low=low, close=close, period=14)