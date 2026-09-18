import numpy as np
import pandas as pd
import pytest

from src.indicators import (
    BoundaryClassification,
    OscillatorMatrixConfig,
    OscillatorMatrixEngine,
    OscillatorMatrixResult,
)
from src.indicators.oscillator_matrix import (
    BoundaryClassification as MatrixBoundaryClassification,
    OscillatorMatrixConfig as MatrixOscillatorMatrixConfig,
    OscillatorMatrixEngine as ModuleOscillatorMatrixEngine,
    OscillatorMatrixResult as MatrixOscillatorMatrixResult,
)


@pytest.fixture
def sample_ohlcv_data() -> pd.DataFrame:
    """Generate 100 bars of deterministic synthetic OHLCV data."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=100, freq="1h")
    close = 100.0 + np.cumsum(np.random.randn(100))
    high = close + np.abs(np.random.randn(100))
    low = close - np.abs(np.random.randn(100))
    open_price = low + (high - low) * np.random.rand(100)
    volume = np.random.uniform(1000, 5000, size=100)

    df = pd.DataFrame(
        {
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=dates,
    )
    return df


@pytest.fixture
def sample_normalized_components(sample_ohlcv_data: pd.DataFrame) -> dict[str, pd.Series]:
    """Generate a set of 4 normalized oscillator series bounded in [-100, 100]."""
    index = sample_ohlcv_data.index
    n = len(index)
    t = np.linspace(0, 4 * np.pi, n)

    # 4 distinct oscillator series oscillating within [-100, 100]
    osc1 = pd.Series(100.0 * np.sin(t), index=index, name="rsi_norm")
    osc2 = pd.Series(100.0 * np.cos(t), index=index, name="mfi_norm")
    osc3 = pd.Series(80.0 * np.sin(2 * t), index=index, name="stoch_norm")
    osc4 = pd.Series(60.0 * np.cos(0.5 * t), index=index, name="cci_norm")

    return {
        "rsi_norm": osc1,
        "mfi_norm": osc2,
        "stoch_norm": osc3,
        "cci_norm": osc4,
    }


class TestOscillatorMatrixExports:
    """Validate module architecture and required exports."""

    def test_package_exports(self) -> None:
        """Verify indicator package exports required matrix symbols."""
        assert OscillatorMatrixEngine is ModuleOscillatorMatrixEngine
        assert OscillatorMatrixConfig is MatrixOscillatorMatrixConfig
        assert OscillatorMatrixResult is MatrixOscillatorMatrixResult
        assert BoundaryClassification is MatrixBoundaryClassification

    def test_classification_enum_values(self) -> None:
        """Verify boundary classification states are defined."""
        assert hasattr(BoundaryClassification, "OVERBOUGHT")
        assert hasattr(BoundaryClassification, "OVERSOLD")
        assert hasattr(BoundaryClassification, "NEUTRAL")


class TestOscillatorMatrixConfigValidation:
    """Validate configuration constraint enforcement."""

    def test_default_config_valid(self) -> None:
        """Test default parameters produce valid configuration."""
        config = OscillatorMatrixConfig()
        assert config.overbought_threshold > config.oversold_threshold
        assert config.overbought_threshold <= 100.0
        assert config.oversold_threshold >= -100.0

    def test_invalid_boundary_order_raises(self) -> None:
        """Test that overbought <= oversold triggers ValueError."""
        with pytest.raises(ValueError):
            OscillatorMatrixConfig(overbought_threshold=30.0, oversold_threshold=50.0)

        with pytest.raises(ValueError):
            OscillatorMatrixConfig(overbought_threshold=50.0, oversold_threshold=50.0)

    def test_boundary_out_of_bounds_raises(self) -> None:
        """Test thresholds outside [-100, 100] raise ValueError."""
        with pytest.raises(ValueError):
            OscillatorMatrixConfig(overbought_threshold=105.0)

        with pytest.raises(ValueError):
            OscillatorMatrixConfig(oversold_threshold=-105.0)

    def test_negative_or_zero_weights_raise(self) -> None:
        """Test invalid weights specification raises ValueError."""
        with pytest.raises(ValueError):
            OscillatorMatrixConfig(weights={"osc1": -1.0, "osc2": 1.0})

        with pytest.raises(ValueError):
            OscillatorMatrixConfig(weights={"osc1": 0.0, "osc2": 0.0})

    def test_invalid_lookback_windows_raise(self) -> None:
        """Test non-positive lookback window values raise ValueError."""
        with pytest.raises(ValueError):
            OscillatorMatrixConfig(lookback_windows=[14, 0, 28])

        with pytest.raises(ValueError):
            OscillatorMatrixConfig(lookback_windows=[14, -5, 28])


class TestOscillatorMatrixEngineEvaluation:
    """Validate calculation, aggregation, normalization, and classification logic."""

    def test_composite_aggregation_with_precomputed_series(
        self,
        sample_ohlcv_data: pd.DataFrame,
        sample_normalized_components: dict[str, pd.Series],
    ) -> None:
        """Verify composite output structure, shape, and boundary normalization [-100, 100]."""
        config = OscillatorMatrixConfig(
            overbought_threshold=60.0,
            oversold_threshold=-60.0,
        )
        engine = OscillatorMatrixEngine(config=config)
        result = engine.evaluate(
            ohlcv=sample_ohlcv_data,
            components=sample_normalized_components,
        )

        assert isinstance(result, OscillatorMatrixResult)
        assert isinstance(result.composite, pd.Series)
        assert isinstance(result.classification, pd.Series)
        assert len(result.composite) == len(sample_ohlcv_data)
        assert len(result.classification) == len(sample_ohlcv_data)

        # Range normalization constraint
        valid_composite = result.composite.dropna()
        assert (valid_composite >= -100.0).all()
        assert (valid_composite <= 100.0).all()

    def test_exact_hand_calculated_weighted_composite(self) -> None:
        """Verify exact aggregation arithmetic with controlled inputs."""
        index = pd.date_range("2024-01-01", periods=3, freq="1D")
        ohlcv = pd.DataFrame(
            {
                "open": [10.0, 11.0, 12.0],
                "high": [12.0, 13.0, 14.0],
                "low": [9.0, 10.0, 11.0],
                "close": [11.0, 12.0, 13.0],
                "volume": [100, 100, 100],
            },
            index=index,
        )

        # Three components with designated values
        # Bar 0: (50*1 + -25*2 + 75*1) / 4 = 75 / 4 = 18.75
        # Bar 1: (100*1 + 100*2 + 100*1) / 4 = 400 / 4 = 100.0
        # Bar 2: (-100*1 + -100*2 + -100*1) / 4 = -400 / 4 = -100.0
        components = {
            "compA": pd.Series([50.0, 100.0, -100.0], index=index),
            "compB": pd.Series([-25.0, 100.0, -100.0], index=index),
            "compC": pd.Series([75.0, 100.0, -100.0], index=index),
        }
        weights = {"compA": 1.0, "compB": 2.0, "compC": 1.0}

        config = OscillatorMatrixConfig(
            overbought_threshold=50.0,
            oversold_threshold=-50.0,
            weights=weights,
        )
        engine = OscillatorMatrixEngine(config=config)
        result = engine.evaluate(ohlcv=ohlcv, components=components)

        np.testing.assert_allclose(
            result.composite.values,
            np.array([18.75, 100.0, -100.0]),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_boundary_classification_logic(self) -> None:
        """Verify states are classified as OVERBOUGHT, OVERSOLD, or NEUTRAL."""
        index = pd.date_range("2024-01-01", periods=5, freq="1D")
        ohlcv = pd.DataFrame(
            {
                "open": [10.0] * 5,
                "high": [12.0] * 5,
                "low": [8.0] * 5,
                "close": [10.0] * 5,
                "volume": [1000] * 5,
            },
            index=index,
        )
        # Components designed to yield composite values: [70.0, 50.0, 0.0, -50.0, -80.0]
        components = {
            "c1": pd.Series([70.0, 50.0, 0.0, -50.0, -80.0], index=index),
        }
        config = OscillatorMatrixConfig(
            overbought_threshold=50.0,
            oversold_threshold=-50.0,
        )
        engine = OscillatorMatrixEngine(config=config)
        result = engine.evaluate(ohlcv=ohlcv, components=components)

        # > 50 -> OVERBOUGHT
        # == 50 -> NEUTRAL (strictly above boundary)
        # 0 -> NEUTRAL
        # == -50 -> NEUTRAL (strictly below boundary)
        # < -50 -> OVERSOLD
        expected = [
            BoundaryClassification.OVERBOUGHT,
            BoundaryClassification.NEUTRAL,
            BoundaryClassification.NEUTRAL,
            BoundaryClassification.NEUTRAL,
            BoundaryClassification.OVERSOLD,
        ]
        assert list(result.classification) == expected

    def test_matrix_evaluation_across_lookback_windows(
        self,
        sample_ohlcv_data: pd.DataFrame,
    ) -> None:
        """Verify engine evaluates component oscillator across matrix of lookback windows."""
        lookbacks = [7, 14, 28]
        config = OscillatorMatrixConfig(
            lookback_windows=lookbacks,
            component_types=["rsi", "stochastic"],
            overbought_threshold=60.0,
            oversold_threshold=-60.0,
        )
        engine = OscillatorMatrixEngine(config=config)
        result = engine.evaluate(ohlcv=sample_ohlcv_data)

        # Confirm matrix details are accessible
        assert hasattr(result, "matrix")
        assert isinstance(result.matrix, pd.DataFrame)
        # 2 components * 3 lookbacks = 6 matrix columns
        assert result.matrix.shape[1] == len(lookbacks) * 2

        # Composite oscillator should aggregate all lookback cells
        assert len(result.composite) == len(sample_ohlcv_data)
        valid = result.composite.dropna()
        assert (valid >= -100.0).all()
        assert (valid <= 100.0).all()

    def test_equal_weighting_when_weights_unspecified(
        self,
        sample_ohlcv_data: pd.DataFrame,
    ) -> None:
        """Verify equal weights are applied by default across components."""
        index = sample_ohlcv_data.index[:4]
        ohlcv = sample_ohlcv_data.iloc[:4]
        comp1 = pd.Series([10.0, 20.0, 30.0, 40.0], index=index)
        comp2 = pd.Series([30.0, 40.0, 50.0, 60.0], index=index)

        components = {"osc1": comp1, "osc2": comp2}
        engine = OscillatorMatrixEngine(OscillatorMatrixConfig(weights=None))
        result = engine.evaluate(ohlcv=ohlcv, components=components)

        expected = (comp1 + comp2) / 2.0
        pd.testing.assert_series_equal(result.composite, expected, check_names=False)


class TestOscillatorMatrixEngineEdgeCases:
    """Validate robustness against input anomalies and edge cases."""

    def test_empty_dataframe_raises(self) -> None:
        """Test engine raises ValueError when provided an empty OHLCV DataFrame."""
        empty_ohlcv = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        engine = OscillatorMatrixEngine(OscillatorMatrixConfig())
        with pytest.raises(ValueError):
            engine.evaluate(ohlcv=empty_ohlcv)

    def test_missing_required_columns_raises(self) -> None:
        """Test missing required OHLCV columns raises ValueError."""
        incomplete_ohlcv = pd.DataFrame(
            {
                "open": [10.0, 11.0],
                "close": [10.5, 11.5],
                # Missing 'high', 'low', 'volume'
            }
        )
        engine = OscillatorMatrixEngine(OscillatorMatrixConfig())
        with pytest.raises(ValueError):
            engine.evaluate(ohlcv=incomplete_ohlcv)

    def test_component_length_mismatch_raises(
        self,
        sample_ohlcv_data: pd.DataFrame,
    ) -> None:
        """Test mismatched component series length raises ValueError."""
        mismatched_component = {
            "osc1": pd.Series(
                np.zeros(len(sample_ohlcv_data) - 5),
                index=sample_ohlcv_data.index[:-5],
            )
        }
        engine = OscillatorMatrixEngine(OscillatorMatrixConfig())
        with pytest.raises(ValueError):
            engine.evaluate(ohlcv=sample_ohlcv_data, components=mismatched_component)

    def test_unaligned_datetime_index_raises(
        self,
        sample_ohlcv_data: pd.DataFrame,
    ) -> None:
        """Test component series with unaligned index raises ValueError."""
        shifted_index = pd.date_range("2025-01-01", periods=len(sample_ohlcv_data), freq="1h")
        misaligned_component = {
            "osc1": pd.Series(np.zeros(len(sample_ohlcv_data)), index=shifted_index)
        }
        engine = OscillatorMatrixEngine(OscillatorMatrixConfig())
        with pytest.raises(ValueError):
            engine.evaluate(ohlcv=sample_ohlcv_data, components=misaligned_component)

    def test_extreme_clamping_behavior(self) -> None:
        """Verify composite remains bounded in [-100, 100] even with extreme components."""
        index = pd.date_range("2024-01-01", periods=3, freq="1D")
        ohlcv = pd.DataFrame(
            {
                "open": [100.0] * 3,
                "high": [110.0] * 3,
                "low": [90.0] * 3,
                "close": [100.0] * 3,
                "volume": [1000] * 3,
            },
            index=index,
        )
        # Components attempting to exceed limits
        extreme_components = {
            "extreme_high": pd.Series([150.0, 200.0, 100.0], index=index),
            "extreme_low": pd.Series([-150.0, -200.0, -100.0], index=index),
        }
        engine = OscillatorMatrixEngine(OscillatorMatrixConfig())
        result = engine.evaluate(ohlcv=ohlcv, components=extreme_components)

        assert (result.composite <= 100.0).all()
        assert (result.composite >= -100.0).all()

    def test_nan_preservation_in_warmup_period(
        self,
        sample_ohlcv_data: pd.DataFrame,
    ) -> None:
        """Verify leading NaN values from indicator lookbacks are handled cleanly."""
        lookback = 20
        config = OscillatorMatrixConfig(lookback_windows=[lookback])
        engine = OscillatorMatrixEngine(config=config)
        result = engine.evaluate(ohlcv=sample_ohlcv_data)

        # Leading warmup elements should be NaN in composite
        assert result.composite.iloc[: lookback - 1].isna().all()
        # Classification for NaN entries should be NEUTRAL or designated neutral placeholder
        assert (
            result.classification.iloc[: lookback - 1] == BoundaryClassification.NEUTRAL
        ).all()
        # Subsequent elements must be non-NaN
        assert not result.composite.iloc[lookback:].isna().any()