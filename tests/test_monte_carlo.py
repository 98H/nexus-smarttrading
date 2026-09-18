"""
Unit tests for the Monte Carlo Simulation Framework.
Story 6.3.3: Implement Monte Carlo Simulation Framework

Acceptance Criteria:
- Given a MonteCarloConfig with valid initial price, drift, volatility, time horizon, and path count
- Given invalid configuration parameters (such as non-positive steps, paths, or horizon)
- Given completed simulation paths
"""

import math
from typing import Generator
import numpy as np
import pytest

from src.simulation import MonteCarloConfig, MonteCarloSimulator, SimulationResult
import src.simulation.monte_carlo as mc_module


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def default_valid_config() -> MonteCarloConfig:
    """Returns a standard, valid MonteCarloConfig for general testing."""
    return MonteCarloConfig(
        initial_price=100.0,
        drift=0.05,
        volatility=0.20,
        time_horizon=1.0,
        num_paths=1000,
        num_steps=252,
        seed=42,
    )


# =====================================================================
# Module & Export Verification
# =====================================================================

class TestModuleExports:
    """Verifies module structure and expected exports from src.simulation."""

    def test_top_level_exports(self) -> None:
        """Verify that core simulation symbols are exported at package level."""
        import src.simulation as sim_pkg

        assert hasattr(sim_pkg, "MonteCarloConfig")
        assert hasattr(sim_pkg, "MonteCarloSimulator")
        assert hasattr(sim_pkg, "SimulationResult")
        assert sim_pkg.MonteCarloConfig is MonteCarloConfig
        assert sim_pkg.MonteCarloSimulator is MonteCarloSimulator
        assert sim_pkg.SimulationResult is SimulationResult

    def test_module_all_defined(self) -> None:
        """Verify __all__ is properly defined in monte_carlo module."""
        assert hasattr(mc_module, "__all__")
        assert "MonteCarloConfig" in mc_module.__all__
        assert "MonteCarloSimulator" in mc_module.__all__
        assert "SimulationResult" in mc_module.__all__


# =====================================================================
# Configuration Validation Tests (AC: Valid and Invalid Configs)
# =====================================================================

class TestMonteCarloConfigValidation:
    """Tests input validation and construction of MonteCarloConfig."""

    def test_valid_configuration_creation(self, default_valid_config: MonteCarloConfig) -> None:
        """Verifies that a config with all valid parameters is properly constructed."""
        config = default_valid_config
        assert config.initial_price == 100.0
        assert config.drift == 0.05
        assert config.volatility == 0.20
        assert config.time_horizon == 1.0
        assert config.num_paths == 1000
        assert config.num_steps == 252
        assert config.seed == 42

    def test_valid_configuration_with_zero_drift_and_zero_volatility(self) -> None:
        """Verifies that zero drift and zero volatility are allowed (e.g. riskless asset / deterministic)."""
        config = MonteCarloConfig(
            initial_price=50.0,
            drift=0.0,
            volatility=0.0,
            time_horizon=0.5,
            num_paths=10,
            num_steps=10,
            seed=None,
        )
        assert config.drift == 0.0
        assert config.volatility == 0.0

    def test_valid_configuration_with_negative_drift(self) -> None:
        """Verifies that negative drift is permissible (downward trending asset)."""
        config = MonteCarloConfig(
            initial_price=100.0,
            drift=-0.15,
            volatility=0.25,
            time_horizon=2.0,
            num_paths=500,
            num_steps=100,
        )
        assert config.drift == -0.15

    @pytest.mark.parametrize("invalid_price", [0.0, -0.01, -100.0])
    def test_invalid_initial_price_raises_value_error(self, invalid_price: float) -> None:
        """Non-positive initial price must raise ValueError."""
        with pytest.raises(ValueError):
            MonteCarloConfig(
                initial_price=invalid_price,
                drift=0.05,
                volatility=0.2,
                time_horizon=1.0,
                num_paths=100,
                num_steps=10,
            )

    @pytest.mark.parametrize("invalid_horizon", [0.0, -0.5, -10.0])
    def test_invalid_time_horizon_raises_value_error(self, invalid_horizon: float) -> None:
        """Non-positive time horizon must raise ValueError."""
        with pytest.raises(ValueError):
            MonteCarloConfig(
                initial_price=100.0,
                drift=0.05,
                volatility=0.2,
                time_horizon=invalid_horizon,
                num_paths=100,
                num_steps=10,
            )

    @pytest.mark.parametrize("invalid_paths", [0, -1, -100])
    def test_invalid_num_paths_raises_value_error(self, invalid_paths: int) -> None:
        """Non-positive number of paths must raise ValueError."""
        with pytest.raises(ValueError):
            MonteCarloConfig(
                initial_price=100.0,
                drift=0.05,
                volatility=0.2,
                time_horizon=1.0,
                num_paths=invalid_paths,
                num_steps=10,
            )

    @pytest.mark.parametrize("invalid_steps", [0, -1, -50])
    def test_invalid_num_steps_raises_value_error(self, invalid_steps: int) -> None:
        """Non-positive number of steps must raise ValueError."""
        with pytest.raises(ValueError):
            MonteCarloConfig(
                initial_price=100.0,
                drift=0.05,
                volatility=0.2,
                time_horizon=1.0,
                num_paths=100,
                num_steps=invalid_steps,
            )

    @pytest.mark.parametrize("invalid_volatility", [-0.0001, -0.2, -1.0])
    def test_invalid_volatility_raises_value_error(self, invalid_volatility: float) -> None:
        """Negative volatility must raise ValueError."""
        with pytest.raises(ValueError):
            MonteCarloConfig(
                initial_price=100.0,
                drift=0.05,
                volatility=invalid_volatility,
                time_horizon=1.0,
                num_paths=100,
                num_steps=10,
            )


# =====================================================================
# Simulation Path Execution & Structure Tests (AC: Completed Paths)
# =====================================================================

class TestMonteCarloSimulationExecution:
    """Tests execution and structural properties of generated simulation paths."""

    def test_simulation_result_type_and_dimensions(self, default_valid_config: MonteCarloConfig) -> None:
        """Verify simulator produces SimulationResult with correct array shapes."""
        simulator = MonteCarloSimulator(default_valid_config)
        result = simulator.simulate()

        assert isinstance(result, SimulationResult)
        assert isinstance(result.paths, np.ndarray)
        # Dimensions must be (num_paths, num_steps + 1) to include t = 0
        expected_shape = (default_valid_config.num_paths, default_valid_config.num_steps + 1)
        assert result.paths.shape == expected_shape

    def test_time_grid_generation(self, default_valid_config: MonteCarloConfig) -> None:
        """Verify time grid starts at 0, ends at time_horizon, with uniform step dt."""
        simulator = MonteCarloSimulator(default_valid_config)
        result = simulator.simulate()

        assert isinstance(result.time_grid, np.ndarray)
        assert len(result.time_grid) == default_valid_config.num_steps + 1
        assert math.isclose(result.time_grid[0], 0.0, abs_tol=1e-9)
        assert math.isclose(result.time_grid[-1], default_valid_config.time_horizon, abs_tol=1e-9)

        expected_dt = default_valid_config.time_horizon / default_valid_config.num_steps
        actual_dts = np.diff(result.time_grid)
        np.testing.assert_allclose(actual_dts, expected_dt, rtol=1e-7)

    def test_initial_price_across_all_paths(self, default_valid_config: MonteCarloConfig) -> None:
        """Verify that every generated path begins exactly at initial_price at t = 0."""
        simulator = MonteCarloSimulator(default_valid_config)
        result = simulator.simulate()

        initial_prices = result.paths[:, 0]
        np.testing.assert_allclose(initial_prices, default_valid_config.initial_price)

    def test_paths_contain_no_nans_or_infinities(self, default_valid_config: MonteCarloConfig) -> None:
        """Verify that simulated paths are finite real numbers."""
        simulator = MonteCarloSimulator(default_valid_config)
        result = simulator.simulate()

        assert not np.isnan(result.paths).any()
        assert not np.isinf(result.paths).any()

    def test_strictly_positive_prices_under_gbm(self) -> None:
        """Geometric Brownian Motion prices must remain strictly positive across all steps."""
        config = MonteCarloConfig(
            initial_price=10.0,
            drift=-0.5,
            volatility=0.8,
            time_horizon=2.0,
            num_paths=500,
            num_steps=100,
            seed=123,
        )
        simulator = MonteCarloSimulator(config)
        result = simulator.simulate()

        assert np.all(result.paths > 0.0)

    def test_terminal_prices_property(self, default_valid_config: MonteCarloConfig) -> None:
        """Verify convenience access to terminal prices (at horizon T)."""
        simulator = MonteCarloSimulator(default_valid_config)
        result = simulator.simulate()

        assert hasattr(result, "terminal_prices")
        np.testing.assert_array_equal(result.terminal_prices, result.paths[:, -1])


# =====================================================================
# Determinism & Reproducibility Tests
# =====================================================================

class TestMonteCarloDeterminism:
    """Tests reproducibility with seeds and randomness divergence."""

    def test_seed_reproducibility(self) -> None:
        """Two simulator instances with identical seeds must produce identical paths."""
        config_a = MonteCarloConfig(
            initial_price=100.0,
            drift=0.08,
            volatility=0.25,
            time_horizon=1.0,
            num_paths=200,
            num_steps=50,
            seed=999,
        )
        config_b = MonteCarloConfig(
            initial_price=100.0,
            drift=0.08,
            volatility=0.25,
            time_horizon=1.0,
            num_paths=200,
            num_steps=50,
            seed=999,
        )

        res_a = MonteCarloSimulator(config_a).simulate()
        res_b = MonteCarloSimulator(config_b).simulate()

        np.testing.assert_array_equal(res_a.paths, res_b.paths)

    def test_different_seeds_produce_distinct_paths(self) -> None:
        """Simulators with different seeds must produce distinct realizations."""
        config_a = MonteCarloConfig(
            initial_price=100.0,
            drift=0.08,
            volatility=0.25,
            time_horizon=1.0,
            num_paths=200,
            num_steps=50,
            seed=111,
        )
        config_b = MonteCarloConfig(
            initial_price=100.0,
            drift=0.08,
            volatility=0.25,
            time_horizon=1.0,
            num_paths=200,
            num_steps=50,
            seed=222,
        )

        res_a = MonteCarloSimulator(config_a).simulate()
        res_b = MonteCarloSimulator(config_b).simulate()

        assert not np.array_equal(res_a.paths, res_b.paths)


# =====================================================================
# Mathematical & Statistical Convergence Tests
# =====================================================================

class TestMonteCarloTheoreticalProperties:
    """Validates mathematical correctness and statistical convergence of GBM."""

    def test_zero_volatility_deterministic_path(self) -> None:
        """With zero volatility, all paths must follow S(t) = S_0 * exp(mu * t) exactly."""
        s0 = 100.0
        mu = 0.06
        t_horizon = 2.0
        steps = 100
        config = MonteCarloConfig(
            initial_price=s0,
            drift=mu,
            volatility=0.0,
            time_horizon=t_horizon,
            num_paths=50,
            num_steps=steps,
            seed=42,
        )
        simulator = MonteCarloSimulator(config)
        result = simulator.simulate()

        times = result.time_grid
        analytical_path = s0 * np.exp(mu * times)

        for path_idx in range(config.num_paths):
            np.testing.assert_allclose(result.paths[path_idx], analytical_path, rtol=1e-5)

    def test_terminal_price_mean_convergence(self) -> None:
        """
        By the Law of Large Numbers, the sample mean of terminal prices
        must converge to E[S_T] = S_0 * exp(mu * T) within statistical bounds.
        """
        s0 = 100.0
        mu = 0.05
        sigma = 0.20
        t_horizon = 1.0
        num_paths = 50_000
        config = MonteCarloConfig(
            initial_price=s0,
            drift=mu,
            volatility=sigma,
            time_horizon=t_horizon,
            num_paths=num_paths,
            num_steps=50,
            seed=777,
        )
        simulator = MonteCarloSimulator(config)
        result = simulator.simulate()

        theoretical_mean = s0 * math.exp(mu * t_horizon)
        theoretical_variance = (s0 ** 2) * math.exp(2 * mu * t_horizon) * (math.exp((sigma ** 2) * t_horizon) - 1.0)
        standard_error = math.sqrt(theoretical_variance / num_paths)

        sample_mean = float(np.mean(result.terminal_prices))

        # 4-sigma tolerance corresponds to ~99.99% confidence interval
        assert abs(sample_mean - theoretical_mean) < 4.0 * standard_error

    def test_terminal_log_returns_distribution(self) -> None:
        """
        Under GBM, ln(S_T / S_0) is normally distributed with:
        mean = (mu - 0.5 * sigma^2) * T
        variance = sigma^2 * T
        """
        s0 = 100.0
        mu = 0.08
        sigma = 0.15
        t_horizon = 1.5
        num_paths = 40_000
        config = MonteCarloConfig(
            initial_price=s0,
            drift=mu,
            volatility=sigma,
            time_horizon=t_horizon,
            num_paths=num_paths,
            num_steps=60,
            seed=12345,
        )
        simulator = MonteCarloSimulator(config)
        result = simulator.simulate()

        log_returns = np.log(result.terminal_prices / s0)

        theoretical_mean = (mu - 0.5 * (sigma ** 2)) * t_horizon
        theoretical_var = (sigma ** 2) * t_horizon

        sample_mean = float(np.mean(log_returns))
        sample_var = float(np.var(log_returns, ddof=1))

        se_mean = math.sqrt(theoretical_var / num_paths)
        assert abs(sample_mean - theoretical_mean) < 4.0 * se_mean
        # Variance convergence within 5% relative tolerance
        assert math.isclose(sample_var, theoretical_var, rel_tol=0.05)


# =====================================================================
# Boundary and Edge Case Tests
# =====================================================================

class TestMonteCarloEdgeCases:
    """Tests boundary conditions and extreme but valid parameter values."""

    def test_minimal_steps_and_paths(self) -> None:
        """Simulation with 1 step and 1 path must execute properly without shape corruption."""
        config = MonteCarloConfig(
            initial_price=50.0,
            drift=0.02,
            volatility=0.1,
            time_horizon=0.1,
            num_paths=1,
            num_steps=1,
            seed=1,
        )
        simulator = MonteCarloSimulator(config)
        result = simulator.simulate()

        assert result.paths.shape == (1, 2)
        assert result.paths[0, 0] == 50.0
        assert result.paths[0, 1] > 0.0

    def test_very_short_time_horizon(self) -> None:
        """Very small horizon (e.g. 1 hour / 1e-4 years) should execute with values near initial price."""
        config = MonteCarloConfig(
            initial_price=200.0,
            drift=0.05,
            volatility=0.15,
            time_horizon=1e-4,
            num_paths=100,
            num_steps=5,
            seed=42,
        )
        simulator = MonteCarloSimulator(config)
        result = simulator.simulate()

        # Over such a small interval, paths should barely deviate from initial price
        np.testing.assert_allclose(result.terminal_prices, 200.0, rtol=1e-2)