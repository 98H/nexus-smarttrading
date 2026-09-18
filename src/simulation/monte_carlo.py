"""Monte Carlo simulation framework for asset price path generation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = [
    "MonteCarloConfig",
    "MonteCarloSimulator",
    "SimulationResult",
]


@dataclass
class MonteCarloConfig:
    """Configuration parameters for Monte Carlo simulation under Geometric Brownian Motion.

    Attributes:
        initial_price: Asset price at t = 0 (must be > 0).
        drift: Expected rate of return (annualized drift mu).
        volatility: Annualized volatility sigma (must be >= 0).
        time_horizon: Total simulation horizon in years (must be > 0).
        num_paths: Number of simulation paths to generate (must be > 0).
        num_steps: Number of discrete time steps across the horizon (must be > 0).
        seed: Optional random seed for reproducible simulations.
    """

    initial_price: float
    drift: float
    volatility: float
    time_horizon: float
    num_paths: int
    num_steps: int
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.initial_price <= 0.0:
            raise ValueError(f"initial_price must be positive, got {self.initial_price}")
        if self.time_horizon <= 0.0:
            raise ValueError(f"time_horizon must be positive, got {self.time_horizon}")
        if self.num_paths <= 0:
            raise ValueError(f"num_paths must be positive, got {self.num_paths}")
        if self.num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {self.num_steps}")
        if self.volatility < 0.0:
            raise ValueError(f"volatility must be non-negative, got {self.volatility}")


@dataclass
class SimulationResult:
    """Encapsulates the outputs of a completed Monte Carlo simulation.

    Attributes:
        paths: Simulated price matrix with shape (num_paths, num_steps + 1).
        time_grid: Array of time points corresponding to path columns.
    """

    paths: np.ndarray
    time_grid: np.ndarray

    @property
    def terminal_prices(self) -> np.ndarray:
        """Convenience accessor for prices at the terminal time horizon (t = T)."""
        return self.paths[:, -1]


class MonteCarloSimulator:
    """Generates asset price paths using Geometric Brownian Motion (GBM)."""

    def __init__(self, config: MonteCarloConfig) -> None:
        """Initialize simulator with a validated configuration.

        Args:
            config: MonteCarloConfig specifying simulation parameters.
        """
        self.config = config

    def simulate(self) -> SimulationResult:
        """Execute the Monte Carlo simulation.

        Returns:
            SimulationResult containing simulated paths and time grid.
        """
        num_paths = self.config.num_paths
        num_steps = self.config.num_steps
        t_horizon = self.config.time_horizon
        initial_price = self.config.initial_price
        drift = self.config.drift
        volatility = self.config.volatility

        time_grid = np.linspace(0.0, t_horizon, num_steps + 1)
        dt = t_horizon / num_steps

        # Deterministic realization for zero volatility
        if volatility == 0.0:
            analytical_path = initial_price * np.exp(drift * time_grid)
            paths = np.tile(analytical_path, (num_paths, 1))
            return SimulationResult(paths=paths, time_grid=time_grid)

        rng = np.random.default_rng(self.config.seed)
        z = rng.standard_normal(size=(num_paths, num_steps))

        drift_term = (drift - 0.5 * (volatility ** 2)) * dt
        diffusion_term = volatility * math.sqrt(dt) * z
        log_increments = drift_term + diffusion_term

        log_paths = np.empty((num_paths, num_steps + 1), dtype=np.float64)
        log_paths[:, 0] = 0.0
        np.cumsum(log_increments, axis=1, out=log_paths[:, 1:])

        paths = initial_price * np.exp(log_paths)
        paths[:, 0] = initial_price

        return SimulationResult(paths=paths, time_grid=time_grid)