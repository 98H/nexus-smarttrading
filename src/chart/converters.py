"""Coordinate converters for linear, logarithmic, and percentage scales."""

from abc import ABC, abstractmethod
import math
from typing import Tuple


class CoordinateConverter(ABC):
    """Abstract base class for 1D coordinate transformations."""

    @abstractmethod
    def transform(self, value: float) -> float:
        """Transform a value from domain space to coordinate range space."""

    @abstractmethod
    def inverse_transform(self, coordinate: float) -> float:
        """Reconstruct the original domain value from a coordinate in range space."""


class LinearCoordinateConverter(CoordinateConverter):
    """Linear coordinate converter between input domain and output range."""

    def __init__(
        self,
        domain: Tuple[float, float],
        range: Tuple[float, float],
    ) -> None:
        if domain[0] == domain[1]:
            raise ValueError(f"Domain limits must not be equal: {domain}")
        if range[0] == range[1]:
            raise ValueError(f"Range limits must not be equal: {range}")

        self.domain = (float(domain[0]), float(domain[1]))
        self.range = (float(range[0]), float(range[1]))

    def transform(self, value: float) -> float:
        val = float(value)
        d0, d1 = self.domain
        r0, r1 = self.range
        t = (val - d0) / (d1 - d0)
        return float(r0 + t * (r1 - r0))

    def inverse_transform(self, coordinate: float) -> float:
        coord = float(coordinate)
        d0, d1 = self.domain
        r0, r1 = self.range
        t = (coord - r0) / (r1 - r0)
        return float(d0 + t * (d1 - d0))


class LogarithmicCoordinateConverter(CoordinateConverter):
    """Logarithmic coordinate converter for positive domains."""

    def __init__(
        self,
        domain: Tuple[float, float],
        range: Tuple[float, float],
    ) -> None:
        if domain[0] <= 0 or domain[1] <= 0:
            raise ValueError(
                f"Domain values must be strictly positive (> 0), got: {domain}"
            )
        if domain[0] == domain[1]:
            raise ValueError(f"Domain limits must not be equal: {domain}")
        if range[0] == range[1]:
            raise ValueError(f"Range limits must not be equal: {range}")

        self.domain = (float(domain[0]), float(domain[1]))
        self.range = (float(range[0]), float(range[1]))
        self._log_d0 = math.log(self.domain[0])
        self._log_d1 = math.log(self.domain[1])
        self._log_span = self._log_d1 - self._log_d0

    def transform(self, value: float) -> float:
        if value <= 0:
            raise ValueError(
                f"Input value must be strictly positive (> 0), got: {value}"
            )
        val = float(value)
        t = (math.log(val) - self._log_d0) / self._log_span
        r0, r1 = self.range
        return float(r0 + t * (r1 - r0))

    def inverse_transform(self, coordinate: float) -> float:
        coord = float(coordinate)
        r0, r1 = self.range
        t = (coord - r0) / (r1 - r0)
        log_val = self._log_d0 + t * self._log_span
        return float(math.exp(log_val))


class PercentageCoordinateConverter(CoordinateConverter):
    """Percentage coordinate converter relative to a reference base value."""

    def __init__(
        self,
        base_value: float,
        domain: Tuple[float, float],
        range: Tuple[float, float],
    ) -> None:
        if base_value <= 0:
            raise ValueError(
                f"Base value must be strictly positive (> 0), got: {base_value}"
            )
        if domain[0] == domain[1]:
            raise ValueError(f"Domain limits must not be equal: {domain}")
        if range[0] == range[1]:
            raise ValueError(f"Range limits must not be equal: {range}")

        self.base_value = float(base_value)
        self.domain = (float(domain[0]), float(domain[1]))
        self.range = (float(range[0]), float(range[1]))

    def transform(self, value: float) -> float:
        val = float(value)
        pct = ((val - self.base_value) / self.base_value) * 100.0
        d0, d1 = self.domain
        r0, r1 = self.range
        t = (pct - d0) / (d1 - d0)
        return float(r0 + t * (r1 - r0))

    def inverse_transform(self, coordinate: float) -> float:
        coord = float(coordinate)
        d0, d1 = self.domain
        r0, r1 = self.range
        t = (coord - r0) / (r1 - r0)
        pct = d0 + t * (d1 - d0)
        return float(self.base_value * (1.0 + pct / 100.0))