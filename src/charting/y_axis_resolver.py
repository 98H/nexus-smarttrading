"""Dynamic Auto-Scaling (Fit-to-Screen) Y-Axis Resolver."""

from __future__ import annotations

from decimal import Decimal
import math
from typing import Iterable, NamedTuple, Union

Numeric = Union[int, float, Decimal]


class YAxisRange(NamedTuple):
    """Immutable representation of a resolved Y-axis display range."""

    min: float
    max: float

    @property
    def span(self) -> float:
        """Calculate the total span between max and min."""
        return self.max - self.min


class YAxisResolver:
    """Calculates auto-scaled fit-to-screen Y-axis bounds with margin padding."""

    def __init__(self, margin_padding_percentage: float = 0.05) -> None:
        """Initialize the resolver with a default margin padding percentage.

        Args:
            margin_padding_percentage: Proportional padding applied to data span (>= 0.0).

        Raises:
            TypeError: If margin_padding_percentage is not numeric.
            ValueError: If margin_padding_percentage is negative or non-finite.
        """
        self._margin_padding_percentage = self._validate_padding(
            margin_padding_percentage
        )

    @property
    def margin_padding_percentage(self) -> float:
        """Return the default margin padding percentage."""
        return self._margin_padding_percentage

    @staticmethod
    def _validate_padding(padding: Numeric) -> float:
        """Validate margin padding percentage parameter."""
        if isinstance(padding, bool) or not isinstance(padding, (int, float, Decimal)):
            raise TypeError("margin_padding_percentage must be a numeric value")

        padding_val = float(padding)
        if padding_val < 0.0 or math.isnan(padding_val) or math.isinf(padding_val):
            raise ValueError(
                f"margin_padding_percentage must be a non-negative finite number, got {padding!r}"
            )
        return padding_val

    def resolve(
        self,
        data: Iterable[Numeric],
        margin_padding_percentage: float | None = None,
    ) -> YAxisRange:
        """Resolve fit-to-screen Y-axis bounds for a sequence of numeric data points.

        Args:
            data: Sequence or iterable of numeric data points.
            margin_padding_percentage: Optional padding override for this calculation.

        Returns:
            YAxisRange containing padded min, max, and span values.

        Raises:
            TypeError: If data contains non-numeric elements.
            ValueError: If data is empty, contains NaN/inf, or padding is invalid.
        """
        padding = (
            self._validate_padding(margin_padding_percentage)
            if margin_padding_percentage is not None
            else self._margin_padding_percentage
        )

        min_val = math.inf
        max_val = -math.inf
        count = 0

        for item in data:
            if isinstance(item, bool) or not isinstance(
                item, (int, float, Decimal)
            ):
                raise TypeError(f"Invalid numeric data point: {item!r}")

            try:
                val = float(item)
            except (ValueError, TypeError) as err:
                raise ValueError(f"Cannot convert {item!r} to float") from err

            if math.isnan(val):
                raise ValueError("Data sequence cannot contain NaN values")
            if math.isinf(val):
                raise ValueError("Data sequence cannot contain infinite values")

            if val < min_val:
                min_val = val
            if val > max_val:
                max_val = val
            count += 1

        if count == 0:
            raise ValueError("Data sequence cannot be empty")

        return self._calculate_bounds(min_val, max_val, padding)

    @staticmethod
    def _calculate_bounds(
        min_val: float, max_val: float, padding: float
    ) -> YAxisRange:
        """Calculate padded bounds handling both variance and zero-variance series."""
        if min_val == max_val:
            constant_val = min_val
            margin = abs(constant_val) * padding
            if margin == 0.0:
                margin = (
                    abs(constant_val) * 0.05 if constant_val != 0.0 else 1.0
                )
                if margin == 0.0:
                    margin = 1.0
            return YAxisRange(
                min=constant_val - margin,
                max=constant_val + margin,
            )

        span = max_val - min_val
        margin = span * padding
        return YAxisRange(min=min_val - margin, max=max_val + margin)