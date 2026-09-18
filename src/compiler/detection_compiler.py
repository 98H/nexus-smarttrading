"""Detection logic compiler module."""

from __future__ import annotations

import operator
from typing import Any, Callable, Mapping, Sequence

from src.compiler.exceptions import CompilationError

THRESHOLD_OPERATORS: dict[str, Callable[[Any, Any], bool]] = {
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
}

CROSSOVER_OPERATORS: set[str] = {"crosses_above", "crosses_below"}

Evaluator = Callable[[Mapping[str, Sequence[Any]]], list[bool]]


class DetectionCompiler:
    """Compiles condition specifications into executable detection evaluators."""

    def compile(self, spec: Any) -> Evaluator:
        """Compile a detection specification into an executable evaluator.

        Args:
            spec: Dictionary specification containing detection parameters.

        Returns:
            An evaluator function that processes data series and returns boolean results.

        Raises:
            CompilationError: If the specification is malformed or unsupported.
        """
        if not isinstance(spec, dict) or not spec:
            raise CompilationError("Specification must be a non-empty dictionary.")

        spec_type = spec.get("type")
        if spec_type == "threshold":
            return self._compile_threshold(spec)
        if spec_type == "crossover":
            return self._compile_crossover(spec)

        raise CompilationError(f"Unsupported detection type: {spec_type!r}")

    def _compile_threshold(self, spec: dict[str, Any]) -> Evaluator:
        """Compile a threshold detection specification."""
        metric = spec.get("metric")
        if not isinstance(metric, str) or not metric:
            raise CompilationError("Threshold specification requires a non-empty string 'metric'.")

        op_name = spec.get("operator")
        if op_name not in THRESHOLD_OPERATORS:
            raise CompilationError(f"Unsupported threshold operator: {op_name!r}")
        op_func = THRESHOLD_OPERATORS[op_name]

        if "target" not in spec:
            raise CompilationError("Threshold specification requires a 'target' field.")

        target = spec["target"]
        if isinstance(target, bool) or not isinstance(target, (int, float)):
            raise CompilationError(f"Threshold target must be a numeric scalar, got: {type(target).__name__}")

        def threshold_evaluator(data: Mapping[str, Sequence[Any]]) -> list[bool]:
            if metric not in data:
                raise KeyError(f"Metric '{metric}' not found in data.")
            return [bool(op_func(val, target)) for val in data[metric]]

        return threshold_evaluator

    def _compile_crossover(self, spec: dict[str, Any]) -> Evaluator:
        """Compile a cross-over detection specification."""
        fast_indicator = spec.get("fast_indicator")
        if not isinstance(fast_indicator, str) or not fast_indicator:
            raise CompilationError("Cross-over specification requires a non-empty string 'fast_indicator'.")

        slow_indicator = spec.get("slow_indicator")
        if not isinstance(slow_indicator, str) or not slow_indicator:
            raise CompilationError("Cross-over specification requires a non-empty string 'slow_indicator'.")

        if fast_indicator == slow_indicator:
            raise CompilationError("Fast and slow indicators must be distinct.")

        op_name = spec.get("operator")
        if op_name not in CROSSOVER_OPERATORS:
            raise CompilationError(f"Unsupported cross-over operator: {op_name!r}")

        def crossover_evaluator(data: Mapping[str, Sequence[Any]]) -> list[bool]:
            if fast_indicator not in data:
                raise KeyError(f"Indicator '{fast_indicator}' not found in data.")
            if slow_indicator not in data:
                raise KeyError(f"Indicator '{slow_indicator}' not found in data.")

            fast_series = data[fast_indicator]
            slow_series = data[slow_indicator]

            if len(fast_series) != len(slow_series):
                raise ValueError(
                    f"Series length mismatch: '{fast_indicator}' length is {len(fast_series)}, "
                    f"'{slow_indicator}' length is {len(slow_series)}."
                )

            length = len(fast_series)
            if length == 0:
                return []

            results = [False] * length
            if op_name == "crosses_above":
                for i in range(1, length):
                    if fast_series[i] > slow_series[i] and fast_series[i - 1] <= slow_series[i - 1]:
                        results[i] = True
            elif op_name == "crosses_below":
                for i in range(1, length):
                    if fast_series[i] < slow_series[i] and fast_series[i - 1] >= slow_series[i - 1]:
                        results[i] = True

            return results

        return crossover_evaluator