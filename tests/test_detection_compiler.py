import pytest
from src.compiler.detection_compiler import DetectionCompiler
from src.compiler.exceptions import CompilationError


@pytest.fixture
def compiler() -> DetectionCompiler:
    """Fixture providing a fresh instance of DetectionCompiler."""
    return DetectionCompiler()


# ============================================================================
# Acceptance Criterion 1: Threshold Detection Logic
# ============================================================================


class TestThresholdDetectionCompilation:
    """Tests verifying compilation and execution of threshold detection evaluators."""

    @pytest.mark.parametrize(
        ("operator", "target", "values", "expected"),
        [
            (">", 100.0, [90.0, 100.0, 105.5, 99.9], [False, False, True, False]),
            (">=", 100.0, [90.0, 100.0, 105.5, 99.9], [False, True, True, False]),
            ("<", 50.0, [60.0, 50.0, 45.0, 55.0], [False, False, True, False]),
            ("<=", 50.0, [60.0, 50.0, 45.0, 55.0], [False, True, True, False]),
            ("==", 0.0, [-1.0, 0.0, 1.0, 0.0], [False, True, False, True]),
            ("!=", 0.0, [-1.0, 0.0, 1.0, 0.0], [True, False, True, False]),
        ],
    )
    def test_threshold_operators_across_time_steps(
        self,
        compiler: DetectionCompiler,
        operator: str,
        target: float,
        values: list[float],
        expected: list[bool],
    ) -> None:
        spec = {
            "type": "threshold",
            "metric": "cpu_utilization",
            "operator": operator,
            "target": target,
        }
        evaluator = compiler.compile(spec)
        assert callable(evaluator)

        data = {"cpu_utilization": values}
        result = list(evaluator(data))
        assert result == expected

    def test_threshold_with_negative_target(self, compiler: DetectionCompiler) -> None:
        spec = {
            "type": "threshold",
            "metric": "pnl",
            "operator": "<",
            "target": -10.5,
        }
        evaluator = compiler.compile(spec)
        data = {"pnl": [-5.0, -10.5, -11.0, 0.0]}
        result = list(evaluator(data))
        assert result == [False, False, True, False]

    def test_threshold_with_integer_target(self, compiler: DetectionCompiler) -> None:
        spec = {
            "type": "threshold",
            "metric": "error_count",
            "operator": ">=",
            "target": 5,
        }
        evaluator = compiler.compile(spec)
        data = {"error_count": [0, 4, 5, 6, 2]}
        result = list(evaluator(data))
        assert result == [False, False, True, True, False]

    def test_threshold_isolated_to_specified_metric(
        self, compiler: DetectionCompiler
    ) -> None:
        spec = {
            "type": "threshold",
            "metric": "target_metric",
            "operator": ">",
            "target": 10.0,
        }
        evaluator = compiler.compile(spec)
        data = {
            "target_metric": [5.0, 15.0],
            "irrelevant_metric": [999.0, 999.0],
        }
        result = list(evaluator(data))
        assert result == [False, True]

    def test_threshold_empty_series(self, compiler: DetectionCompiler) -> None:
        spec = {
            "type": "threshold",
            "metric": "empty_metric",
            "operator": ">",
            "target": 10.0,
        }
        evaluator = compiler.compile(spec)
        result = list(evaluator({"empty_metric": []}))
        assert result == []

    def test_threshold_evaluator_is_reusable(
        self, compiler: DetectionCompiler
    ) -> None:
        spec = {
            "type": "threshold",
            "metric": "reusable_metric",
            "operator": ">=",
            "target": 50.0,
        }
        evaluator = compiler.compile(spec)
        run_1 = list(evaluator({"reusable_metric": [40.0, 60.0]}))
        run_2 = list(evaluator({"reusable_metric": [70.0, 30.0]}))
        assert run_1 == [False, True]
        assert run_2 == [True, False]


# ============================================================================
# Acceptance Criterion 2: Cross-Over Detection Logic
# ============================================================================


class TestCrossOverDetectionCompilation:
    """Tests verifying cross-over evaluator compilation and exact-index triggers."""

    def test_crosses_above_exact_index_trigger(
        self, compiler: DetectionCompiler
    ) -> None:
        spec = {
            "type": "crossover",
            "fast_indicator": "fast_ma",
            "slow_indicator": "slow_ma",
            "operator": "crosses_above",
        }
        evaluator = compiler.compile(spec)
        assert callable(evaluator)

        # Index 0: fast below slow -> cannot cross on first step
        # Index 1: fast remains below slow -> False
        # Index 2: fast crosses strictly above slow -> True (exact index)
        # Index 3: fast remains above slow -> False (already crossed)
        # Index 4: fast crosses back below -> False
        # Index 5: fast crosses above slow again -> True
        data = {
            "fast_ma": [10.0, 15.0, 25.0, 30.0, 18.0, 22.0],
            "slow_ma": [20.0, 20.0, 20.0, 20.0, 20.0, 20.0],
        }
        result = list(evaluator(data))
        assert result == [False, False, True, False, False, True]

    def test_crosses_below_exact_index_trigger(
        self, compiler: DetectionCompiler
    ) -> None:
        spec = {
            "type": "crossover",
            "fast_indicator": "fast_ma",
            "slow_indicator": "slow_ma",
            "operator": "crosses_below",
        }
        evaluator = compiler.compile(spec)

        # Index 0: fast above slow -> False
        # Index 1: fast remains above slow -> False
        # Index 2: fast crosses below slow -> True (exact index)
        # Index 3: fast remains below slow -> False (already crossed)
        data = {
            "fast_ma": [30.0, 25.0, 15.0, 10.0],
            "slow_ma": [20.0, 20.0, 20.0, 20.0],
        }
        result = list(evaluator(data))
        assert result == [False, False, True, False]

    def test_crosses_above_transition_from_equal_to_above(
        self, compiler: DetectionCompiler
    ) -> None:
        spec = {
            "type": "crossover",
            "fast_indicator": "fast",
            "slow_indicator": "slow",
            "operator": "crosses_above",
        }
        evaluator = compiler.compile(spec)

        # Index 0: fast < slow (10 < 15)
        # Index 1: fast == slow (15 == 15, touched, not above) -> False
        # Index 2: fast > slow (20 > 15, broke above) -> True
        data = {
            "fast": [10.0, 15.0, 20.0],
            "slow": [15.0, 15.0, 15.0],
        }
        result = list(evaluator(data))
        assert result == [False, False, True]

    def test_crosses_below_transition_from_equal_to_below(
        self, compiler: DetectionCompiler
    ) -> None:
        spec = {
            "type": "crossover",
            "fast_indicator": "fast",
            "slow_indicator": "slow",
            "operator": "crosses_below",
        }
        evaluator = compiler.compile(spec)

        # Index 0: fast > slow (20 > 15)
        # Index 1: fast == slow (15 == 15, touched, not below) -> False
        # Index 2: fast < slow (10 < 15, broke below) -> True
        data = {
            "fast": [20.0, 15.0, 10.0],
            "slow": [15.0, 15.0, 15.0],
        }
        result = list(evaluator(data))
        assert result == [False, False, True]

    def test_no_crossover_when_touching_and_bouncing_away(
        self, compiler: DetectionCompiler
    ) -> None:
        spec = {
            "type": "crossover",
            "fast_indicator": "fast",
            "slow_indicator": "slow",
            "operator": "crosses_above",
        }
        evaluator = compiler.compile(spec)

        # Fast approaches slow, touches it, but falls back below
        data = {
            "fast": [10.0, 20.0, 10.0],
            "slow": [20.0, 20.0, 20.0],
        }
        result = list(evaluator(data))
        assert result == [False, False, False]

    def test_first_index_never_triggers_crossover(
        self, compiler: DetectionCompiler
    ) -> None:
        spec = {
            "type": "crossover",
            "fast_indicator": "fast",
            "slow_indicator": "slow",
            "operator": "crosses_above",
        }
        evaluator = compiler.compile(spec)

        # Even if fast > slow at index 0, there is no previous step to cross over
        data = {
            "fast": [100.0],
            "slow": [50.0],
        }
        result = list(evaluator(data))
        assert result == [False]

    def test_crossover_empty_series(self, compiler: DetectionCompiler) -> None:
        spec = {
            "type": "crossover",
            "fast_indicator": "fast",
            "slow_indicator": "slow",
            "operator": "crosses_above",
        }
        evaluator = compiler.compile(spec)
        result = list(evaluator({"fast": [], "slow": []}))
        assert result == []


# ============================================================================
# Acceptance Criterion 3: Compilation Validation and CompilationError
# ============================================================================


class TestCompilationErrorValidation:
    """Tests verifying CompilationError is raised on invalid specifications."""

    @pytest.mark.parametrize("non_dict_spec", [None, "invalid_spec", 123, [], True])
    def test_non_dict_spec_raises_compilation_error(
        self, compiler: DetectionCompiler, non_dict_spec: object
    ) -> None:
        with pytest.raises(CompilationError):
            compiler.compile(non_dict_spec)  # type: ignore[arg-type]

    def test_empty_dict_spec_raises_compilation_error(
        self, compiler: DetectionCompiler
    ) -> None:
        with pytest.raises(CompilationError):
            compiler.compile({})

    def test_unsupported_condition_type_raises_compilation_error(
        self, compiler: DetectionCompiler
    ) -> None:
        spec = {
            "type": "unsupported_detection_type",
            "metric": "latency",
            "operator": ">",
            "target": 100.0,
        }
        with pytest.raises(CompilationError):
            compiler.compile(spec)

    # ------------------ Threshold Validation Failures ------------------

    @pytest.mark.parametrize(
        "missing_field_spec",
        [
            # Missing metric
            {"type": "threshold", "operator": ">", "target": 100.0},
            # Missing operator
            {"type": "threshold", "metric": "cpu", "target": 100.0},
            # Missing target
            {"type": "threshold", "metric": "cpu", "operator": ">"},
            # Empty metric name
            {"type": "threshold", "metric": "", "operator": ">", "target": 100.0},
        ],
    )
    def test_threshold_missing_fields_raises_compilation_error(
        self, compiler: DetectionCompiler, missing_field_spec: dict
    ) -> None:
        with pytest.raises(CompilationError):
            compiler.compile(missing_field_spec)

    @pytest.mark.parametrize(
        "invalid_operator",
        ["crosses_above", "crosses_below", "contains", "in", "~=", "===", "LIKE"],
    )
    def test_threshold_unsupported_operator_raises_compilation_error(
        self, compiler: DetectionCompiler, invalid_operator: str
    ) -> None:
        spec = {
            "type": "threshold",
            "metric": "cpu",
            "operator": invalid_operator,
            "target": 50.0,
        }
        with pytest.raises(CompilationError):
            compiler.compile(spec)

    @pytest.mark.parametrize(
        "invalid_target",
        ["fifty", [50.0], {"val": 50}, None, (50,)],
    )
    def test_threshold_non_scalar_target_raises_compilation_error(
        self, compiler: DetectionCompiler, invalid_target: object
    ) -> None:
        spec = {
            "type": "threshold",
            "metric": "cpu",
            "operator": ">",
            "target": invalid_target,
        }
        with pytest.raises(CompilationError):
            compiler.compile(spec)

    # ------------------ Cross-Over Validation Failures -----------------

    @pytest.mark.parametrize(
        "missing_field_spec",
        [
            # Missing fast_indicator
            {
                "type": "crossover",
                "slow_indicator": "slow",
                "operator": "crosses_above",
            },
            # Missing slow_indicator
            {
                "type": "crossover",
                "fast_indicator": "fast",
                "operator": "crosses_above",
            },
            # Missing operator
            {
                "type": "crossover",
                "fast_indicator": "fast",
                "slow_indicator": "slow",
            },
            # Empty indicator names
            {
                "type": "crossover",
                "fast_indicator": "",
                "slow_indicator": "slow",
                "operator": "crosses_above",
            },
            {
                "type": "crossover",
                "fast_indicator": "fast",
                "slow_indicator": "",
                "operator": "crosses_above",
            },
        ],
    )
    def test_crossover_missing_fields_raises_compilation_error(
        self, compiler: DetectionCompiler, missing_field_spec: dict
    ) -> None:
        with pytest.raises(CompilationError):
            compiler.compile(missing_field_spec)

    @pytest.mark.parametrize(
        "unsupported_crossover_operator",
        [">", "<", ">=", "<=", "==", "!=", "crosses", "touches", "above", "below"],
    )
    def test_crossover_unsupported_operator_raises_compilation_error(
        self, compiler: DetectionCompiler, unsupported_crossover_operator: str
    ) -> None:
        spec = {
            "type": "crossover",
            "fast_indicator": "fast",
            "slow_indicator": "slow",
            "operator": unsupported_crossover_operator,
        }
        with pytest.raises(CompilationError):
            compiler.compile(spec)

    def test_crossover_identical_indicators_raises_compilation_error(
        self, compiler: DetectionCompiler
    ) -> None:
        spec = {
            "type": "crossover",
            "fast_indicator": "same_metric",
            "slow_indicator": "same_metric",
            "operator": "crosses_above",
        }
        with pytest.raises(CompilationError):
            compiler.compile(spec)


# ============================================================================
# Evaluator Runtime Robustness Tests
# ============================================================================


class TestEvaluatorRuntimeExecution:
    """Tests runtime behavior and errors when executing compiled evaluators."""

    def test_threshold_evaluator_missing_metric_in_data_raises_key_error(
        self, compiler: DetectionCompiler
    ) -> None:
        spec = {
            "type": "threshold",
            "metric": "expected_metric",
            "operator": ">",
            "target": 10.0,
        }
        evaluator = compiler.compile(spec)
        with pytest.raises(KeyError):
            evaluator({"unexpected_metric": [1.0, 2.0]})

    def test_crossover_evaluator_missing_indicator_raises_key_error(
        self, compiler: DetectionCompiler
    ) -> None:
        spec = {
            "type": "crossover",
            "fast_indicator": "fast",
            "slow_indicator": "slow",
            "operator": "crosses_above",
        }
        evaluator = compiler.compile(spec)
        with pytest.raises(KeyError):
            evaluator({"fast": [10.0, 20.0]})  # slow indicator omitted

    def test_crossover_mismatched_series_lengths_raises_value_error(
        self, compiler: DetectionCompiler
    ) -> None:
        spec = {
            "type": "crossover",
            "fast_indicator": "fast",
            "slow_indicator": "slow",
            "operator": "crosses_above",
        }
        evaluator = compiler.compile(spec)
        data = {
            "fast": [10.0, 20.0, 30.0],
            "slow": [15.0, 15.0],
        }
        with pytest.raises(ValueError):
            evaluator(data)