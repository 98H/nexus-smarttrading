"""
Unit tests for Scoped Execution Pipeline (var, varip, and standard globals).

Requirement: Story 3.2.2 - Implement Scoped Execution Pipeline for var, varip,
and Global Re-evaluations.
"""

from typing import Callable, Optional
import pytest

from src.engine.scope import Scope, VarModifier
from src.engine.pipeline import ExecutionPipeline, Bar, Tick


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def empty_scope() -> Scope:
    """Provides a fresh Scope instance."""
    return Scope()


@pytest.fixture
def sample_bars() -> list[Bar]:
    """Provides a deterministic sequence of historical OHLCV bars."""
    return [
        Bar(index=0, open=100.0, high=105.0, low=95.0, close=102.0, volume=1000.0),
        Bar(index=1, open=102.0, high=110.0, low=101.0, close=108.0, volume=1500.0),
        Bar(index=2, open=108.0, high=112.0, low=104.0, close=106.0, volume=1200.0),
    ]


# ============================================================================
# Scope Unit Tests (src/engine/scope.py)
# ============================================================================

class TestScopeDeclarations:
    """Tests variable declaration, type enforcement, and access rules in Scope."""

    def test_declare_and_get_standard_variable(self, empty_scope: Scope) -> None:
        empty_scope.declare("std_var", 42, VarModifier.STANDARD)
        assert empty_scope.get("std_var") == 42
        assert empty_scope.has("std_var") is True

    def test_declare_and_get_var_variable(self, empty_scope: Scope) -> None:
        empty_scope.declare("var_var", 100, VarModifier.VAR)
        assert empty_scope.get("var_var") == 100
        assert empty_scope.has("var_var") is True

    def test_declare_and_get_varip_variable(self, empty_scope: Scope) -> None:
        empty_scope.declare("varip_var", 999, VarModifier.VARIP)
        assert empty_scope.get("varip_var") == 999
        assert empty_scope.has("varip_var") is True

    def test_declare_accepts_string_modifier(self, empty_scope: Scope) -> None:
        empty_scope.declare("str_mod_var", 1, "var")
        assert empty_scope.get("str_mod_var") == 1

    def test_declare_duplicate_variable_raises_error(self, empty_scope: Scope) -> None:
        empty_scope.declare("x", 10, VarModifier.STANDARD)
        with pytest.raises(ValueError):
            empty_scope.declare("x", 20, VarModifier.STANDARD)

    def test_declare_invalid_modifier_raises_error(self, empty_scope: Scope) -> None:
        with pytest.raises(ValueError):
            empty_scope.declare("bad_var", 10, "invalid_modifier")  # type: ignore[arg-type]

    def test_get_undeclared_variable_raises_error(self, empty_scope: Scope) -> None:
        with pytest.raises(KeyError):
            empty_scope.get("non_existent")

    def test_set_undeclared_variable_raises_error(self, empty_scope: Scope) -> None:
        with pytest.raises(KeyError):
            empty_scope.set("non_existent", 123)

    def test_set_updates_variable_value(self, empty_scope: Scope) -> None:
        empty_scope.declare("val", 10, VarModifier.STANDARD)
        empty_scope.set("val", 20)
        assert empty_scope.get("val") == 20


class TestScopeLifecycleMechanics:
    """Tests Scope bar and tick lifecycle operations for standard, var, and varip."""

    def test_standard_variable_resets_to_initial_value_on_new_bar(self, empty_scope: Scope) -> None:
        empty_scope.declare("x", 10, VarModifier.STANDARD)
        empty_scope.on_bar_start(bar_index=0)
        empty_scope.set("x", 50)
        assert empty_scope.get("x") == 50

        empty_scope.on_bar_close()
        empty_scope.on_bar_start(bar_index=1)
        # Standard variables re-evaluate/reset on new bar
        assert empty_scope.get("x") == 10

    def test_standard_variable_reevaluates_initializer_callable_on_new_bar(self, empty_scope: Scope) -> None:
        counter = 0

        def dynamic_initializer() -> int:
            nonlocal counter
            counter += 10
            return counter

        empty_scope.declare("dyn_var", 10, VarModifier.STANDARD, initializer=dynamic_initializer)
        empty_scope.on_bar_start(bar_index=0)
        assert empty_scope.get("dyn_var") == 10

        empty_scope.set("dyn_var", 999)
        empty_scope.on_bar_close()

        empty_scope.on_bar_start(bar_index=1)
        assert empty_scope.get("dyn_var") == 20

    def test_var_persists_across_bars_when_committed(self, empty_scope: Scope) -> None:
        empty_scope.declare("v", 0, VarModifier.VAR)

        # Bar 0
        empty_scope.on_bar_start(bar_index=0)
        empty_scope.on_tick_start()
        empty_scope.set("v", 100)
        empty_scope.on_tick_end(is_confirmed=True)
        empty_scope.on_bar_close()

        # Bar 1
        empty_scope.on_bar_start(bar_index=1)
        assert empty_scope.get("v") == 100

        empty_scope.on_tick_start()
        empty_scope.set("v", 250)
        empty_scope.on_tick_end(is_confirmed=True)
        empty_scope.on_bar_close()

        # Bar 2
        empty_scope.on_bar_start(bar_index=2)
        assert empty_scope.get("v") == 250

    def test_var_rolls_back_on_unconfirmed_intrabar_ticks(self, empty_scope: Scope) -> None:
        empty_scope.declare("v", 10, VarModifier.VAR)

        # Setup committed value at Bar 0 close
        empty_scope.on_bar_start(bar_index=0)
        empty_scope.on_tick_start()
        empty_scope.set("v", 50)
        empty_scope.on_tick_end(is_confirmed=True)
        empty_scope.on_bar_close()

        # Start Bar 1
        empty_scope.on_bar_start(bar_index=1)
        assert empty_scope.get("v") == 50

        # Tick 1 (unconfirmed)
        empty_scope.on_tick_start()
        empty_scope.set("v", 75)
        assert empty_scope.get("v") == 75
        empty_scope.on_tick_end(is_confirmed=False)
        # Should rollback to bar start (committed baseline)
        assert empty_scope.get("v") == 50

        # Tick 2 (unconfirmed)
        empty_scope.on_tick_start()
        empty_scope.set("v", 80)
        assert empty_scope.get("v") == 80
        empty_scope.on_tick_end(is_confirmed=False)
        assert empty_scope.get("v") == 50

        # Tick 3 (confirmed close)
        empty_scope.on_tick_start()
        empty_scope.set("v", 95)
        empty_scope.on_tick_end(is_confirmed=True)
        empty_scope.on_bar_close()
        # Committed value at Bar 1 close
        assert empty_scope.get("v") == 95

        # Bar 2 sees committed value
        empty_scope.on_bar_start(bar_index=2)
        assert empty_scope.get("v") == 95

    def test_varip_persists_across_unconfirmed_ticks_without_rollback(self, empty_scope: Scope) -> None:
        empty_scope.declare("vip", 0, VarModifier.VARIP)

        empty_scope.on_bar_start(bar_index=0)

        # Tick 1 (unconfirmed)
        empty_scope.on_tick_start()
        empty_scope.set("vip", empty_scope.get("vip") + 1)
        assert empty_scope.get("vip") == 1
        empty_scope.on_tick_end(is_confirmed=False)
        # varip must NOT rollback
        assert empty_scope.get("vip") == 1

        # Tick 2 (unconfirmed)
        empty_scope.on_tick_start()
        empty_scope.set("vip", empty_scope.get("vip") + 1)
        assert empty_scope.get("vip") == 2
        empty_scope.on_tick_end(is_confirmed=False)
        assert empty_scope.get("vip") == 2

        # Tick 3 (confirmed)
        empty_scope.on_tick_start()
        empty_scope.set("vip", empty_scope.get("vip") + 1)
        assert empty_scope.get("vip") == 3
        empty_scope.on_tick_end(is_confirmed=True)
        empty_scope.on_bar_close()
        assert empty_scope.get("vip") == 3

        # Bar 1 maintains varip state
        empty_scope.on_bar_start(bar_index=1)
        assert empty_scope.get("vip") == 3

    def test_mixed_scope_lifecycle_isolation(self, empty_scope: Scope) -> None:
        empty_scope.declare("std", 0, VarModifier.STANDARD, initializer=lambda: 0)
        empty_scope.declare("v", 0, VarModifier.VAR)
        empty_scope.declare("vip", 0, VarModifier.VARIP)

        empty_scope.on_bar_start(bar_index=0)

        # Intrabar Tick 1 (unconfirmed)
        empty_scope.on_tick_start()
        empty_scope.set("std", 10)
        empty_scope.set("v", 10)
        empty_scope.set("vip", 10)
        empty_scope.on_tick_end(is_confirmed=False)

        assert empty_scope.get("v") == 0      # Rolled back
        assert empty_scope.get("vip") == 10   # Persisted without rollback

        # Intrabar Tick 2 (confirmed)
        empty_scope.on_tick_start()
        empty_scope.set("std", 20)
        empty_scope.set("v", empty_scope.get("v") + 5)     # 0 + 5 = 5
        empty_scope.set("vip", empty_scope.get("vip") + 5) # 10 + 5 = 15
        empty_scope.on_tick_end(is_confirmed=True)
        empty_scope.on_bar_close()

        assert empty_scope.get("std") == 20
        assert empty_scope.get("v") == 5
        assert empty_scope.get("vip") == 15

        # Bar 1 starts
        empty_scope.on_bar_start(bar_index=1)
        assert empty_scope.get("std") == 0    # Reset by initializer
        assert empty_scope.get("v") == 5      # Retained from Bar 0 close
        assert empty_scope.get("vip") == 15   # Retained


# ============================================================================
# ExecutionPipeline Unit Tests (src/engine/pipeline.py)
# ============================================================================

class TestExecutionPipelineHistorical:
    """Tests pipeline historical bar execution semantics."""

    def test_pipeline_historical_execution_evaluates_all_modifiers(self, sample_bars: list[Bar]) -> None:
        scope = Scope()
        scope.declare("std_mult", 0.0, VarModifier.STANDARD)
        scope.declare("var_counter", 0, VarModifier.VAR)
        scope.declare("varip_counter", 0, VarModifier.VARIP)

        history_snapshots: list[dict[str, float | int]] = []

        def script(s: Scope, bar: Bar, tick: Optional[Tick] = None) -> None:
            # Standard re-evaluates each bar based on bar close
            s.set("std_mult", bar.close * 2)
            # var and varip accumulate across bars
            s.set("var_counter", s.get("var_counter") + 1)
            s.set("varip_counter", s.get("varip_counter") + 1)

            history_snapshots.append({
                "bar_index": bar.index,
                "std_mult": s.get("std_mult"),
                "var_counter": s.get("var_counter"),
                "varip_counter": s.get("varip_counter"),
            })

        pipeline = ExecutionPipeline(scope=scope, script=script)
        pipeline.process_historical(sample_bars)

        assert len(history_snapshots) == 3

        # Bar 0 close: 102.0
        assert history_snapshots[0] == {
            "bar_index": 0,
            "std_mult": 204.0,
            "var_counter": 1,
            "varip_counter": 1,
        }
        # Bar 1 close: 108.0
        assert history_snapshots[1] == {
            "bar_index": 1,
            "std_mult": 216.0,
            "var_counter": 2,
            "varip_counter": 2,
        }
        # Bar 2 close: 106.0
        assert history_snapshots[2] == {
            "bar_index": 2,
            "std_mult": 212.0,
            "var_counter": 3,
            "varip_counter": 3,
        }

    def test_pipeline_empty_historical_bars_raises_error(self, empty_scope: Scope) -> None:
        pipeline = ExecutionPipeline(scope=empty_scope)
        with pytest.raises(ValueError):
            pipeline.process_historical([])

    def test_pipeline_execution_without_script_raises_error(
        self, empty_scope: Scope, sample_bars: list[Bar]
    ) -> None:
        pipeline = ExecutionPipeline(scope=empty_scope)
        with pytest.raises(ValueError):
            pipeline.process_historical(sample_bars)


class TestExecutionPipelineIntrabarTicks:
    """Tests pipeline intrabar tick execution, rollbacks, and bar confirmation."""

    def test_pipeline_processes_intrabar_ticks_with_var_rollback_and_varip_persistence(self) -> None:
        scope = Scope()
        scope.declare("std_val", 0.0, VarModifier.STANDARD, initializer=lambda: 0.0)
        scope.declare("var_count", 0, VarModifier.VAR)
        scope.declare("varip_count", 0, VarModifier.VARIP)

        def script(s: Scope, bar: Bar, tick: Optional[Tick] = None) -> None:
            price = tick.price if tick is not None else bar.close
            s.set("std_val", price)
            s.set("var_count", s.get("var_count") + 1)
            s.set("varip_count", s.get("varip_count") + 1)

        pipeline = ExecutionPipeline(scope=scope, script=script)

        bar = Bar(index=0, open=100.0, high=105.0, low=99.0, close=102.0)
        ticks = [
            Tick(price=100.5, volume=10.0, is_confirmed=False),
            Tick(price=101.0, volume=15.0, is_confirmed=False),
            Tick(price=102.0, volume=20.0, is_confirmed=True),
        ]

        # Tick 1: unconfirmed
        pipeline.process_tick(bar, ticks[0])
        # Script calculated: var_count = 0 + 1 = 1, varip_count = 0 + 1 = 1
        # Tick completed with is_confirmed=False -> var_count rolled back to 0, varip persists at 1
        assert scope.get("var_count") == 0
        assert scope.get("varip_count") == 1

        # Tick 2: unconfirmed
        pipeline.process_tick(bar, ticks[1])
        # Script sees: var_count = 0 -> calculates 1; varip_count = 1 -> calculates 2
        # Tick completed with is_confirmed=False -> var_count rolled back to 0, varip persists at 2
        assert scope.get("var_count") == 0
        assert scope.get("varip_count") == 2

        # Tick 3: confirmed close
        pipeline.process_tick(bar, ticks[2])
        # Script sees: var_count = 0 -> calculates 1; varip_count = 2 -> calculates 3
        # Confirmed -> committed to bar close
        assert scope.get("var_count") == 1
        assert scope.get("varip_count") == 3
        assert scope.get("std_val") == 102.0

    def test_pipeline_multi_bar_intrabar_transitions(self) -> None:
        """Verifies state consistency across multi-bar transitions with multiple ticks per bar."""
        scope = Scope()
        scope.declare("var_acc", 10, VarModifier.VAR)
        scope.declare("varip_acc", 10, VarModifier.VARIP)

        def script(s: Scope, bar: Bar, tick: Optional[Tick] = None) -> None:
            s.set("var_acc", s.get("var_acc") + 1)
            s.set("varip_acc", s.get("varip_acc") + 1)

        pipeline = ExecutionPipeline(scope=scope, script=script)

        # Bar 0 with 2 unconfirmed ticks, then 1 confirmed tick
        bar_0 = Bar(index=0, open=10.0, high=12.0, low=9.0, close=11.0)
        bar_0_ticks = [
            Tick(price=10.2, is_confirmed=False),
            Tick(price=10.8, is_confirmed=False),
            Tick(price=11.0, is_confirmed=True),
        ]
        pipeline.process_bar(bar_0, ticks=bar_0_ticks)

        # In Bar 0:
        # Initial: var=10, varip=10
        # Tick 1: var calculates 11 -> rollbacks to 10. varip calculates 11 -> keeps 11.
        # Tick 2: var calculates 11 -> rollbacks to 10. varip calculates 12 -> keeps 12.
        # Tick 3 (confirmed): var calculates 11 -> committed! varip calculates 13 -> committed!
        assert scope.get("var_acc") == 11
        assert scope.get("varip_acc") == 13

        # Bar 1 with 1 unconfirmed tick, then 1 confirmed tick
        bar_1 = Bar(index=1, open=11.0, high=15.0, low=10.5, close=14.0)
        bar_1_ticks = [
            Tick(price=12.5, is_confirmed=False),
            Tick(price=14.0, is_confirmed=True),
        ]
        pipeline.process_bar(bar_1, ticks=bar_1_ticks)

        # In Bar 1:
        # Starting base: var=11, varip=13
        # Tick 1: var calculates 12 -> rollbacks to 11. varip calculates 14 -> keeps 14.
        # Tick 2 (confirmed): var calculates 12 -> committed! varip calculates 15 -> committed!
        assert scope.get("var_acc") == 12
        assert scope.get("varip_acc") == 15

    def test_pipeline_standard_global_reset_between_bars_with_ticks(self) -> None:
        scope = Scope()
        scope.declare("tick_counter", 0, VarModifier.STANDARD, initializer=lambda: 0)

        def script(s: Scope, bar: Bar, tick: Optional[Tick] = None) -> None:
            s.set("tick_counter", s.get("tick_counter") + 1)

        pipeline = ExecutionPipeline(scope=scope, script=script)

        bar_0 = Bar(index=0, open=100.0, high=101.0, low=99.0, close=100.0)
        pipeline.process_bar(bar_0, ticks=[
            Tick(price=100.2, is_confirmed=False),
            Tick(price=100.0, is_confirmed=True),
        ])
        # Bar 0 closed: final tick confirmation set tick_counter to 1
        assert scope.get("tick_counter") == 1

        # Processing Bar 1 should reset tick_counter to 0 at bar start
        bar_1 = Bar(index=1, open=100.0, high=102.0, low=100.0, close=101.5)
        pipeline.process_bar(bar_1, ticks=[
            Tick(price=101.5, is_confirmed=True),
        ])
        # Bar 1 ran 1 confirmed tick starting from 0 -> final value is 1
        assert scope.get("tick_counter") == 1


class TestExecutionPipelineRobustness:
    """Tests error handling, unexpected aborts, and state isolation in the pipeline."""

    def test_pipeline_script_exception_aborts_without_corrupting_committed_scope(self) -> None:
        scope = Scope()
        scope.declare("v", 100, VarModifier.VAR)
        scope.declare("vip", 100, VarModifier.VARIP)

        # Bar 0 commits successfully
        bar_0 = Bar(index=0, open=50.0, high=55.0, low=49.0, close=52.0)
        pipeline = ExecutionPipeline(
            scope=scope,
            script=lambda s, b, t: [
                s.set("v", s.get("v") + 10),
                s.set("vip", s.get("vip") + 10),
            ]
        )
        pipeline.process_bar(bar_0, ticks=[Tick(price=52.0, is_confirmed=True)])
        assert scope.get("v") == 110
        assert scope.get("vip") == 110

        # Bar 1 script raises an unhandled error
        def failing_script(s: Scope, b: Bar, t: Optional[Tick]) -> None:
            s.set("v", 999)
            s.set("vip", 999)
            raise RuntimeError("Script computation error")

        pipeline.set_script(failing_script)
        bar_1 = Bar(index=1, open=52.0, high=53.0, low=50.0, close=51.0)

        with pytest.raises(RuntimeError):
            pipeline.process_tick(bar_1, Tick(price=51.0, is_confirmed=False))

        # v must still reflect the uncorrupted state before tick failure
        assert scope.get("v") == 110

    def test_pipeline_sets_script_via_setter_method(self, empty_scope: Scope) -> None:
        pipeline = ExecutionPipeline(scope=empty_scope)
        called = False

        def custom_script(s: Scope, b: Bar, t: Optional[Tick] = None) -> None:
            nonlocal called
            called = True

        pipeline.set_script(custom_script)
        bar = Bar(index=0, open=1.0, high=2.0, low=0.5, close=1.5)
        pipeline.process_bar(bar)
        assert called is True

    def test_independent_scopes_do_not_leak_state(self) -> None:
        scope_a = Scope()
        scope_b = Scope()

        scope_a.declare("val", 10, VarModifier.VAR)
        scope_b.declare("val", 20, VarModifier.VAR)

        pipeline_a = ExecutionPipeline(scope=scope_a, script=lambda s, b, t: s.set("val", s.get("val") + 5))
        pipeline_b = ExecutionPipeline(scope=scope_b, script=lambda s, b, t: s.set("val", s.get("val") + 50))

        bar = Bar(index=0, open=100.0, high=101.0, low=99.0, close=100.0)
        pipeline_a.process_bar(bar)
        pipeline_b.process_bar(bar)

        assert scope_a.get("val") == 15
        assert scope_b.get("val") == 70