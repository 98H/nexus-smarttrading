"""Scoped variable management supporting standard globals, var, and varip modifiers."""

from __future__ import annotations

import copy
from enum import Enum
from typing import Any, Callable, Optional


class VarModifier(str, Enum):
    """Supported variable modifiers in Pine Script execution semantics."""

    STANDARD = "standard"
    VAR = "var"
    VARIP = "varip"


class Variable:
    """Internal container representing variable state and lifecycle baselines."""

    def __init__(
        self,
        name: str,
        value: Any,
        modifier: VarModifier,
        initial_value: Any,
        initializer: Optional[Callable[[], Any]] = None,
        bar_start_value: Any = None,
        committed_value: Any = None,
    ) -> None:
        self.name = name
        self.value = value
        self.modifier = modifier
        self.initial_value = initial_value
        self.initializer = initializer
        self.bar_start_value = bar_start_value
        self.committed_value = committed_value


class Scope:
    """Manages scoped variables with STANDARD, VAR, and VARIP modifiers across bar and tick lifecycles."""

    def __init__(self) -> None:
        self._variables: dict[str, Variable] = {}
        self._current_bar_index: Optional[int] = None
        self._has_started: bool = False
        self._last_snapshot_context: tuple[Optional[int], bool] = (None, False)

    @staticmethod
    def _normalize_modifier(modifier: VarModifier | str) -> VarModifier:
        if isinstance(modifier, VarModifier):
            return modifier
        if isinstance(modifier, str):
            try:
                return VarModifier(modifier.lower())
            except ValueError:
                raise ValueError(f"Invalid modifier: '{modifier}'")
        raise ValueError(f"Invalid modifier type: {type(modifier).__name__}")

    def declare(
        self,
        name: str,
        initial_value: Any,
        modifier: VarModifier | str = VarModifier.STANDARD,
        initializer: Optional[Callable[[], Any]] = None,
    ) -> None:
        """Declares a new variable in the scope."""
        if name in self._variables:
            raise ValueError(f"Variable '{name}' is already declared.")

        mod = self._normalize_modifier(modifier)
        try:
            stored_initial = copy.deepcopy(initial_value)
        except Exception:
            stored_initial = initial_value

        self._variables[name] = Variable(
            name=name,
            value=initial_value,
            modifier=mod,
            initial_value=stored_initial,
            initializer=initializer,
            bar_start_value=initial_value,
            committed_value=initial_value,
        )

    def get(self, name: str) -> Any:
        """Returns the current value of the declared variable."""
        if name not in self._variables:
            raise KeyError(f"Variable '{name}' is not declared.")
        return self._variables[name].value

    def set(self, name: str, value: Any) -> None:
        """Sets the current value of the declared variable."""
        if name not in self._variables:
            raise KeyError(f"Variable '{name}' is not declared.")
        self._variables[name].value = value

    def has(self, name: str) -> bool:
        """Checks whether a variable is declared in this scope."""
        return name in self._variables

    def on_bar_start(self, bar_index: int) -> None:
        """Lifecycle hook invoked at the opening of a new bar."""
        self._current_bar_index = bar_index
        is_first_bar = not self._has_started
        self._has_started = True

        for var in self._variables.values():
            if var.modifier == VarModifier.STANDARD:
                if var.initializer is not None:
                    var.value = var.initializer()
                else:
                    try:
                        var.value = copy.deepcopy(var.initial_value)
                    except Exception:
                        var.value = var.initial_value
                var.bar_start_value = var.value
            elif var.modifier == VarModifier.VAR:
                if is_first_bar and var.initializer is not None:
                    var.value = var.initializer()
                    var.committed_value = var.value
                else:
                    var.value = var.committed_value
                var.bar_start_value = var.value
            elif var.modifier == VarModifier.VARIP:
                if is_first_bar and var.initializer is not None:
                    var.value = var.initializer()
                    var.committed_value = var.value
                var.bar_start_value = var.value

    def on_tick_start(self) -> None:
        """Lifecycle hook invoked before executing a tick calculation."""

    def on_tick_end(self, is_confirmed: bool) -> None:
        """Lifecycle hook invoked after completing a tick calculation."""
        if is_confirmed:
            for var in self._variables.values():
                var.bar_start_value = var.value
        else:
            for var in self._variables.values():
                if var.modifier in (VarModifier.VAR, VarModifier.STANDARD):
                    var.value = var.bar_start_value

    def on_bar_close(self) -> None:
        """Lifecycle hook invoked at bar close to commit persistent state."""
        for var in self._variables.values():
            var.committed_value = var.value

    def snapshot(self) -> dict[str, tuple[Any, Any, Any]]:
        """Takes an isolated snapshot of all variable states for transactional rollback."""
        self._last_snapshot_context = (self._current_bar_index, self._has_started)
        state: dict[str, tuple[Any, Any, Any]] = {}
        for name, var in self._variables.items():
            try:
                v_copy = copy.deepcopy(var.value)
            except Exception:
                v_copy = var.value
            try:
                b_copy = copy.deepcopy(var.bar_start_value)
            except Exception:
                b_copy = var.bar_start_value
            try:
                c_copy = copy.deepcopy(var.committed_value)
            except Exception:
                c_copy = var.committed_value
            state[name] = (v_copy, b_copy, c_copy)
        return state

    def restore_snapshot(self, state: dict[str, tuple[Any, Any, Any]]) -> None:
        """Restores variable states from a snapshot."""
        self._current_bar_index, self._has_started = self._last_snapshot_context
        for name in list(self._variables.keys()):
            if name not in state:
                del self._variables[name]
        for name, (val, b_val, c_val) in state.items():
            if name in self._variables:
                var = self._variables[name]
                try:
                    var.value = copy.deepcopy(val)
                except Exception:
                    var.value = val
                try:
                    var.bar_start_value = copy.deepcopy(b_val)
                except Exception:
                    var.bar_start_value = b_val
                try:
                    var.committed_value = copy.deepcopy(c_val)
                except Exception:
                    var.committed_value = c_val