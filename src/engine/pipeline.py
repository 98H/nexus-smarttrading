"""Execution pipeline for historical and intrabar Pine Script simulation."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Optional

from src.engine.scope import Scope


@dataclass
class Bar:
    """Historical or streaming OHLCV bar representation."""

    index: int
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    time: Optional[Any] = None


@dataclass
class Tick:
    """Intrabar tick update."""

    price: float
    volume: float = 0.0
    is_confirmed: bool = False
    time: Optional[Any] = None


class ExecutionPipeline:
    """Coordinates execution across bars and ticks, maintaining scoped lifecycle state."""

    def __init__(
        self,
        scope: Scope,
        script: Optional[Callable[..., None]] = None,
    ) -> None:
        self.scope = scope
        self._script = script
        self._current_bar_index: Optional[int] = None
        self._current_bar_closed: bool = False

    def set_script(self, script: Callable[..., None]) -> None:
        """Sets or updates the execution script callable."""
        self._script = script

    def _invoke_script(self, bar: Bar, tick: Optional[Tick]) -> None:
        if self._script is None:
            raise ValueError("Script is not set.")
        try:
            sig = inspect.signature(self._script)
            params = list(sig.parameters.values())
            if len(params) == 2 and not any(p.kind == p.VAR_POSITIONAL for p in params):
                self._script(self.scope, bar)
                return
        except (ValueError, TypeError):
            pass
        self._script(self.scope, bar, tick)

    def process_tick(self, bar: Bar, tick: Optional[Tick] = None) -> None:
        """Processes a single intrabar tick or bar close update transactionally."""
        if self._script is None:
            raise ValueError("Script is not set.")

        snapshot = self.scope.snapshot()
        prev_bar_index = self._current_bar_index
        prev_bar_closed = self._current_bar_closed

        try:
            if self._current_bar_index is not None and self._current_bar_index != bar.index:
                if not self._current_bar_closed:
                    self.scope.on_bar_close()
                    self._current_bar_closed = True

            if self._current_bar_index != bar.index or self._current_bar_closed:
                self.scope.on_bar_start(bar.index)
                self._current_bar_index = bar.index
                self._current_bar_closed = False

            self.scope.on_tick_start()
            self._invoke_script(bar, tick)

            is_confirmed = tick.is_confirmed if tick is not None else True
            self.scope.on_tick_end(is_confirmed=is_confirmed)

            if is_confirmed:
                self.scope.on_bar_close()
                self._current_bar_closed = True
        except Exception:
            self.scope.restore_snapshot(snapshot)
            self._current_bar_index = prev_bar_index
            self._current_bar_closed = prev_bar_closed
            raise

    def process_bar(self, bar: Bar, ticks: Optional[list[Tick]] = None) -> None:
        """Processes a bar across its intrabar ticks or directly at bar close."""
        if self._script is None:
            raise ValueError("Script is not set.")

        if ticks:
            for tick in ticks:
                self.process_tick(bar, tick)
        else:
            self.process_tick(bar, None)

    def process_historical(self, bars: list[Bar]) -> None:
        """Processes a batch of historical OHLCV bars sequentially."""
        if not bars:
            raise ValueError("Historical bars list cannot be empty.")
        if self._script is None:
            raise ValueError("Script is not set.")

        for bar in bars:
            self.process_bar(bar)