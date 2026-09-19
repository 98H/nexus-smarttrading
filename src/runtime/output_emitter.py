from __future__ import annotations

import copy
import math
from typing import Any

VALID_LOCATIONS: set[str] = {"abovebar", "belowbar", "top", "bottom", "absolute"}


class OutputEmitter:
    """Collects visual outputs (plots, shapes, chars, candles) during script evaluation."""

    def __init__(self) -> None:
        self._current_bar_index: int | None = None
        self._current_time: int | float | None = None
        self._plots: dict[str, dict[str, Any]] = {}
        self._plotshapes: list[dict[str, Any]] = []
        self._plotchars: list[dict[str, Any]] = []
        self._plotcandles: list[dict[str, Any]] = []

    def set_bar(self, bar_index: int, time: int | float) -> None:
        """Sets the active evaluation bar context."""
        self._current_bar_index = bar_index
        self._current_time = time

    def clear(self) -> None:
        """Purges all recorded visual series back to empty baseline."""
        self._plots.clear()
        self._plotshapes.clear()
        self._plotchars.clear()
        self._plotcandles.clear()

    def _resolve_bar_context(
        self, bar_index: int | None, time: int | float | None
    ) -> tuple[int, int | float]:
        """Resolves bar_index and time from kwargs or active bar context."""
        resolved_bar_index = self._current_bar_index if bar_index is None else bar_index
        resolved_time = self._current_time if time is None else time
        if resolved_bar_index is None or resolved_time is None:
            raise ValueError("Active bar context or explicit bar_index and time required.")
        return resolved_bar_index, resolved_time

    def emit_plot(
        self,
        title: str = "Plot",
        value: float | None = None,
        color: str | None = None,
        style: str = "line",
        linewidth: int = 1,
        offset: int = 0,
        bar_index: int | None = None,
        time: int | float | None = None,
    ) -> None:
        """Records a plot series point with styling and time-indexed values."""
        if linewidth <= 0:
            raise ValueError(f"linewidth must be positive, got {linewidth}")

        b_idx, t = self._resolve_bar_context(bar_index, time)

        if title not in self._plots:
            self._plots[title] = {
                "title": title,
                "color": color if color is not None else "#2196F3",
                "style": style,
                "linewidth": linewidth,
                "offset": offset,
                "values": [],
            }

        val_point: dict[str, Any] = {
            "bar_index": b_idx,
            "time": t,
            "value": value,
        }
        if color is not None:
            val_point["color"] = color

        self._plots[title]["values"].append(val_point)

    def emit_plotshape(
        self,
        condition: Any = True,
        title: str = "",
        style: str = "shape_circle",
        location: str = "abovebar",
        color: str | None = None,
        offset: int = 0,
        text: str = "",
        textcolor: str = "#FFFFFF",
        size: str = "auto",
        bar_index: int | None = None,
        time: int | float | None = None,
    ) -> None:
        """Captures a shape marker at the current bar when condition evaluates to True."""
        if location not in VALID_LOCATIONS:
            raise ValueError(f"Invalid location '{location}'. Valid locations: {VALID_LOCATIONS}")

        b_idx, t = self._resolve_bar_context(bar_index, time)

        if not condition:
            return

        marker: dict[str, Any] = {
            "title": title,
            "style": style,
            "location": location,
            "color": color if color is not None else "#2196F3",
            "offset": offset,
            "text": text,
            "textcolor": textcolor,
            "size": size,
            "bar_index": b_idx,
            "time": t,
        }
        self._plotshapes.append(marker)

    def emit_plotchar(
        self,
        condition: Any = True,
        char: str = "★",
        title: str = "",
        location: str = "abovebar",
        color: str | None = None,
        offset: int = 0,
        text: str = "",
        textcolor: str = "#FFFFFF",
        size: str = "auto",
        bar_index: int | None = None,
        time: int | float | None = None,
    ) -> None:
        """Captures a character marker at the current bar when condition evaluates to True."""
        if not char:
            raise ValueError("char cannot be an empty string.")
        if location not in VALID_LOCATIONS:
            raise ValueError(f"Invalid location '{location}'. Valid locations: {VALID_LOCATIONS}")

        b_idx, t = self._resolve_bar_context(bar_index, time)

        if not condition:
            return

        marker: dict[str, Any] = {
            "char": char,
            "title": title,
            "location": location,
            "color": color if color is not None else "#2196F3",
            "offset": offset,
            "text": text,
            "textcolor": textcolor,
            "size": size,
            "bar_index": b_idx,
            "time": t,
        }
        self._plotchars.append(marker)

    def emit_plotcandle(
        self,
        open: float,
        high: float,
        low: float,
        close: float,
        title: str = "",
        color: str | None = None,
        wickcolor: str | None = None,
        bordercolor: str | None = None,
        offset: int = 0,
        bar_index: int | None = None,
        time: int | float | None = None,
    ) -> None:
        """Records candle data points with OHLC values and respective border/wick colors."""
        if (
            high is not None
            and low is not None
            and not (math.isnan(high) or math.isnan(low))
            and high < low
        ):
            raise ValueError(f"high ({high}) cannot be less than low ({low})")

        b_idx, t = self._resolve_bar_context(bar_index, time)

        candle: dict[str, Any] = {
            "title": title,
            "open": open,
            "high": high,
            "low": low,
            "close": close,
            "color": color,
            "wickcolor": wickcolor,
            "bordercolor": bordercolor,
            "offset": offset,
            "bar_index": b_idx,
            "time": t,
        }
        self._plotcandles.append(candle)

    def get_outputs(self) -> dict[str, Any]:
        """Returns an isolated snapshot of all categorized visual series."""
        return copy.deepcopy(
            {
                "plots": self._plots,
                "plotshapes": self._plotshapes,
                "plotchars": self._plotchars,
                "plotcandles": self._plotcandles,
            }
        )