"""Multi-Pane Grid Layout System."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PaneSlot:
    """Represents an immutable rectangular region allocated for a pane."""

    x: int
    y: int
    width: int
    height: int


class GridLayout:
    """Manages multi-pane grid layout configurations and slot calculations."""

    def __init__(
        self,
        rows: int = 1,
        cols: int = 1,
        container_bounds: tuple[int, int] | None = None,
        preset: str | None = None,
    ) -> None:
        self._container_bounds: tuple[int, int] | None = None
        if container_bounds is not None:
            self.container_bounds = container_bounds

        if preset is not None:
            self.switch_preset(preset)
        else:
            self.update_dimensions(rows, cols)

    @property
    def rows(self) -> int:
        """Get the number of grid rows."""
        return self._rows

    @rows.setter
    def rows(self, value: int) -> None:
        """Set the number of grid rows."""
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"Rows must be a positive integer, got {value}")
        self._rows = value

    @property
    def cols(self) -> int:
        """Get the number of grid columns."""
        return self._cols

    @cols.setter
    def cols(self, value: int) -> None:
        """Set the number of grid columns."""
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"Cols must be a positive integer, got {value}")
        self._cols = value

    @property
    def container_bounds(self) -> tuple[int, int] | None:
        """Get the container bounds (width, height)."""
        return self._container_bounds

    @container_bounds.setter
    def container_bounds(self, bounds: tuple[int, int] | None) -> None:
        """Set the container bounds (width, height)."""
        if bounds is not None:
            if len(bounds) != 2 or bounds[0] <= 0 or bounds[1] <= 0:
                raise ValueError(f"Container bounds must be positive (width, height), got {bounds}")
            self._container_bounds = (int(bounds[0]), int(bounds[1]))
        else:
            self._container_bounds = None

    def update_dimensions(self, rows: int, cols: int) -> None:
        """Update both row and column dimensions atomically."""
        if not isinstance(rows, int) or isinstance(rows, bool) or rows <= 0:
            raise ValueError(f"Rows must be a positive integer, got {rows}")
        if not isinstance(cols, int) or isinstance(cols, bool) or cols <= 0:
            raise ValueError(f"Cols must be a positive integer, got {cols}")
        self._rows = rows
        self._cols = cols

    def switch_preset(self, preset: str) -> None:
        """Switch layout dimensions based on a preset string (e.g., '1x3', '2x2')."""
        if not isinstance(preset, str):
            raise ValueError(f"Preset must be a string, got {type(preset).__name__}")

        parts = preset.strip().lower().split("x")
        if len(parts) != 2:
            raise ValueError(
                f"Invalid preset format '{preset}'. Expected format 'RxC' (e.g., '1x3')."
            )

        try:
            rows = int(parts[0])
            cols = int(parts[1])
        except ValueError:
            raise ValueError(
                f"Invalid preset format '{preset}'. Rows and columns must be integers."
            )

        if rows <= 0 or cols <= 0:
            raise ValueError(
                f"Preset dimensions must be positive integers, got {rows}x{cols}."
            )

        self._rows = rows
        self._cols = cols

    def calculate_pane_slots(
        self, container_bounds: tuple[int, int] | None = None
    ) -> list[PaneSlot]:
        """
        Calculate non-overlapping rectangular pane regions partitioned equally.

        Args:
            container_bounds: Optional (width, height) override for calculation.

        Returns:
            A list of PaneSlot instances representing the pane slots.

        Raises:
            ValueError: If container bounds are missing or non-positive.
        """
        bounds = container_bounds if container_bounds is not None else self._container_bounds
        if bounds is None:
            raise ValueError("Container bounds must be provided to calculate pane slots.")

        if len(bounds) != 2 or bounds[0] <= 0 or bounds[1] <= 0:
            raise ValueError(f"Container bounds must be positive (width, height), got {bounds}")

        width, height = bounds
        slot_width = width // self._cols
        slot_height = height // self._rows

        slots: list[PaneSlot] = []
        for row in range(self._rows):
            for col in range(self._cols):
                x = col * slot_width
                y = row * slot_height
                slots.append(
                    PaneSlot(
                        x=x,
                        y=y,
                        width=slot_width,
                        height=slot_height,
                    )
                )
        return slots