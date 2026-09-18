"""
Unit tests for the Multi-Pane Grid Layout System.

Requirement: Story 9.1.1: Implement Multi-Pane Grid Layout System (1x1, 1x2, 2x2, 1x3, etc.)
Modules tested:
- src.layout
- src.layout.grid_layout
"""

import itertools
import pytest

from src.layout import GridLayout, PaneSlot
import src.layout as layout_pkg
import src.layout.grid_layout as grid_layout_module


def _rectangles_overlap(slot1: PaneSlot, slot2: PaneSlot) -> bool:
    """
    Helper function to determine if two rectangular pane slots overlap with positive area.
    Adjacent boundaries (touching edges) do not count as overlapping.
    """
    horizontal_overlap = max(0, min(slot1.x + slot1.width, slot2.x + slot2.width) - max(slot1.x, slot2.x))
    vertical_overlap = max(0, min(slot1.y + slot1.height, slot2.y + slot2.height) - max(slot1.y, slot2.y))
    return horizontal_overlap > 0 and vertical_overlap > 0


class TestLayoutModuleExports:
    """Verifies that the package interface exposes layout primitives properly."""

    def test_package_exports(self):
        """Ensure GridLayout and PaneSlot are re-exported in src.layout."""
        assert hasattr(layout_pkg, "GridLayout")
        assert hasattr(layout_pkg, "PaneSlot")
        assert layout_pkg.GridLayout is grid_layout_module.GridLayout
        assert layout_pkg.PaneSlot is grid_layout_module.PaneSlot


class TestGridLayoutCalculation:
    """
    AC-1: Given a grid layout configured with dimensions 2x2 and a container
    bounds of (width=800, height=600), when calculating pane slots, then 4
    non-overlapping rectangular pane regions of equal size (400x300) are
    returned with their respective coordinates.
    """

    def test_2x2_pane_slots_calculation(self):
        layout = GridLayout(rows=2, cols=2, container_bounds=(800, 600))
        slots = layout.calculate_pane_slots()

        assert len(slots) == 4

        # Verify each slot has equal size of 400x300
        for slot in slots:
            assert slot.width == 400
            assert slot.height == 300

        # Verify expected coordinates for a 2x2 grid
        expected_coordinates = {
            (0, 0, 400, 300),
            (400, 0, 400, 300),
            (0, 300, 400, 300),
            (400, 300, 400, 300),
        }
        actual_coordinates = {(s.x, s.y, s.width, s.height) for s in slots}
        assert actual_coordinates == expected_coordinates

        # Verify no two slots overlap
        for s1, s2 in itertools.combinations(slots, 2):
            assert not _rectangles_overlap(s1, s2), f"Slot {s1} overlaps with {s2}"

        # Verify total covered area equals container bounds area
        total_pane_area = sum(s.width * s.height for s in slots)
        assert total_pane_area == 800 * 600

    def test_calculation_with_explicit_bounds_override(self):
        """Verify that calculate_pane_slots accepts bounds dynamically."""
        layout = GridLayout(rows=2, cols=2)
        slots = layout.calculate_pane_slots(container_bounds=(800, 600))

        assert len(slots) == 4
        assert {(s.x, s.y, s.width, s.height) for s in slots} == {
            (0, 0, 400, 300),
            (400, 0, 400, 300),
            (0, 300, 400, 300),
            (400, 300, 400, 300),
        }


class TestPresetSwitching:
    """
    AC-2: Given an active layout instance, when switching presets to "1x3",
    then the layout updates to 1 row and 3 columns and allocates 3 pane slots
    partitioned horizontally across the full container height.
    """

    def test_switch_preset_to_1x3(self):
        # Start with an active 2x2 layout
        layout = GridLayout(rows=2, cols=2, container_bounds=(900, 600))
        assert layout.rows == 2
        assert layout.cols == 2

        # Switch preset to "1x3"
        layout.switch_preset("1x3")

        # Layout state updates
        assert layout.rows == 1
        assert layout.cols == 3

        slots = layout.calculate_pane_slots()
        assert len(slots) == 3

        # Partitioned horizontally across the full container height (600)
        expected_width = 300
        expected_height = 600

        for i, slot in enumerate(sorted(slots, key=lambda s: s.x)):
            assert slot.y == 0
            assert slot.height == expected_height
            assert slot.width == expected_width
            assert slot.x == i * expected_width

        # Verify non-overlapping
        for s1, s2 in itertools.combinations(slots, 2):
            assert not _rectangles_overlap(s1, s2)

    def test_preset_case_insensitivity_and_spacing(self):
        """Presets such as ' 1X3 ' should be parsed cleanly."""
        layout = GridLayout(container_bounds=(900, 600))
        layout.switch_preset(" 1X3 ")

        assert layout.rows == 1
        assert layout.cols == 3
        slots = layout.calculate_pane_slots()
        assert len(slots) == 3

    @pytest.mark.parametrize(
        "preset, expected_rows, expected_cols, expected_slot_count",
        [
            ("1x1", 1, 1, 1),
            ("1x2", 1, 2, 2),
            ("2x1", 2, 1, 2),
            ("2x2", 2, 2, 4),
            ("1x3", 1, 3, 3),
            ("3x1", 3, 1, 3),
            ("3x3", 3, 3, 9),
        ],
    )
    def test_supported_presets_switching(
        self, preset, expected_rows, expected_cols, expected_slot_count
    ):
        layout = GridLayout(container_bounds=(600, 600))
        layout.switch_preset(preset)

        assert layout.rows == expected_rows
        assert layout.cols == expected_cols
        slots = layout.calculate_pane_slots()
        assert len(slots) == expected_slot_count


class TestInvalidGridDimensions:
    """
    AC-3: Given invalid grid dimensions such as rows <= 0 or cols <= 0,
    when instantiating or updating the grid layout, then a ValueError is raised.
    """

    @pytest.mark.parametrize(
        "rows, cols",
        [
            (0, 1),
            (1, 0),
            (0, 0),
            (-1, 2),
            (2, -1),
            (-3, -3),
        ],
    )
    def test_instantiate_invalid_dimensions_raises_value_error(self, rows, cols):
        with pytest.raises(ValueError):
            GridLayout(rows=rows, cols=cols)

    @pytest.mark.parametrize(
        "rows, cols",
        [
            (0, 1),
            (1, 0),
            (0, 0),
            (-1, 2),
            (2, -1),
            (-2, -5),
        ],
    )
    def test_update_invalid_dimensions_raises_value_error(self, rows, cols):
        layout = GridLayout(rows=2, cols=2)
        with pytest.raises(ValueError):
            layout.update_dimensions(rows=rows, cols=cols)

    @pytest.mark.parametrize(
        "invalid_rows",
        [0, -1, -10],
    )
    def test_row_setter_invalid_raises_value_error(self, invalid_rows):
        layout = GridLayout(rows=2, cols=2)
        with pytest.raises(ValueError):
            layout.rows = invalid_rows

    @pytest.mark.parametrize(
        "invalid_cols",
        [0, -1, -10],
    )
    def test_col_setter_invalid_raises_value_error(self, invalid_cols):
        layout = GridLayout(rows=2, cols=2)
        with pytest.raises(ValueError):
            layout.cols = invalid_cols

    @pytest.mark.parametrize(
        "invalid_preset",
        [
            "0x1",
            "1x0",
            "0x0",
            "-1x2",
            "2x-1",
            "-2x-2",
            "invalid",
            "1x",
            "x2",
            "axb",
            "",
            "1x2x3",
        ],
    )
    def test_switch_preset_invalid_format_raises_value_error(self, invalid_preset):
        layout = GridLayout(rows=2, cols=2)
        with pytest.raises(ValueError):
            layout.switch_preset(invalid_preset)

    @pytest.mark.parametrize(
        "invalid_preset",
        [
            "0x2",
            "2x0",
            "-1x3",
            "bad_preset",
        ],
    )
    def test_instantiate_with_invalid_preset_raises_value_error(self, invalid_preset):
        with pytest.raises(ValueError):
            GridLayout(preset=invalid_preset)


class TestPaneSlotModel:
    """Validates the structure and immutability of PaneSlot objects."""

    def test_pane_slot_attributes(self):
        slot = PaneSlot(x=10, y=20, width=300, height=400)
        assert slot.x == 10
        assert slot.y == 20
        assert slot.width == 300
        assert slot.height == 400

    def test_calculate_slots_without_bounds_raises_value_error(self):
        """Calculating slots without setting or providing bounds must raise a ValueError."""
        layout = GridLayout(rows=2, cols=2)
        with pytest.raises(ValueError):
            layout.calculate_pane_slots()