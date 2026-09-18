"""
Unit tests for Elliott Wave (Impulse 1-5 and Correction A-B-C) Vector Visualizer.

Story 5.3.3: Implement Elliott Wave (Impulse 1-5 and Correction A-B-C) Vector Visualizer.
Target modules:
- src/visualization/elliott_wave_visualizer.py
- src/visualization/__init__.py
"""

from datetime import datetime, timedelta, timezone
from typing import List

import pytest

from src.visualization import (
    ElliottWaveVisualizer,
    VectorSegment,
    WavePoint,
)


@pytest.fixture
def visualizer() -> ElliottWaveVisualizer:
    """Fixture providing a fresh instance of ElliottWaveVisualizer."""
    return ElliottWaveVisualizer()


@pytest.fixture
def base_time() -> datetime:
    """Fixture providing an anchor UTC timestamp."""
    return datetime(2023, 1, 1, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def valid_impulse_points(base_time: datetime) -> List[WavePoint]:
    """
    Fixture providing standard 5-point Elliott Impulse Wave coordinates (1-5).
    Points follow sequential chronological order and standard impulse progression.
    """
    return [
        WavePoint(label="1", timestamp=base_time, price=100.0),
        WavePoint(label="2", timestamp=base_time + timedelta(hours=1), price=120.0),
        WavePoint(label="3", timestamp=base_time + timedelta(hours=2), price=110.0),
        WavePoint(label="4", timestamp=base_time + timedelta(hours=3), price=135.0),
        WavePoint(label="5", timestamp=base_time + timedelta(hours=4), price=125.0),
    ]


@pytest.fixture
def valid_corrective_points(base_time: datetime) -> List[WavePoint]:
    """
    Fixture providing standard 3-point Elliott Corrective Wave coordinates (A-C).
    Points follow sequential chronological order.
    """
    return [
        WavePoint(label="A", timestamp=base_time, price=125.0),
        WavePoint(label="B", timestamp=base_time + timedelta(hours=1), price=115.0),
        WavePoint(label="C", timestamp=base_time + timedelta(hours=2), price=122.0),
    ]


# ============================================================================
# Package and Export Verification
# ============================================================================


def test_visualization_package_exports() -> None:
    """Verify core visualizer types are exported from the top-level visualization package."""
    import src.visualization as viz

    assert hasattr(viz, "ElliottWaveVisualizer")
    assert hasattr(viz, "WavePoint")
    assert hasattr(viz, "VectorSegment")


# ============================================================================
# Elliott Impulse Wave (1-5) Rendering Tests
# ============================================================================


def test_render_impulse_wave_produces_four_sequential_segments(
    visualizer: ElliottWaveVisualizer,
    valid_impulse_points: List[WavePoint],
) -> None:
    """
    AC: Given an Elliott Impulse Wave with points (1, 2, 3, 4, 5) defined by timestamps and prices,
    When the vector visualizer renders the impulse pattern,
    Then it produces 4 sequential vector segments labeled 1-5.
    """
    segments: List[VectorSegment] = visualizer.render_impulse_wave(valid_impulse_points)

    assert isinstance(segments, list)
    assert len(segments) == 4

    expected_label_pairs = [("1", "2"), ("2", "3"), ("3", "4"), ("4", "5")]
    for i, (start_lbl, end_lbl) in enumerate(expected_label_pairs):
        seg = segments[i]
        assert isinstance(seg, VectorSegment)
        assert seg.start_point.label == start_lbl
        assert seg.end_point.label == end_lbl
        assert f"{start_lbl}-{end_lbl}" in seg.label or (
            seg.label == f"{start_lbl}->{end_lbl}"
        ) or seg.label == f"{start_lbl}-{end_lbl}"


def test_render_impulse_wave_coordinates_continuity_and_integrity(
    visualizer: ElliottWaveVisualizer,
    valid_impulse_points: List[WavePoint],
) -> None:
    """Verify coordinate continuity between adjacent vector segments in impulse pattern."""
    segments = visualizer.render_impulse_wave(valid_impulse_points)

    for i in range(len(segments) - 1):
        # End of current segment must match start of subsequent segment
        assert segments[i].end_point == segments[i + 1].start_point
        assert segments[i].end_point.price == segments[i + 1].start_point.price
        assert segments[i].end_point.timestamp == segments[i + 1].start_point.timestamp

    # Verify boundary points match original input
    assert segments[0].start_point == valid_impulse_points[0]
    assert segments[-1].end_point == valid_impulse_points[-1]


def test_render_impulse_wave_styling_metadata(
    visualizer: ElliottWaveVisualizer,
    valid_impulse_points: List[WavePoint],
) -> None:
    """
    AC: Verify impulse vector segments contain distinct impulse styling metadata
    (e.g., wave_type/pattern_type identifying impulse pattern).
    """
    segments = visualizer.render_impulse_wave(valid_impulse_points)

    for segment in segments:
        assert isinstance(segment.styling, dict)
        assert len(segment.styling) > 0
        pattern_type = (
            segment.styling.get("pattern_type")
            or segment.styling.get("wave_type")
            or segment.styling.get("type")
        )
        assert pattern_type is not None
        assert "impulse" in str(pattern_type).lower()


# ============================================================================
# Elliott Corrective Wave (A-B-C) Rendering Tests
# ============================================================================


def test_render_corrective_wave_produces_two_sequential_segments(
    visualizer: ElliottWaveVisualizer,
    valid_corrective_points: List[WavePoint],
) -> None:
    """
    AC: Given an Elliott Corrective Wave with points (A, B, C) defined by timestamps and prices,
    When the vector visualizer renders the corrective pattern,
    Then it produces 2 sequential vector segments labeled A-C.
    """
    segments: List[VectorSegment] = visualizer.render_corrective_wave(
        valid_corrective_points
    )

    assert isinstance(segments, list)
    assert len(segments) == 2

    expected_label_pairs = [("A", "B"), ("B", "C")]
    for i, (start_lbl, end_lbl) in enumerate(expected_label_pairs):
        seg = segments[i]
        assert isinstance(seg, VectorSegment)
        assert seg.start_point.label == start_lbl
        assert seg.end_point.label == end_lbl
        assert f"{start_lbl}-{end_lbl}" in seg.label or (
            seg.label == f"{start_lbl}->{end_lbl}"
        ) or seg.label == f"{start_lbl}-{end_lbl}"


def test_render_corrective_wave_coordinates_continuity_and_integrity(
    visualizer: ElliottWaveVisualizer,
    valid_corrective_points: List[WavePoint],
) -> None:
    """Verify coordinate continuity between adjacent vector segments in corrective pattern."""
    segments = visualizer.render_corrective_wave(valid_corrective_points)

    assert segments[0].end_point == segments[1].start_point
    assert segments[0].end_point.price == segments[1].start_point.price
    assert segments[0].end_point.timestamp == segments[1].start_point.timestamp

    assert segments[0].start_point == valid_corrective_points[0]
    assert segments[-1].end_point == valid_corrective_points[-1]


def test_render_corrective_wave_styling_metadata(
    visualizer: ElliottWaveVisualizer,
    valid_corrective_points: List[WavePoint],
) -> None:
    """
    AC: Verify corrective vector segments contain distinct corrective styling metadata.
    """
    segments = visualizer.render_corrective_wave(valid_corrective_points)

    for segment in segments:
        assert isinstance(segment.styling, dict)
        assert len(segment.styling) > 0
        pattern_type = (
            segment.styling.get("pattern_type")
            or segment.styling.get("wave_type")
            or segment.styling.get("type")
        )
        assert pattern_type is not None
        assert "corrective" in str(pattern_type).lower()


def test_impulse_and_corrective_styling_metadata_are_distinct(
    visualizer: ElliottWaveVisualizer,
    valid_impulse_points: List[WavePoint],
    valid_corrective_points: List[WavePoint],
) -> None:
    """Verify impulse and corrective styling metadata are mutually distinct."""
    impulse_segments = visualizer.render_impulse_wave(valid_impulse_points)
    corrective_segments = visualizer.render_corrective_wave(valid_corrective_points)

    impulse_styling = impulse_segments[0].styling
    corrective_styling = corrective_segments[0].styling

    assert impulse_styling != corrective_styling


# ============================================================================
# Validation and Error Handling Tests (ValueError)
# ============================================================================


@pytest.mark.parametrize(
    "invalid_point_count",
    [
        0,
        1,
        2,
        3,
        4,
        6,
        10,
    ],
)
def test_render_impulse_wave_invalid_point_count_raises_value_error(
    visualizer: ElliottWaveVisualizer,
    base_time: datetime,
    invalid_point_count: int,
) -> None:
    """
    AC: Given incomplete wave coordinate points (fewer or more than 5 points),
    When vector generation is invoked, Then a ValueError is raised.
    """
    points = [
        WavePoint(label=str(i + 1), timestamp=base_time + timedelta(hours=i), price=100.0 + i)
        for i in range(invalid_point_count)
    ]
    with pytest.raises(ValueError) as exc_info:
        visualizer.render_impulse_wave(points)
    assert str(exc_info.value).strip() != ""


@pytest.mark.parametrize(
    "invalid_point_count",
    [
        0,
        1,
        2,
        4,
        5,
    ],
)
def test_render_corrective_wave_invalid_point_count_raises_value_error(
    visualizer: ElliottWaveVisualizer,
    base_time: datetime,
    invalid_point_count: int,
) -> None:
    """
    AC: Given incomplete wave coordinate points (fewer or more than 3 points),
    When vector generation is invoked, Then a ValueError is raised.
    """
    labels = ["A", "B", "C", "D", "E"]
    points = [
        WavePoint(
            label=labels[i % len(labels)],
            timestamp=base_time + timedelta(hours=i),
            price=100.0 + i,
        )
        for i in range(invalid_point_count)
    ]
    with pytest.raises(ValueError) as exc_info:
        visualizer.render_corrective_wave(points)
    assert str(exc_info.value).strip() != ""


def test_render_impulse_wave_out_of_order_timestamps_raises_value_error(
    visualizer: ElliottWaveVisualizer,
    base_time: datetime,
) -> None:
    """
    AC: Given out-of-order wave coordinate points (timestamps non-chronological),
    When vector generation is invoked, Then a ValueError is raised.
    """
    out_of_order_points = [
        WavePoint(label="1", timestamp=base_time, price=100.0),
        WavePoint(label="2", timestamp=base_time + timedelta(hours=2), price=120.0),
        WavePoint(label="3", timestamp=base_time + timedelta(hours=1), price=110.0),  # timestamp regression
        WavePoint(label="4", timestamp=base_time + timedelta(hours=3), price=130.0),
        WavePoint(label="5", timestamp=base_time + timedelta(hours=4), price=125.0),
    ]
    with pytest.raises(ValueError) as exc_info:
        visualizer.render_impulse_wave(out_of_order_points)
    assert str(exc_info.value).strip() != ""


def test_render_corrective_wave_out_of_order_timestamps_raises_value_error(
    visualizer: ElliottWaveVisualizer,
    base_time: datetime,
) -> None:
    """
    AC: Given out-of-order wave coordinate points (timestamps non-chronological),
    When vector generation is invoked, Then a ValueError is raised.
    """
    out_of_order_points = [
        WavePoint(label="A", timestamp=base_time, price=125.0),
        WavePoint(label="B", timestamp=base_time - timedelta(hours=1), price=115.0),  # earlier than A
        WavePoint(label="C", timestamp=base_time + timedelta(hours=2), price=120.0),
    ]
    with pytest.raises(ValueError) as exc_info:
        visualizer.render_corrective_wave(out_of_order_points)
    assert str(exc_info.value).strip() != ""


def test_render_impulse_wave_duplicate_timestamps_raises_value_error(
    visualizer: ElliottWaveVisualizer,
    base_time: datetime,
) -> None:
    """Verify points with identical timestamps are rejected with a ValueError."""
    points = [
        WavePoint(label="1", timestamp=base_time, price=100.0),
        WavePoint(label="2", timestamp=base_time, price=120.0),  # duplicate timestamp
        WavePoint(label="3", timestamp=base_time + timedelta(hours=2), price=110.0),
        WavePoint(label="4", timestamp=base_time + timedelta(hours=3), price=130.0),
        WavePoint(label="5", timestamp=base_time + timedelta(hours=4), price=125.0),
    ]
    with pytest.raises(ValueError) as exc_info:
        visualizer.render_impulse_wave(points)
    assert str(exc_info.value).strip() != ""


def test_render_corrective_wave_duplicate_timestamps_raises_value_error(
    visualizer: ElliottWaveVisualizer,
    base_time: datetime,
) -> None:
    """Verify corrective points with identical timestamps are rejected with a ValueError."""
    points = [
        WavePoint(label="A", timestamp=base_time, price=125.0),
        WavePoint(label="B", timestamp=base_time + timedelta(hours=1), price=115.0),
        WavePoint(label="C", timestamp=base_time + timedelta(hours=1), price=120.0),  # duplicate timestamp
    ]
    with pytest.raises(ValueError) as exc_info:
        visualizer.render_corrective_wave(points)
    assert str(exc_info.value).strip() != ""


def test_render_impulse_wave_out_of_order_labels_raises_value_error(
    visualizer: ElliottWaveVisualizer,
    base_time: datetime,
) -> None:
    """
    AC: Given out-of-order wave coordinate labels (e.g. ['1', '3', '2', '4', '5']),
    When vector generation is invoked, Then a ValueError is raised.
    """
    scrambled_points = [
        WavePoint(label="1", timestamp=base_time, price=100.0),
        WavePoint(label="3", timestamp=base_time + timedelta(hours=1), price=120.0),  # wrong sequence
        WavePoint(label="2", timestamp=base_time + timedelta(hours=2), price=110.0),
        WavePoint(label="4", timestamp=base_time + timedelta(hours=3), price=130.0),
        WavePoint(label="5", timestamp=base_time + timedelta(hours=4), price=125.0),
    ]
    with pytest.raises(ValueError) as exc_info:
        visualizer.render_impulse_wave(scrambled_points)
    assert str(exc_info.value).strip() != ""


def test_render_corrective_wave_out_of_order_labels_raises_value_error(
    visualizer: ElliottWaveVisualizer,
    base_time: datetime,
) -> None:
    """
    AC: Given out-of-order wave coordinate labels (e.g. ['A', 'C', 'B']),
    When vector generation is invoked, Then a ValueError is raised.
    """
    scrambled_points = [
        WavePoint(label="A", timestamp=base_time, price=125.0),
        WavePoint(label="C", timestamp=base_time + timedelta(hours=1), price=115.0),  # wrong sequence
        WavePoint(label="B", timestamp=base_time + timedelta(hours=2), price=120.0),
    ]
    with pytest.raises(ValueError) as exc_info:
        visualizer.render_corrective_wave(scrambled_points)
    assert str(exc_info.value).strip() != ""


def test_render_impulse_wave_invalid_labels_raises_value_error(
    visualizer: ElliottWaveVisualizer,
    base_time: datetime,
) -> None:
    """Verify impulse points with completely unexpected labels raise a ValueError."""
    invalid_labeled_points = [
        WavePoint(label="0", timestamp=base_time, price=100.0),
        WavePoint(label="1", timestamp=base_time + timedelta(hours=1), price=120.0),
        WavePoint(label="2", timestamp=base_time + timedelta(hours=2), price=110.0),
        WavePoint(label="3", timestamp=base_time + timedelta(hours=3), price=130.0),
        WavePoint(label="4", timestamp=base_time + timedelta(hours=4), price=125.0),
    ]
    with pytest.raises(ValueError) as exc_info:
        visualizer.render_impulse_wave(invalid_labeled_points)
    assert str(exc_info.value).strip() != ""


def test_render_corrective_wave_invalid_labels_raises_value_error(
    visualizer: ElliottWaveVisualizer,
    base_time: datetime,
) -> None:
    """Verify corrective points with completely unexpected labels raise a ValueError."""
    invalid_labeled_points = [
        WavePoint(label="X", timestamp=base_time, price=125.0),
        WavePoint(label="Y", timestamp=base_time + timedelta(hours=1), price=115.0),
        WavePoint(label="Z", timestamp=base_time + timedelta(hours=2), price=120.0),
    ]
    with pytest.raises(ValueError) as exc_info:
        visualizer.render_corrective_wave(invalid_labeled_points)
    assert str(exc_info.value).strip() != ""


@pytest.mark.parametrize("invalid_price", [float("nan"), float("inf"), float("-inf")])
def test_render_wave_invalid_price_raises_value_error(
    visualizer: ElliottWaveVisualizer,
    base_time: datetime,
    invalid_price: float,
) -> None:
    """Verify points with NaN or infinite price values are rejected with ValueError."""
    points = [
        WavePoint(label="1", timestamp=base_time, price=100.0),
        WavePoint(label="2", timestamp=base_time + timedelta(hours=1), price=invalid_price),
        WavePoint(label="3", timestamp=base_time + timedelta(hours=2), price=110.0),
        WavePoint(label="4", timestamp=base_time + timedelta(hours=3), price=130.0),
        WavePoint(label="5", timestamp=base_time + timedelta(hours=4), price=125.0),
    ]
    with pytest.raises(ValueError) as exc_info:
        visualizer.render_impulse_wave(points)
    assert str(exc_info.value).strip() != ""