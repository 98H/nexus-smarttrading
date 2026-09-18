from datetime import datetime, timedelta, timezone
import re
import pytest

from src.charts.formatters import AdaptiveTimeIntervalFormatter

# Regex patterns matching required formats per Acceptance Criteria:
# Under 24h: HH:MM or HH:MM:SS
PATTERN_UNDER_24_HOURS = re.compile(r"^(?:[01]\d|2[0-3]):[0-5]\d(?::[0-5]\d)?$")

# 1 day to 30 days: MMM DD (e.g., 'Jan 01' or 'Jan 1') or YYYY-MM-DD
PATTERN_1_TO_30_DAYS = re.compile(
    r"^(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}|\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01]))$"
)

# Exceeding 365 days: YYYY or YYYY-MM
PATTERN_EXCEEDING_365_DAYS = re.compile(r"^(?:\d{4}|\d{4}-(?:0[1-9]|1[0-2]))$")


@pytest.fixture
def formatter() -> AdaptiveTimeIntervalFormatter:
    """Fixture providing an instance of AdaptiveTimeIntervalFormatter."""
    return AdaptiveTimeIntervalFormatter()


# ============================================================================
# Acceptance Criterion 1: Interval range < 24 hours -> HH:MM or HH:MM:SS
# ============================================================================


@pytest.mark.parametrize(
    "span,step",
    [
        (timedelta(minutes=15), timedelta(minutes=5)),
        (timedelta(hours=6), timedelta(hours=1)),
        (timedelta(hours=12), timedelta(hours=2)),
        (timedelta(hours=23, minutes=59, seconds=59), timedelta(hours=4)),
    ],
)
def test_format_labels_interval_under_24_hours_matches_expected_patterns(
    formatter: AdaptiveTimeIntervalFormatter, span: timedelta, step: timedelta
) -> None:
    start = datetime(2023, 6, 15, 8, 0, 0)
    timestamps = []
    curr = start
    while curr <= start + span:
        timestamps.append(curr)
        curr += step

    labels = formatter.format_labels(timestamps)

    assert len(labels) == len(timestamps)
    for label in labels:
        assert isinstance(label, str)
        assert PATTERN_UNDER_24_HOURS.match(label), (
            f"Label '{label}' does not match HH:MM or HH:MM:SS format"
        )


def test_format_labels_interval_under_24_hours_across_midnight(
    formatter: AdaptiveTimeIntervalFormatter,
) -> None:
    # 22:00 to 04:00 spans 6 hours across midnight
    timestamps = [
        datetime(2023, 1, 1, 22, 0, 0),
        datetime(2023, 1, 2, 0, 0, 0),
        datetime(2023, 1, 2, 2, 0, 0),
        datetime(2023, 1, 2, 4, 0, 0),
    ]

    labels = formatter.format_labels(timestamps)

    assert len(labels) == 4
    for label in labels:
        assert PATTERN_UNDER_24_HOURS.match(label)


def test_format_labels_interval_under_24_hours_single_timestamp(
    formatter: AdaptiveTimeIntervalFormatter,
) -> None:
    # Single timestamp: range is 0 seconds (< 24 hours)
    timestamps = [datetime(2023, 5, 10, 14, 30, 45)]

    labels = formatter.format_labels(timestamps)

    assert len(labels) == 1
    assert PATTERN_UNDER_24_HOURS.match(labels[0])


# ============================================================================
# Acceptance Criterion 2: Interval range 1 day to 30 days -> MMM DD or YYYY-MM-DD
# ============================================================================


@pytest.mark.parametrize(
    "days_offset",
    [
        1,   # Boundary: exactly 1 day (24 hours)
        5,   # Typical mid-range
        14,  # Two weeks
        29,  # Just below 30 days
        30,  # Boundary: exactly 30 days
    ],
)
def test_format_labels_interval_1_to_30_days_matches_expected_patterns(
    formatter: AdaptiveTimeIntervalFormatter, days_offset: int
) -> None:
    start = datetime(2023, 4, 1, 0, 0, 0)
    timestamps = [
        start,
        start + timedelta(days=days_offset / 2),
        start + timedelta(days=days_offset),
    ]

    labels = formatter.format_labels(timestamps)

    assert len(labels) == len(timestamps)
    for label in labels:
        assert isinstance(label, str)
        assert PATTERN_1_TO_30_DAYS.match(label), (
            f"Label '{label}' does not match MMM DD or YYYY-MM-DD format"
        )


def test_format_labels_interval_1_day_exact_boundary(
    formatter: AdaptiveTimeIntervalFormatter,
) -> None:
    # Exactly 24 hours between min and max
    timestamps = [
        datetime(2023, 10, 1, 12, 0, 0),
        datetime(2023, 10, 2, 0, 0, 0),
        datetime(2023, 10, 2, 12, 0, 0),
    ]

    labels = formatter.format_labels(timestamps)

    assert len(labels) == 3
    for label in labels:
        assert PATTERN_1_TO_30_DAYS.match(label)


def test_format_labels_interval_30_days_exact_boundary(
    formatter: AdaptiveTimeIntervalFormatter,
) -> None:
    # Exactly 30 days between min and max
    start = datetime(2023, 3, 1, 0, 0, 0)
    end = start + timedelta(days=30)
    timestamps = [start, start + timedelta(days=15), end]

    labels = formatter.format_labels(timestamps)

    assert len(labels) == 3
    for label in labels:
        assert PATTERN_1_TO_30_DAYS.match(label)


# ============================================================================
# Acceptance Criterion 3: Interval range > 365 days -> YYYY-MM or YYYY
# ============================================================================


@pytest.mark.parametrize(
    "days_offset",
    [
        366,   # Boundary: just exceeding 365 days
        730,   # ~2 years
        1825,  # ~5 years
        3650,  # ~10 years
    ],
)
def test_format_labels_interval_exceeding_365_days_matches_expected_patterns(
    formatter: AdaptiveTimeIntervalFormatter, days_offset: int
) -> None:
    start = datetime(2015, 1, 1, 0, 0, 0)
    timestamps = [
        start,
        start + timedelta(days=days_offset // 2),
        start + timedelta(days=days_offset),
    ]

    labels = formatter.format_labels(timestamps)

    assert len(labels) == len(timestamps)
    for label in labels:
        assert isinstance(label, str)
        assert PATTERN_EXCEEDING_365_DAYS.match(label), (
            f"Label '{label}' does not match YYYY-MM or YYYY format"
        )


def test_format_labels_exceeding_365_days_prevents_visual_overlap(
    formatter: AdaptiveTimeIntervalFormatter,
) -> None:
    # When spanning 5 years with annual ticks, formatted labels must be concise
    # (maximum 7 characters: len('YYYY-MM') == 7, len('YYYY') == 4) and distinct
    timestamps = [
        datetime(2018, 1, 1),
        datetime(2019, 1, 1),
        datetime(2020, 1, 1),
        datetime(2021, 1, 1),
        datetime(2022, 1, 1),
        datetime(2023, 1, 1),
    ]

    labels = formatter.format_labels(timestamps)

    assert len(labels) == len(timestamps)
    # Ensure compact representation without visual overlap: string length <= 7
    for label in labels:
        assert len(label) <= 7, f"Label '{label}' exceeds compact visual length"
        assert PATTERN_EXCEEDING_365_DAYS.match(label)

    # Verify that labels across distinct years do not produce duplicate overlap
    assert len(set(labels)) == len(timestamps), "Distinct yearly ticks produced duplicate labels"


# ============================================================================
# General Behavior, Ordering, Edge Cases & Error Handling
# ============================================================================


def test_format_labels_preserves_input_ordering(
    formatter: AdaptiveTimeIntervalFormatter,
) -> None:
    # Range is 2 days (between 1 and 30 days), but inputs are shuffled
    ts1 = datetime(2023, 7, 2, 12, 0, 0)
    ts2 = datetime(2023, 7, 1, 10, 0, 0)
    ts3 = datetime(2023, 7, 3, 14, 0, 0)
    unsorted_timestamps = [ts1, ts2, ts3]

    labels = formatter.format_labels(unsorted_timestamps)

    assert len(labels) == 3
    # Format labels independently ordered as matching the input sequence
    sorted_labels = formatter.format_labels([ts2, ts1, ts3])
    assert labels == [sorted_labels[1], sorted_labels[0], sorted_labels[2]]


def test_format_labels_empty_input_returns_empty_list(
    formatter: AdaptiveTimeIntervalFormatter,
) -> None:
    labels = formatter.format_labels([])
    assert labels == []


def test_format_labels_supports_timezone_aware_datetimes(
    formatter: AdaptiveTimeIntervalFormatter,
) -> None:
    # 4 hours span with timezone awareness
    tz = timezone.utc
    timestamps = [
        datetime(2023, 5, 1, 8, 0, tzinfo=tz),
        datetime(2023, 5, 1, 10, 0, tzinfo=tz),
        datetime(2023, 5, 1, 12, 0, tzinfo=tz),
    ]

    labels = formatter.format_labels(timestamps)

    assert len(labels) == 3
    for label in labels:
        assert PATTERN_UNDER_24_HOURS.match(label)


def test_format_labels_invalid_elements_raise_type_error(
    formatter: AdaptiveTimeIntervalFormatter,
) -> None:
    invalid_timestamps = [
        datetime(2023, 1, 1, 10, 0, 0),
        "2023-01-01 12:00:00",  # invalid string instead of datetime
    ]

    with pytest.raises(TypeError):
        formatter.format_labels(invalid_timestamps)  # type: ignore[arg-type]


def test_format_labels_none_input_raises_type_error(
    formatter: AdaptiveTimeIntervalFormatter,
) -> None:
    with pytest.raises(TypeError):
        formatter.format_labels(None)  # type: ignore[arg-type]