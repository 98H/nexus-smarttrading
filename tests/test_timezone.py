"""
Unit tests for Timezone Normalizer and Daylight Saving Time Handler.

Story 2.3.3: Implement Timezone Normalizer and Daylight Saving Time Handler
Acceptance Criteria:
- Given a naive or timezone-aware datetime object and a target timezone identifier
- Given an ambiguous local datetime during a daylight saving time fall-back transition (repeated hour)
- Given a nonexistent local datetime during a daylight saving time spring-forward transition (skipped hour)
"""

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import pytest

from src.utils.timezone import (
    AmbiguousTimeError,
    DSTHandler,
    InvalidTimezoneError,
    NonExistentTimeError,
    TimezoneNormalizer,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def normalizer() -> TimezoneNormalizer:
    """Fixture providing a fresh TimezoneNormalizer instance."""
    return TimezoneNormalizer()


@pytest.fixture
def dst_handler() -> DSTHandler:
    """Fixture providing a fresh DSTHandler instance."""
    return DSTHandler()


# ============================================================================
# AC 1: Naive or Timezone-Aware Datetime and Target Timezone Identifier
# ============================================================================


class TestTimezoneNormalization:
    """Tests covering normalization of naive and timezone-aware datetimes to a target timezone."""

    def test_normalize_aware_utc_to_target_tz(self, normalizer: TimezoneNormalizer):
        aware_utc = datetime(2023, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        result = normalizer.normalize(aware_utc, target_tz="America/New_York")

        expected = datetime(2023, 6, 15, 8, 0, 0, tzinfo=ZoneInfo("America/New_York"))
        assert result == expected
        assert result.tzinfo == ZoneInfo("America/New_York")
        assert result.utcoffset() == timedelta(hours=-4)  # EDT is UTC-4

    def test_normalize_aware_between_non_utc_timezones(
        self, normalizer: TimezoneNormalizer
    ):
        tokyo_tz = ZoneInfo("Asia/Tokyo")  # UTC+9 year-round
        aware_tokyo = datetime(2023, 1, 15, 18, 30, 0, tzinfo=tokyo_tz)

        result = normalizer.normalize(aware_tokyo, target_tz="Europe/London")

        expected = datetime(2023, 1, 15, 9, 30, 0, tzinfo=ZoneInfo("Europe/London"))
        assert result == expected
        assert result.tzinfo == ZoneInfo("Europe/London")
        assert result.utcoffset() == timedelta(hours=0)  # GMT in January

    def test_normalize_naive_datetime_with_explicit_source_tz(
        self, normalizer: TimezoneNormalizer
    ):
        naive_dt = datetime(2023, 7, 20, 15, 0, 0)

        result = normalizer.normalize(
            naive_dt, target_tz="UTC", source_tz="America/Los_Angeles"
        )

        # America/Los_Angeles is PDT (UTC-7) in July; 15:00 PDT -> 22:00 UTC
        expected = datetime(2023, 7, 20, 22, 0, 0, tzinfo=timezone.utc)
        assert result == expected
        assert result.utcoffset() == timedelta(0)

    def test_normalize_naive_datetime_defaults_source_to_utc(
        self, normalizer: TimezoneNormalizer
    ):
        naive_dt = datetime(2023, 5, 10, 10, 0, 0)

        result = normalizer.normalize(naive_dt, target_tz="America/New_York")

        # Assumed UTC source: 10:00 UTC -> 06:00 EDT (UTC-4)
        expected = datetime(2023, 5, 10, 6, 0, 0, tzinfo=ZoneInfo("America/New_York"))
        assert result == expected
        assert result.utcoffset() == timedelta(hours=-4)

    def test_normalize_preserves_subsecond_precision(
        self, normalizer: TimezoneNormalizer
    ):
        aware_dt = datetime(2023, 8, 1, 12, 34, 56, 789123, tzinfo=timezone.utc)

        result = normalizer.normalize(aware_dt, target_tz="America/Chicago")

        assert result.microsecond == 789123
        expected = datetime(
            2023, 8, 1, 7, 34, 56, 789123, tzinfo=ZoneInfo("America/Chicago")
        )
        assert result == expected

    def test_normalize_across_date_boundary(self, normalizer: TimezoneNormalizer):
        aware_utc = datetime(2023, 12, 31, 23, 30, 0, tzinfo=timezone.utc)

        result = normalizer.normalize(aware_utc, target_tz="Asia/Tokyo")

        # 23:30 UTC Dec 31 -> 08:30 JST Jan 1 (+9 hours)
        expected = datetime(2024, 1, 1, 8, 30, 0, tzinfo=ZoneInfo("Asia/Tokyo"))
        assert result == expected
        assert result.year == 2024
        assert result.day == 1

    def test_normalize_leap_year_transition(self, normalizer: TimezoneNormalizer):
        aware_dt = datetime(2024, 2, 29, 2, 0, 0, tzinfo=timezone.utc)

        result = normalizer.normalize(aware_dt, target_tz="America/New_York")

        # 02:00 UTC Feb 29 -> 21:00 EST Feb 28 (-5 hours)
        expected = datetime(2024, 2, 28, 21, 0, 0, tzinfo=ZoneInfo("America/New_York"))
        assert result == expected
        assert result.month == 2
        assert result.day == 28

    def test_normalize_same_source_and_target_timezone(
        self, normalizer: TimezoneNormalizer
    ):
        aware_dt = datetime(
            2023, 4, 10, 14, 0, 0, tzinfo=ZoneInfo("America/New_York")
        )

        result = normalizer.normalize(aware_dt, target_tz="America/New_York")

        assert result == aware_dt
        assert result.tzinfo == ZoneInfo("America/New_York")

    def test_normalize_accepts_zoneinfo_instances(
        self, normalizer: TimezoneNormalizer
    ):
        aware_dt = datetime(2023, 6, 1, 12, 0, 0, tzinfo=ZoneInfo("UTC"))
        target_zone = ZoneInfo("Europe/Paris")

        result = normalizer.normalize(aware_dt, target_tz=target_zone)

        expected = datetime(2023, 6, 1, 14, 0, 0, tzinfo=target_zone)
        assert result == expected


# ============================================================================
# AC 2: Ambiguous Local Datetime During DST Fall-Back Transition (Repeated Hour)
# ============================================================================


class TestDSTFallBackAmbiguousTime:
    """Tests covering ambiguous local datetime handling during fall-back DST transitions."""

    # In America/New_York on 2023-11-05:
    # At 02:00 EDT (UTC-4), clocks fall back to 01:00 EST (UTC-5).
    # 01:00:00 through 01:59:59 is repeated (ambiguous).

    def test_is_ambiguous_returns_true_for_repeated_hour(
        self, dst_handler: DSTHandler
    ):
        ambiguous_dt = datetime(2023, 11, 5, 1, 30, 0)
        assert dst_handler.is_ambiguous(ambiguous_dt, "America/New_York") is True

    @pytest.mark.parametrize(
        "dt",
        [
            datetime(2023, 11, 5, 1, 0, 0),  # Start of repeated hour
            datetime(2023, 11, 5, 1, 30, 0),  # Middle of repeated hour
            datetime(2023, 11, 5, 1, 59, 59, 999999),  # End of repeated hour
        ],
    )
    def test_is_ambiguous_boundary_times(
        self, dst_handler: DSTHandler, dt: datetime
    ):
        assert dst_handler.is_ambiguous(dt, "America/New_York") is True

    @pytest.mark.parametrize(
        "dt",
        [
            datetime(2023, 11, 5, 0, 59, 59),  # Before transition window
            datetime(2023, 11, 5, 2, 0, 0),  # After transition window
            datetime(2023, 11, 4, 1, 30, 0),  # Day before
            datetime(2023, 11, 6, 1, 30, 0),  # Day after
        ],
    )
    def test_is_ambiguous_returns_false_for_unambiguous_times(
        self, dst_handler: DSTHandler, dt: datetime
    ):
        assert dst_handler.is_ambiguous(dt, "America/New_York") is False

    def test_is_ambiguous_returns_false_for_non_dst_timezone(
        self, dst_handler: DSTHandler
    ):
        dt = datetime(2023, 11, 5, 1, 30, 0)
        assert dst_handler.is_ambiguous(dt, "Asia/Tokyo") is False
        assert dst_handler.is_ambiguous(dt, "UTC") is False

    def test_normalize_ambiguous_time_default_raises_error(
        self, normalizer: TimezoneNormalizer
    ):
        ambiguous_dt = datetime(2023, 11, 5, 1, 30, 0)
        with pytest.raises(AmbiguousTimeError):
            normalizer.normalize(
                ambiguous_dt,
                target_tz="UTC",
                source_tz="America/New_York",
                ambiguous="raise",
            )

    def test_normalize_ambiguous_time_resolve_earliest(
        self, normalizer: TimezoneNormalizer
    ):
        # Earliest corresponds to EDT (fold=0, UTC-4): 01:30 EDT -> 05:30 UTC
        ambiguous_dt = datetime(2023, 11, 5, 1, 30, 0)

        result = normalizer.normalize(
            ambiguous_dt,
            target_tz="UTC",
            source_tz="America/New_York",
            ambiguous="earliest",
        )

        expected = datetime(2023, 11, 5, 5, 30, 0, tzinfo=timezone.utc)
        assert result == expected

    def test_normalize_ambiguous_time_resolve_latest(
        self, normalizer: TimezoneNormalizer
    ):
        # Latest corresponds to EST (fold=1, UTC-5): 01:30 EST -> 06:30 UTC
        ambiguous_dt = datetime(2023, 11, 5, 1, 30, 0)

        result = normalizer.normalize(
            ambiguous_dt,
            target_tz="UTC",
            source_tz="America/New_York",
            ambiguous="latest",
        )

        expected = datetime(2023, 11, 5, 6, 30, 0, tzinfo=timezone.utc)
        assert result == expected

    def test_normalize_aware_ambiguous_dt_respects_fold_0(
        self, normalizer: TimezoneNormalizer
    ):
        # Datetime is already aware and disambiguated via fold=0
        aware_fold_0 = datetime(
            2023, 11, 5, 1, 30, 0, fold=0, tzinfo=ZoneInfo("America/New_York")
        )

        result = normalizer.normalize(
            aware_fold_0, target_tz="UTC", ambiguous="raise"
        )

        expected = datetime(2023, 11, 5, 5, 30, 0, tzinfo=timezone.utc)
        assert result == expected

    def test_normalize_aware_ambiguous_dt_respects_fold_1(
        self, normalizer: TimezoneNormalizer
    ):
        # Datetime is already aware and disambiguated via fold=1
        aware_fold_1 = datetime(
            2023, 11, 5, 1, 30, 0, fold=1, tzinfo=ZoneInfo("America/New_York")
        )

        result = normalizer.normalize(
            aware_fold_1, target_tz="UTC", ambiguous="raise"
        )

        expected = datetime(2023, 11, 5, 6, 30, 0, tzinfo=timezone.utc)
        assert result == expected

    def test_resolve_ambiguous_direct_handler(self, dst_handler: DSTHandler):
        dt = datetime(2023, 11, 5, 1, 45, 0)

        resolved_early = dst_handler.resolve_ambiguous(
            dt, "America/New_York", strategy="earliest"
        )
        assert resolved_early.fold == 0
        assert resolved_early.utcoffset() == timedelta(hours=-4)

        resolved_late = dst_handler.resolve_ambiguous(
            dt, "America/New_York", strategy="latest"
        )
        assert resolved_late.fold == 1
        assert resolved_late.utcoffset() == timedelta(hours=-5)

    def test_resolve_ambiguous_invalid_strategy_raises_value_error(
        self, dst_handler: DSTHandler
    ):
        dt = datetime(2023, 11, 5, 1, 30, 0)
        with pytest.raises(ValueError):
            dst_handler.resolve_ambiguous(
                dt, "America/New_York", strategy="unsupported_strategy"
            )

    def test_european_fall_back_transition(self, normalizer: TimezoneNormalizer):
        # Europe/London on 2023-10-29: 02:00 BST (UTC+1) falls back to 01:00 GMT (UTC+0)
        dt = datetime(2023, 10, 29, 1, 30, 0)

        early = normalizer.normalize(
            dt, target_tz="UTC", source_tz="Europe/London", ambiguous="earliest"
        )
        late = normalizer.normalize(
            dt, target_tz="UTC", source_tz="Europe/London", ambiguous="latest"
        )

        # Earliest (BST, UTC+1): 01:30 BST -> 00:30 UTC
        assert early == datetime(2023, 10, 29, 0, 30, 0, tzinfo=timezone.utc)
        # Latest (GMT, UTC+0): 01:30 GMT -> 01:30 UTC
        assert late == datetime(2023, 10, 29, 1, 30, 0, tzinfo=timezone.utc)


# ============================================================================
# AC 3: Nonexistent Local Datetime During DST Spring-Forward (Skipped Hour)
# ============================================================================


class TestDSTSpringForwardNonexistentTime:
    """Tests covering nonexistent local datetime handling during spring-forward DST transitions."""

    # In America/New_York on 2023-03-12:
    # At 02:00 EST (UTC-5), clocks jump forward to 03:00 EDT (UTC-4).
    # 02:00:00 through 02:59:59 does not exist.

    def test_is_nonexistent_returns_true_for_skipped_hour(
        self, dst_handler: DSTHandler
    ):
        nonexistent_dt = datetime(2023, 3, 12, 2, 30, 0)
        assert dst_handler.is_nonexistent(nonexistent_dt, "America/New_York") is True

    @pytest.mark.parametrize(
        "dt",
        [
            datetime(2023, 3, 12, 2, 0, 0),  # Start of skipped hour
            datetime(2023, 3, 12, 2, 30, 0),  # Middle of skipped hour
            datetime(2023, 3, 12, 2, 59, 59, 999999),  # End of skipped hour
        ],
    )
    def test_is_nonexistent_boundary_times(
        self, dst_handler: DSTHandler, dt: datetime
    ):
        assert dst_handler.is_nonexistent(dt, "America/New_York") is True

    @pytest.mark.parametrize(
        "dt",
        [
            datetime(2023, 3, 12, 1, 59, 59),  # Just before transition
            datetime(2023, 3, 12, 3, 0, 0),  # Jump target
            datetime(2023, 3, 11, 2, 30, 0),  # Day before
            datetime(2023, 3, 13, 2, 30, 0),  # Day after
        ],
    )
    def test_is_nonexistent_returns_false_for_valid_times(
        self, dst_handler: DSTHandler, dt: datetime
    ):
        assert dst_handler.is_nonexistent(dt, "America/New_York") is False

    def test_is_nonexistent_returns_false_for_non_dst_timezone(
        self, dst_handler: DSTHandler
    ):
        dt = datetime(2023, 3, 12, 2, 30, 0)
        assert dst_handler.is_nonexistent(dt, "Asia/Tokyo") is False
        assert dst_handler.is_nonexistent(dt, "UTC") is False

    def test_normalize_nonexistent_time_default_raises_error(
        self, normalizer: TimezoneNormalizer
    ):
        nonexistent_dt = datetime(2023, 3, 12, 2, 30, 0)
        with pytest.raises(NonExistentTimeError):
            normalizer.normalize(
                nonexistent_dt,
                target_tz="UTC",
                source_tz="America/New_York",
                nonexistent="raise",
            )

    def test_normalize_nonexistent_time_resolve_shift_forward(
        self, normalizer: TimezoneNormalizer
    ):
        # 02:30 shifted forward across the 1-hour gap -> 03:30 EDT (UTC-4) -> 07:30 UTC
        nonexistent_dt = datetime(2023, 3, 12, 2, 30, 0)

        result = normalizer.normalize(
            nonexistent_dt,
            target_tz="UTC",
            source_tz="America/New_York",
            nonexistent="shift_forward",
        )

        expected = datetime(2023, 3, 12, 7, 30, 0, tzinfo=timezone.utc)
        assert result == expected

    def test_normalize_nonexistent_time_resolve_shift_backward(
        self, normalizer: TimezoneNormalizer
    ):
        # 02:30 shifted backward before the gap -> 01:30 EST (UTC-5) -> 06:30 UTC
        nonexistent_dt = datetime(2023, 3, 12, 2, 30, 0)

        result = normalizer.normalize(
            nonexistent_dt,
            target_tz="UTC",
            source_tz="America/New_York",
            nonexistent="shift_backward",
        )

        expected = datetime(2023, 3, 12, 6, 30, 0, tzinfo=timezone.utc)
        assert result == expected

    def test_resolve_nonexistent_direct_handler(self, dst_handler: DSTHandler):
        nonexistent_dt = datetime(2023, 3, 12, 2, 15, 0)

        res_fwd = dst_handler.resolve_nonexistent(
            nonexistent_dt, "America/New_York", strategy="shift_forward"
        )
        assert res_fwd.hour == 3
        assert res_fwd.minute == 15
        assert res_fwd.tzinfo == ZoneInfo("America/New_York")
        assert res_fwd.utcoffset() == timedelta(hours=-4)  # EDT

        res_back = dst_handler.resolve_nonexistent(
            nonexistent_dt, "America/New_York", strategy="shift_backward"
        )
        assert res_back.hour == 1
        assert res_back.minute == 15
        assert res_back.tzinfo == ZoneInfo("America/New_York")
        assert res_back.utcoffset() == timedelta(hours=-5)  # EST

    def test_resolve_nonexistent_invalid_strategy_raises_value_error(
        self, dst_handler: DSTHandler
    ):
        nonexistent_dt = datetime(2023, 3, 12, 2, 30, 0)
        with pytest.raises(ValueError):
            dst_handler.resolve_nonexistent(
                nonexistent_dt, "America/New_York", strategy="invalid_strategy"
            )

    def test_european_spring_forward_transition(
        self, normalizer: TimezoneNormalizer, dst_handler: DSTHandler
    ):
        # Europe/Berlin on 2023-03-26: 02:00 CET (UTC+1) jumps to 03:00 CEST (UTC+2)
        dt = datetime(2023, 3, 26, 2, 45, 0)
        assert dst_handler.is_nonexistent(dt, "Europe/Berlin") is True

        fwd_result = normalizer.normalize(
            dt,
            target_tz="UTC",
            source_tz="Europe/Berlin",
            nonexistent="shift_forward",
        )
        # 02:45 shifted forward -> 03:45 CEST (UTC+2) -> 01:45 UTC
        assert fwd_result == datetime(2023, 3, 26, 1, 45, 0, tzinfo=timezone.utc)


# ============================================================================
# Validation, Error Handling, and Edge Cases
# ============================================================================


class TestTimezoneValidationAndEdgeCases:
    """Tests covering invalid inputs, custom exceptions, and edge cases."""

    def test_invalid_target_timezone_raises_error(
        self, normalizer: TimezoneNormalizer
    ):
        dt = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        with pytest.raises(InvalidTimezoneError):
            normalizer.normalize(dt, target_tz="Invalid/Nonexistent_Timezone")

    def test_invalid_source_timezone_raises_error(
        self, normalizer: TimezoneNormalizer
    ):
        naive_dt = datetime(2023, 1, 1, 12, 0, 0)
        with pytest.raises(InvalidTimezoneError):
            normalizer.normalize(
                naive_dt, target_tz="UTC", source_tz="Bad/Timezone"
            )

    def test_empty_timezone_string_raises_error(
        self, normalizer: TimezoneNormalizer
    ):
        dt = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        with pytest.raises(InvalidTimezoneError):
            normalizer.normalize(dt, target_tz="")

    def test_non_datetime_input_raises_type_error(
        self, normalizer: TimezoneNormalizer
    ):
        with pytest.raises(TypeError):
            normalizer.normalize("2023-01-01 12:00:00", target_tz="UTC")  # type: ignore

    def test_custom_exceptions_inherit_from_value_error(self):
        assert issubclass(AmbiguousTimeError, ValueError)
        assert issubclass(NonExistentTimeError, ValueError)
        assert issubclass(InvalidTimezoneError, ValueError)

    def test_normalizer_configured_with_default_target_tz(self):
        tokyo_normalizer = TimezoneNormalizer(default_target_tz="Asia/Tokyo")
        aware_utc = datetime(2023, 5, 1, 0, 0, 0, tzinfo=timezone.utc)

        # Uses default target when not explicitly specified
        result = tokyo_normalizer.normalize(aware_utc)

        expected = datetime(2023, 5, 1, 9, 0, 0, tzinfo=ZoneInfo("Asia/Tokyo"))
        assert result == expected