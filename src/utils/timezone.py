"""Timezone normalization and Daylight Saving Time (DST) handling utilities."""

from datetime import datetime, timedelta, timezone, tzinfo
from typing import Optional, Union
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

__all__ = [
    "AmbiguousTimeError",
    "DSTHandler",
    "InvalidTimezoneError",
    "NonExistentTimeError",
    "TimezoneNormalizer",
]


class TimezoneError(ValueError):
    """Base exception for timezone-related errors."""


class InvalidTimezoneError(TimezoneError):
    """Raised when an invalid timezone identifier is provided."""


class AmbiguousTimeError(TimezoneError):
    """Raised when a local datetime is ambiguous during a DST fall-back transition."""


class NonExistentTimeError(TimezoneError):
    """Raised when a local datetime does not exist during a DST spring-forward transition."""


def _get_tz(tz: Union[str, tzinfo]) -> tzinfo:
    """Resolve a timezone identifier or tzinfo instance to a tzinfo object."""
    if isinstance(tz, tzinfo):
        return tz
    if not isinstance(tz, str) or not tz.strip():
        raise InvalidTimezoneError(f"Invalid timezone identifier: {tz!r}")
    if tz == "UTC":
        return timezone.utc
    try:
        return ZoneInfo(tz)
    except (ZoneInfoNotFoundError, KeyError, ValueError) as exc:
        raise InvalidTimezoneError(f"Invalid timezone identifier: {tz!r}") from exc


class DSTHandler:
    """Detects and resolves daylight saving time transition anomalies."""

    def is_ambiguous(self, dt: datetime, tz: Union[str, tzinfo]) -> bool:
        """Check if a datetime falls within a repeated hour (fall-back transition)."""
        if not isinstance(dt, datetime):
            raise TypeError(f"Expected datetime object, got {type(dt).__name__}")
        tz_obj = _get_tz(tz)
        dt_naive = dt.replace(tzinfo=None)
        off0 = dt_naive.replace(tzinfo=tz_obj, fold=0).utcoffset()
        off1 = dt_naive.replace(tzinfo=tz_obj, fold=1).utcoffset()
        if off0 is None or off1 is None or off0 == off1:
            return False
        return off0 > off1

    def is_nonexistent(self, dt: datetime, tz: Union[str, tzinfo]) -> bool:
        """Check if a datetime falls within a skipped hour (spring-forward transition)."""
        if not isinstance(dt, datetime):
            raise TypeError(f"Expected datetime object, got {type(dt).__name__}")
        tz_obj = _get_tz(tz)
        dt_naive = dt.replace(tzinfo=None)
        off0 = dt_naive.replace(tzinfo=tz_obj, fold=0).utcoffset()
        off1 = dt_naive.replace(tzinfo=tz_obj, fold=1).utcoffset()
        if off0 is None or off1 is None or off0 == off1:
            return False
        return off0 < off1

    def resolve_ambiguous(
        self,
        dt: datetime,
        tz: Union[str, tzinfo],
        strategy: str = "earliest",
    ) -> datetime:
        """Resolve an ambiguous datetime using the specified strategy ('earliest' or 'latest')."""
        if not isinstance(dt, datetime):
            raise TypeError(f"Expected datetime object, got {type(dt).__name__}")
        if strategy not in ("earliest", "latest"):
            raise ValueError(
                f"Invalid ambiguous strategy: {strategy!r}. Must be 'earliest' or 'latest'."
            )
        tz_obj = _get_tz(tz)
        fold = 0 if strategy == "earliest" else 1
        return dt.replace(tzinfo=tz_obj, fold=fold)

    def resolve_nonexistent(
        self,
        dt: datetime,
        tz: Union[str, tzinfo],
        strategy: str = "shift_forward",
    ) -> datetime:
        """Resolve a nonexistent datetime by shifting forward or backward across the DST gap."""
        if not isinstance(dt, datetime):
            raise TypeError(f"Expected datetime object, got {type(dt).__name__}")
        if strategy not in ("shift_forward", "shift_backward"):
            raise ValueError(
                f"Invalid nonexistent strategy: {strategy!r}. Must be 'shift_forward' or 'shift_backward'."
            )
        tz_obj = _get_tz(tz)
        dt_naive = dt.replace(tzinfo=None)
        off0 = dt_naive.replace(tzinfo=tz_obj, fold=0).utcoffset()
        off1 = dt_naive.replace(tzinfo=tz_obj, fold=1).utcoffset()
        if off0 is not None and off1 is not None and off1 != off0:
            gap = abs(off1 - off0)
        else:
            gap = timedelta(hours=1)

        if strategy == "shift_forward":
            return (dt_naive + gap).replace(tzinfo=tz_obj)
        return (dt_naive - gap).replace(tzinfo=tz_obj)


class TimezoneNormalizer:
    """Normalizes naive and timezone-aware datetimes to target timezones."""

    def __init__(
        self,
        default_target_tz: Optional[Union[str, tzinfo]] = None,
        dst_handler: Optional[DSTHandler] = None,
    ) -> None:
        self.default_target_tz = (
            _get_tz(default_target_tz) if default_target_tz is not None else None
        )
        self.dst_handler = dst_handler or DSTHandler()

    def normalize(
        self,
        dt: datetime,
        target_tz: Optional[Union[str, tzinfo]] = None,
        source_tz: Optional[Union[str, tzinfo]] = None,
        ambiguous: str = "raise",
        nonexistent: str = "raise",
    ) -> datetime:
        """
        Normalize a datetime to a target timezone.

        Handles naive and timezone-aware datetimes, disambiguating or adjusting
        for daylight saving time transitions according to the specified strategies.
        """
        if not isinstance(dt, datetime):
            raise TypeError(f"Expected datetime object, got {type(dt).__name__}")

        if target_tz is None:
            if self.default_target_tz is None:
                raise InvalidTimezoneError("Target timezone must be specified.")
            target_zone = self.default_target_tz
        else:
            target_zone = _get_tz(target_tz)

        source_zone = _get_tz(source_tz) if source_tz is not None else None

        if dt.tzinfo is None:
            effective_source = source_zone if source_zone is not None else timezone.utc

            if self.dst_handler.is_ambiguous(dt, effective_source):
                if ambiguous == "raise":
                    raise AmbiguousTimeError(
                        f"Ambiguous local datetime: {dt} in timezone {source_tz or 'UTC'}"
                    )
                localized_dt = self.dst_handler.resolve_ambiguous(
                    dt, effective_source, strategy=ambiguous
                )
            elif self.dst_handler.is_nonexistent(dt, effective_source):
                if nonexistent == "raise":
                    raise NonExistentTimeError(
                        f"Nonexistent local datetime: {dt} in timezone {source_tz or 'UTC'}"
                    )
                localized_dt = self.dst_handler.resolve_nonexistent(
                    dt, effective_source, strategy=nonexistent
                )
            else:
                localized_dt = dt.replace(tzinfo=effective_source)
        else:
            if self.dst_handler.is_nonexistent(dt, dt.tzinfo):
                if nonexistent == "raise":
                    raise NonExistentTimeError(
                        f"Nonexistent local datetime: {dt} in timezone {dt.tzinfo}"
                    )
                localized_dt = self.dst_handler.resolve_nonexistent(
                    dt, dt.tzinfo, strategy=nonexistent
                )
            else:
                localized_dt = dt

        return localized_dt.astimezone(target_zone)