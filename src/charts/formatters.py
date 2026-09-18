"""Adaptive x-axis time interval label formatters for charts."""

from collections.abc import Sequence
from datetime import datetime, timedelta

_MONTH_NAMES = (
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
)


class AdaptiveTimeIntervalFormatter:
    """Formatter that adaptively formats x-axis datetime labels based on interval range."""

    def format_labels(self, timestamps: Sequence[datetime]) -> list[str]:
        """Format a sequence of datetime objects into compact x-axis labels.

        Args:
            timestamps: Sequence of datetime objects to format.

        Returns:
            List of formatted string labels corresponding to input timestamps.

        Raises:
            TypeError: If timestamps is None, not iterable, or contains non-datetime items.
        """
        if timestamps is None:
            raise TypeError("timestamps cannot be None")

        try:
            ts_list = list(timestamps)
        except TypeError:
            raise TypeError("timestamps must be an iterable of datetime instances")

        for ts in ts_list:
            if not isinstance(ts, datetime):
                raise TypeError(f"Expected datetime instance, got {type(ts).__name__}")

        if not ts_list:
            return []

        min_ts = min(ts_list)
        max_ts = max(ts_list)
        time_range = max_ts - min_ts

        # Criterion 1: Interval range < 24 hours -> HH:MM or HH:MM:SS
        if time_range < timedelta(days=1):
            if any(ts.second != 0 or ts.microsecond != 0 for ts in ts_list):
                return [ts.strftime("%H:%M:%S") for ts in ts_list]
            return [ts.strftime("%H:%M") for ts in ts_list]

        # Criterion 3: Interval range exceeding 365 days -> YYYY or YYYY-MM
        if time_range > timedelta(days=365):
            year_labels = [ts.strftime("%Y") for ts in ts_list]
            if len(set(year_labels)) == len(set(ts_list)):
                return year_labels
            return [ts.strftime("%Y-%m") for ts in ts_list]

        # Criterion 2: Interval range between 1 day and 30 days (up to 365 days) -> MMM DD
        return [f"{_MONTH_NAMES[ts.month - 1]} {ts.day:02d}" for ts in ts_list]