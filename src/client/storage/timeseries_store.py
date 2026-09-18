"""Client-side time-series cache backed by IndexedDB storage."""

from dataclasses import dataclass
import heapq
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

from src.client.storage.indexeddb_cache import IndexedDBCache


@dataclass
class MetricPoint:
    """Representation of a single time-series metric data point."""

    metric_name: str
    timestamp: float
    value: float
    tags: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.tags is not None:
            self.tags = dict(self.tags)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MetricPoint":
        """Construct MetricPoint from a dictionary representation."""
        raw_tags = data.get("tags")
        return cls(
            metric_name=str(data["metric_name"]),
            timestamp=float(data["timestamp"]),
            value=float(data.get("value", 0.0)),
            tags=dict(raw_tags) if raw_tags is not None else None,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert MetricPoint to a dictionary."""
        d: Dict[str, Any] = {
            "metric_name": self.metric_name,
            "timestamp": self.timestamp,
            "value": self.value,
        }
        if self.tags is not None:
            d["tags"] = dict(self.tags)
        return d


class TimeSeriesStore:
    """High-level client-side time-series store backed by IndexedDBCache."""

    def __init__(
        self,
        backend: IndexedDBCache,
        retention_seconds: Optional[float] = None,
        max_records: Optional[int] = None,
    ) -> None:
        self.backend = backend
        self.retention_seconds = retention_seconds
        self.max_records = max_records
        self._latest_timestamp: Optional[float] = None

    def _get_metric_timestamp_index(self) -> Optional[str]:
        """Resolve index covering (metric_name, timestamp)."""
        idx = self.backend.find_index_for_fields(("metric_name", "timestamp"))
        if idx is not None:
            return idx
        if "by_metric_timestamp" in self.backend.indexes:
            return "by_metric_timestamp"
        return None

    def _get_timestamp_index(self) -> Optional[str]:
        """Resolve index covering (timestamp,)."""
        idx = self.backend.find_index_for_fields(("timestamp",))
        if idx is not None:
            return idx
        if "by_timestamp" in self.backend.indexes:
            return "by_timestamp"
        return None

    def _normalize_point(self, item: Any) -> Dict[str, Any]:
        """Validate and normalize an incoming point into a persistable dictionary record."""
        if isinstance(item, MetricPoint):
            metric_name = item.metric_name
            timestamp = item.timestamp
            value = item.value
            tags = item.tags
            extra_fields: Dict[str, Any] = {}
        elif isinstance(item, dict):
            if "metric_name" not in item:
                raise ValueError("Point missing required composite key field 'metric_name'")
            if "timestamp" not in item:
                raise ValueError("Point missing required composite key field 'timestamp'")
            metric_name = item["metric_name"]
            timestamp = item["timestamp"]
            value = item.get("value", 0.0)
            tags = item.get("tags")
            extra_fields = {
                k: v for k, v in item.items() if k not in ("metric_name", "timestamp", "value", "tags")
            }
        else:
            raise ValueError(f"Invalid metric point type: {type(item)}")

        if not isinstance(metric_name, str) or not metric_name:
            raise ValueError("metric_name must be a non-empty string")

        if timestamp is None or isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)):
            raise ValueError("timestamp must be numeric")

        record: Dict[str, Any] = {
            "metric_name": metric_name,
            "timestamp": float(timestamp),
            "value": float(value),
            **extra_fields,
        }
        if tags is not None:
            record["tags"] = dict(tags)

        return record

    def write_batch(self, points: Iterable[Union[MetricPoint, Dict[str, Any]]]) -> int:
        """Store points with composite keys and automatically trigger retention/capacity pruning."""
        records = [self._normalize_point(pt) for pt in points]
        if not records:
            return 0

        self.backend.put_batch(records)

        batch_max_time = max(r["timestamp"] for r in records)
        if self._latest_timestamp is None or batch_max_time > self._latest_timestamp:
            self._latest_timestamp = batch_max_time

        if self.retention_seconds is not None or self.max_records is not None:
            self.prune(current_time=self._latest_timestamp)

        return len(records)

    def query(
        self,
        metric_name: str,
        start_time: float,
        end_time: float,
        server_fetcher: Optional[Callable[..., Any]] = None,
    ) -> List[MetricPoint]:
        """Query metric data points within [start_time, end_time] interval."""
        if not isinstance(start_time, (int, float)) or isinstance(start_time, bool):
            raise ValueError("start_time must be numeric")
        if not isinstance(end_time, (int, float)) or isinstance(end_time, bool):
            raise ValueError("end_time must be numeric")
        if start_time > end_time:
            raise ValueError(f"start_time ({start_time}) cannot be greater than end_time ({end_time})")

        metric_ts_idx = self._get_metric_timestamp_index()
        ts_idx = self._get_timestamp_index()

        if metric_ts_idx is not None:
            candidates = self.backend.get_range(
                index_name=metric_ts_idx,
                lower=(metric_name, start_time),
                upper=(metric_name, end_time),
                include_lower=True,
                include_upper=True,
            )
        elif ts_idx is not None:
            candidates = self.backend.get_range(
                index_name=ts_idx,
                lower=start_time,
                upper=end_time,
                include_lower=True,
                include_upper=True,
            )
        else:
            candidates = self.backend.get_all()

        matching = [
            r for r in candidates
            if r.get("metric_name") == metric_name
            and r.get("timestamp") is not None
            and start_time <= r["timestamp"] <= end_time
        ]

        if not matching and server_fetcher is not None:
            fetched = server_fetcher(metric_name, start_time, end_time)
            if fetched:
                self.write_batch(fetched)
                return self.query(metric_name, start_time, end_time)

        matching.sort(key=lambda r: (r["timestamp"], r.get("metric_name", "")))
        return [MetricPoint.from_dict(r) for r in matching]

    def prune(self, current_time: Optional[float] = None) -> int:
        """Prune entries exceeding retention window or max records threshold."""
        deleted_count = 0

        # Enforce retention window
        if self.retention_seconds is not None:
            ref_time = (
                current_time
                if current_time is not None
                else (self._latest_timestamp if self._latest_timestamp is not None else time.time())
            )
            cutoff = ref_time - self.retention_seconds

            ts_idx = self._get_timestamp_index()
            if ts_idx is not None:
                expired = self.backend.get_range(
                    index_name=ts_idx,
                    upper=cutoff,
                    include_upper=False,
                )
            else:
                expired = [
                    r for r in self.backend.get_all()
                    if r.get("timestamp") is not None and r["timestamp"] < cutoff
                ]

            if expired:
                deleted_count += self.backend.delete_records(expired)

        # Enforce max record capacity (FIFO eviction)
        if self.max_records is not None and self.max_records >= 0:
            current_count = self.backend.count()
            if current_count > self.max_records:
                excess = current_count - self.max_records
                ts_idx = self._get_timestamp_index()
                if ts_idx is not None:
                    oldest = self.backend.get_range(
                        index_name=ts_idx,
                        limit=excess,
                    )
                else:
                    all_records = self.backend.get_all()
                    oldest = heapq.nsmallest(
                        excess,
                        all_records,
                        key=lambda r: (
                            r.get("timestamp", float("-inf")),
                            self.backend.extract_primary_key(r),
                        ),
                    )

                if oldest:
                    deleted_count += self.backend.delete_records(oldest)

        if self.backend.count() == 0:
            self._latest_timestamp = None

        return deleted_count

    def count(self) -> int:
        """Return total number of points in store."""
        return self.backend.count()

    def clear(self) -> None:
        """Clear all stored data."""
        self.backend.clear()
        self._latest_timestamp = None