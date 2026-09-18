"""
Unit tests for Client-Side IndexedDB Time-Series Cache.

Requirements covered:
- AC 1: Persist time-series points with composite keys (metric_name, timestamp)
        and indexed for range scanning.
- AC 2: Query metrics within [t_start, t_end] interval served directly from
        local store without server requests.
- AC 3: Prune entries older than retention threshold or exceeding max records
        triggered on write or explicit pruning operations.
"""

from typing import Any, Callable, Dict, List
from unittest.mock import Mock
import pytest

from src.client.storage.indexeddb_cache import IndexedDBCache
from src.client.storage.timeseries_store import MetricPoint, TimeSeriesStore


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def idb_cache() -> IndexedDBCache:
    """Fixture providing a fresh low-level IndexedDBCache instance."""
    cache = IndexedDBCache(
        db_name="test_telemetry_db",
        store_name="metrics_store",
        key_path=("metric_name", "timestamp"),
        indexes={
            "by_timestamp": ("timestamp",),
            "by_metric_timestamp": ("metric_name", "timestamp"),
        },
    )
    cache.clear()
    return cache


@pytest.fixture
def store(idb_cache: IndexedDBCache) -> TimeSeriesStore:
    """Fixture providing a TimeSeriesStore backed by the IndexedDBCache."""
    return TimeSeriesStore(backend=idb_cache)


@pytest.fixture
def sample_metric_points() -> List[MetricPoint]:
    """Fixture providing deterministic metric points for testing."""
    return [
        MetricPoint(
            metric_name="system.cpu.usage",
            timestamp=1700000000.0,
            value=45.2,
            tags={"host": "app-server-01", "region": "us-east-1"},
        ),
        MetricPoint(
            metric_name="system.cpu.usage",
            timestamp=1700000010.0,
            value=52.1,
            tags={"host": "app-server-01", "region": "us-east-1"},
        ),
        MetricPoint(
            metric_name="system.cpu.usage",
            timestamp=1700000020.0,
            value=60.8,
            tags={"host": "app-server-01", "region": "us-east-1"},
        ),
        MetricPoint(
            metric_name="system.memory.free",
            timestamp=1700000000.0,
            value=1024.0,
            tags={"host": "app-server-01"},
        ),
        MetricPoint(
            metric_name="system.memory.free",
            timestamp=1700000020.0,
            value=950.0,
            tags={"host": "app-server-01"},
        ),
    ]


# ============================================================================
# AC 1: Persistence with Composite Keys & Indexing for Range Scanning
# ============================================================================

class TestIndexedDBPersistenceAndIndexing:
    """Tests for composite key storage and indexing in IndexedDB cache."""

    def test_low_level_cache_stores_records_with_composite_key(
        self, idb_cache: IndexedDBCache
    ) -> None:
        """Verify IndexedDBCache persists records identified by (metric_name, timestamp)."""
        record = {
            "metric_name": "network.ingress",
            "timestamp": 1700000100.0,
            "value": 1500.0,
        }
        idb_cache.put(record)

        composite_key = ("network.ingress", 1700000100.0)
        retrieved = idb_cache.get(composite_key)

        assert retrieved is not None
        assert retrieved["metric_name"] == "network.ingress"
        assert retrieved["timestamp"] == 1700000100.0
        assert retrieved["value"] == 1500.0

    def test_write_batch_persists_all_metric_points(
        self, store: TimeSeriesStore, sample_metric_points: List[MetricPoint]
    ) -> None:
        """Verify write_batch stores all points and returns total written count."""
        written_count = store.write_batch(sample_metric_points)

        assert written_count == len(sample_metric_points)
        assert store.count() == len(sample_metric_points)

    def test_write_batch_stores_dict_or_metric_point_objects(
        self, store: TimeSeriesStore
    ) -> None:
        """Verify write_batch supports both MetricPoint objects and raw dictionaries."""
        batch: List[Any] = [
            MetricPoint(metric_name="disk.io", timestamp=1700000050.0, value=12.0),
            {
                "metric_name": "disk.io",
                "timestamp": 1700000060.0,
                "value": 14.5,
                "tags": {"device": "nvme0n1"},
            },
        ]
        written = store.write_batch(batch)
        assert written == 2

        results = store.query("disk.io", start_time=1700000050.0, end_time=1700000060.0)
        assert len(results) == 2
        assert results[0].value == 12.0
        assert results[1].value == 14.5

    def test_write_batch_raises_on_missing_composite_key_fields(
        self, store: TimeSeriesStore
    ) -> None:
        """Verify write_batch raises ValueError if metric_name or timestamp is missing."""
        invalid_point = {"value": 100.0}  # Missing metric_name and timestamp

        with pytest.raises(ValueError):
            store.write_batch([invalid_point])  # type: ignore

    def test_write_batch_raises_on_invalid_timestamp_type(
        self, store: TimeSeriesStore
    ) -> None:
        """Verify write_batch rejects records with non-numeric timestamps."""
        invalid_point = {
            "metric_name": "system.cpu",
            "timestamp": "invalid_timestamp",
            "value": 20.0,
        }

        with pytest.raises(ValueError):
            store.write_batch([invalid_point])  # type: ignore

    def test_write_batch_upserts_duplicate_composite_keys(
        self, store: TimeSeriesStore
    ) -> None:
        """Verify that duplicate points with the exact same composite key are upserted."""
        point_initial = MetricPoint("api.latency", 1700000000.0, 150.0)
        point_updated = MetricPoint("api.latency", 1700000000.0, 250.0)

        store.write_batch([point_initial])
        assert store.count() == 1

        store.write_batch([point_updated])
        assert store.count() == 1

        results = store.query("api.latency", 1700000000.0, 1700000000.0)
        assert len(results) == 1
        assert results[0].value == 250.0

    def test_indexed_range_scan_on_cache_backend(
        self, idb_cache: IndexedDBCache
    ) -> None:
        """Verify low-level index scan returns records ordered by indexed key."""
        records = [
            {"metric_name": "m1", "timestamp": 100.0, "val": 1},
            {"metric_name": "m1", "timestamp": 200.0, "val": 2},
            {"metric_name": "m1", "timestamp": 300.0, "val": 3},
        ]
        idb_cache.put_batch(records)

        scanned = idb_cache.get_range(
            index_name="by_timestamp",
            lower=100.0,
            upper=250.0,
            include_lower=True,
            include_upper=True,
        )

        assert len(scanned) == 2
        assert [r["timestamp"] for r in scanned] == [100.0, 200.0]


# ============================================================================
# AC 2: Cached Range Queries Served Locally Without Server Requests
# ============================================================================

class TestTimeSeriesStoreRangeQueries:
    """Tests for range querying time-series data locally within [t_start, t_end]."""

    def test_query_returns_points_within_closed_interval(
        self, store: TimeSeriesStore, sample_metric_points: List[MetricPoint]
    ) -> None:
        """Verify query returns points inside [t_start, t_end] inclusive."""
        store.write_batch(sample_metric_points)

        # Query interval [1700000000.0, 1700000010.0] for 'system.cpu.usage'
        results = store.query(
            metric_name="system.cpu.usage",
            start_time=1700000000.0,
            end_time=1700000010.0,
        )

        assert len(results) == 2
        assert results[0].timestamp == 1700000000.0
        assert results[1].timestamp == 1700000010.0
        assert all(pt.metric_name == "system.cpu.usage" for pt in results)

    def test_query_excludes_points_strictly_outside_interval(
        self, store: TimeSeriesStore, sample_metric_points: List[MetricPoint]
    ) -> None:
        """Verify points before t_start and after t_end are strictly excluded."""
        store.write_batch(sample_metric_points)

        # Query sub-interval [1700000005.0, 1700000015.0]
        # Only 1700000010.0 should match
        results = store.query(
            metric_name="system.cpu.usage",
            start_time=1700000005.0,
            end_time=1700000015.0,
        )

        assert len(results) == 1
        assert results[0].timestamp == 1700000010.0

    def test_query_filters_strictly_by_metric_name(
        self, store: TimeSeriesStore, sample_metric_points: List[MetricPoint]
    ) -> None:
        """Verify query only returns points for the requested metric_name."""
        store.write_batch(sample_metric_points)

        results = store.query(
            metric_name="system.memory.free",
            start_time=1700000000.0,
            end_time=1700000020.0,
        )

        assert len(results) == 2
        assert all(pt.metric_name == "system.memory.free" for pt in results)
        timestamps = [pt.timestamp for pt in results]
        assert 1700000010.0 not in timestamps  # cpu had a point at 10, memory didn't

    def test_query_served_directly_without_server_request(
        self, store: TimeSeriesStore, sample_metric_points: List[MetricPoint]
    ) -> None:
        """Verify that querying cached data does NOT invoke the server fetcher callback."""
        store.write_batch(sample_metric_points)

        mock_server_fetcher: Mock = Mock()

        results = store.query(
            metric_name="system.cpu.usage",
            start_time=1700000000.0,
            end_time=1700000020.0,
            server_fetcher=mock_server_fetcher,
        )

        assert len(results) == 3
        mock_server_fetcher.assert_not_called()

    def test_query_returns_sorted_by_timestamp_ascending(
        self, store: TimeSeriesStore
    ) -> None:
        """Verify results are always returned sorted chronologically."""
        unordered_points = [
            MetricPoint("temp", 1030.0, 22.0),
            MetricPoint("temp", 1010.0, 20.0),
            MetricPoint("temp", 1040.0, 23.0),
            MetricPoint("temp", 1020.0, 21.0),
        ]
        store.write_batch(unordered_points)

        results = store.query("temp", 1000.0, 1050.0)

        timestamps = [pt.timestamp for pt in results]
        assert timestamps == [1010.0, 1020.0, 1030.0, 1040.0]

    def test_query_empty_interval_returns_empty_list(
        self, store: TimeSeriesStore, sample_metric_points: List[MetricPoint]
    ) -> None:
        """Verify query returns empty list when no points exist in interval."""
        store.write_batch(sample_metric_points)

        results = store.query("system.cpu.usage", 1800000000.0, 1800000100.0)
        assert results == []

    def test_query_with_start_time_greater_than_end_time_raises_error(
        self, store: TimeSeriesStore
    ) -> None:
        """Verify query raises ValueError when start_time > end_time."""
        with pytest.raises(ValueError):
            store.query("system.cpu.usage", start_time=200.0, end_time=100.0)


# ============================================================================
# AC 3: Retention Policy & Pruning Operations
# ============================================================================

class TestRetentionAndPruning:
    """Tests for retention window and max-record threshold pruning."""

    def test_manual_prune_deletes_entries_older_than_retention_window(
        self, idb_cache: IndexedDBCache
    ) -> None:
        """Verify explicit prune() removes entries older than (current_time - retention_seconds)."""
        retention_seconds = 3600.0  # 1 hour
        store = TimeSeriesStore(backend=idb_cache, retention_seconds=retention_seconds)

        now = 1700003600.0  # reference time
        points = [
            # Expired: timestamp < now - retention_seconds (1700000000.0)
            MetricPoint("http.requests", 1699999990.0, 10.0),
            MetricPoint("http.requests", 1699999999.0, 20.0),
            # Valid: boundary and newer
            MetricPoint("http.requests", 1700000000.0, 30.0),
            MetricPoint("http.requests", 1700001000.0, 40.0),
        ]
        store.write_batch(points)
        assert store.count() == 4

        # Run pruning explicitly
        deleted_count = store.prune(current_time=now)

        assert deleted_count == 2
        assert store.count() == 2

        # Expired items must be gone
        expired_query = store.query("http.requests", 1699999900.0, 1699999999.0)
        assert len(expired_query) == 0

        # Valid items must remain
        valid_query = store.query("http.requests", 1700000000.0, 1700003600.0)
        assert len(valid_query) == 2

    def test_write_batch_triggers_automatic_retention_pruning(
        self, idb_cache: IndexedDBCache
    ) -> None:
        """Verify write_batch automatically prunes entries older than retention window."""
        store = TimeSeriesStore(backend=idb_cache, retention_seconds=60.0)

        t0 = 1000.0
        old_batch = [
            MetricPoint("device.battery", t0, 99.0),
            MetricPoint("device.battery", t0 + 10.0, 98.0),
        ]
        store.write_batch(old_batch)
        assert store.count() == 2

        # Writing points at t0 + 100.0; retention cutoff becomes (1000.0 + 100.0 - 60.0) = 1040.0
        new_batch = [
            MetricPoint("device.battery", t0 + 100.0, 90.0),
        ]
        store.write_batch(new_batch)

        # Points at t0 (1000.0) and t0 + 10.0 (1010.0) should be pruned automatically
        assert store.count() == 1
        remaining = store.query("device.battery", 0.0, 2000.0)
        assert len(remaining) == 1
        assert remaining[0].timestamp == t0 + 100.0

    def test_pruning_enforces_max_record_threshold_fifo(
        self, idb_cache: IndexedDBCache
    ) -> None:
        """Verify that when max_records is exceeded, oldest entries are evicted."""
        max_capacity = 5
        store = TimeSeriesStore(backend=idb_cache, max_records=max_capacity)

        points = [
            MetricPoint("db.connections", 1000.0 + i, float(i))
            for i in range(10)  # 10 records, exceeds capacity of 5
        ]
        store.write_batch(points)

        # Cache must not exceed max_records
        assert store.count() <= max_capacity

        # Remaining records must be the 5 newest points: timestamps 1005 to 1009
        results = store.query("db.connections", 1000.0, 1010.0)
        assert len(results) == max_capacity
        assert [pt.timestamp for pt in results] == [1005.0, 1006.0, 1007.0, 1008.0, 1009.0]

    def test_prune_with_both_retention_and_max_records(
        self, idb_cache: IndexedDBCache
    ) -> None:
        """Verify store enforces both retention window and max record cap together."""
        store = TimeSeriesStore(
            backend=idb_cache,
            retention_seconds=50.0,
            max_records=3,
        )

        now = 1000.0
        points = [
            MetricPoint("net.err", 900.0, 1.0),  # Expired (> 50s old relative to 1000.0)
            MetricPoint("net.err", 960.0, 2.0),  # Valid retention
            MetricPoint("net.err", 970.0, 3.0),  # Valid retention
            MetricPoint("net.err", 980.0, 4.0),  # Valid retention
            MetricPoint("net.err", 990.0, 5.0),  # Valid retention
            MetricPoint("net.err", 1000.0, 6.0), # Valid retention
        ]
        store.write_batch(points)

        # After retention: 5 points remain (960, 970, 980, 990, 1000)
        # After max_records=3: only 3 newest points remain (980, 990, 1000)
        assert store.count() == 3

        results = store.query("net.err", 900.0, 1000.0)
        assert len(results) == 3
        assert [pt.timestamp for pt in results] == [980.0, 990.0, 1000.0]

    def test_pruning_multi_metric_data_preserves_unexpired_across_metrics(
        self, idb_cache: IndexedDBCache
    ) -> None:
        """Verify pruning applies globally across multiple metric series."""
        store = TimeSeriesStore(backend=idb_cache, retention_seconds=100.0)

        now = 1000.0
        points = [
            # Expired
            MetricPoint("metric.a", 850.0, 1.0),
            MetricPoint("metric.b", 850.0, 10.0),
            # Unexpired
            MetricPoint("metric.a", 950.0, 2.0),
            MetricPoint("metric.b", 950.0, 20.0),
        ]
        store.write_batch(points)
        deleted = store.prune(current_time=now)

        assert deleted == 2
        assert len(store.query("metric.a", 0.0, 2000.0)) == 1
        assert len(store.query("metric.b", 0.0, 2000.0)) == 1

    def test_pruning_on_empty_store_returns_zero(
        self, store: TimeSeriesStore
    ) -> None:
        """Verify prune on an empty store completes cleanly returning 0."""
        deleted = store.prune(current_time=1700000000.0)
        assert deleted == 0
        assert store.count() == 0