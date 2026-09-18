"""
Unit tests for TimescaleDB Distributed Hypertable deployment for OHLCV ticks.
Covers Story 11.2.1:
- Base table initialization and distributed hypertable creation on access node.
- Partitioning by time dimension and hash partitioning on symbol dimension.
- Idempotency and verification via timescaledb_information.hypertables.
"""

from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, call
import pytest

from src.db.ohlcv_schema import (
    OHLCV_TICKS_TABLE_NAME,
    get_create_ohlcv_ticks_table_ddl,
    create_ohlcv_ticks_table,
)
from src.db.timescale_cluster import (
    initialize_ohlcv_distributed_hypertable,
    is_distributed_hypertable,
    TimescaleClusterError,
)


class MockDatabaseCursor:
    """Deterministic mock for DB-API 2.0 cursor handling TimescaleDB queries."""

    def __init__(self) -> None:
        self.executed_statements: List[Tuple[str, Optional[Tuple[Any, ...]]]] = []
        self._mock_records: List[Any] = []

    def execute(self, query: str, params: Optional[Tuple[Any, ...]] = None) -> None:
        normalized_query = " ".join(query.strip().split())
        self.executed_statements.append((normalized_query, params))

    def fetchone(self) -> Optional[Tuple[Any, ...]]:
        if self._mock_records:
            return self._mock_records.pop(0)
        return None

    def fetchall(self) -> List[Tuple[Any, ...]]:
        records = list(self._mock_records)
        self._mock_records.clear()
        return records

    def set_query_result(self, record: Optional[Tuple[Any, ...]]) -> None:
        self._mock_records = [record] if record is not None else []

    def __enter__(self) -> "MockDatabaseCursor":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass


class MockDatabaseConnection:
    """Deterministic mock for DB-API 2.0 database connection."""

    def __init__(self, cursor: MockDatabaseCursor) -> None:
        self.cursor_instance = cursor
        self.commit_count = 0
        self.rollback_count = 0

    def cursor(self) -> MockDatabaseCursor:
        return self.cursor_instance

    def commit(self) -> None:
        self.commit_count += 1

    def rollback(self) -> None:
        self.rollback_count += 1


@pytest.fixture
def mock_cursor() -> MockDatabaseCursor:
    return MockDatabaseCursor()


@pytest.fixture
def mock_conn(mock_cursor: MockDatabaseCursor) -> MockDatabaseConnection:
    return MockDatabaseConnection(mock_cursor)


# ==============================================================================
# Tests for src/db/ohlcv_schema.py
# ==============================================================================


class TestOHLCVSchemaDDL:
    """Verifies DDL structure and creation logic for OHLCV ticks base table."""

    def test_default_table_name_is_ohlcv_ticks(self) -> None:
        assert OHLCV_TICKS_TABLE_NAME == "ohlcv_ticks"

    def test_ddl_contains_required_tick_columns(self) -> None:
        ddl = get_create_ohlcv_ticks_table_ddl().lower()

        required_columns = ["time", "symbol", "open", "high", "low", "close", "volume"]
        for col in required_columns:
            assert col in ddl, f"Expected column '{col}' to be defined in OHLCV ticks DDL"

        # Verify time column type supports TimescaleDB time partitioning
        assert "timestamptz" in ddl or "timestamp with time zone" in ddl

    def test_create_ohlcv_ticks_table_executes_ddl(
        self, mock_conn: MockDatabaseConnection, mock_cursor: MockDatabaseCursor
    ) -> None:
        create_ohlcv_ticks_table(mock_conn)

        assert len(mock_cursor.executed_statements) == 1
        executed_query, _ = mock_cursor.executed_statements[0]
        assert "create table" in executed_query.lower()
        assert OHLCV_TICKS_TABLE_NAME in executed_query.lower()
        assert mock_conn.commit_count == 1


# ==============================================================================
# Tests for src/db/timescale_cluster.py
# ==============================================================================


class TestIsDistributedHypertable:
    """Verifies hypertable inspection via timescaledb_information.hypertables."""

    def test_queries_timescaledb_information_hypertables_catalog(
        self, mock_conn: MockDatabaseConnection, mock_cursor: MockDatabaseCursor
    ) -> None:
        mock_cursor.set_query_result(("ohlcv_ticks", "public", True))

        result = is_distributed_hypertable(mock_conn, table_name="ohlcv_ticks", schema="public")

        assert result is True
        assert len(mock_cursor.executed_statements) == 1
        query, params = mock_cursor.executed_statements[0]
        assert "timescaledb_information.hypertables" in query.lower()
        assert "is_distributed" in query.lower()
        assert params == ("public", "ohlcv_ticks")

    def test_returns_false_when_hypertable_does_not_exist(
        self, mock_conn: MockDatabaseConnection, mock_cursor: MockDatabaseCursor
    ) -> None:
        mock_cursor.set_query_result(None)

        result = is_distributed_hypertable(mock_conn, table_name="ohlcv_ticks", schema="public")

        assert result is False

    def test_returns_false_when_table_is_hypertable_but_not_distributed(
        self, mock_conn: MockDatabaseConnection, mock_cursor: MockDatabaseCursor
    ) -> None:
        # Hypertable exists, but is_distributed is False
        mock_cursor.set_query_result(("ohlcv_ticks", "public", False))

        result = is_distributed_hypertable(mock_conn, table_name="ohlcv_ticks", schema="public")

        assert result is False


class TestInitializeOHLCVDistributedHypertable:
    """Verifies creation and idempotency of OHLCV distributed hypertable cluster."""

    def test_creates_distributed_hypertable_on_unpartitioned_table(
        self, mock_conn: MockDatabaseConnection, mock_cursor: MockDatabaseCursor
    ) -> None:
        # Setup: hypertable does not exist yet
        mock_cursor.set_query_result(None)

        result = initialize_ohlcv_distributed_hypertable(
            mock_conn,
            table_name="ohlcv_ticks",
            time_column="time",
            partition_column="symbol",
            chunk_time_interval="1 day",
            number_partitions=8,
        )

        assert result is True

        # First query checks existence in catalog, second creates hypertable
        catalog_query, catalog_params = mock_cursor.executed_statements[0]
        assert "timescaledb_information.hypertables" in catalog_query.lower()
        assert catalog_params == ("public", "ohlcv_ticks")

        create_query, _ = mock_cursor.executed_statements[1]
        normalized_create = create_query.lower()
        assert "create_distributed_hypertable" in normalized_create
        assert "ohlcv_ticks" in normalized_create
        assert "time" in normalized_create
        assert "symbol" in normalized_create
        assert "1 day" in normalized_create
        assert "8" in normalized_create
        assert mock_conn.commit_count >= 1

    def test_already_configured_with_if_not_exists_true_safely_returns_without_repartitioning(
        self, mock_conn: MockDatabaseConnection, mock_cursor: MockDatabaseCursor
    ) -> None:
        # Setup: table is already a distributed hypertable
        mock_cursor.set_query_result(("ohlcv_ticks", "public", True))

        result = initialize_ohlcv_distributed_hypertable(
            mock_conn,
            table_name="ohlcv_ticks",
            time_column="time",
            partition_column="symbol",
            chunk_time_interval="1 day",
            number_partitions=8,
            if_not_exists=True,
        )

        assert result is False
        # Must only execute the existence check
        assert len(mock_cursor.executed_statements) == 1
        query, _ = mock_cursor.executed_statements[0]
        assert "timescaledb_information.hypertables" in query.lower()
        assert "create_distributed_hypertable" not in query.lower()

    def test_already_configured_with_if_not_exists_false_raises_exception(
        self, mock_conn: MockDatabaseConnection, mock_cursor: MockDatabaseCursor
    ) -> None:
        # Setup: table is already a distributed hypertable
        mock_cursor.set_query_result(("ohlcv_ticks", "public", True))

        with pytest.raises(TimescaleClusterError):
            initialize_ohlcv_distributed_hypertable(
                mock_conn,
                table_name="ohlcv_ticks",
                time_column="time",
                partition_column="symbol",
                chunk_time_interval="1 day",
                number_partitions=8,
                if_not_exists=False,
            )

        # No hypertable creation command should be executed
        for query, _ in mock_cursor.executed_statements:
            assert "create_distributed_hypertable" not in query.lower()

    def test_supports_custom_schema_and_defaults(
        self, mock_conn: MockDatabaseConnection, mock_cursor: MockDatabaseCursor
    ) -> None:
        mock_cursor.set_query_result(None)

        result = initialize_ohlcv_distributed_hypertable(
            mock_conn,
            table_name="custom_ticks",
            schema="market_data",
            chunk_time_interval="2 hours",
            partition_column="symbol",
            number_partitions=16,
        )

        assert result is True
        catalog_query, catalog_params = mock_cursor.executed_statements[0]
        assert catalog_params == ("market_data", "custom_ticks")

        create_query, _ = mock_cursor.executed_statements[1]
        normalized = create_query.lower()
        assert "market_data.custom_ticks" in normalized or "market_data" in normalized
        assert "2 hours" in normalized
        assert "16" in normalized

    def test_invalid_number_partitions_raises_value_error(
        self, mock_conn: MockDatabaseConnection
    ) -> None:
        with pytest.raises(ValueError):
            initialize_ohlcv_distributed_hypertable(
                mock_conn,
                table_name="ohlcv_ticks",
                number_partitions=0,
            )

        with pytest.raises(ValueError):
            initialize_ohlcv_distributed_hypertable(
                mock_conn,
                table_name="ohlcv_ticks",
                number_partitions=-4,
            )

    def test_invalid_chunk_interval_raises_value_error(
        self, mock_conn: MockDatabaseConnection
    ) -> None:
        with pytest.raises(ValueError):
            initialize_ohlcv_distributed_hypertable(
                mock_conn,
                table_name="ohlcv_ticks",
                chunk_time_interval="",
            )

        with pytest.raises(ValueError):
            initialize_ohlcv_distributed_hypertable(
                mock_conn,
                table_name="ohlcv_ticks",
                chunk_time_interval="   ",
            )

    def test_missing_partition_column_raises_value_error(
        self, mock_conn: MockDatabaseConnection
    ) -> None:
        with pytest.raises(ValueError):
            initialize_ohlcv_distributed_hypertable(
                mock_conn,
                table_name="ohlcv_ticks",
                partition_column="",
            )

    def test_missing_time_column_raises_value_error(
        self, mock_conn: MockDatabaseConnection
    ) -> None:
        with pytest.raises(ValueError):
            initialize_ohlcv_distributed_hypertable(
                mock_conn,
                table_name="ohlcv_ticks",
                time_column="",
            )