"""TimescaleDB distributed hypertable deployment and cluster management for OHLCV ticks."""

from typing import Any

from src.db.ohlcv_schema import OHLCV_TICKS_TABLE_NAME


class TimescaleClusterError(Exception):
    """Exception raised when TimescaleDB cluster operations fail."""


def is_distributed_hypertable(
    conn: Any,
    table_name: str = OHLCV_TICKS_TABLE_NAME,
    schema: str = "public",
) -> bool:
    """Check whether a table is already configured as a distributed hypertable.

    Queries the timescaledb_information.hypertables catalog view.

    Args:
        conn: DB-API 2.0 database connection to access node.
        table_name: Name of the target table.
        schema: Schema of the target table.

    Returns:
        True if the table exists as a distributed hypertable, False otherwise.
    """
    if not table_name or not table_name.strip():
        raise ValueError("table_name cannot be empty")
    if not schema or not schema.strip():
        raise ValueError("schema cannot be empty")

    query = (
        "SELECT hypertable_name, hypertable_schema, is_distributed "
        "FROM timescaledb_information.hypertables "
        "WHERE hypertable_schema = %s AND hypertable_name = %s"
    )
    with conn.cursor() as cursor:
        cursor.execute(query, (schema.strip(), table_name.strip()))
        row = cursor.fetchone()
        if not row:
            return False
        for val in row:
            if isinstance(val, bool):
                return val
        return bool(row[-1])


def initialize_ohlcv_distributed_hypertable(
    conn: Any,
    table_name: str = OHLCV_TICKS_TABLE_NAME,
    schema: str = "public",
    time_column: str = "time",
    partition_column: str = "symbol",
    chunk_time_interval: str = "1 day",
    number_partitions: int = 8,
    if_not_exists: bool = True,
) -> bool:
    """Initialize a TimescaleDB distributed hypertable across cluster data nodes.

    Configures time dimension partitioning on `time_column` and hash space
    partitioning on `partition_column` for distributed multi-node storage.

    Args:
        conn: DB-API 2.0 connection to the TimescaleDB access node.
        table_name: Name of the base OHLCV ticks table.
        schema: Schema containing the table.
        time_column: Column name for time dimension partitioning.
        partition_column: Column name for hash space partitioning.
        chunk_time_interval: Interval for time chunk partitioning (e.g., '1 day').
        number_partitions: Number of hash partitions across data nodes.
        if_not_exists: If True, return False when already distributed; if False, raise error.

    Returns:
        True if the distributed hypertable was newly created, False if it already existed.

    Raises:
        ValueError: If partition parameters or column names are invalid.
        TimescaleClusterError: If already distributed and if_not_exists is False.
    """
    if not table_name or not table_name.strip():
        raise ValueError("table_name cannot be empty")
    if not schema or not schema.strip():
        raise ValueError("schema cannot be empty")
    if not time_column or not time_column.strip():
        raise ValueError("time_column cannot be empty")
    if not partition_column or not partition_column.strip():
        raise ValueError("partition_column cannot be empty")
    if not chunk_time_interval or not chunk_time_interval.strip():
        raise ValueError("chunk_time_interval cannot be empty")
    if number_partitions is None or number_partitions <= 0:
        raise ValueError("number_partitions must be a positive integer greater than 0")

    clean_table = table_name.strip()
    clean_schema = schema.strip()
    clean_time = time_column.strip()
    clean_partition = partition_column.strip()
    clean_interval = chunk_time_interval.strip()

    if is_distributed_hypertable(conn, table_name=clean_table, schema=clean_schema):
        if if_not_exists:
            return False
        raise TimescaleClusterError(
            f"Table '{clean_schema}.{clean_table}' is already configured as a distributed hypertable."
        )

    qualified_table = f"{clean_schema}.{clean_table}"
    create_sql = (
        f"SELECT create_distributed_hypertable("
        f"'{qualified_table}', "
        f"'{clean_time}', "
        f"partitioning_column => '{clean_partition}', "
        f"number_partitions => {number_partitions}, "
        f"chunk_time_interval => INTERVAL '{clean_interval}'"
        f");"
    )

    try:
        with conn.cursor() as cursor:
            cursor.execute(create_sql)
        conn.commit()
    except Exception:
        conn.rollback()
        raise

    return True